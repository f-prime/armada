from __future__ import annotations
import base64
import contextlib
import hashlib
import json
import re
import secrets
import struct
from collections.abc import Iterator
from typing import Any, BinaryIO, cast
import gguf
import numpy as np
from .residency import PackedTensor, Weights
from .common import catalog, check, home, ident, open_source, readn, rpc, state
from .schema import Fields, FloatArray, Model, Payload, QwenConfig, Tensor
from .bpe import gg_tokenizer
from .kv import KV, attention
from .progress import Progress
import math


def gg_header(f: BinaryIO) -> tuple[Fields, list[Tensor]]:
    import gguf

    check(readn(f, 4) == b"GGUF", "not a GGUF file")
    version, nt, nk = struct.unpack("<IQQ", readn(f, 20))
    check(
        version == 3 and nt < 100000 and nk < 100000, "expected little-endian GGUF v3"
    )
    pos = 24

    def take(n: int) -> bytes:
        nonlocal pos
        pos += n
        return readn(f, n)

    def num(fmt: str) -> Any:
        return struct.unpack("<" + fmt, take(struct.calcsize("<" + fmt)))[0]

    def string() -> str:
        n = num("Q")
        check(n < 64 * 2**20, "oversized GGUF string")
        return take(n).decode("utf8")

    def value(t: int) -> Any:
        if t == 8:
            return string()
        if t == 9:
            subtype, length = num("I"), num("Q")
            check(subtype != 9 and length < 4000000, "invalid GGUF array")
            return [value(subtype) for _ in range(length)]
        return num(("B", "b", "H", "h", "I", "i", "f", "?", "", "", "Q", "q", "d")[t])

    meta = {}
    for _ in range(nk):
        key = string()
        meta[key] = value(num("I"))
    tensors = []
    for _ in range(nt):
        name, rank = string(), num("I")
        check(1 <= rank <= 3, f"unsupported rank for {name}")
        shape = [num("Q") for _ in range(rank)][::-1]
        qt, offset = gguf.GGMLQuantizationType(num("I")), num("Q")
        check(
            qt.name
            in {
                "F32",
                "F16",
                "BF16",
                "Q8_0",
                "Q4_0",
                "Q4_1",
                "Q5_0",
                "Q5_1",
                "Q2_K",
                "Q3_K",
                "Q4_K",
                "Q5_K",
                "Q6_K",
            },
            f"{name}: unsupported {qt.name}",
        )
        block, size = gguf.GGML_QUANT_SIZES[qt]
        check(
            min(shape) > 0 and shape[-1] % block == 0, f"invalid dimensions for {name}"
        )
        tensors.append((offset, name, shape, int(qt), math.prod(shape) // block * size))
    align = meta.get("general.alignment", 32)
    check(0 < align <= 65536, "invalid GGUF alignment")
    take((-pos) % align)
    return meta, sorted(tensors)


def gg_config(meta: Fields) -> QwenConfig:
    arch = meta.get("general.architecture")
    check(arch in ("qwen3", "qwen35", "qwen3moe", "llama"), f"unsupported GGUF architecture: {arch}")
    moe = arch == "qwen3moe"
    prefix = f"{arch}."
    p = {k.removeprefix(prefix): v for k, v in meta.items() if k.startswith(prefix)}
    d, heads = p["embedding_length"], p["attention.head_count"]
    hd = p.get("attention.key_length", d // heads)
    c = QwenConfig(
        dim=d,
        heads=heads,
        hd=hd,
        kv=p["attention.head_count_kv"],
        layers=p["block_count"],
        experts=p["expert_count"] if moe else 0,
        top=p["expert_used_count"] if moe else 0,
        hidden=p["expert_feed_forward_length" if moe else "feed_forward_length"],
        ctx=p["context_length"],
        eps=p["attention.layer_norm_rms_epsilon"],
        theta=p.get("rope.freq_base", 10000.0),
        scale=p.get("expert_weights_scale", 1.0),
        vocab=len(meta["tokenizer.ggml.tokens"]),
    )
    if arch == "qwen35":
        interval = p.get("full_attention_interval", 4)
        check(interval > 0, "invalid full attention interval")
        c["recurrent"] = p.get("attention.recurrent_layers", [(i+1) % interval != 0 for i in range(c["layers"])])
        c["rotary"] = p["rope.dimension_count"]
        c["ssm_k_heads"] = p["ssm.group_count"]
        c["ssm_heads"], c["ssm_dim"], c["conv"] = p["ssm.time_step_rank"], p["ssm.state_size"], p["ssm.conv_kernel"]
        check(len(c["recurrent"]) == c["layers"] and 0 < c["rotary"] <= hd and not c["rotary"] % 2
              and min(c["ssm_k_heads"], c["ssm_heads"], c["ssm_dim"], c["conv"]) > 0
              and c["ssm_heads"] % c["ssm_k_heads"] == 0
              and p["ssm.inner_size"] == c["ssm_heads"] * c["ssm_dim"]
              and not p.get("nextn_predict_layers"), "unsupported Qwen3.5 recurrent or MTP layout")
    if arch == "llama":
        check(meta.get("tokenizer.ggml.pre") == "llama-bpe" and not p.get("expert_count"),
              "expected a dense Llama 3 GGUF with llama-bpe tokenizer")
        c["llama3"] = True
    check(
        hd % 2 == 0
        and heads % c["kv"] == 0
        and (not moe or 0 < c["top"] <= c["experts"])
        and p.get("attention.value_length", hd) == hd
        and p.get("rope.dimension_count", hd) == c.get("rotary", hd)
        and p.get("rope.scaling.type", "none") == "none"
        and p.get("expert_weights_norm", True)
        and not p.get("attention.sliding_window"),
        "unsupported attention, RoPE scaling or expert configuration",
    )
    return c


def gg_shapes(c: QwenConfig) -> Iterator[tuple[str, list[int], int]]:
    d, hd, heads, kv, e, h, v = (
        c["dim"],
        c["hd"],
        c["heads"],
        c["kv"],
        c["experts"],
        c["hidden"],
        c["vocab"],
    )
    yield "token_embd.weight", [v, d], 0
    yield "output_norm.weight", [d], c["layers"] - 1
    yield "output.weight", [v, d], c["layers"] - 1
    ffn = (
        dict(ffn_gate_inp=[e, d], ffn_gate_exps=[e, h, d],
             ffn_up_exps=[e, h, d], ffn_down_exps=[e, d, h])
        if e else dict(ffn_gate=[h, d], ffn_up=[h, d], ffn_down=[d, h])
    )
    for i in range(c["layers"]):
        if "recurrent" in c and c["recurrent"][i]:
            heads_ssm, width = c["ssm_heads"], c["ssm_dim"]
            inner = heads_ssm * width
            channels = inner + 2 * c["ssm_k_heads"] * width
            for key, shape in dict(attn_norm=[d], post_attention_norm=[d], attn_gate=[inner,d],
                                   attn_qkv=[channels,d], ssm_alpha=[heads_ssm,d], ssm_beta=[heads_ssm,d],
                                   ssm_conv1d=[channels,c["conv"]], ssm_norm=[width],
                                   ssm_out=[d,inner], **ffn).items():
                yield f"blk.{i}.{key}.weight", shape, i
            yield f"blk.{i}.ssm_a", [heads_ssm], i
            yield f"blk.{i}.ssm_dt.bias", [heads_ssm], i
            continue
        for key, shape in dict(
            attn_norm=[d],
            attn_q=[heads * hd * (2 if "recurrent" in c else 1), d],
            attn_k=[kv * hd, d],
            attn_v=[kv * hd, d],
            **({} if c.get("llama3") else dict(attn_q_norm=[hd], attn_k_norm=[hd])),
            attn_output=[d, heads * hd],
            **{"post_attention_norm" if "recurrent" in c else "ffn_norm": [d]},
            **ffn,
        ).items():
            yield f"blk.{i}.{key}.weight", shape, i


def gg_parts(source: str) -> list[str]:
    m = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})(\.gguf)(\?.*)?", source, re.I)
    if not m:
        return [source]
    check(int(m[2]) == 1 and int(m[3]) > 0, "specify the first GGUF shard")
    return [
        f"{m[1]}-{i:05}-of-{m[3]}{m[4]}{m[5] or ''}" for i in range(1, int(m[3]) + 1)
    ]


from .mesh import online, publish
from .placement import location, verify


def import_gg(name: str, source: str, *, exclude_self: bool = False,
              progress: Progress | None = None) -> Model:
    ident(name)
    s = state()
    cat = catalog(s)
    check(name not in cat["models"], "model name already exists")
    uid = secrets.token_hex(12)
    parts = gg_parts(source)
    seen = set()
    manifests: dict[str, dict[str, tuple[str, list[int], int]]] = {}
    try:
        for part, path in enumerate(parts):
            if progress and len(parts) > 1:
                progress.phase(f"Preparing part {part + 1}/{len(parts)}…")
            with open_source(path) as f:
                meta, tensors = gg_header(f)
                check(
                    meta.get("split.count", 1) == len(parts)
                    and meta.get("split.no", 0) == part,
                    "GGUF split metadata does not match filenames",
                )
                if part == 0:
                    c = gg_config(meta)
                    nodes = online(exclude_self)[: c["layers"]]
                    for node in nodes:
                        rpc(node, "qwen35-ready" if "recurrent" in c else "llama3-ready" if c.get("llama3") else
                            "qwen3-ready" if not c["experts"] else "qwen-ready")
                    owners = [
                        nodes[i * len(nodes) // c["layers"]] for i in range(c["layers"])
                    ]
                    expected = {
                        key: (shape, owners[i]) for key, shape, i in gg_shapes(c)
                    }
                    manifests = {node: {} for node in nodes}
                    stops = [
                        v
                        for k, v in meta.items()
                        if k
                        in (
                            "tokenizer.ggml.eos_token_id",
                            "tokenizer.ggml.eot_token_id",
                            "tokenizer.ggml.eom_token_id",
                            "tokenizer.ggml.padding_token_id",
                        )
                    ]
                    if c.get("llama3"):
                        stops += [i for i, word in enumerate(meta["tokenizer.ggml.tokens"])
                                  if word in ("<|eot_id|>", "<|eom_id|>") and i not in stops]
                    m = Model(
                        id=uid,
                        backend=meta["general.architecture"],
                        qwen=c,
                        owners=owners,
                        stop=stops,
                        source=location(source),
                    )
                    tok = gg_tokenizer(meta)
                    rpc(
                        owners[0],
                        f"weight/{uid}/tokenizer/{hashlib.sha256(tok).hexdigest()}",
                        body=tok,
                    )
                pos = 0
                needs = dict.fromkeys(nodes, 0)
                for _, name_, shape, qt, size in tensors:
                    check(
                        name_ in expected
                        and shape == expected[name_][0]
                        and name_ not in seen,
                        f"unexpected, duplicate or misshaped tensor: {name_}",
                    )
                    seen.add(name_)
                    if name_ == "token_embd.weight":
                        embedding_size = size
                    needs[expected[name_][1]] += size
                for node, size in needs.items():
                    check(
                        rpc(node, "stats")["free_disk"] >= size,
                        f"{node}: need {size} free disk bytes",
                    )
                if progress:
                    label = f"Distributing {name} · peers: {len(nodes)}"
                    if len(parts) > 1:
                        label += f" · part {part + 1}/{len(parts)} (part ETA)"
                    progress.start(label, sum(needs.values()))
                for offset, name_, shape, qt, size in tensors:
                    check(0 <= offset - pos <= 65536, "invalid GGUF tensor offset")
                    readn(f, offset - pos)
                    node = expected[name_][1]
                    key = name_.replace(".", "_")
                    manifests[node][name_] = (key, shape, qt)
                    for start in range(0, size, 8 * 2**20):
                        b = readn(f, min(8 * 2**20, size - start))
                        rpc(
                            node,
                            f"chunk/{uid}/{key}/{start}/{hashlib.sha256(b).hexdigest()}",
                            body=b,
                        )
                        if progress:
                            progress.advance(len(b))
                    pos = offset + size
        # An output projection may still appear in a later split. Only infer
        # tying once all headers are accounted for.
        if not c["experts"] and "output.weight" not in seen:
            del expected["output.weight"]
            c["tied_output"] = True
        check(
            seen == set(expected), f"missing tensors: {sorted(set(expected)-seen)[:5]}"
        )
        first, last = owners[0], owners[-1]
        if c.get("tied_output") and first != last:
            if progress:
                progress.start("Copying tied output embeddings", embedding_size)
            check(rpc(last, "stats")["free_disk"] >= embedding_size,
                  f"{last}: need {embedding_size} free disk bytes for tied output")
            entry = manifests[first]["token_embd.weight"]
            key = entry[0]
            for start in range(0, embedding_size, 8 * 2**20):
                b = base64.b64decode(str(rpc(first, "read-chunk", dict(id=uid, file=key, offset=start))))
                check(len(b) == min(8 * 2**20, embedding_size - start), "incomplete embedding chunk")
                rpc(last, f"chunk/{uid}/{key}/{start}/{hashlib.sha256(b).hexdigest()}", body=b)
                if progress:
                    progress.advance(len(b))
            manifests[last]["token_embd.weight"] = entry
        if progress:
            progress.phase("Finalizing manifests…")
        for node, manifest in manifests.items():
            b = json.dumps(manifest).encode()
            rpc(node, f"weight/{uid}/manifest/{hashlib.sha256(b).hexdigest()}", body=b)
        if progress:
            progress.phase("Verifying weights…")
        verify(m, nodes, {})
    except Exception:
        for node in manifests:
            with contextlib.suppress(Exception):
                rpc(node, "discard", {"id": uid})
        raise
    if progress:
        progress.phase("Publishing model…")
    publish(name, m)
    if progress:
        progress.phase(f"{name} is ready.")
    return m


def gg_stage(
    m: Model, start: int, end: int, payload: Payload, return_logits: bool = False,
    *, weights_cache: Weights | None = None, cache: KV | None = None,
) -> FloatArray | int | list[list[float]]:
    c = m["qwen"]
    folder = home() / "weights" / ident(m["id"])
    manifest: dict[str, tuple[str, list[int], int]] = json.loads(
        (folder / "manifest.bin").read_bytes()
    ) if weights_cache is None else {}

    def tensor(name: str, suffix: str = ".weight") -> PackedTensor:
        if weights_cache is not None:
            return weights_cache[name + suffix]
        key, shape, code = manifest[name + suffix]
        return PackedTensor.load(folder / (ident(key) + ".bin"), tuple(shape), code)

    def mm(x: FloatArray, name: str, expert: int | None = None) -> FloatArray:
        return tensor(name).multiply(x, expert)

    def norm(x: FloatArray, name: str) -> FloatArray:
        scale = tensor(name).decode()
        return np.asarray(
            (x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + c["eps"]))
            * scale,
            dtype=np.float32,
        )

    if start == 0:
        idx = np.asarray(payload, dtype=np.intp)
        x = tensor("token_embd").decode(idx)
    else:
        x = np.asarray(payload, dtype="f4")
    hd, heads, kv = c["hd"], c["heads"], c["kv"]
    seq = len(x)
    offset = cache.offset if cache else 0
    rot = c.get("rotary", hd)
    pos = np.outer(np.arange(offset, offset + seq), c["theta"] ** (-np.arange(0, rot, 2) / rot)).astype(
        "f4"
    )[:, None, :]

    def rope(a: FloatArray) -> FloatArray:
        if c.get("llama3"):
            even, odd = a[..., ::2], a[..., 1::2]
            return np.stack((even * np.cos(pos) - odd * np.sin(pos),
                             even * np.sin(pos) + odd * np.cos(pos)), axis=-1).reshape(a.shape)
        # GGUF Qwen uses NeoX layout (two half-vectors), not llama2.c's interleaved pairs.
        l, r = np.split(a[..., :rot], 2, axis=-1)
        return np.concatenate(
            [l * np.cos(pos) - r * np.sin(pos), r * np.cos(pos) + l * np.sin(pos), a[..., rot:]],
            axis=-1,
        )

    def full_attention(a: FloatArray, pre: str, i: int) -> FloatArray:
        query = mm(a, pre + "attn_q").reshape(seq, heads, -1)
        q = query[..., :hd]
        k = mm(a, pre + "attn_k").reshape(seq, kv, hd)
        if not c.get("llama3"):
            q, k = norm(q, pre + "attn_q_norm"), norm(k, pre + "attn_k_norm")
        q, k = rope(q), rope(k)
        v = mm(a, pre + "attn_v").reshape(seq, kv, hd)
        if cache:
            k, v = cache.append(i, k, v, c["ctx"])
        a = attention(q, k, v)
        if "recurrent" in c:
            a *= sigmoid(query[..., hd:])
        return mm(a.reshape(seq, -1), pre + "attn_output")

    def sigmoid(a: FloatArray) -> FloatArray:
        return np.asarray(1 / (1 + np.exp(-np.clip(a, -80, 80))), dtype="f4")

    def linear_attention(a: FloatArray, pre: str, i: int) -> FloatArray:
        h, w, taps = c["ssm_heads"], c["ssm_dim"], c["conv"]
        raw = mm(a, pre + "attn_qkv")
        saved = cache.recurrent.get(i) if cache else None
        history, memory = saved if saved else (
            np.zeros((taps-1, raw.shape[1]), dtype="f4"), np.zeros((h,w,w), dtype="f4"))
        signal = np.concatenate((history, raw))
        filt = tensor(pre + "ssm_conv1d").decode()
        conv = sum((signal[j:j+seq] * filt[:,j] for j in range(1,taps)), signal[:seq] * filt[:,0])
        kh = c["ssm_k_heads"]
        q, k, v = np.split((conv * sigmoid(conv)).reshape(seq, 2*kh+h, w), [kh,2*kh], axis=1)
        q /= np.sqrt((q*q).sum(axis=-1, keepdims=True) + c["eps"]) * np.sqrt(np.float32(w))
        k /= np.sqrt((k*k).sum(axis=-1, keepdims=True) + c["eps"])
        if kh != h:
            q, k = np.tile(q, (1,h//kh,1)), np.tile(k, (1,h//kh,1))
        beta = sigmoid(mm(a, pre + "ssm_beta"))
        dt = mm(a, pre + "ssm_alpha") + tensor(pre + "ssm_dt", ".bias").decode()
        decay = np.exp(np.logaddexp(0, dt) * tensor(pre + "ssm_a", "").decode())
        out = np.empty_like(v)
        for t in range(seq):
            memory *= decay[t,:,None,None]
            residual = (v[t] - np.einsum("hi,hij->hj", k[t], memory)) * beta[t,:,None]
            memory += k[t,:,:,None] * residual[:,None,:]
            out[t] = np.einsum("hi,hij->hj", q[t], memory)
        if cache:
            cache.recurrent[i] = signal[seq:].copy(), memory
        z = mm(a, pre + "attn_gate").reshape(seq,h,w)
        gated = norm(out, pre + "ssm_norm") * z * sigmoid(z)
        return mm(gated.reshape(seq,-1), pre + "ssm_out")

    for i in range(start, end):
        pre = f"blk.{i}."
        a = norm(x, pre + "attn_norm")
        x = x + (linear_attention(a, pre, i) if "recurrent" in c and c["recurrent"][i]
                 else full_attention(a, pre, i))
        a = norm(x, pre + ("post_attention_norm" if "recurrent" in c else "ffn_norm"))
        if not c["experts"]:
            g = mm(a, pre + "ffn_gate")
            act = mm(a, pre + "ffn_up") * g / (1 + np.exp(-np.clip(g, -80, 80)))
            x += mm(act, pre + "ffn_down")
            continue
        scores = mm(a, pre + "ffn_gate_inp")
        top = np.argsort(-scores, axis=-1, kind="stable")[:, : c["top"]]
        weights = np.take_along_axis(scores, top, axis=1)
        weights = np.exp(weights - weights.max(axis=1, keepdims=True))
        weights = weights / weights.sum(axis=1, keepdims=True) * c["scale"]
        for expert in np.unique(top):
            idx, slot = np.where(top == expert)
            g = mm(a[idx], pre + "ffn_gate_exps", expert)
            up = mm(a[idx], pre + "ffn_up_exps", expert)
            act = up * g / (1 + np.exp(-np.clip(g, -80, 80)))
            x[idx] += weights[idx, slot, None] * mm(act, pre + "ffn_down_exps", expert)
    if end == c["layers"]:
        logits = mm(norm(x[-1], "output_norm"), "token_embd" if c.get("tied_output") else "output")
        return logits if return_logits else int(logits.argmax())
    return cast(list[list[float]], x.tolist())
