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
from numpy.typing import NDArray
from .common import catalog, check, home, ident, open_source, readn, rpc, state
from .schema import Fields, FloatArray, Model, Payload, QwenConfig, Tensor
from .bpe import gg_tokenizer
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
    check(arch in ("qwen3", "qwen3moe"), "expected qwen3 or qwen3moe GGUF")
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
    check(
        hd % 2 == 0
        and heads % c["kv"] == 0
        and (not moe or 0 < c["top"] <= c["experts"])
        and p.get("attention.value_length", hd) == hd
        and p.get("rope.dimension_count", hd) == hd
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
        for key, shape in dict(
            attn_norm=[d],
            attn_q=[heads * hd, d],
            attn_k=[kv * hd, d],
            attn_v=[kv * hd, d],
            attn_q_norm=[hd],
            attn_k_norm=[hd],
            attn_output=[d, heads * hd],
            ffn_norm=[d],
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
                        rpc(node, "qwen3-ready" if not c["experts"] else "qwen-ready")
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
    *, weights_cache: dict[str, FloatArray] | None = None,
) -> FloatArray | int | list[list[float]]:
    import gguf

    c = m["qwen"]
    folder = home() / "weights" / ident(m["id"])
    manifest: dict[str, tuple[str, list[int], int]] = json.loads(
        (folder / "manifest.bin").read_bytes()
    ) if weights_cache is None else {}

    def rows(
        name: str, expert: int | None = None
    ) -> tuple[NDArray[np.uint8], gguf.GGMLQuantizationType]:
        key, shape, code = manifest[name + ".weight"]
        qt = gguf.GGMLQuantizationType(code)
        block, size = gguf.GGML_QUANT_SIZES[qt]
        data = np.memmap(
            folder / (ident(key) + ".bin"),
            dtype="u1",
            mode="r",
            shape=(*shape[:-1], shape[-1] // block * size),
        )
        return (data if expert is None else data[expert]), qt

    def mm(x: FloatArray, name: str, expert: int | None = None) -> FloatArray:
        if weights_cache is not None:
            weight = weights_cache[name + ".weight"]
            return x @ (weight if expert is None else weight[expert]).T
        data, qt = rows(name, expert)
        # Bound float32 scratch, including for the vocabulary projection.
        out = np.empty((*x.shape[:-1], len(data)), dtype="f4")
        for j in range(0, len(data), 256):
            out[..., j : j + 256] = x @ gguf.dequantize(data[j : j + 256], qt).T
        return out

    def norm(x: FloatArray, name: str) -> FloatArray:
        if weights_cache is None:
            w, qt = rows(name)
            scale = gguf.dequantize(w, qt)
        else:
            scale = weights_cache[name + ".weight"]
        return np.asarray(
            (x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + c["eps"]))
            * scale,
            dtype=np.float32,
        )

    if start == 0:
        idx = np.asarray(payload, dtype=np.intp)
        if weights_cache is None:
            emb, qt = rows("token_embd")
            x = gguf.dequantize(emb[idx], qt)
        else:
            x = weights_cache["token_embd.weight"][idx]
    else:
        x = np.asarray(payload, dtype="f4")
    hd, heads, kv = c["hd"], c["heads"], c["kv"]
    seq = len(x)
    pos = np.outer(np.arange(seq), c["theta"] ** (-np.arange(0, hd, 2) / hd)).astype(
        "f4"
    )[:, None, :]

    def rope(a: FloatArray) -> FloatArray:
        # GGUF Qwen uses NeoX layout (two half-vectors), not llama2.c's interleaved pairs.
        l, r = np.split(a, 2, axis=-1)
        return np.concatenate(
            [l * np.cos(pos) - r * np.sin(pos), r * np.cos(pos) + l * np.sin(pos)],
            axis=-1,
        )

    for i in range(start, end):
        pre = f"blk.{i}."
        a = norm(x, pre + "attn_norm")
        q = rope(
            norm(mm(a, pre + "attn_q").reshape(seq, heads, hd), pre + "attn_q_norm")
        )
        k = rope(norm(mm(a, pre + "attn_k").reshape(seq, kv, hd), pre + "attn_k_norm"))
        v = mm(a, pre + "attn_v").reshape(seq, kv, hd)
        score = np.einsum("thd,shd->hts", q, k.repeat(heads // kv, axis=1)) / np.sqrt(
            np.float32(hd)
        )
        score[:, np.triu(np.ones((seq, seq), bool), 1)] = -np.inf
        att = np.exp(score - score.max(axis=-1, keepdims=True))
        a = np.einsum(
            "hts,shd->thd",
            att / att.sum(axis=-1, keepdims=True),
            v.repeat(heads // kv, axis=1),
        )
        x = x + mm(a.reshape(seq, -1), pre + "attn_output")
        a = norm(x, pre + "ffn_norm")
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
