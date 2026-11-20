import contextlib
import codecs
import fcntl
import getpass
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator, Sequence
from typing import cast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
from .schema import Fields, FloatArray, Model, NodeState, Payload
from .common import (
    address,
    catalog,
    check,
    home,
    ident,
    open_source,
    readn,
    rpc,
    save,
    state,
)
from .qwen import gg_stage, import_gg
from .bpe import load_tokenizer
from .gpu import gpu_stats
from .progress import Progress, terminal_progress
from . import chat
from .keys import authorized
from .mesh import delete_model, key_command

from .mesh import (
    LOCK,
    initialize,
    enroll,
    join,
    merge,
    gossip,
    online,
    publish,
    commit,
    snapshot,
    unavailable,
)
from .placement import file_rpc, location, redistribute, verify

ACTIVE: dict[str, int] = {}


def memory_stats() -> tuple[int | None, int | None]:
    """Available and total RAM in bytes, as reported by the local OS."""
    free = total = None
    if sys.platform == "linux":
        with contextlib.suppress(OSError):
            text = Path("/proc/meminfo").read_text()
            sizes = dict(re.findall(r"^(\w+):\s+(\d+) kB$", text, re.MULTILINE))
            if "MemTotal" in sizes:
                total = int(sizes["MemTotal"]) * 1024
            if "MemAvailable" in sizes:
                free = int(sizes["MemAvailable"]) * 1024
    elif sys.platform == "darwin":
        with contextlib.suppress(OSError, ValueError, subprocess.SubprocessError):
            total = int(subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True, timeout=3
            ))
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            text = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=3)
            page_size = re.search(r"page size of (\d+) bytes", text)
            pages = dict(re.findall(r"^Pages (\w+):\s+(\d+)\.?$", text, re.MULTILINE))
            if page_size and "free" in pages and "inactive" in pages:
                free = (int(pages["free"]) + int(pages["inactive"])) * int(page_size[1])
    return free, total


def stats() -> Fields:
    free_ram, total_ram = memory_stats()
    gpus = gpu_stats()
    with LOCK:
        active = list(ACTIVE)
    return dict(
        free_ram=free_ram,
        total_ram=total_ram,
        cpu_threads=os.cpu_count(),
        cpu_load_1m=os.getloadavg()[0],
        gpus=gpus,
        nvidia_gpus=[{k: g[k] for k in ("name", "free_vram", "total_vram") if k in g}
                     for g in gpus if g.get("vendor") == "NVIDIA"],
        free_disk=shutil.disk_usage(home()).free,
        shards={
            p.name: sum(f.stat().st_size for f in p.glob("*.bin"))
            for p in home().glob("weights/*")
        },
        active_models=active,
        load_policy="mapped per stage request",
    )


def layout(c: Sequence[int]) -> Iterator[tuple[str, tuple[int, ...], int | None]]:
    d, h, n, heads, kv, vocab, ctx = c
    k = d * kv // heads
    yield "emb", (abs(vocab), d), -1
    for key, shape in [
        ("an", (d,)),
        ("q", (d, d)),
        ("k", (k, d)),
        ("v", (k, d)),
        ("o", (d, d)),
        ("fn", (d,)),
        ("w1", (h, d)),
        ("w2", (d, h)),
        ("w3", (h, d)),
    ]:
        for i in range(n):
            yield f"{key}_{i}", shape, i
    yield "norm", (d,), n
    yield "rope_r", (ctx, d // heads // 2), None
    yield "rope_i", (ctx, d // heads // 2), None
    if vocab < 0:
        yield "out", (-vocab, d), n


def import_model(
    name: str, source: str, tokenizer: str, *, exclude_self: bool = False,
    progress: Progress | None = None,
) -> Model:
    ident(name)
    s = state()
    cat = catalog(s)
    check(name not in cat["models"], "model name already exists")
    uid = secrets.token_hex(12)
    with open_source(source) as f:
        c: tuple[int, ...] = struct.unpack("<7i", readn(f, 28))
        d, h, n, heads, kv, vocab, ctx = c
        check(
            all(0 < x <= 1000000 for x in (d, h, n, heads, kv, abs(vocab), ctx))
            and d % (2 * heads) == 0
            and heads % kv == 0,
            "not a supported llama2.c checkpoint",
        )
        nodes = online(exclude_self)[:n]
        owners = [nodes[i * len(nodes) // n] for i in range(n)]

        def owner(i: int) -> str:
            return owners[min(n - 1, max(0, i))]

        m = Model(
            id=uid,
            config=list(c),
            owners=owners,
            source=location(source),
            tokenizer_source=location(tokenizer),
        )
        needed = dict.fromkeys(nodes, 0)
        for key, shape, i in layout(c):
            if i is None:
                continue
            size = shape[0] * (shape[1] + 4) if len(shape) == 2 else shape[0] * 4
            needed[owner(i)] += size
            if key == "emb" and vocab > 0 and owner(0) != owner(n):
                needed[owner(n)] += size
        for node, size in needed.items():
            available = rpc(node, "stats")
            check(size < available["free_disk"], f"{node}: need {size} free disk bytes")

        def upload(node: str, key: str, b: bytes) -> None:
            digest = hashlib.sha256(b).hexdigest()
            rpc(node, f"weight/{uid}/{key}/{digest}", body=b)
            if progress:
                progress.advance(len(b))

        with open_source(tokenizer) as t:
            tok = t.read(16 * 1024 * 1024 + 1)
            parse_tokenizer(tok, abs(vocab))
        if progress:
            progress.start(f"Distributing {name} · peers: {len(nodes)}", sum(needed.values()) + len(tok))
        upload(owner(0), "tokenizer", tok)
        try:
            for key, shape, i in layout(c):
                raw = readn(f, 4 * int(np.prod(shape)))
                if i is None:
                    continue
                if len(shape) == 2:
                    w = np.frombuffer(raw, "<f4").reshape(shape)
                    scale = np.maximum(np.max(np.abs(w), axis=1), 1e-30) / 127
                    raw = (
                        scale.astype("<f4").tobytes()
                        + np.rint(w / scale[:, None]).astype("i1").tobytes()
                    )
                upload(owner(i), key, raw)
                if key == "emb" and vocab > 0 and owner(0) != owner(n):
                    upload(owner(n), "emb", raw)
            check(not f.read(1), "unexpected trailing checkpoint data")
            if progress:
                progress.phase("Verifying weights…")
            verify(m, nodes, {})
        except Exception:
            for node in nodes:
                with contextlib.suppress(Exception):
                    rpc(node, "discard", {"id": uid})
            raise
    if progress:
        progress.phase("Publishing model…")
    publish(name, m)
    if progress:
        progress.phase(f"{name} is ready.")
    return m


def parse_tokenizer(b: bytes, vocab: int) -> tuple[list[bytes], list[float]]:
    check(4 <= len(b) <= 16 * 1024 * 1024, "invalid tokenizer size")
    off = 4
    pieces, scores = [], []
    for _ in range(vocab):
        score, size = struct.unpack_from("<fi", b, off)
        off += 8
        check(0 <= size <= len(b) - off, "invalid tokenizer entry")
        pieces.append(b[off : off + size])
        scores.append(score)
        off += size
    check(off == len(b), "tokenizer vocabulary mismatch")
    return pieces, scores


def encode(prompt: str, pieces: list[bytes], scores: list[float]) -> list[int]:
    ids = {b: i for i, b in enumerate(pieces)}
    tokens = [1]
    for ch in (" " + prompt if prompt else ""):
        b = ch.encode()
        tokens.extend([ids[b]] if b in ids else [v + 3 for v in b])
    while len(tokens) > 1:
        pairs = [
            (ids.get(pieces[a] + pieces[b]), i)
            for i, (a, b) in enumerate(zip(tokens, tokens[1:]))
        ]
        best = max(
            ((scores[k], -i, k) for k, i in pairs if k is not None), default=None
        )
        if best is None:
            break
        _, i, k = best
        tokens[-i : -i + 2] = [k]
    return tokens


def stage(m: Model, start: int, end: int, payload: Payload) -> int | list[list[float]]:
    """Evaluate one contiguous decoder stage, recomputing the prefix (no KV cache)."""
    d, h, n, heads, kv, vocab, ctx = m["config"]
    folder = home() / "weights" / ident(m["id"])
    hd = d // heads

    def matrix(
        key: str, rows: int, cols: int, index: Payload | None = None
    ) -> FloatArray:
        path = folder / (key + ".bin")
        scale = np.memmap(path, dtype="<f4", mode="r", shape=(rows, 1))
        q = np.memmap(path, dtype="i1", mode="r", offset=4 * rows, shape=(rows, cols))
        if index is not None:
            idx = np.asarray(index, dtype=np.intp)
            return np.asarray(q[idx] * scale[idx], dtype=np.float32)
        return np.asarray(q * scale, dtype=np.float32)

    def norm(x: FloatArray, key: str) -> FloatArray:
        return np.asarray(
            x
            / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
            * np.fromfile(folder / (key + ".bin"), "<f4"),
            dtype=np.float32,
        )

    x = (
        matrix("emb", abs(vocab), d, payload)
        if start == 0
        else np.asarray(payload, dtype=np.float32)
    )
    seq = len(x)
    check(0 < seq <= ctx, "context limit exceeded")
    angles = np.outer(np.arange(seq), 10000 ** (-np.arange(0, hd, 2) / hd)).astype(
        np.float32
    )[:, None, :]

    def rotate(a: FloatArray) -> FloatArray:
        even, odd = a[..., ::2].copy(), a[..., 1::2].copy()
        a[..., ::2] = even * np.cos(angles) - odd * np.sin(angles)
        a[..., 1::2] = odd * np.cos(angles) + even * np.sin(angles)
        return a

    for i in range(start, end):

        def mm(x: FloatArray, key: str, rows: int, cols: int) -> FloatArray:
            return x @ matrix(f"{key}_{i}", rows, cols).T

        a = norm(x, f"an_{i}")
        q = rotate(mm(a, "q", d, d).reshape(seq, heads, hd))
        k = rotate(mm(a, "k", kv * hd, d).reshape(seq, kv, hd)).repeat(
            heads // kv, axis=1
        )
        v = mm(a, "v", kv * hd, d).reshape(seq, kv, hd).repeat(heads // kv, axis=1)
        att = np.einsum("thd,shd->hts", q, k) / np.sqrt(np.float32(hd))
        att[:, np.triu_indices(seq, 1)[0], np.triu_indices(seq, 1)[1]] = -np.inf
        att = np.exp(att - att.max(axis=-1, keepdims=True))
        a = np.einsum("hts,shd->thd", att / att.sum(axis=-1, keepdims=True), v)
        x = x + mm(a.reshape(seq, d), "o", d, d)
        a = norm(x, f"fn_{i}")
        gate = mm(a, "w1", h, d)
        x = x + mm(
            (gate / (1 + np.exp(-np.clip(gate, -80, 80)))) * mm(a, "w3", h, d),
            "w2",
            d,
            h,
        )
    if end == n:
        logits = (
            norm(x[-1], "norm") @ matrix("emb" if vocab > 0 else "out", abs(vocab), d).T
        )
        return int(np.argmax(logits))
    return cast(list[list[float]], x.tolist())


def generation(m: Model, prompt: str, count: int, stats: Fields) -> Iterator[int]:
    import itertools

    m = m.copy()
    m.pop("files", None)
    missing = unavailable(m)
    check(
        not missing,
        f"unavailable layers by owner: {missing}; bring the peer back or use model redistribute",
    )
    owners = m["owners"]
    ctx = m["qwen"]["ctx"] if m.get("backend") == "qwen3moe" else m["config"][-1]
    check(0 < count <= ctx, "invalid token count")
    tokens = rpc(owners[0], "tokenize", dict(model=m, prompt=prompt))
    check(0 < len(tokens) < ctx, "prompt is empty or exceeds context window")
    prompt_len = len(tokens)

    def steps() -> Iterator[int]:
        reason = "length"
        sampled = 0
        for _ in range(min(count, ctx - prompt_len)):
            payload: Payload | int = tokens
            start = 0
            for node, group in itertools.groupby(owners):
                end = start + sum(1 for _ in group)
                payload = rpc(node, "stage", dict(model=m, start=start, end=end, payload=payload))
                start = end
            if not isinstance(payload, int):
                raise ValueError("final stage did not return a token ID")
            sampled += 1
            if payload in m.get("stop", (1, 2)):
                reason = "stop"
                break
            tokens.append(payload)
            yield payload
        stats.update(finish_reason=reason, usage=dict(prompt_tokens=prompt_len,
                     completion_tokens=sampled, total_tokens=prompt_len + sampled))
    return steps()


def generate(m: Model, prompt: str, count: int) -> str:
    output = list(generation(m, prompt, count, {}))
    metadata = m.copy()
    metadata.pop("files", None)
    return str(rpc(m["owners"][0], "decode", dict(model=metadata, tokens=output)))


def chat_pieces(m: Model, tokens: Iterator[int]) -> Iterator[str]:
    # The tokenizer may live on another machine. Decode raw bytes so a UTF-8
    # character split across tokens is emitted exactly once.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for token in tokens:
        data = rpc(m["owners"][0], "chat-token", dict(id=m["id"], token=token))
        yield decoder.decode(bytes.fromhex(str(data)))
    yield decoder.decode(b"", final=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.inference("GET")

    def json_reply(self, status: int, result: object) -> None:
        payload = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def inference(self, method: str) -> None:
        path = self.path.partition("?")[0]
        streaming = False
        try:
            s = state()
            auth = self.headers.get("Authorization", "").encode()
            keys = [os.environ.get("ARMADA_API_KEY", "")]
            if path == "/v1/completions":
                keys.append(s["key"])  # Existing Armada CLI and older fleet clients.
            if not authorized(s, auth) and not any(key and hmac.compare_digest(auth, ("Bearer " + key).encode()) for key in keys):
                self.json_reply(401, {"error": {"type": "authentication_error", "message": "Invalid API key"}})
                return
            models = s["models"]
            if path == "/v1/models" and method == "GET":
                self.json_reply(200, dict(object="list", data=[dict(id=name, object="model", created=0,
                                owned_by="armada") for name in sorted(models)]))
                return
            if path not in ("/v1/chat/completions", "/v1/completions") or method != "POST":
                self.json_reply(404, {"error": {"type": "invalid_request_error", "message": "Unknown endpoint"}})
                return
            length = int(self.headers.get("Content-Length", "0"))
            check(0 < length <= 16 * 2**20, "expected a JSON body of at most 16 MiB")
            b = json.loads(readn(self.rfile, length))
            check(isinstance(b, dict) and isinstance(b.get("model"), str), "expected a model name")
            if b["model"] not in models:
                self.json_reply(404, {"error": {"type": "invalid_request_error", "code": "model_not_found",
                                                "message": "Unknown fleet model: " + b["model"]}})
                return
            if path == "/v1/completions":
                check(not b.get("stream"), "streaming is supported on /v1/chat/completions")
                check(isinstance(b.get("prompt"), str), "expected a text prompt")
                self.json_reply(200, self.dispatch("v1/completions", b, s))
                return
            model = models[b["model"]]
            check(model.get("backend") == "qwen3moe", "chat requires a Qwen instruction model; use /v1/completions for base models")
            check(b.get("n", 1) == 1, "only one completion is supported")
            check(isinstance(b.get("stream", False), bool), "stream must be a boolean")
            check(not b.get("stop"), "custom stop sequences are not supported")
            prompt, names = chat.prompt(b)
            count = b.get("max_completion_tokens", b.get("max_tokens", min(1024, model["qwen"]["ctx"])))
            check(type(count) is int and count > 0, "expected a positive token limit")
            stats: Fields = {}
            tokens = generation(model, prompt, count, stats)
            options = b.get("stream_options") or {}
            check(isinstance(options, dict), "invalid stream_options")
            chunks = chat.chunks(b["model"], chat_pieces(model, tokens), names, stats,
                                 not b.get("stream") or options.get("include_usage", False))
            if not b.get("stream"):
                self.json_reply(200, chat.collect(chunks))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            streaming = True
            for chunk in chunks:
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            status = (503 if "unavailable layers" in str(exc) or isinstance(exc, OSError)
                      else 400 if isinstance(exc, (ValueError, KeyError, TypeError)) else 500)
            error = {"error": {"type": "invalid_request_error" if status == 400 else "server_error",
                               "message": str(exc)}}
            if streaming:
                with contextlib.suppress(OSError):
                    self.wfile.write(b"data: " + json.dumps(error).encode() + b"\n\ndata: [DONE]\n\n")
            else:
                self.json_reply(status, error)

    def do_POST(self) -> None:
        if self.path.partition("?")[0].startswith("/v1/"):
            self.inference("POST")
            return
        try:
            op = self.path.strip("/")
            s = state()
            enrollment = op == "enroll"
            auth = s.get("pair", "") if enrollment else s["key"]
            check(
                auth
                and hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + auth
                ),
                "unauthorized",
            )
            length = int(self.headers.get("Content-Length", 0))
            check(0 <= length <= 2**31, "invalid request length")
            raw = readn(self.rfile, length)
            if op.startswith("chunk/"):
                _, uid, key, offset, digest = op.split("/")
                folder = home() / "weights" / ident(uid)
                folder.mkdir(parents=True, exist_ok=True)
                check(
                    hashlib.sha256(raw).hexdigest() == digest, "invalid chunk checksum"
                )
                with (folder / (ident(key) + ".bin")).open("ab") as dest:
                    check(dest.tell() == int(offset), "chunk offset mismatch")
                    dest.write(raw)
                result: object = True
            elif op.startswith("weight/"):
                _, uid, key, digest = op.split("/")
                folder = home() / "weights" / ident(uid)
                folder.mkdir(parents=True, exist_ok=True)
                check(hashlib.sha256(raw).hexdigest() == digest, "invalid checksum")
                (folder / (ident(key) + ".bin")).write_bytes(raw)
                result = True
            else:
                result = self.dispatch(op, json.loads(raw), s)
            status, body = 200, json.dumps(result).encode()
        except Exception as e:
            status, body = (403 if str(e) == "unauthorized" else 400), str(e).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def dispatch(self, op: str, b: Fields, s: NodeState) -> object:
        if op == "manage-key":
            return key_command(b)
        if op == "stats":
            return stats()
        if op == "qwen-ready":
            try:
                import gguf
                import regex
            except ImportError as e:
                raise ValueError(
                    f"GGUF runtime unavailable; reinstall Armada with uv: {e}"
                ) from e
            return True
        if op == "ping":
            return s["uid"]
        if op == "sync":
            with LOCK:
                pending = merge(b)
                return dict(**snapshot(state()), cleanup_pending=pending)
        if op == "model-delete":
            return delete_model(b["name"], b["id"])
        if op == "commit":
            commit(b)
            return True
        if op == "enroll":
            return enroll(b, self.headers["Authorization"])
        if op == "join":
            join(b["node"], b["code"], b.get("return_address"))
            return True
        if op in ("inventory", "read-chunk", "link"):
            return file_rpc(op, b)
        if op == "pair":
            with LOCK:
                s = state()
                check(len(s["peers"]) == 1, "node already paired")
                s["pair"] = secrets.token_urlsafe(24)
                save(home() / "node.json", s)
                return s["pair"]
        if op == "discard":
            shutil.rmtree(home() / "weights" / ident(b["id"]), ignore_errors=True)
            return True
        if op == "stage":
            uid = b["model"]["id"]
            with LOCK:
                ACTIVE[uid] = ACTIVE.get(uid, 0) + 1
            try:
                fn = gg_stage if b["model"].get("backend") == "qwen3moe" else stage
                return fn(b["model"], b["start"], b["end"], b["payload"])
            finally:
                with LOCK:
                    ACTIVE[uid] -= 1
                    if not ACTIVE[uid]:
                        del ACTIVE[uid]
        if op == "chat-token":
            tok = load_tokenizer(home() / "weights" / ident(b["id"]) / "tokenizer.bin")
            return tok.decode_bytes([b["token"]], keep_special=True).hex()
        if op in ("tokenize", "decode"):
            m = b["model"]
            if m.get("backend") == "qwen3moe":
                tok = load_tokenizer(
                    home() / "weights" / ident(m["id"]) / "tokenizer.bin"
                )
                return (
                    tok.encode(b["prompt"])
                    if op == "tokenize"
                    else tok.decode(b["tokens"])
                )
            pieces, scores = parse_tokenizer(
                (home() / "weights" / ident(m["id"]) / "tokenizer.bin").read_bytes(),
                abs(m["config"][5]),
            )
            if op == "tokenize":
                return encode(b["prompt"], pieces, scores)

            def decode(p: bytes) -> bytes:
                return (
                    bytes([int(p[3:5], 16)])
                    if re.fullmatch(b"<0x[0-9A-Fa-f]{2}>", p)
                    else p
                )

            return b"".join(decode(pieces[i]) for i in b["tokens"]).decode(
                errors="replace"
            )
        if op == "v1/completions":
            m = catalog(s)["models"][b["model"]]
            text = generate(m, b["prompt"], int(b.get("max_tokens", 32)))
            return dict(
                object="text_completion",
                model=b["model"],
                choices=[dict(index=0, text=text)],
            )
        raise ValueError("unknown command")


def advertised_host() -> str:
    # UDP connect asks the kernel which local address its default route uses.
    # No packet is sent.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("192.0.2.1", 9))
            return str(sock.getsockname()[0])
        except OSError:
            try:
                return socket.gethostbyname(socket.gethostname())
            except OSError:
                return "127.0.0.1"


def execute(args: list[str]) -> object:
    os.umask(0o077)
    home().mkdir(parents=True, exist_ok=True)
    op, *a = args
    if op == "daemon":
        host, port = a[0], int(a[1])
        check(1 <= port <= 65535, "port must be between 1 and 65535")
        bind_addr = address(f"{host}:{port}")
        with (home() / "daemon.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise ValueError(
                    "a daemon is already running in this ARMADA_HOME"
                ) from e
            path = home() / "node.json"
            node: NodeState = (
                state()
                if path.exists()
                else {"addr": bind_addr, "key": secrets.token_urlsafe(32)}
            )
            addr = address(f"{advertised_host() if host=='0.0.0.0' else host}:{port}")
            advertise = address(f"{a[2]}:{port}") if len(a) > 2 else None
            initialize(node, addr, advertise)
            with ThreadingHTTPServer((host, port), Handler) as server:
                save(path, node)
                token = secrets.token_hex(16)
                save(home() / "daemon.json", dict(host=host, port=port,
                     advertise_host=a[2] if len(a) > 2 else None, token=token,
                     mode=os.environ.get("_ARMADA_WORKER_MODE", "foreground")))
                threading.Thread(target=gossip, daemon=True).start()
                print(
                    f"Armada listening on {bind_addr}; peer address {node['peers'][node['uid']]['endpoint']}",
                    flush=True,
                )
                # A local stop request names this exact run; stale requests cannot
                # stop a replacement process (including after a crash or reboot).
                server.timeout = 0.25
                request = home() / ("stop-" + token)
                while not request.exists():
                    server.handle_request()
                request.unlink()
                return "Armada stopped."
    s = state()
    if op == "pair":
        return rpc(s["addr"], "pair")
    if op.startswith("key-"):
        action = op.removeprefix("key-")
        return rpc(s["addr"], "manage-key", dict(action=action,
                   id=a[0] if a and action != "create" else "",
                   name=a[-1] if action in ("create", "rename") else ""))
    if op == "add":
        code = (
            getpass.getpass("Peer pairing code: ")
            if sys.stdin.isatty()
            else sys.stdin.readline()
        )
        return rpc(
            s["addr"],
            "join",
            dict(
                node=a[0],
                code=code.strip(),
                return_address=a[1] if len(a) > 1 else None,
            ),
        )
    if op in ("import", "import-gguf", "redistribute"):
        with terminal_progress() as progress:
            if op == "redistribute":
                return redistribute(*a, progress=progress)
            if progress:
                progress.phase(f"Preparing {a[0]}…")
            if op == "import":
                return import_model(*a[:3], exclude_self=len(a) > 3, progress=progress)
            return import_gg(*a[:2], exclude_self=len(a) > 2, progress=progress)
    if op == "run":
        return rpc(
            s["addr"],
            "v1/completions",
            dict(model=a[0], prompt=a[1], max_tokens=int(a[2])),
        )["choices"][0]["text"]
    cat = catalog(s)
    if op == "model-delete":
        check(a[0] in cat["models"], f"unknown model: {a[0]}")
        return rpc(s["addr"], op, dict(name=a[0], id=cat["models"][a[0]]["id"]))
    if op == "model-list":
        return [
            {"name": name, "peer_count": len(set(model["owners"]))}
            for name, model in sorted(cat["models"].items())
        ]
    if op == "model":
        model = cat["models"][a[0]].copy()
        count = len(model.pop("files", {}))
        missing = unavailable(model)
        peers = list(dict.fromkeys([*model["owners"], *cat["nodes"]]))

        def stored(peer: str) -> int | None:
            if peer not in missing:
                try:
                    report = rpc(peer, "stats", timeout=12)
                    if isinstance(report, dict) and isinstance(report.get("shards"), dict):
                        size = report["shards"].get(model["id"], 0)
                        if type(size) is int and size >= 0:
                            return size
                except (OSError, ValueError):
                    pass
            return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            storage = dict(zip(peers, pool.map(stored, peers)))
        return dict(
            **model, verified_artifacts=count, unavailable_layers=missing, storage_bytes_by_peer=storage
        )
    resources = cat.setdefault("resources", {})
    for peer in cat["nodes"]:
        try:
            resources[peer] = rpc(peer, "stats")
        except Exception as e:
            resources[peer] = {"error": str(e)}
    return dict(**cat, peers=s["peers"], self=s["uid"])
