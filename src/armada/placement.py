"""Verified artifact inventory and copy-before-retire model redistribution."""

import base64
import contextlib
import hashlib
import json
import os
import secrets
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import cast
import numpy as np

from .common import check, home, ident, open_source, readn, rpc, state
from .mesh import online, publish
from .progress import Progress
from .schema import Artifact, Fields, Model

CHUNK = 8 * 2**20


def location(path: str) -> str:
    return path if path.startswith("https://") else str(Path(path).resolve())


def inventory(uid: str, hashes: bool) -> dict[str, Artifact]:
    folder = home() / "weights" / ident(uid)
    manifest = (
        json.loads((folder / "manifest.bin").read_bytes())
        if (folder / "manifest.bin").exists()
        else {}
    )
    result = {}
    for p in folder.glob("*.bin"):
        if p.stem == "manifest":
            continue
        with p.open("rb") as f:
            result[p.stem] = Artifact(
                size=p.stat().st_size,
                digest=hashlib.file_digest(f, "sha256").hexdigest() if hashes else "",
            )
    for name, (key, shape, qt) in manifest.items():
        result[key]["tensor"] = (name, shape, qt)
    return result


def verify(m: Model, nodes: list[str], expected: dict[str, Artifact]) -> None:
    found: dict[str, Artifact] = {}
    for node in nodes:
        local = cast(
            dict[str, Artifact], rpc(node, "inventory", dict(id=m["id"], hashes=True))
        )
        if expected:
            wanted = {k for k, f in expected.items() if node in destinations(m, k, f)}
            check(set(local) == wanted, f"{node}: incomplete staged placement")
        for k, v in local.items():
            if expected:
                check(v == expected[k], f"{node}: source or copy mismatch for {k}")
            found[k] = v
    check(not expected or found == expected, "incomplete replacement model")
    m["files"] = found


def layers(m: Model, k: str, f: Artifact) -> list[int]:
    last = len(m["owners"]) - 1
    if k == "tokenizer":
        return [0]
    if "tensor" in f:
        name = f["tensor"][0]
        if name.startswith("blk."):
            return [int(name.split(".")[1])]
        if name == "token_embd.weight" and m["qwen"].get("tied_output"):
            return [0, last]
        return [0 if name == "token_embd.weight" else last]
    if k == "emb":
        return [0, last] if m["config"][5] > 0 else [0]
    return [last if k in ("norm", "out") else int(k.rsplit("_", 1)[1])]


def destinations(m: Model, k: str, f: Artifact) -> set[str]:
    return {m["owners"][i] for i in layers(m, k, f)}


def file_rpc(op: str, b: Fields) -> object:
    if op == "inventory":
        return inventory(b["id"], b["hashes"])
    path = home() / "weights" / ident(b["id"]) / (ident(b["file"]) + ".bin")
    if op == "read-chunk":
        with path.open("rb") as f:
            f.seek(int(b["offset"]))
            return base64.b64encode(f.read(CHUNK)).decode()
    target = home() / "weights" / ident(b["new"]) / path.name
    target.parent.mkdir(exist_ok=True, parents=True)
    os.link(path, target)
    return True


def plan(old: Model, nodes: list[str], local: dict[str, set[str]]) -> Model:
    """Find a contiguous partition that fits free disk. Hard links use no tensor space."""
    from functools import lru_cache

    n = len(old["owners"])
    sizes = {node: int(rpc(node, "stats")["free_disk"]) for node in nodes}
    costs = {node: [0] * n for node in nodes}
    shared = {}
    for key, file in old["files"].items():
        owned_layers = set(layers(old, key, file))
        if len(owned_layers) > 1:
            shared[key] = file["size"]
        for node in nodes:
            if key not in local[node]:
                for i in owned_layers:
                    costs[node][i] += file["size"]

    @lru_cache(None)
    def assign(j: int, start: int) -> list[str] | None:
        if start == n:
            return []
        if j == len(nodes):
            return None
        node = nodes[j]
        target = (n - start) / min(n - start, len(nodes) - j)
        for count in sorted(range(n - start + 1), key=lambda c: abs(c - target)):
            need = sum(costs[node][start : start + count])
            if start == 0 and count == n:
                need -= sum(size for key, size in shared.items() if key not in local[node])
            if need > sizes[node]:
                continue
            rest = assign(j + 1, start + count)
            if rest is not None:
                return [node] * count + rest
        return None

    owners = assign(0, 0)
    check(
        owners is not None,
        "insufficient free disk for a verified replacement on online peers",
    )
    assert owners is not None
    new = old.copy()
    new.update(
        {
            "id": secrets.token_hex(12),
            "owners": owners,
            "revision": old.get("revision", 0) + 1,
            "retired": [*old.get("retired", []), old["id"]],
        }
    )
    return new


def recovery(
    m: Model, missing: set[str], source: str, tokenizer: str
) -> Iterator[tuple[str, int, bytes]]:
    from .qwen import gg_parts, gg_header
    from .bpe import gg_tokenizer
    from .runtime import layout

    if "qwen" in m:
        for part, path in enumerate(gg_parts(source)):
            with open_source(path) as f:
                meta, tensors = gg_header(f)
                if not part and "tokenizer" in missing:
                    yield "tokenizer", 0, gg_tokenizer(meta)
                pos = 0
                for offset, name, shape, qt, size in tensors:
                    key = name.replace(".", "_")
                    readn(f, offset - pos)
                    if key in missing:
                        ref = m["files"][key]
                        check(
                            ref["size"] == size
                            and list(ref["tensor"]) == [name, shape, qt],
                            "recovery GGUF tensor mismatch",
                        )
                    for j in range(0, size, CHUNK):
                        data = readn(f, min(CHUNK, size - j))
                        if key in missing:
                            yield key, j, data
                    pos = offset + size
        return
    if "tokenizer" in missing:
        with open_source(tokenizer) as f:
            yield "tokenizer", 0, f.read(16 * 2**20 + 1)
    if not missing - {"tokenizer"}:
        return
    with open_source(source) as f:
        c = list(struct.unpack("<7i", readn(f, 28)))
        check(c == m["config"], "recovery checkpoint config mismatch")
        for key, dims, _ in layout(c):
            raw = readn(f, 4 * int(np.prod(dims)))
            if key not in missing:
                continue
            if len(dims) == 2:
                w = np.frombuffer(raw, "<f4").reshape(dims)
                scale = np.maximum(np.max(np.abs(w), axis=1), 1e-30) / 127
                raw = (
                    scale.astype("<f4").tobytes()
                    + np.rint(w / scale[:, None]).astype("i1").tobytes()
                )
            for j in range(0, len(raw), CHUNK):
                yield key, j, raw[j : j + CHUNK]


def redistribute(
    name: str, source: str = "", tokenizer: str = "", *, progress: Progress | None = None
) -> Model:
    old = state()["models"][name]
    if progress:
        progress.phase(f"Finding reachable peers for {name}…")
    nodes = online()
    if "files" not in old:
        if progress:
            progress.phase("Indexing existing weights…")
        check(
            set(old["owners"]) <= set(nodes),
            "legacy manifest: bring all original owners online once to index weights",
        )
        verify(old, list(set(old["owners"])), {})
    files = old["files"]
    if progress:
        progress.phase("Checking peer inventories and planning placement…")
    local = {
        p: set(cast(Fields, rpc(p, "inventory", dict(id=old["id"], hashes=False))))
        for p in nodes
    }
    m = plan(old, nodes, local)
    if source:
        m["source"] = location(source)
    if tokenizer:
        m["tokenizer_source"] = location(tokenizer)
    targets = list(dict.fromkeys(m["owners"]))
    if m.get("backend") == "llama":
        for owner in targets:
            rpc(owner, "llama3-ready")
    sources = {k: next((p for p in nodes if k in local[p]), None) for k in files}
    missing = {k for k, src in sources.items() if src is None}
    if progress:
        reuse = transfer = 0
        for k, f in files.items():
            for p in destinations(m, k, f):
                if k in local[p]:
                    reuse += 1
                elif k not in missing:
                    transfer += f["size"]
        if reuse:
            progress.phase(f"Reusing {reuse} local file copies…")
        if transfer:
            progress.start(f"Redistributing {name} · peers: {len(targets)}", transfer)

    def send(p: str, k: str, j: int, data: bytes) -> None:
        rpc(p, f"chunk/{m['id']}/{k}/{j}/{hashlib.sha256(data).hexdigest()}", body=data)
        if progress:
            progress.advance(len(data))

    try:
        for k, f in files.items():
            src = sources[k]
            if src is None:
                continue
            for p in destinations(m, k, f):
                if k in local[p]:
                    rpc(p, "link", dict(id=old["id"], file=k, new=m["id"]))
                else:
                    for j in range(0, f["size"], CHUNK):
                        data = base64.b64decode(
                            str(
                                rpc(
                                    src,
                                    "read-chunk",
                                    dict(id=old["id"], file=k, offset=j),
                                )
                            )
                        )
                        send(p, k, j, data)
        if missing:
            if progress:
                progress.start(
                    f"Recovering {len(missing)} files from source…",
                    sum(files[k]["size"] * len(destinations(m, k, files[k])) for k in missing),
                )
            try:
                for k, j, data in recovery(
                    m, missing, m.get("source", ""), m.get("tokenizer_source", "")
                ):
                    for p in destinations(m, k, files[k]):
                        send(p, k, j, data)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"cannot recover {len(missing)} files (including {min(missing)}); supply --source / --tokenizer: {exc}"
                ) from exc
        if "qwen" in m:
            if progress:
                progress.phase("Finalizing manifests…")
            for p in targets:
                manifest = {
                    f["tensor"][0]: [k, *f["tensor"][1:]]
                    for k, f in files.items()
                    if "tensor" in f and p in destinations(m, k, f)
                }
                b = json.dumps(manifest).encode()
                rpc(
                    p,
                    f"weight/{m['id']}/manifest/{hashlib.sha256(b).hexdigest()}",
                    body=b,
                )
        if progress:
            progress.phase("Verifying weights…")
        verify(m, targets, files)
    except Exception:
        for p in targets:
            with contextlib.suppress(OSError, ValueError):
                rpc(p, "discard", dict(id=m["id"]))
        raise
    if progress:
        progress.phase("Publishing model…")
    publish(name, m, old["id"])
    if progress:
        progress.phase(f"{name} is ready.")
    return m
