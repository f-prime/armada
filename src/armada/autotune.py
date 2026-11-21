"""Node-local Q8 decode calibration. Placement generations have immutable shard files."""
import hashlib
import json
import os
import platform
import statistics
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import gguf

from . import residency
from .common import home, ident, save
from .residency import DEFAULT_THREADS, MAX_THREADS, Weights, q8_threads


@lru_cache(maxsize=1)
def _kernel() -> str:
    from . import _q8

    digest = hashlib.sha256()
    for module in (_q8, residency):
        filename = module.__file__
        assert filename is not None
        digest.update(Path(filename).read_bytes())
    return digest.hexdigest()


def _environment() -> tuple[str, int]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    cpus = len(affinity) if affinity else os.cpu_count() or 1
    key = json.dumps([1, platform.machine(), platform.node(), _kernel(), cpus, affinity])
    return key, min(MAX_THREADS, cpus)


def _path(uid: str) -> Path:
    return home() / "weights" / ident(uid) / "decode-tuning.json"


@lru_cache(maxsize=128)
def _read(path: Path, signature: str) -> int | None:
    try:
        record = json.loads(path.read_text())
        if isinstance(record, dict) and record.get("environment") == signature:
            n = record.get("threads")
            if type(n) is int and 1 <= n <= MAX_THREADS:
                return n
    except (OSError, ValueError):
        pass
    return None


def selected(uid: str) -> int:
    if not residency.NATIVE_Q8 or "ARMADA_Q8_THREADS" in os.environ:
        return DEFAULT_THREADS
    return _read(_path(uid), _environment()[0]) or DEFAULT_THREADS


def calibrate(uid: str, weights: Weights, decode: Callable[[], None]) -> None:
    if not residency.NATIVE_Q8 or "ARMADA_Q8_THREADS" in os.environ:
        return
    # Single-expert bytes matter, not the combined size of all experts.
    if not any(t.qt == gguf.GGMLQuantizationType.Q8_0 and len(t.shape) in (2, 3)
               and t.shape[-1] // 32 * 34 * t.shape[-2] >= 1024**2 for t in weights.values()):
        return
    signature, limit = _environment()
    path = _path(uid)
    if _read(path, signature) is not None:
        return
    counts = sorted({limit, *(n for n in (1, 2, 4, 8, 16, 32) if n <= limit)})
    times: dict[int, list[float]] = {n: [] for n in counts}
    ticket = q8_threads.set(1)
    try:
        # Prime CPU and thread pool; only measure steady-state single-token work.
        q8_threads.set(counts[-1])
        decode()
        for order in (counts, counts[::-1], counts):
            for n in order:
                q8_threads.set(n)
                start = perf_counter()
                decode()
                times[n].append(perf_counter() - start)
    finally:
        q8_threads.reset(ticket)
    medians = {n: statistics.median(t) for n, t in times.items()}
    best = min(counts, key=medians.__getitem__)
    save(path, dict(environment=signature, threads=best, median_decode_seconds=medians))
    _read.cache_clear()
