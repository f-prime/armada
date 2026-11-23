"""Request-local decoder KV state. Store KV heads, not repeated query heads."""
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic

import numpy as np
from .common import check
from .schema import FloatArray


@dataclass
class KV:
    offset: int = 0
    layers: dict[int, tuple[FloatArray, FloatArray]] = field(default_factory=dict)
    recurrent: dict[int, tuple[FloatArray, FloatArray]] = field(default_factory=dict)

    def append(self, layer: int, k: FloatArray, v: FloatArray, ctx: int) -> tuple[FloatArray, FloatArray]:
        end = self.offset + len(k)
        check(end <= ctx, "context limit exceeded")
        prev = self.layers.get(layer)
        if prev is None or len(prev[0]) < end:
            capacity = min(ctx, max(end, 2 * len(prev[0]) if prev else 16))
            shape = (capacity, *k.shape[1:])
            keys, values = np.empty(shape, dtype="f4"), np.empty(shape, dtype="f4")
            if prev:
                keys[:self.offset], values[:self.offset] = prev[0][:self.offset], prev[1][:self.offset]
            self.layers[layer] = keys, values
        keys, values = self.layers[layer]
        keys[self.offset:end], values[self.offset:end] = k, v
        return keys[:end], values[:end]


def attention(q: FloatArray, k: FloatArray, v: FloatArray) -> FloatArray:
    seq, heads, hd = q.shape
    queries = q.reshape(seq, k.shape[1], heads // k.shape[1], hd)
    scores = np.einsum("thgd,shd->hgts", queries, k) / np.sqrt(np.float32(hd))
    if seq > 1:
        scores[..., np.triu(np.ones((seq, len(k)), bool), 1 + len(k) - seq)] = -np.inf
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    output = np.einsum("hgts,shd->thgd", probs / probs.sum(axis=-1, keepdims=True), v)
    return np.asarray(output.reshape(q.shape), dtype=np.float32)


type Key = tuple[str, str, int, int]  # request, model generation, layer range
_idle: dict[Key, tuple[float, KV]] = {}
_busy: set[Key] = set()
_lock = Lock()


def expire() -> None:
    with _lock:
        now = monotonic()
        for expired in [k for k, (t, _) in _idle.items() if now - t > 900]:
            del _idle[expired]


@contextmanager
def session(key: Key, offset: int, length: int) -> Iterator[KV]:
    with _lock:
        check(key not in _busy, "concurrent decode on the same stage")
        saved = _idle.get(key)
        check((offset == 0 and saved is None) or (saved and saved[1].offset == offset),
              "decode session expired or out of sequence")
        check(saved or len(_idle) + len(_busy) < 128, "too many active decode sessions")
        cache = _idle.pop(key)[1] if saved else KV()
        _busy.add(key)
    try:
        yield cache
    except BaseException:
        with _lock:
            _busy.remove(key)
        raise
    else:
        cache.offset += length
        with _lock:
            _busy.remove(key)
            _idle[key] = monotonic(), cache


def release(request: str) -> None:
    with _lock:
        for key in [k for k in _idle if k[0] == request]:
            del _idle[key]
