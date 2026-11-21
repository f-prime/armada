"""Prepared weights keyed by immutable model generation. Mutations use mesh.LOCK."""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np
from numpy.typing import NDArray

from .schema import FloatArray

try:
    from ._q8 import matvec
except ImportError:
    matvec = None  # type: ignore[assignment]

NATIVE_Q8 = matvec is not None and os.getenv("ARMADA_NATIVE_Q8", "1") != "0"
MAX_THREADS = 32
_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
DEFAULT_THREADS = int(os.getenv("ARMADA_Q8_THREADS", str(min(4, _cpus))))
if not 1 <= DEFAULT_THREADS <= MAX_THREADS:
    raise ValueError(f"ARMADA_Q8_THREADS must be between 1 and {MAX_THREADS}")
q8_threads = ContextVar("q8_threads", default=DEFAULT_THREADS)
# The native extension releases the GIL. Share a lazy pool across requests;
# every job owns disjoint complete rows, so the summation order is unchanged.
_pool = ThreadPoolExecutor(max_workers=MAX_THREADS, thread_name_prefix="q8")


@dataclass
class PackedTensor:
    data: NDArray[np.uint8]
    shape: tuple[int, ...]
    qt: gguf.GGMLQuantizationType | None

    @classmethod
    def load(cls, path: Path, shape: tuple[int, ...], code: int | None = None,
             *, warm: bool = False) -> "PackedTensor":
        raw = np.fromfile(path, dtype="u1") if warm else np.memmap(path, dtype="u1", mode="r")
        qt = gguf.GGMLQuantizationType(code) if code is not None else None
        if qt is not None:
            block, size = gguf.GGML_QUANT_SIZES[qt]
            raw = raw.reshape(*shape[:-1], shape[-1] // block * size)
        elif raw.size != (4 * shape[0] if len(shape) == 1 else 4 * shape[0] + math.prod(shape)):
            raise ValueError("invalid row-quantized tensor size")
        return cls(raw, shape, qt)

    def decode(self, index: slice | NDArray[np.intp] = slice(None), expert: int | None = None) -> FloatArray:
        if self.qt is not None:
            data = self.data if expert is None else self.data[expert]
            return gguf.dequantize(data[index], self.qt)
        if len(self.shape) == 1:
            return self.data.view("<f4")[index]
        rows = self.shape[0]
        scales = self.data[:4 * rows].view("<f4").reshape(rows, 1)
        quantized = self.data[4 * rows:].view("i1").reshape(self.shape)
        return np.asarray(quantized[index] * scales[index], dtype=np.float32)

    def multiply(self, x: FloatArray, expert: int | None = None) -> FloatArray:
        rows, cols = self.shape[-2:]
        out = np.empty((*x.shape[:-1], rows), dtype="f4")
        if (NATIVE_Q8 and matvec is not None and x.size == cols
                and self.qt == gguf.GGMLQuantizationType.Q8_0):
            packed = self.data if expert is None else self.data[expert]
            vector = np.require(x, dtype=np.float32, requirements=["C", "A"])
            workers = min(q8_threads.get(), rows, max(1, packed.nbytes // (512 * 1024)))
            if workers == 1:
                matvec(packed, vector, out)
            else:
                result = out.reshape(-1)
                jobs = []
                for i in range(workers):
                    span = slice(i * rows // workers, (i + 1) * rows // workers)
                    if i + 1 == workers:
                        matvec(packed[span], vector, result[span])
                    else:
                        jobs.append(_pool.submit(matvec, packed[span], vector, result[span]))
                for job in jobs:
                    job.result()
            return out
        # Target at most 4 MiB of decoded weights (or one very wide row).
        step = max(1, min(256, 2**20 // cols))
        for j in range(0, rows, step):
            block = slice(j, j + step)
            out[..., block] = x @ self.decode(block, expert).T
        return out


type Weights = dict[str, PackedTensor]
resident: dict[str, Weights] = {}
