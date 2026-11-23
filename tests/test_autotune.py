import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gguf
import numpy as np

from armada import autotune, residency
from armada.residency import DEFAULT_THREADS, PackedTensor, q8_threads


class DecodeTuningTest(unittest.TestCase):
    def test_persistence_invalidation_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ARMADA_HOME=tmp), \
             patch.object(residency, "NATIVE_LOWBIT", True), patch.object(residency, "NATIVE_Q8", False), \
             patch.object(autotune, "_environment", return_value=("cpu-A", 4)) as env:
            os.environ.pop("ARMADA_Q8_THREADS", None)
            (Path(tmp) / "weights" / "model").mkdir(parents=True)
            w = {"w": PackedTensor(np.empty((2048, 4*144), dtype="u1"), (2048, 1024),
                                   gguf.GGMLQuantizationType.Q4_K)}
            clock = 0.0
            calls = []

            def decode() -> None:
                nonlocal clock
                n = q8_threads.get()
                calls.append(n)
                clock += {1: 3, 2: 1, 4: 2}[n]

            self.assertEqual(autotune.selected("model"), DEFAULT_THREADS)
            with patch.object(autotune, "perf_counter", side_effect=lambda: clock):
                autotune.calibrate("model", w, decode)
            self.assertEqual(set(calls), {1, 2, 4})
            autotune._read.cache_clear()  # simulate a restarted daemon
            self.assertEqual(autotune.selected("model"), 2)
            autotune.calibrate("model", w, lambda: self.fail("reran valid calibration"))
            self.assertEqual(autotune.selected("new-placement"), DEFAULT_THREADS)
            with patch.dict(os.environ, ARMADA_Q8_THREADS=str(DEFAULT_THREADS)):
                self.assertEqual(autotune.selected("model"), DEFAULT_THREADS)
                autotune.calibrate("model", w, lambda: self.fail("ignored explicit setting"))
            env.return_value = ("cpu-B", 1)
            self.assertEqual(autotune.selected("model"), DEFAULT_THREADS)
            autotune.calibrate("model", w, decode)
            self.assertEqual(autotune.selected("model"), 1)
            self.assertEqual(q8_threads.get(), DEFAULT_THREADS)
