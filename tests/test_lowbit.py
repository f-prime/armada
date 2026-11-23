import unittest
from unittest.mock import patch

import gguf
import numpy as np

from armada import residency
from armada.residency import PackedTensor, QT


@unittest.skipUnless(residency.NATIVE_LOWBIT, "native low-bit kernels not enabled")
class PackedLowbitTest(unittest.TestCase):
    def test_reject_incompatible_buffers(self) -> None:
        from armada._q8 import lowbit_matvec

        w = bytes(144)
        x = np.ones(256, dtype="f4")
        y = np.zeros(1, dtype="f4")
        for data, vector, batch, qt in ((w[:-1], x, 1, QT.Q4_K), (w, x[:-1], 1, QT.Q4_K),
                                        (w, x, 0, QT.Q4_K), (w, x, 1, QT.Q2_K)):
            with self.assertRaises(ValueError):
                lowbit_matvec(data, vector, y, batch, qt)
        self.assertEqual(y[0], 0)

    def test_block_layouts_against_gguf(self) -> None:
        rng = np.random.default_rng(253)
        for qt in (QT.Q4_K, QT.Q5_K, QT.Q6_K, QT.Q4_0, QT.Q4_1):
            with self.subTest(qt=qt):
                width, size = gguf.GGML_QUANT_SIZES[qt]
                # Random payloads exercise high bits, signed subscales and both
                # nibbles. Supply finite fp16 superblock scales/minimums.
                b = rng.integers(256, size=(2, 19, 3, size), dtype="u1")
                offsets = (208,) if qt == QT.Q6_K else (0,) if qt == QT.Q4_0 else (0, 2)
                for j in offsets:
                    b[..., j:j+2] = rng.uniform(-.02, .02, b.shape[:-1]).astype("<f2").view("u1").reshape(*b.shape[:-1], 2)
                data = b.reshape(2, 19, -1)
                tensor = PackedTensor(data, (2, 19, width*3), qt)
                w = gguf.dequantize(data[1], qt)
                for batch in (1, 7, 32):
                    x = rng.normal(size=(batch, width*3*2)).astype("f4")[:, ::2]
                    with patch.object(PackedTensor, "decode", side_effect=AssertionError("used NumPy fallback")), \
                         patch.object(residency, "_PREFILL_LIMIT", 32):
                        got = tensor.multiply(x if batch > 1 else x[0], expert=1)
                    np.testing.assert_allclose(got, (x if batch > 1 else x[0]) @ w.T,
                                               atol=0.002, rtol=2e-5)
