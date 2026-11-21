import unittest
from unittest.mock import patch

import gguf
import numpy as np

from armada import residency
from armada.residency import PackedTensor


@unittest.skipUnless(residency.NATIVE_Q8, "native Q8 extension not enabled")
class PackedMatvecTest(unittest.TestCase):
    def test_parallel_rows_identical_for_uneven_expert(self) -> None:
        rng = np.random.default_rng(10)
        qt = gguf.GGMLQuantizationType.Q8_0
        shape = (2, 1031, 2048)
        packed = gguf.quantize(rng.normal(size=shape).astype("f4"), qt)
        tensor = PackedTensor(packed, shape, qt)
        x = rng.normal(size=(1, shape[-1])).astype("f4")
        ticket = residency.q8_threads.set(1)
        try:
            reference = tensor.multiply(x, 1)
            residency.q8_threads.set(4)
            np.testing.assert_array_equal(tensor.multiply(x, 1), reference)
        finally:
            residency.q8_threads.reset(ticket)

    def test_vector_and_expert_against_dequantized_reference(self) -> None:
        rng = np.random.default_rng(42)
        qt = gguf.GGMLQuantizationType.Q8_0
        for cols in (32, 128, 2048):
            weights = rng.normal(size=(2, 13, cols)).astype("f4")
            weights[1, 0] = 0
            data = gguf.quantize(weights, qt)
            tensor = PackedTensor(data, weights.shape, qt)
            x = rng.normal(size=cols * 2).astype("f4")[::2]
            for vector in (x, x[None, :]):
                reference = vector @ gguf.dequantize(data[1], qt).T
                with patch.object(residency, "NATIVE_Q8", True), patch.object(
                    PackedTensor, "decode", side_effect=AssertionError("decoded a weight tile")
                ):
                    actual = tensor.multiply(vector, expert=1)
                np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=5e-5)
