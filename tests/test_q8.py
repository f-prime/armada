import unittest
from unittest.mock import patch

import gguf
import numpy as np

from armada import residency
from armada.residency import PackedTensor


@unittest.skipUnless(residency.NATIVE_Q8, "native Q8 extension not enabled")
class PackedMatvecTest(unittest.TestCase):
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
