import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from armada.runtime import prepare_weights
from armada.schema import Model


class PackedWeightsTest(unittest.TestCase):
    def test_warm_budget_and_blocked_row_quantized_projection(self) -> None:
        rows, cols = 513, 32
        rng = np.random.default_rng(12)
        q = rng.integers(-127, 128, (rows, cols), dtype=np.int8)
        scales = rng.uniform(0.01, 1, (rows, 1)).astype("<f4")
        size = q.nbytes + scales.nbytes
        model: Model = dict(id="tiny", owners=["first", "last"],
                            config=[cols, cols, 2, 1, 1, rows, 16],
                            files={"emb": dict(size=size, digest=""),
                                   "q_1": dict(size=cols * (cols + 4), digest="")})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "weights" / "tiny"
            folder.mkdir(parents=True)
            path = folder / "emb.bin"
            path.write_bytes(scales.tobytes() + q.tobytes())
            with (patch("armada.runtime.home", return_value=Path(tmp)),
                  patch("armada.runtime.memory_stats", return_value=(size + 1, 4 * size))):
                weights = prepare_weights(model, "first")
            path.unlink()
        self.assertEqual(set(weights), {"emb"})
        emb = weights["emb"]
        self.assertEqual(emb.data.nbytes, size)
        dense = q * scales
        idx = np.array([512, 0, 512], dtype=np.intp)
        np.testing.assert_array_equal(emb.decode(idx), dense[idx])
        x = rng.normal(size=(2, cols)).astype("f4")
        np.testing.assert_allclose(emb.multiply(x), x @ dense.T, rtol=1e-5, atol=1e-4)
