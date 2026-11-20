import io
import unittest
from unittest.mock import patch

from armada.progress import Progress, terminal_progress


class ProgressTest(unittest.TestCase):
    def test_acknowledged_bytes_and_eta_reset_per_part(self) -> None:
        stream = io.StringIO()
        p = Progress(stream)
        with patch("armada.progress.time.monotonic", side_effect=[0, 0, 2, 2.1, 3, 3]):
            p.start("part 1/2", 8192)
            p.advance(2048)
            p.advance(6144)
            p.start("part 2/2", 4096)
        lines = stream.getvalue().splitlines()
        self.assertTrue(any("25%" in s and "1.0 KiB/s" in s and "ETA 0m 06s" in s for s in lines))
        self.assertTrue(any("100%" in s for s in lines))
        self.assertIn("0% · 0 B / 4.0 KiB", lines[-1])

    def test_redirected_stderr(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stream, terminal_progress() as p:
            self.assertIsNone(p)
            self.assertEqual(stream.getvalue(), "")
