import unittest
from unittest.mock import patch

from armada.kv import expire, release, session


class SessionTest(unittest.TestCase):
    def test_interleaved_requests_and_offset_validation(self) -> None:
        one = ("one", "model", 0, 1)
        two = ("two", "model", 0, 1)
        self.addCleanup(release, "one")
        self.addCleanup(release, "two")
        with session(one, 0, 3) as first, session(two, 0, 7) as second:
            self.assertIsNot(first, second)
            with self.assertRaisesRegex(ValueError, "concurrent"), session(one, 0, 3):
                pass
        with self.assertRaisesRegex(ValueError, "sequence"), session(one, 7, 1):
            pass
        with session(one, 3, 1) as resumed:
            self.assertIs(resumed, first)
        release("one")
        with self.assertRaisesRegex(ValueError, "expired"), session(one, 4, 1):
            pass
        with session(two, 7, 1) as resumed:
            self.assertIs(resumed, second)

    def test_failed_and_abandoned_sessions(self) -> None:
        key = ("abandoned", "model", 1, 3)
        self.addCleanup(release, key[0])
        with self.assertRaises(ArithmeticError), session(key, 0, 1):
            raise ArithmeticError
        with patch("armada.kv.monotonic", return_value=10), session(key, 0, 2):
            pass
        with patch("armada.kv.monotonic", return_value=920):
            expire()
        with self.assertRaisesRegex(ValueError, "expired"), session(key, 2, 1):
            pass
