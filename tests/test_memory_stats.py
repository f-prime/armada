import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from armada import runtime


class MemoryStatsTest(unittest.TestCase):
    def test_linux(self) -> None:
        with patch.object(sys, "platform", "linux"), patch.object(
            Path, "read_text", return_value="MemTotal:  8192 kB\nMemAvailable:  512 kB\n"
        ):
            self.assertEqual(runtime.memory_stats(), (512 * 1024, 8192 * 1024))

    def test_mac(self) -> None:
        vm = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
              "Pages free:  123.\nPages inactive:  45.\n")
        with patch.object(sys, "platform", "darwin"), patch.object(
            subprocess, "check_output", side_effect=["8589934592\n", vm]
        ), patch.object(Path, "read_text", side_effect=AssertionError("unexpected file probe")):
            self.assertEqual(runtime.memory_stats(), (168 * 16384, 8589934592))

    def test_mac_missing_vm_stat(self) -> None:
        with patch.object(sys, "platform", "darwin"), patch.object(
            subprocess, "check_output", side_effect=["8589934592\n", FileNotFoundError()]
        ):
            self.assertEqual(runtime.memory_stats(), (None, 8589934592))
