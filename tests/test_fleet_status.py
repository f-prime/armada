import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from armada import cli, gpu
from armada.schema import Fields


class FleetStatusTest(unittest.TestCase):
    def test_macos_inventory(self) -> None:
        report = {"SPDisplaysDataType": [{"sppci_model": "Apple M4"},
                  {"sppci_model": "Radeon Pro", "spdisplays_vram": "8 GB"}]}
        with patch.object(sys, "platform", "darwin"), patch.object(
            gpu, "_output", return_value=json.dumps(report)
        ):
            self.assertEqual(gpu.gpu_stats(), [
                {"name": "Apple M4", "unified_memory": True},
                {"name": "Radeon Pro", "total_vram": 8 * 2**30}])

    def test_linux_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for bus, vendor in [("01", "0x10de"), ("02", "0x1002"), ("03", "0x8086")]:
                dev = root / f"bus/pci/devices/0000:{bus}:00.0"
                dev.mkdir(parents=True)
                (dev / "class").write_text("0x030000")
                (dev / "vendor").write_text(vendor)
                if bus == "02":
                    (dev / "mem_info_vram_total").write_text("4096")
                    (dev / "mem_info_vram_used").write_text("1024")
            with patch("shutil.which", return_value="nvidia-smi"), patch.object(
                gpu, "_output", return_value="00000000:01:00.0, T4, [N/A], 100"
            ):
                found = gpu._linux(root)
            self.assertEqual([g["vendor"] for g in found], ["NVIDIA", "AMD", "Intel"])
            self.assertEqual(found[0]["total_vram"], 100 * 2**20)
            self.assertNotIn("free_vram", found[0])
            self.assertEqual(found[1]["free_vram"], 3072)

    def test_cli_formats(self) -> None:
        snapshot: Fields = dict(fleet="home", nodes=["mac", "old", "failed"], models={},
                                self="mac", resources={
                                    "mac": {"gpus": [{"name": "Apple M4", "unified_memory": True}]},
                                    "old": {"nvidia_gpus": []}, "failed": {"error": "timeout"}})
        for args in (["status"], ["peer", "status"], ["fleet", "status", "--json"]):
            out = io.StringIO()
            with patch.object(sys, "argv", ["armada", *args]), redirect_stdout(out), patch(
                "armada.runtime.execute", return_value=snapshot
            ):
                self.assertEqual(cli.main(), 0)
            if "--json" in args:
                self.assertEqual(json.loads(out.getvalue()), snapshot)
            else:
                for text in ("mac (you)", "Apple M4", "shared with RAM", "older peer", "timeout"):
                    self.assertIn(text, out.getvalue())
