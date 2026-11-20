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
from armada.status import fleet_totals


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

    def test_fleet_totals(self) -> None:
        resources: dict[str, Fields] = {
            "mac": dict(cpu_threads=8, total_ram=128, free_ram=0, total_disk=512,
                        free_disk=32, gpus=[{"unified_memory": True, "total_vram": 128}]),
            "pc": dict(cpu_threads=4, total_ram=64, free_ram=16, total_disk=256,
                       free_disk=24, gpus=[{"free_vram": 4, "total_vram": 8}])}
        self.assertEqual(fleet_totals(resources), dict(cpu_threads=12, total_ram=192,
                         free_ram=16, total_disk=768, free_disk=56, gpu_count=2,
                         free_vram=4, total_vram=8, partial=[]))
        resources["old"] = {"cpu_threads": 2, "nvidia_gpus": [{"total_vram": 12}]}
        totals = fleet_totals(resources)
        self.assertEqual((totals["gpu_count"], totals["total_vram"]), (3, 20))
        self.assertNotIn("cpu_threads", totals["partial"])
        self.assertIn("total_vram", totals["partial"])
        missing = fleet_totals({"offline": {"error": "timeout"}})
        self.assertIsNone(missing["cpu_threads"])
        self.assertIn("cpu_threads", missing["partial"])

    def test_cli_formats(self) -> None:
        snapshot: Fields = dict(fleet="home", nodes=["mac", "old", "failed"], models={},
                                self="mac", peers={"failed": {"endpoint": "10.0.0.3:7318"}}, resources={
                                    "mac": {"gpus": [{"name": "Apple M4", "unified_memory": True}]},
                                    "old": {"nvidia_gpus": []}, "failed": {"error": "timeout"}})
        for args in (["status"], ["fleet", "status"], ["fleet", "status", "--json"]):
            out = io.StringIO()
            with patch.object(sys, "argv", ["armada", *args]), redirect_stdout(out), patch(
                "armada.runtime.execute", return_value=snapshot
            ):
                self.assertEqual(cli.main(), 0)
            if "--json" in args:
                self.assertEqual(json.loads(out.getvalue()), snapshot)
            else:
                for text in ("mac (you)", "Apple M4", "shared with RAM", "older peer", "timeout",
                             "10.0.0.3", "Fleet totals:", "(partial)"):
                    self.assertIn(text, out.getvalue())
