"""Local GPU inventory for fleet status. Memory figures are bytes."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .schema import Fields


def _output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, timeout=8, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ""


def _mac() -> list[Fields]:
    try:
        devices = json.loads(_output(["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"]))
    except ValueError:
        return []
    result: list[Fields] = []
    for device in devices.get("SPDisplaysDataType", []):
        name = device.get("sppci_model") or device.get("_name") or "Unknown GPU"
        gpu: Fields = {"name": name}
        if "nvidia" in (name + device.get("spdisplays_vendor", "")).lower():
            gpu["vendor"] = "NVIDIA"
        if name.startswith("Apple "):
            gpu["unified_memory"] = True
        else:
            size = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(MB|GB)", device.get("spdisplays_vram", ""))
            if size:
                gpu["total_vram"] = int(float(size[1]) * (2**20 if size[2] == "MB" else 2**30))
        result.append(gpu)
    return result


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _linux(sysfs: Path) -> list[Fields]:
    devices: dict[str, Fields] = {}
    paths = {p.resolve() for p in (sysfs / "bus/pci/devices").glob("*")
             if _read(p / "class").startswith("0x03")}
    paths.update((card / "device").resolve() for card in (sysfs / "class/drm").glob("card*")
                 if re.fullmatch(r"card\d+", card.name) and (card / "device").exists())
    vendors = {"0x1002": "AMD", "0x8086": "Intel", "0x10de": "NVIDIA"}
    for path in sorted(paths):
        vendor = vendors.get(_read(path / "vendor"), "")
        gpu: Fields = {"name": " ".join(filter(None, [vendor, "GPU", _read(path / "device") or path.name]))}
        if vendor:
            gpu["vendor"] = vendor
        total, used = (_read(path / f"mem_info_vram_{key}") for key in ("total", "used"))
        if total.isdigit() and int(total) > 0:
            gpu["total_vram"] = int(total)
            if used.isdigit() and int(used) <= int(total):
                gpu["free_vram"] = int(total) - int(used)
        devices[path.name] = gpu
    if shutil.which("nvidia-smi"):
        lines = _output(["nvidia-smi", "--query-gpu=pci.bus_id,name,memory.free,memory.total",
                         "--format=csv,noheader,nounits"])
        for line in lines.splitlines():
            parts = [s.strip() for s in line.split(",")]
            if len(parts) != 4:
                continue
            bus, name, free, total = parts
            # NVIDIA pads the PCI domain to eight digits; sysfs uses four.
            bus = re.sub(r"^0000(?=[0-9a-f]{4}:)", "", bus.lower())
            gpu = devices.setdefault(bus, {})
            gpu["name"] = name
            gpu["vendor"] = "NVIDIA"
            for key, value in (("free_vram", free), ("total_vram", total)):
                if value.isdigit():
                    gpu[key] = int(value) * 2**20
    return list(devices.values())


def gpu_stats() -> list[Fields]:
    if sys.platform == "linux":
        return _linux(Path("/sys"))
    if sys.platform == "darwin":
        return _mac()
    return []
