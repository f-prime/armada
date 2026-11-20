"""Human-readable fleet snapshot (including snapshots from older peers)."""

from itertools import groupby

from .schema import Fields


def format_models(models: list[Fields]) -> str:
    if not models:
        return "No models registered in the fleet."
    width = max(len("MODEL"), max(len(m["name"]) for m in models))
    rows = [f"{'MODEL':<{width}}  PEERS"]
    rows.extend(f"{m['name']:<{width}}  {m['peer_count']}" for m in models)
    return "\n".join(rows)


def format_model_status(name: str, model: Fields) -> str:
    owners: list[str] = model["owners"]
    missing: dict[str, list[int]] = model["unavailable_layers"]
    layers, peers = len(owners), len(set(owners))
    unavailable = sum(map(len, missing.values()))
    status = (f"{unavailable} layer{'s' if unavailable != 1 else ''} unavailable"
              if unavailable else "All layer owners reachable")
    storage: dict[str, int | None] = model["storage_bytes_by_peer"]
    total = format_size(sum(size for size in storage.values() if size is not None))
    unknown = sum(size is None for size in storage.values())
    summary = (f"{total} known · {unknown} peer{'s' if unknown != 1 else ''} unreported"
               if unknown else f"{total} total")
    lines = [f"Model: {name} · {layers} layer{'s' if layers != 1 else ''} across "
             f"{peers} peer{'s' if peers != 1 else ''}", f"Status: {status}", f"Storage: {summary}", ""]
    spans: dict[str, list[str]] = {}
    start = 0
    for owner, group in groupby(owners):
        end = start + sum(1 for _ in group)
        span = str(start) if end == start + 1 else f"{start}–{end - 1}"
        spans.setdefault(owner, []).append(span)
        start = end
    rows = [("PEER", "LAYERS", "STORED", "STATUS")]
    for peer, size in storage.items():
        reach = ("Unreachable" if peer in missing else "Reachable"
                 if peer in spans or size is not None else "Unreported")
        rows.append((peer[:8], ", ".join(spans.get(peer, [])) or "—",
                     "Unknown" if size is None else format_size(size), reach))
    widths = [max(map(len, column)) for column in zip(*rows)]
    lines.extend("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows)
    return "\n".join(lines)


def format_size(value: int | None) -> str:
    if value is None:
        return "—"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024 or unit == "PiB":
            return f"{size:g} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError


def format_status(snapshot: Fields) -> str:
    nodes, models = snapshot["nodes"], snapshot["models"]
    lines = [f"Fleet: {snapshot['fleet']} · {len(nodes)} node{'s' if len(nodes) != 1 else ''}"
             f" · {len(models)} model{'s' if len(models) != 1 else ''}"]
    if models:
        lines.append("Models: " + ", ".join(sorted(models)))
    rows = [("NODE", "THREADS", "LOAD (1m)", "RAM FREE / TOTAL", "DISK FREE")]
    details = []
    for node in nodes:
        label = node[:8]
        stats = snapshot.get("resources", {}).get(node, {})
        rows.append((label + (" (you)" if node == snapshot.get("self") else ""),
                     str(stats.get("cpu_threads") or "—"),
                     f"{stats['cpu_load_1m']:.2f}" if stats.get("cpu_load_1m") is not None else "—",
                     f"{format_size(stats.get('free_ram'))} / {format_size(stats.get('total_ram'))}",
                     format_size(stats.get("free_disk"))))
        if "error" in stats or not stats:
            details.append(f"{label}  Resource check failed: {stats.get('error', 'no report')}")
            continue
        gpus = stats.get("gpus", stats.get("nvidia_gpus", []))
        if not gpus:
            details.append(f"{label}  No GPUs reported" if "gpus" in stats else
                           f"{label}  No NVIDIA GPUs reported (older peer)")
        for gpu in gpus:
            memory = ("unified (shared with RAM)" if gpu.get("unified_memory") else
                      f"{format_size(gpu.get('free_vram'))} free / {format_size(gpu.get('total_vram'))} total"
                      if "total_vram" in gpu or "free_vram" in gpu else "memory unknown")
            details.append(f"{label}  GPU: {gpu['name']} · {memory}")
        if stats.get("active_models"):
            details.append(f"{label}  Active models: {', '.join(stats['active_models'])}")
    widths = [max(map(len, column)) for column in zip(*rows)]
    lines.append("")
    lines.extend("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows)
    if details:
        lines.extend(["", *details])
    return "\n".join(lines)
