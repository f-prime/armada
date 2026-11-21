"""Human-readable fleet snapshot (including snapshots from older peers)."""

from collections.abc import Sequence
from itertools import groupby

from .schema import Fields, Peer


def format_timings(report: Fields, peers: dict[str, Peer]) -> str:
    def ms(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1f} ms"

    rate = report["tokens_per_second"]
    speed = "n/a" if rate is None else f"{rate:.2f} tokens/s"
    lines = [f"First token: {ms(report['first_ms'])} · Generation after first token: {speed}"]
    def table(rows: Sequence[tuple[str, ...]]) -> None:
        widths = [max(map(len, column)) for column in zip(*rows)]
        lines.extend("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows)

    if "prompt_tokens" in report:
        lines.append(f"Prompt: {report['prompt_tokens']} tokens · Setup before first stage: {ms(report['setup_ms'])}")
        prefill = [("OWNER", "LAYERS", "PREFILL ROUND TRIP", "WORKER", "RPC OVERHEAD")]
        for row in report["owners"]:
            owner = row["owner"]
            label = peers[owner]["endpoint"] if owner in peers else owner
            worker = row["prefill_compute_ms"]
            overhead = max(0.0, row["prefill_ms"] - worker) if worker is not None else None
            prefill.append((label, row["layers"], ms(row["prefill_ms"]), ms(worker), ms(overhead)))
        table(prefill)
        lines.append("")
    rows = [("OWNER", "LAYERS", "AVG ROUND TRIP / TOKEN AFTER FIRST", "WORKER", "RPC OVERHEAD", "Q8 THREADS")]
    for row in report["owners"]:
        owner = row["owner"]
        label = peers[owner]["endpoint"] if owner in peers else owner
        compute = row.get("compute_ms")
        overhead = max(0.0, row["avg_ms"] - compute) if compute is not None else None
        rows.append((label, row["layers"], ms(row["avg_ms"]), ms(compute), ms(overhead),
                     str(row["threads"]) if row.get("threads") is not None else "n/a"))
    table(rows)
    return "\n".join(lines)


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
    warmth = model.get("warmth_by_owner", {})
    ready = sum(value == "warm" for value in warmth.values())
    lines.insert(-1, f"Memory: {model.get('warmth', 'unknown')} ({ready}/{peers} owners ready)")
    spans: dict[str, list[str]] = {}
    start = 0
    for owner, group in groupby(owners):
        end = start + sum(1 for _ in group)
        span = str(start) if end == start + 1 else f"{start}–{end - 1}"
        spans.setdefault(owner, []).append(span)
        start = end
    rows = [("PEER", "LAYERS", "STORED", "STATUS", "MEMORY")]
    for peer, size in storage.items():
        reach = ("Unreachable" if peer in missing else "Reachable"
                 if peer in spans or size is not None else "Unreported")
        rows.append((peer[:8], ", ".join(spans.get(peer, [])) or "—",
                     "Unknown" if size is None else format_size(size), reach,
                     warmth.get(peer, "unknown") if peer in spans else "—"))
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


def fleet_totals(resources: dict[str, Fields]) -> Fields:
    """Sum known capacity, tracking incomplete metrics independently."""
    host_metrics = ("cpu_threads", "free_ram", "total_ram", "free_disk", "total_disk")
    gpu_metrics = ("gpu_count", "free_vram", "total_vram")
    values: dict[str, list[int | None]] = {key: [] for key in (*host_metrics, *gpu_metrics)}
    for report in resources.values():
        stats = {} if "error" in report else report
        for key in host_metrics:
            values[key].append(stats.get(key))
        if "gpus" not in stats:
            # Older peers only enumerate NVIDIA GPUs.
            for key in gpu_metrics:
                values[key].append(None)
        gpus = stats.get("gpus", stats.get("nvidia_gpus"))
        if gpus is None:
            continue
        values["gpu_count"].append(len(gpus))
        dedicated = [gpu for gpu in gpus if not gpu.get("unified_memory")]
        for key in ("free_vram", "total_vram"):
            if dedicated:
                values[key].extend(gpu.get(key) for gpu in dedicated)
            else:
                values[key].append(0)
    totals: Fields = {}
    for key, items in values.items():
        known = [value for value in items if value is not None]
        totals[key] = sum(known) if known else None
    totals["partial"] = [key for key, items in values.items() if None in items]
    return totals


def format_status(snapshot: Fields) -> str:
    nodes, models = snapshot["nodes"], snapshot["models"]
    lines = [f"Fleet: {snapshot['fleet']} · {len(nodes)} node{'s' if len(nodes) != 1 else ''}"
             f" · {len(models)} model{'s' if len(models) != 1 else ''}"]
    if models:
        lines.append("Models: " + ", ".join(sorted(models)))
    totals = snapshot.get("totals")
    if totals is None:
        totals = fleet_totals({n: snapshot.get("resources", {}).get(n, {}) for n in nodes})

    def metric(key: str) -> str:
        value = totals[key]
        text = ("—" if value is None else str(value)) if key in ("cpu_threads", "gpu_count") else format_size(value)
        return text + (" (partial)" if key in totals["partial"] else "")

    lines.extend([f"Fleet totals: {metric('cpu_threads')} CPU threads · {metric('gpu_count')} GPUs",
                  "Free / total: " + " · ".join(
                      f"{name}: {metric('free_' + key)} / {metric('total_' + key)}"
                      for name, key in (("RAM", "ram"), ("disk", "disk"), ("dedicated VRAM", "vram")))])
    rows = [("NODE", "STATUS", "ADDRESS", "THREADS", "LOAD (1m)", "RAM FREE / TOTAL", "DISK FREE")]
    details = []
    resources = snapshot.get("resources", {})
    for node in nodes:
        label = node[:8]
        stats = resources.get(node, {})
        status = "Unknown" if node not in resources else "Offline" if "error" in stats else "Online"
        endpoint = snapshot.get("peers", {}).get(node, {}).get("endpoint", "")
        rows.append((label + (" (you)" if node == snapshot.get("self") else ""),
                     status, endpoint.rsplit(":", 1)[0] or "—",
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
