"""Benchmark a running warm fleet: uv run python tools/bench_decode.py ablit."""
import argparse
import statistics

from armada.common import rpc, state


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--prompt", default="Write a numbered list of the US states and their capital cities, starting with Alabama.")
    args = p.parse_args()
    if args.tokens < 2 or args.repeats < 1 or not all(1 <= n <= 32 for n in args.counts):
        p.error("need >=2 tokens, >=1 repeat and thread counts between 1 and 32")
    s = state()
    owners = list(dict.fromkeys(s["models"][args.model]["owners"]))
    samples: dict[tuple[int, ...], list[float]] = {}
    costs: dict[tuple[int, int], list[float]] = {}
    baseline = None

    def trial(counts: tuple[int, ...]) -> None:
        nonlocal baseline
        response = rpc(s["addr"], "v1/completions", dict(
            model=args.model, prompt=args.prompt, max_tokens=args.tokens, diagnostics=True, q8_threads=counts))
        text = response["choices"][0]["text"]
        if baseline is not None and baseline != text:
            raise SystemExit("Output changed between trials")
        baseline = text
        diag = response["diagnostics"]
        rows = {row["owner"]: row for row in diag["owners"]}
        if any(rows[uid].get("threads") != n for uid, n in zip(owners, counts)):
            raise SystemExit("A worker did not confirm native Q8 threads; update and restart the fleet")
        speed = diag["tokens_per_second"]
        if not speed:
            raise SystemExit("Fewer than two tokens emitted: choose a different prompt")
        samples.setdefault(counts, []).append(speed)
        elapsed = [rows[uid]["avg_ms"] for uid in owners]
        for i, (n, ms) in enumerate(zip(counts, elapsed)):
            costs.setdefault((i, n), []).append(ms)
        print(f"{counts}: {speed:.2f} tok/s; stage ms {[round(t, 1) for t in elapsed]}", flush=True)

    for r in range(args.repeats):
        for n in args.counts[::(-1 if r % 2 else 1)]:
            trial((n,) * len(owners))
    mixed = tuple(min(args.counts, key=lambda n: statistics.median(costs[i, n])) for i in range(len(owners)))
    if mixed not in samples:
        for _ in range(args.repeats):
            trial(mixed)
    best = max(samples, key=lambda k: statistics.median(samples[k]))
    print(f"Best measured median: {statistics.median(samples[best]):.2f} tok/s; "
          f"--q8-threads {','.join(map(str, best))}")


if __name__ == "__main__":
    main()
