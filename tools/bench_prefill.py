"""Compare warm first-stage prefill paths in isolated processes (does not restart workers)."""
import argparse
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter


def measure(model: str, prompt: str, repeats: int, threads: int, mode: str, ref: Path) -> None:
    import numpy as np
    from armada.bpe import load_tokenizer
    from armada.common import home, state
    from armada.kv import KV
    from armada.qwen import gg_stage
    from armada.residency import NATIVE_Q8, q8_threads
    from armada.runtime import prepare_weights

    s = state()
    m = s["models"][model]
    if "qwen" not in m or m["owners"][0] != s["uid"]:
        raise SystemExit("run on the first layer owner of a supported GGUF model")
    if not NATIVE_Q8:
        raise SystemExit("build and enable the native Q8 extension before comparing paths")
    end = next((i for i, owner in enumerate(m["owners"]) if owner != s["uid"]), len(m["owners"]))
    tokens = load_tokenizer(home() / "weights" / m["id"] / "tokenizer.bin").encode(prompt)
    if not 2 <= len(tokens) <= min(32, m["qwen"]["ctx"]):
        raise SystemExit(f"expected a 2–32 token short prompt, got {len(tokens)}")
    w = prepare_weights(m, s["uid"])
    q8_threads.set(threads)
    times = []
    for _ in range(repeats):
        cache = KV()
        start = perf_counter()
        if mode == "sequential":
            rows = []
            for token in tokens:
                output = gg_stage(m, 0, end, [token], weights_cache=w, cache=cache, return_logits=True)
                if isinstance(output, list):
                    rows.extend(output)
                cache.offset += 1
            value = np.asarray(rows if rows else output)
        else:
            value = np.asarray(gg_stage(m, 0, end, tokens, weights_cache=w, cache=cache, return_logits=True))
        times.append(perf_counter() - start)
        if ref.exists():
            np.testing.assert_allclose(value, np.load(ref), atol=2e-3, rtol=2e-3)
        else:
            np.save(ref, value)
    print(f"{mode}: {statistics.median(times):.3f} s median ({len(tokens)} tokens, layers 0–{end-1})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("prompt")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, help="native Q8 threads (default: calibrated count)")
    p.add_argument("--mode", choices=["blas-default", "blas-1", "sequential", "packed"], help=argparse.SUPPRESS)
    p.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    from armada.autotune import selected
    from armada.common import state
    threads = args.threads if args.threads is not None else selected(state()["models"][args.model]["id"])
    if args.repeats < 1 or not 1 <= threads <= 32:
        p.error("expected positive repeats and 1–32 threads")
    if args.mode:
        measure(args.model, args.prompt, args.repeats, threads, args.mode, args.reference)
        return
    with tempfile.TemporaryDirectory() as folder:
        for mode in ("blas-default", "blas-1", "sequential", "packed"):
            env = dict(os.environ, ARMADA_Q8_PREFILL="1" if mode == "packed" else "0")
            if mode == "blas-1":
                env["OPENBLAS_NUM_THREADS"] = "1"
            subprocess.run([sys.executable, __file__, args.model, args.prompt, "--mode", mode, "--threads",
                            str(threads), "--repeats", str(args.repeats), "--reference", f"{folder}/ref.npy"],
                           env=env, check=True)


if __name__ == "__main__":
    main()
