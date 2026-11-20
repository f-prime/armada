"""Integration check: pass paths to stories260K.bin and tok512.bin."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable
from armada.schema import Model
from contextlib import ExitStack


def test(
    checkpoint: str,
    tokenizer: str | None = None,
    reference: str | None = None,
    inspect_shards: Callable[[Model, Path], None] | None = None,
) -> None:
    with ExitStack() as stack:
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        cli = str(Path(sys.executable).with_name("armada"))

        def env(i: int) -> dict[str, str]:
            return dict(
                os.environ, ARMADA_HOME=str(root / str(i)), OPENBLAS_NUM_THREADS="1"
            )

        def cmd(i: int, *args: str, stdin: str | None = None) -> str:
            return subprocess.check_output(
                [cli, *args], env=env(i), input=stdin, text=True, timeout=30
            ).strip()

        addresses: list[str] = []

        def start(i: int) -> Callable[[], None]:
            if len(addresses) <= i:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    addresses.append(f"127.0.0.1:{sock.getsockname()[1]}")
            host, port = addresses[i].split(":")
            p = subprocess.Popen(
                [cli, "daemon", "--host", host, "--port", port],
                env=env(i),
                stdout=subprocess.PIPE,
                text=True,
            )

            def stop() -> None:
                p.terminate()
                p.wait(timeout=5)

            stack.callback(stop)
            assert p.stdout and "listening" in p.stdout.readline()
            return stop

        stop0, stop1 = start(0), start(1)
        cmd(0, "fleet", "init", "test")
        cmd(
            0,
            "fleet",
            "add",
            addresses[1],
            stdin=cmd(1, "fleet", "pairing-code") + "\n",
        )
        add = ("model", "add", checkpoint, "--name", "tiny") + (
            ("--tokenizer", tokenizer) if tokenizer else ()
        )
        m: Model = json.loads(cmd(1, *add))
        assert set(m["owners"]) == set(addresses)
        for i in (0, 1):
            shard = root / str(i) / "weights" / m["id"]
            layers = {j for j, a in enumerate(m["owners"]) if a == addresses[i]}
            if tokenizer:
                assert {p.name for p in shard.glob("q_*.bin")} == {
                    f"q_{j}.bin" for j in layers
                }
            else:
                names = {p.name for p in shard.glob("blk_*.bin")}
                assert names and {int(name.split("_")[1]) for name in names} == layers
                assert (shard / "token_embd_weight.bin").exists() == (i == 0)
                assert (shard / "output_weight.bin").exists() == (i == 1)
        stop0()
        stop1()
        if inspect_shards:
            inspect_shards(m, root)
        start(2)
        cmd(2, "fleet", "init", "solo")
        cmd(2, *add)
        if reference:
            # Remove the entire import source before restarting either worker.
            for path in Path(checkpoint).parent.glob("*.gguf"):
                path.unlink()
        start(0)
        start(1)
        args = ("run", "tiny", "Once upon a time", "-n", "16")
        first = cmd(0, *args)
        assert first and first == cmd(1, *args)
        print("Two-node completion:", first)
        # Compare against the same quantization with no network boundary.
        assert first == cmd(2, *args)
        if reference:
            assert first == reference.strip(), (first, reference)


if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2])
