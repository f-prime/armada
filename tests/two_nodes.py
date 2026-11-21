"""Integration check: pass paths to stories260K.bin and tok512.bin."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
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
        m: Model = json.loads(cmd(0, *add))
        ids = [
            json.loads((root / str(i) / "node.json").read_text())["uid"] for i in (0, 1)
        ]
        assert set(m["owners"]) == set(ids)
        for i in (0, 1):
            shard = root / str(i) / "weights" / m["id"]
            layers = {j for j, a in enumerate(m["owners"]) if a == ids[i]}
            if tokenizer:
                assert {p.name for p in shard.glob("q_*.bin")} == {
                    f"q_{j}.bin" for j in layers
                }
            else:
                names = {p.name for p in shard.glob("blk_*.bin")}
                assert names and {int(name.split("_")[1]) for name in names} == layers
                tied = bool(m["qwen"].get("tied_output"))
                assert (shard / "token_embd_weight.bin").exists() == (i == 0 or tied)
                assert (shard / "output_weight.bin").exists() == (i == 1 and not tied)
        stop0()
        stop1()
        if inspect_shards:
            inspect_shards(m, root)
        start(2)
        cmd(2, *add)
        args = ("run", "tiny", "Once upon a time", "-n", "16")
        expected = cmd(2, *args)
        assert cmd(2, *args, "--no-stream") == expected
        stop0, stop1 = start(0), start(1)
        cmd(1, "model", "warm", "tiny")
        status = json.loads(cmd(0, "model", "status", "tiny", "--json"))
        assert status["warmth"] == "warm" and status["warmth_by_owner"] == dict.fromkeys(ids, "warm")
        # A warm generation must actually use prepared tensors on both owners.
        with ExitStack() as restore:
            for i in (0, 1):
                folder = root / str(i) / "weights" / m["id"]
                for p in folder.glob("*.bin"):
                    if p.stem != "tokenizer":
                        restore.callback(p.rename(p.with_suffix(".held")).rename, p)
            assert cmd(0, *args, "--q8-threads", "2,4", "-v") == expected
        stop1()
        offline = json.loads(cmd(0, "model", "status", "tiny", "--json"))
        assert offline["warmth_by_owner"][ids[1]] == "unknown"
        stop1 = start(1)
        assert json.loads(cmd(0, "model", "status", "tiny", "--json"))["warmth"] == "partially warm"
        cmd(0, "model", "cold", "tiny")
        assert json.loads(cmd(1, "model", "status", "tiny", "--json"))["warmth"] == "cold"
        assert cmd(1, *args) == expected
        stop3 = start(3)
        # Third peer drives import but stores no weights.
        cmd(0, "fleet", "add", addresses[3], stdin=cmd(3, "fleet", "pairing-code") + "\n")
        other = list(add)
        other[other.index("--name") + 1] = "recovery"
        recovery = json.loads(cmd(3, *other, "--no-local-weights"))
        assert set(recovery["owners"]) == set(ids)
        stop3()
        run_recovery = ("run", "recovery", *args[2:])
        assert cmd(0, *run_recovery) == expected
        stop1()
        # Missing source must not destroy the only remaining good shard.
        failure = subprocess.run(
            [
                cli,
                "model",
                "redistribute",
                "recovery",
                "--source",
                str(root / "absent"),
            ],
            env=env(0),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert failure.returncode and (root / "0" / "weights" / recovery["id"]).is_dir()
        assert json.loads(cmd(0, "model", "status", "recovery", "--json"))["id"] == recovery["id"]
        new = json.loads(cmd(0, "model", "redistribute", "recovery"))
        assert set(new["owners"]) == {ids[0]}
        assert not (root / "0" / "weights" / recovery["id"]).exists()
        assert cmd(0, *run_recovery) == expected
        stop1 = start(1)
        until = time.monotonic() + 20
        while (root / "1" / "weights" / recovery["id"]).exists():
            assert time.monotonic() < until, "returning worker kept obsolete weights"
            time.sleep(0.1)
        assert cmd(1, *run_recovery) == expected
        if reference:
            # No GGUF source remains for the next redistribution.
            for path in Path(checkpoint).parent.glob("*.gguf"):
                path.unlink()
        moved = json.loads(
            cmd(
                1, "model", "redistribute", "recovery", "--source", str(root / "absent")
            )
        )
        assert set(moved["owners"]) == set(ids)
        assert cmd(1, *run_recovery) == expected
        # Rebind a layer owner to another port. Its ID and placement survive.
        stop1()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            addresses[1] = f"127.0.0.1:{sock.getsockname()[1]}"
        stop1 = start(1)
        until = time.monotonic() + 20
        while (
            json.loads((root / "0" / "node.json").read_text())["peers"][ids[1]][
                "endpoint"
            ]
            != addresses[1]
        ):
            assert time.monotonic() < until, "peer retained old endpoint"
            time.sleep(0.1)
        first = cmd(0, *args)
        assert first and first == cmd(1, *args)
        print("Two-node completion:", first)
        # Compare against the same quantization with no network boundary.
        assert first == cmd(2, *args)
        if reference:
            assert first == reference.strip(), (first, reference)
        if tokenizer:
            stop1()
            removed = json.loads(cmd(0, "model", "delete", "tiny", "--json"))
            assert ids[0] in removed["confirmed"] and ids[1] in removed["pending"]
            assert not (root / "0" / "weights" / m["id"]).exists()
            assert (root / "1" / "weights" / m["id"]).is_dir()
            assert "tiny" not in [r["name"] for r in json.loads(cmd(0, "model", "list", "--json"))]
            replacement = json.loads(cmd(0, *add))
            assert replacement["id"] != m["id"]
            start(1)
            until = time.monotonic() + 20
            while (root / "1" / "weights" / m["id"]).exists():
                assert time.monotonic() < until, "returning peer retained deleted weights"
                time.sleep(0.1)
            assert json.loads(cmd(1, "model", "status", "tiny", "--json"))["id"] == replacement["id"]
            assert cmd(1, *args) == first


if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2])
