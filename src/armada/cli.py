"""Armada's installed CLI entry point."""

import argparse
import json
import os
import sys
from pathlib import Path
from importlib.metadata import version


def main() -> int:
    # Parse --home at any nesting level, as in the original CLI.
    global_args = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    global_args.add_argument("--home", type=Path)
    options, rest = global_args.parse_known_args()
    parser = argparse.ArgumentParser(prog="armada", parents=[global_args])
    parser.add_argument(
        "--version", action="version", version=f"armada {version('armada')}"
    )
    groups = parser.add_subparsers(required=True)
    daemon = groups.add_parser("daemon", help="run a foreground worker")
    daemon.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    daemon.add_argument("--port", type=int, default=7318, help="TCP port (default: 7318)")
    daemon.set_defaults(op="daemon", keys=["host", "port"])
    fleet = groups.add_parser("fleet").add_subparsers(required=True)
    for name, op, keys in [
        ("init", "init", ["name"]),
        ("add", "add", ["address"]),
        ("pairing-code", "pair", []),
        ("status", "status", []),
    ]:
        cmd = fleet.add_parser(name)
        for key in keys:
            cmd.add_argument(key)
        cmd.set_defaults(op=op, keys=keys)
    model = groups.add_parser("model").add_subparsers(required=True)
    add = model.add_parser("add")
    add.add_argument("source")
    add.add_argument("--name", required=True)
    add.add_argument("--tokenizer", help="legacy llama2.c binary tokenizer")
    add.set_defaults(op="import-gguf", keys=["name", "source"])
    status = model.add_parser("status")
    status.add_argument("name")
    status.set_defaults(op="model", keys=["name"])
    run = groups.add_parser("run")
    run.add_argument("name")
    run.add_argument("prompt")
    run.add_argument("-n", "--tokens", type=int, default=32)
    run.set_defaults(op="run", keys=["name", "prompt", "tokens"])
    args = parser.parse_args(rest)
    if options.home is not None:
        os.environ["ARMADA_HOME"] = str(options.home)
    if args.op == "import-gguf" and args.tokenizer:
        args.op = "import"
        args.keys.append("tokenizer")
    from .runtime import execute

    try:
        result = execute([args.op, *(str(getattr(args, k)) for k in args.keys)])
        print(result if isinstance(result, str) else json.dumps(result, indent=2))
    except (Exception, KeyboardInterrupt) as e:
        print(f"armada: {e}", file=sys.stderr)
        return 1
    return 0
