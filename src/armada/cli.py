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
    status_cmd = groups.add_parser("status", help="show the fleet summary")
    status_cmd.add_argument("--json", action="store_true", help="print the full fleet snapshot")
    status_cmd.set_defaults(op="status", keys=[])
    daemon = groups.add_parser("daemon", help="run a foreground worker")
    daemon.add_argument(
        "--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)"
    )
    daemon.add_argument(
        "--port", type=int, default=7318, help="TCP port (default: 7318)"
    )
    daemon.add_argument(
        "--advertise-host",
        help="reachable hostname to announce to peers (e.g. MagicDNS)",
    )
    daemon.set_defaults(op="daemon", keys=["host", "port"])
    fleet = groups.add_parser("peer", aliases=["fleet"]).add_subparsers(required=True)
    for name, op, keys in [
        ("add", "add", ["address"]),
        ("pairing-code", "pair", []),
        ("status", "status", []),
    ]:
        cmd = fleet.add_parser(name)
        if op == "status":
            cmd.add_argument("--json", action="store_true", help="print the full fleet snapshot")
        for key in keys:
            cmd.add_argument(key)
        if op == "add":
            cmd.add_argument(
                "--return-address",
                help="host:port peers should use to reach this daemon",
            )
        cmd.set_defaults(op=op, keys=keys)
    model = groups.add_parser("model").add_subparsers(required=True)
    add = model.add_parser("add")
    add.add_argument("source")
    add.add_argument("--name", required=True)
    add.add_argument("--tokenizer", help="legacy llama2.c binary tokenizer")
    add.add_argument(
        "--no-local-weights",
        action="store_true",
        help="place weights on other peers only",
    )
    add.set_defaults(op="import-gguf", keys=["name", "source"])
    status = model.add_parser("status")
    status.add_argument("name")
    status.set_defaults(op="model", keys=["name"])
    red = model.add_parser(
        "redistribute", help="rebuild placement across reachable peers"
    )
    red.add_argument("name")
    red.add_argument(
        "--source", default="", help="replacement location for the original checkpoint"
    )
    red.add_argument(
        "--tokenizer", default="", help="replacement location for the legacy tokenizer"
    )
    red.set_defaults(op="redistribute", keys=["name", "source", "tokenizer"])
    run = groups.add_parser("run")
    run.add_argument("name")
    run.add_argument("prompt")
    run.add_argument("-n", "--tokens", type=int, default=32)
    run.set_defaults(op="run", keys=["name", "prompt", "tokens"])
    args = parser.parse_args(rest)
    if args.op == "daemon" and args.advertise_host:
        args.keys.append("advertise_host")
    if args.op == "add" and args.return_address:
        args.keys.append("return_address")
    if options.home is not None:
        os.environ["ARMADA_HOME"] = str(options.home)
    if args.op == "import-gguf" and args.tokenizer:
        args.op = "import"
        args.keys.append("tokenizer")
    if args.op in ("import", "import-gguf") and args.no_local_weights:
        args.keys.append("no_local_weights")
    from .runtime import execute

    try:
        result = execute([args.op, *(str(getattr(args, k)) for k in args.keys)])
        if args.op == "status" and not args.json and isinstance(result, dict):
            from .status import format_status

            result = format_status(result)
        print(result if isinstance(result, str) else json.dumps(result, indent=2))
    except (Exception, KeyboardInterrupt) as e:
        print(f"armada: {e}", file=sys.stderr)
        return 1
    return 0
