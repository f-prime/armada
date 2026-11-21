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
    key_commands = groups.add_parser("key", help="manage fleet inference keys").add_subparsers(required=True)
    for action in ("create", "list", "show", "rename", "rotate", "revoke"):
        sub = key_commands.add_parser(action)
        fields = []
        if action not in ("create", "list"):
            sub.add_argument("id")
            fields.append("id")
        if action in ("create", "rename"):
            if action == "create":
                sub.add_argument("--name", required=True)
            else:
                sub.add_argument("name")
            fields.append("name")
        sub.add_argument("--json", action="store_true", help="print JSON")
        sub.set_defaults(op="key-" + action, keys=fields)
    status_cmd = groups.add_parser("status", help="show the fleet summary")
    status_cmd.add_argument("--json", action="store_true", help="print the full fleet snapshot")
    status_cmd.set_defaults(op="status", keys=[])
    daemon = argparse.ArgumentParser(add_help=False)
    daemon.add_argument(
        "--host", help="bind host (default: 127.0.0.1)"
    )
    daemon.add_argument(
        "--port", type=int, help="TCP port (default: 7318)"
    )
    daemon.add_argument(
        "--advertise-host",
        help="reachable hostname to announce to peers (e.g. MagicDNS)",
    )
    worker = groups.add_parser("daemon", parents=[daemon], help="view and control the local worker")
    worker.add_argument("action", nargs="?", choices=["status", "start", "stop", "restart"])
    worker.add_argument("--foreground", action="store_true", help="serve in this terminal")
    worker.set_defaults(op="daemon", keys=["host", "port"])
    groups.add_parser("startup", parents=[daemon], help="start a worker now and at boot").set_defaults(op="startup")
    groups.add_parser("unstartup", help="stop and remove the boot service").set_defaults(op="unstartup")
    fleet = groups.add_parser("fleet").add_subparsers(required=True)
    fleet.add_parser("leave", help="leave the current fleet and unpair this worker").set_defaults(op="leave", keys=[])
    remove = fleet.add_parser("remove", help="remove another worker, even if it is offline")
    remove.add_argument("peer", help="node ID, unique ID prefix from status, or host:port")
    remove.set_defaults(op="remove-peer", keys=["peer"])
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
    inventory = model.add_parser("list", help="list models registered in the fleet")
    inventory.add_argument("--json", action="store_true", help="print the inventory as JSON")
    inventory.set_defaults(op="model-list", keys=[])
    for action, help_text in (("warm", "prepare and keep local shards in RAM on all owners"),
                              ("cold", "release prepared weights on all owners")):
        command = model.add_parser(action, help=help_text)
        command.add_argument("name")
        command.set_defaults(op="model-" + action, keys=["name"])
    delete = model.add_parser("delete", help="delete a model and its shards across the fleet")
    delete.add_argument("name")
    delete.add_argument("--json", action="store_true", help="print the cleanup report as JSON")
    delete.set_defaults(op="model-delete", keys=["name"])
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
    status.add_argument("--json", action="store_true", help="print the full model status")
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
    run.add_argument("--q8-threads", help="Q8 decode threads: one count for all owners, or comma-separated counts in layer order")
    run.add_argument("--no-stream", action="store_true", help="print only the completed response")
    run.add_argument("-v", "--verbose", action="store_true", help="report fleet and per-owner timings on stderr")
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
    try:
        if args.op == "daemon":
            bind_flags = any(x is not None for x in (args.host, args.port, args.advertise_host))
            if args.action and args.foreground:
                parser.error("--foreground cannot be combined with a daemon action")
            if args.action or not (bind_flags or args.foreground):
                from .control import control

                control(args.action, args.host, args.port, args.advertise_host)
                return 0
            args.host = args.host or "127.0.0.1"
            args.port = 7318 if args.port is None else args.port
            if not args.foreground and (os.getenv("SYSTEMD_EXEC_PID") == str(os.getpid()) or
                                       os.getenv("XPC_SERVICE_NAME", "").startswith("local.armada.worker.")):
                # Units written before --foreground existed use this legacy invocation.
                os.environ["_ARMADA_WORKER_MODE"] = "service"
        if args.op in ("startup", "unstartup"):
            from .startup import configure

            print(configure(args.op == "startup", getattr(args, "host", None) or "127.0.0.1",
                            7318 if getattr(args, "port", None) is None else args.port,
                            getattr(args, "advertise_host", None)))
            return 0
        if args.op == "run":
            from .common import rpc, state, stream_completion
            from .schema import Fields
            from .status import format_timings

            timings: Fields | None = {} if args.verbose else None
            threads = [int(n) for n in args.q8_threads.split(",")] if args.q8_threads else None
            if args.no_stream:
                request: Fields = dict(model=args.name, prompt=args.prompt, max_tokens=args.tokens)
                if threads:
                    request["q8_threads"] = threads
                if args.verbose:
                    request["diagnostics"] = True
                response = rpc(state()["addr"], "v1/completions", request)
                print(response["choices"][0]["text"])
                if timings is not None:
                    timings.update(response.get("diagnostics", {}))
            else:
                for piece in stream_completion(args.name, args.prompt, args.tokens, timings, threads):
                    print(piece, end="", flush=True)
                print(flush=True)
            if timings is not None:
                print(format_timings(timings, state()["peers"]) if timings else
                      "Run diagnostics unavailable; update and restart the coordinating daemon.", file=sys.stderr)
            return 0
        from .runtime import execute

        result = execute([args.op, *(str(getattr(args, k)) for k in args.keys)])
        if args.op == "status" and not args.json and isinstance(result, dict):
            from .status import format_status

            result = format_status(result)
        elif args.op == "model" and not args.json and isinstance(result, dict):
            from .status import format_model_status

            result = format_model_status(args.name, result)
        elif args.op == "model-list" and not args.json and isinstance(result, list):
            from .status import format_models

            result = format_models(result)
        elif args.op == "model-delete" and not args.json and isinstance(result, dict):
            lines = [f"Deleted model: {result['name']}",
                     "Cleanup confirmed: " + (", ".join(result["confirmed"]) or "none")]
            if result["pending"]:
                lines.append("Cleanup pending: " + ", ".join(result["pending"]))
            result = "\n".join(lines)
        elif args.op.startswith("key-") and not args.json:
            if isinstance(result, list):
                result = "\n".join(["ID                        STATUS   NAME",
                                    *(f"{r['id']}  {r['status']:<7}  {r['name']}" for r in result)]
                                   ) if result else "No fleet API keys."
            elif isinstance(result, dict):
                lines = [f"{result['id']}  {result['status']}  {result['name']}"]
                if "acknowledged" in result:
                    lines.append("Acknowledged: " + ", ".join(result["acknowledged"]))
                    if result["pending"]:
                        lines.append("Pending sync: " + ", ".join(result["pending"]))
                if "token" in result:
                    lines.append("Token (shown once): " + result["token"])
                result = "\n".join(lines)
        print(result if isinstance(result, str) else json.dumps(result, indent=2))
    except (Exception, KeyboardInterrupt) as e:
        print(f"armada: {e}", file=sys.stderr)
        return 1
    return 0
