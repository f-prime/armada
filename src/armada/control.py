"""Local daemon control. The worker's advisory lock is its liveness indicator."""

import fcntl
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .common import home
from .startup import registered_args, service_action


@dataclass
class Worker:
    on: bool
    host: str
    port: int
    advertise: str | None
    mode: str
    token: str


def flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def snapshot(directory: Path, service: list[str]) -> Worker:
    on = False
    try:
        with (directory / "daemon.lock").open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                on = True
    except FileNotFoundError:
        pass
    try:
        data = json.loads((directory / "daemon.json").read_text())
    except FileNotFoundError:
        data = {}
    host = str(data.get("host", "127.0.0.1"))
    port = int(data.get("port", 7318))
    advertise = data.get("advertise_host")
    if service and (not on or not data):
        host = flag(service, "--host") or host
        port = int(flag(service, "--port") or port)
        advertise = flag(service, "--advertise-host")
    elif on and not data and (directory / "node.json").exists():
        addr = json.loads((directory / "node.json").read_text())["addr"]
        host, portstr = addr.rsplit(":", 1)
        port = int(portstr)
    return Worker(on, host, port, advertise, data.get("mode", "service" if service else "foreground"),
                  data.get("token", ""))


def wait(directory: Path, service: list[str], on: bool, previous: str = "",
         child: subprocess.Popen[bytes] | None = None) -> Worker:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if child and child.poll() is not None:
            raise ValueError(f"worker exited during startup; see {directory / 'daemon.log'}")
        w = snapshot(directory, service)
        if w.on == on and (not on or (w.token and w.token != previous)):
            return w
        time.sleep(0.1)
    raise ValueError("worker did not finish starting or stopping; check daemon status and logs")


def change(action: str, directory: Path, service: list[str], host: str | None,
           port: int | None, advertise: str | None) -> None:
    w = snapshot(directory, service)
    overrides = host is not None or port is not None or advertise is not None
    if overrides and (service or action not in ("start", "restart")):
        raise ValueError("use startup to change boot service settings; bind flags apply to standalone start/restart")
    if action == "start" and w.on:
        if overrides:
            raise ValueError("worker is already running; use daemon restart to change its settings")
        return
    if action in ("stop", "restart"):
        if service and (not w.on or w.mode == "service"):
            service_action("stop")
        elif w.on:
            if not re.fullmatch("[0-9a-f]{32}", w.token):
                raise ValueError("stop the older foreground worker in its original terminal first")
            (directory / ("stop-" + w.token)).touch(mode=0o600)
        if w.on:
            wait(directory, service, False)
        if action == "stop":
            return
    if service:
        service_action("start")
        wait(directory, service, True, w.token)
        return
    argv = [sys.executable, "-c", "from armada.cli import main; raise SystemExit(main())",
            "daemon", "--foreground", "--home", str(directory), "--host", host or w.host,
            "--port", str(w.port if port is None else port)]
    if advertised := advertise or w.advertise:
        argv += ["--advertise-host", advertised]
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    log = directory / "daemon.log"
    with os.fdopen(os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "ab") as stream:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                 start_new_session=True, env={**os.environ, "_ARMADA_WORKER_MODE": "background"})
    wait(directory, service, True, w.token, child)


def control(action: str | None, host: str | None, port: int | None, advertise: str | None) -> None:
    service = registered_args()
    registered = flag(service, "--home")
    directory = (Path(registered) if registered and not os.environ.get("ARMADA_HOME") else home()).resolve()
    if registered and Path(registered).resolve() != directory:
        service = []
    interactive = action is None and sys.stdin.isatty() and sys.stdout.isatty()
    while True:
        if action and action != "status":
            change(action, directory, service, host, port, advertise)
        elif any(x is not None for x in (host, port, advertise)):
            raise ValueError("bind options apply to daemon start or restart")
        w = snapshot(directory, service)
        print(f"Daemon:  {'ON (' + w.mode + ')' if w.on else 'OFF'}\nHost:    {w.host}\nPort:    {w.port}"
              f"\nStartup: {'registered' if service else 'not registered'}")
        if not interactive:
            return
        try:
            key = input(("[s] Stop   [r] Restart" if w.on else "[s] Start in background")
                        + "   [Enter] Refresh   [q] Quit: ").strip().lower()
        except EOFError:
            return
        if key in ("q", "quit"):
            return
        action = {"s": "stop" if w.on else "start"}.get(key)
        if w.on and key == "r":
            action = "restart"
