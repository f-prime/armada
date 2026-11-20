"""Install the local worker with the machine's boot service manager."""

import base64
import errno
import hashlib
import json
import os
import platform
import plistlib
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import home

UNIT = "armada-worker.service"
MARK = "# Managed by armada startup\n"


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def check_port(host: str, port: int) -> None:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise ValueError(f"{host}:{port} is in use; stop the foreground daemon first") from exc
            raise


def unit_text(argv: list[str]) -> str:
    def quote(arg: str) -> str:
        if any(ord(c) < 32 for c in arg):
            raise ValueError("control characters are not supported in service arguments")
        return '"' + arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'

    # The ':' prefix disables systemd environment-variable substitution.
    return (MARK + "[Unit]\nDescription=Armada worker\n\n[Service]\nEnvironment=_ARMADA_WORKER_MODE=service\nExecStart="
            + " ".join(map(quote, [":" + argv[0], *argv[1:]])) + "\nRestart=on-failure\nRestartSec=5\n\n"
            "[Install]\nWantedBy=default.target\n")


def unit_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "systemd/user" / UNIT


def launch_label() -> str:
    return f"local.armada.worker.{os.getuid()}"


def plist_path() -> Path:
    return Path("/Library/LaunchDaemons") / (launch_label() + ".plist")


def registered_args() -> list[str]:
    """Read our installed launch command, including pre-control-panel units."""
    if sys.platform == "darwin":
        path = plist_path()
        if path.exists():
            with path.open("rb") as f:
                return list(plistlib.load(f)["ProgramArguments"])
    elif sys.platform.startswith("linux"):
        path = unit_path()
        if path.exists():
            text = path.read_text()
            if text.startswith(MARK):
                line = next(line for line in text.splitlines() if line.startswith("ExecStart="))
                return shlex.split(line.removeprefix("ExecStart=").replace("%%", "%"))
    return []


def service_action(action: str) -> None:
    if sys.platform != "darwin":
        run("systemctl", "--user", action, UNIT)
        return
    target = "system/" + launch_label()
    loaded = subprocess.run(["launchctl", "print", target], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0
    if action == "stop":
        if loaded:
            run("sudo", "launchctl", "bootout", target)
    elif not loaded:
        run("sudo", "launchctl", "bootstrap", "system", str(plist_path()))
    elif action == "restart":
        run("sudo", "launchctl", "kickstart", "-k", target)


def linux(argv: list[str], host: str, port: int) -> None:
    import pwd

    path = unit_path()
    ctl = ["systemctl", "--user"]
    if path.exists() and not path.read_text().startswith(MARK):
        raise ValueError(f"{path} is not managed by Armada")
    if not argv:
        if path.exists():
            run(*ctl, "disable", "--now", UNIT)
            path.unlink()
            run(*ctl, "daemon-reload")
        return
    text = unit_text(argv)
    if not path.exists() or subprocess.run([*ctl, "is-active", "--quiet", UNIT]).returncode:
        check_port(host, port)
    run("loginctl", "enable-linger", pwd.getpwuid(os.getuid()).pw_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        f.write(text)
    os.replace(f.name, path)
    run(*ctl, "daemon-reload")
    run(*ctl, "enable", UNIT)
    run(*ctl, "restart", UNIT)


def mac(argv: list[str], host: str, port: int) -> None:
    import pwd

    user = pwd.getpwuid(os.getuid())
    label = launch_label()
    path = plist_path()
    target = "system/" + label
    loaded = subprocess.run(["launchctl", "print", target], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0
    if argv and not loaded:
        check_port(host, port)
    if loaded:
        run("sudo", "launchctl", "bootout", target)
    if not argv:
        if path.exists():
            run("sudo", "rm", str(path))
        return
    data = dict(Label=label, UserName=user.pw_name, ProgramArguments=argv,
                 EnvironmentVariables={"HOME": user.pw_dir, "_ARMADA_WORKER_MODE": "service"},
                 KeepAlive=True, ThrottleInterval=5)
    with tempfile.NamedTemporaryFile() as f:
        f.write(plistlib.dumps(data))
        f.flush()
        run("sudo", "install", "-o", "root", "-g", "wheel", "-m", "644", f.name, str(path))
    run("sudo", "launchctl", "bootstrap", "system", str(path))


def windows_task(install: bool, distro: str) -> None:
    """Start WSL at Windows boot; keep a client attached while the service runs."""
    import pwd

    user = pwd.getpwuid(os.getuid()).pw_name
    name = "ArmadaWorker-" + hashlib.sha256(f"{distro}\0{user}".encode()).hexdigest()[:16]
    args = subprocess.list2cmdline(["-d", distro, "-u", user, "--exec", "systemctl",
                                    "--user", "start", "--wait", UNIT])
    payload = base64.b64encode(json.dumps([name, args]).encode()).decode()
    script = f"""
$ErrorActionPreference = 'Stop'
$n,$a = ConvertFrom-Json ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{payload}')))
$me = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]$me).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {{
    throw 'Run WSL from an elevated Windows terminal under the account owning this distribution.'
}}
$old = Get-ScheduledTask -TaskPath '\\' | Where-Object TaskName -eq $n
if ($old -and $old.State -eq 'Running') {{ Stop-ScheduledTask -InputObject $old }}
"""
    if install:
        script += """
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\\System32\\wsl.exe" -Argument $a
$principal = New-ScheduledTaskPrincipal -UserId $me.Name -LogonType S4U
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $n -Action $action -Principal $principal -Trigger (New-ScheduledTaskTrigger -AtStartup) -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $n
"""
    else:
        script += "if ($old) { Unregister-ScheduledTask -InputObject $old -Confirm:$false }"
    run("powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand",
        base64.b64encode(script.encode("utf-16-le")).decode())


def configure(install: bool, host: str | None, port: int | None, advertise: str | None) -> str:
    system = platform.system()
    if system not in ("Linux", "Darwin"):
        raise ValueError("on Windows, run this command inside a systemd-enabled WSL2 distribution")
    argv: list[str] = []
    if install:
        assert host is not None and port is not None
        executable = shutil.which(sys.argv[0])
        if not executable:
            raise ValueError("run startup using the installed armada command")
        argv = [os.path.abspath(executable), "daemon", "--foreground", "--home", str(home().resolve()),
                "--host", host, "--port", str(port)]
        if advertise:
            argv.extend(["--advertise-host", advertise])
    if system == "Darwin":
        mac(argv, host or "", port or 0)
    else:
        distro = os.environ.get("WSL_DISTRO_NAME")
        if distro and not install:
            windows_task(False, distro)
        linux(argv, host or "", port or 0)
        if distro and install:
            try:
                windows_task(True, distro)
            except (OSError, subprocess.CalledProcessError) as exc:
                raise ValueError("WSL service installed, but Windows task setup failed; rerun startup "
                                 "in WSL from an elevated Windows terminal with interop enabled") from exc
    return "Armada boot service started and enabled." if install else "Armada boot service removed."
