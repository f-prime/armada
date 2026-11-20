import base64
import os
import plistlib
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from armada import startup


class StartupTests(unittest.TestCase):
    def test_linux_install_update_remove(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, XDG_CONFIG_HOME=directory), \
                patch.object(startup, "run") as run, patch("subprocess.run") as probe:
            argv = ["/path with spaces/armada", "daemon", "--home", '/dir/%n/"$HOME"']
            startup.linux(argv, "127.0.0.1", 0)
            unit = Path(directory, "systemd/user", startup.UNIT)
            self.assertIn('ExecStart=":/path with spaces/armada" "daemon" "--home" "/dir/%%n/\\"$HOME\\""', unit.read_text())
            probe.return_value.returncode = 0
            startup.linux(argv + ["--port", "4000"], "127.0.0.1", 4000)
            self.assertIn('"--port" "4000"', unit.read_text())
            startup.linux([], "", 0)
            run.assert_any_call("systemctl", "--user", "disable", "--now", startup.UNIT)
            self.assertFalse(unit.exists())
            startup.linux([], "", 0)

    def test_busy_port_leaves_service_uninstalled(self) -> None:
        with socket.socket() as sock, tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, XDG_CONFIG_HOME=directory), patch.object(startup, "run") as run:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            with self.assertRaisesRegex(ValueError, "foreground daemon"):
                startup.linux(["/bin/armada"], "127.0.0.1", sock.getsockname()[1])
            run.assert_not_called()

    def test_mac_installs_plist_with_literal_arguments(self) -> None:
        argv = ["/Applications/My Tools/armada", "daemon", "--home", "/Users/test/A&B"]
        def inspect(*cmd: str) -> None:
            if cmd[1] == "install":
                obj = plistlib.loads(Path(cmd[-2]).read_bytes())
                self.assertEqual(obj["ProgramArguments"], argv)
                self.assertTrue(obj["UserName"])
        with patch.object(startup, "run", side_effect=inspect) as run, \
                patch("subprocess.run") as probe:
            probe.return_value.returncode = 1
            startup.mac(argv, "127.0.0.1", 0)
            self.assertEqual(run.call_args.args[:4], ("sudo", "launchctl", "bootstrap", "system"))

    def test_wsl_uses_same_task_for_install_and_removal(self) -> None:
        with patch.object(startup, "run") as run:
            distro = "Dev WSL '$(foo)'"
            startup.windows_task(True, distro)
            install = base64.b64decode(run.call_args.args[-1]).decode("utf-16-le")
            startup.windows_task(False, distro)
            remove = base64.b64decode(run.call_args.args[-1]).decode("utf-16-le")
            self.assertNotIn(distro, install)
            self.assertIn("New-ScheduledTaskTrigger -AtStartup", install)
            self.assertIn("Unregister-ScheduledTask", remove)
            self.assertEqual(install.split("$old =")[0], remove.split("$old =")[0])
