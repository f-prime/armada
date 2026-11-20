import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from armada.control import change, snapshot


class DaemonControlTests(unittest.TestCase):
    def test_panel_can_recognize_and_stop_foreground_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory, socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            path = Path(directory)
            with subprocess.Popen([sys.executable, "-c", "from armada.cli import main; raise SystemExit(main())",
                                   "daemon", "--foreground", "--port", str(port), "--home", directory],
                                  stdout=subprocess.PIPE, text=True) as worker:
                try:
                    assert worker.stdout is not None
                    self.assertIn("Armada listening", worker.stdout.readline())
                    state = snapshot(path, [])
                    self.assertEqual((state.on, state.mode, state.port), (True, "foreground", port))
                    change("stop", path, [], None, None, None)
                    self.assertEqual(worker.wait(timeout=5), 0)
                finally:
                    if worker.poll() is None:
                        worker.kill()

    def test_background_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "ARMADA_HOME": directory, "XDG_CONFIG_HOME": directory}
            def cli(*args: str) -> str:
                return subprocess.check_output([sys.executable, "-c",
                    "from armada.cli import main; raise SystemExit(main())", "daemon", *args],
                    env=env, stdin=subprocess.DEVNULL, text=True, timeout=25)

            self.assertIn("Daemon:  OFF", cli())
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = str(sock.getsockname()[1])
            try:
                self.assertIn("ON (background)", cli("start", "--port", port))
                self.assertIn("Port:    " + port, cli("status"))
                with socket.create_connection(("127.0.0.1", int(port)), timeout=2):
                    pass
                self.assertIn("ON (background)", cli("restart"))
            finally:
                self.assertIn("Daemon:  OFF", cli("stop"))
            self.assertIn("Port:    " + port, cli())
            self.assertIn("Startup: not registered", cli())
