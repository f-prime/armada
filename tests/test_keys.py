"""Exercise key management through real CLI processes and two daemon homes."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path

from armada.keys import combine, mutate
from armada.schema import Fields, NodeState


class FleetKeysTest(unittest.TestCase):
    def test_conflicts(self) -> None:
        s = NodeState(addr="test", key="fleet", uid="a")
        kid = mutate(s, "create", "", "old")["id"]
        other: NodeState = json.loads(json.dumps(s))
        other["uid"] = "b"
        mutate(s, "rename", kid, "new")
        mutate(other, "rotate", kid, "")
        a, b = s["api_keys"][kid], other["api_keys"][kid]
        self.assertEqual(combine(a, b), combine(b, a))
        self.assertEqual(combine(a, b)["name"], "new")
        self.assertEqual(combine(a, b)["digest"], b["digest"])
        mutate(s, "revoke", kid, "")
        self.assertTrue(combine(b, a)["revoked"])
        self.assertEqual(combine(a, b)["digest"], "")

    def test_lifecycle(self) -> None:
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        cli = [sys.executable, "-c", "from armada.cli import main; raise SystemExit(main())"]

        def env(i: int) -> dict[str, str]:
            return dict(os.environ, ARMADA_HOME=f"{tmp}/{i}", ARMADA_API_KEY="")

        def command(i: int, *args: str, stdin: str = "") -> str:
            return subprocess.check_output(cli + list(args), env=env(i), input=stdin, text=True, timeout=15).strip()

        def start(i: int, port: int = 0) -> tuple[subprocess.Popen[str], int]:
            if not port:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
            p = subprocess.Popen(cli + ["daemon", "--port", str(port)], env=env(i), stdout=subprocess.PIPE, text=True)
            def stop() -> None:
                p.terminate()
                p.wait(timeout=5)
                if p.stdout:
                    p.stdout.close()
            self.addCleanup(stop)
            assert p.stdout
            self.assertIn("listening", p.stdout.readline())
            return p, port

        def status(port: int, token: str, admin: bool = False) -> int:
            conn = HTTPConnection("127.0.0.1", port, timeout=3)
            try:
                conn.request("POST" if admin else "GET", "/manage-key" if admin else "/v1/models",
                             headers={"Authorization": "Bearer " + token})
                return conn.getresponse().status
            finally:
                conn.close()

        def key(i: int, *args: str) -> Fields:
            return json.loads(command(i, "key", *args, "--json"))  # type: ignore[no-any-return]

        _, first = start(0)
        p, second = start(1)
        r = key(0, "create", "--name", "client")
        kid, old = r["id"], r["token"]
        command(0, "fleet", "add", f"127.0.0.1:{second}", stdin=command(1, "fleet", "pairing-code") + "\n")
        self.assertEqual(status(second, old), 200)
        self.assertEqual(status(first, old, admin=True), 403)
        self.assertEqual(key(1, "rename", kid, "editor")["pending"], [])
        self.assertEqual(key(0, "show", kid), dict(id=kid, status="active", name="editor"))
        rotate = key(1, "rotate", kid)
        new = rotate["token"]
        self.assertEqual(len(rotate["acknowledged"]), 2)
        for port in (first, second):
            self.assertEqual(status(port, old), 401)
            self.assertEqual(status(port, new), 200)
        for i in (0, 1):
            disk = Path(tmp, str(i), "node.json").read_text()
            self.assertNotIn(old, disk)
            self.assertNotIn(new, disk)
        p.terminate()
        p.wait(timeout=5)
        self.assertEqual(len(key(0, "revoke", kid)["pending"]), 1)
        start(1, second)
        until = time.monotonic() + 15
        while status(second, new) != 401 and time.monotonic() < until:
            time.sleep(0.1)
        self.assertEqual(status(second, new), 401)
        self.assertEqual(json.loads(command(1, "key", "list", "--json")),
                         [dict(id=kid, status="revoked", name="editor")])
