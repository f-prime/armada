import json
import os
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from armada.runtime import Handler
from armada.common import stream_completion
from armada.schema import Fields


class InferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        model = dict(id="x", owners=["a", "b"], backend="qwen3moe", qwen={"ctx": 1024}, stop=[0])
        self.enterContext(patch("armada.runtime.state", return_value=dict(key="fleet", peers={}, models={"coder": model})))
        self.enterContext(patch.dict(os.environ, ARMADA_API_KEY="inference"))
        self.enterContext(patch("armada.runtime.unavailable", return_value={}))
        self.prompts: list[str] = []
        self.resume: threading.Event | None = None

        def rpc(node: str, op: str, body: Fields, **kwargs: object) -> object:
            if op == "tokenize":
                self.prompts.append(body["prompt"])
                return [999]
            if op == "decode":
                return bytes(body["tokens"]).decode()
            if op == "release-kv":
                return True
            if op == "stage-kv":
                if node == "a":
                    self.assertEqual(len(body["payload"]), 1)
                    hidden = [[body["offset"]]]
                    if body.get("profile"):
                        return dict(payload=hidden, compute_s=0.75 if not body["offset"] else 0.25, q8_threads=4)
                    return hidden
                i = body["payload"][0][0]
                if i == 1 and self.resume is not None:
                    self.assertTrue(self.resume.wait(timeout=5), "client buffered the first token")
                return self.output[i] if i < len(self.output) else 0
            self.assertIn(op, ("chat-token", "completion-token"))
            return bytes([body["token"]]).hex()

        self.enterContext(patch("armada.runtime.rpc", side_effect=rpc))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(self.server.server_close)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)

    def request(self, path: str, body: Fields | None = None, key: str = "inference") -> tuple[int, str]:
        conn = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            conn.request("GET" if body is None else "POST", path, None if body is None else json.dumps(body),
                         {"Authorization": "Bearer " + key})
            response = conn.getresponse()
            return response.status, response.read().decode()
        finally:
            conn.close()

    def test_access(self) -> None:
        self.assertEqual(self.request("/v1/models", key="wrong")[0], 401)
        self.assertEqual(json.loads(self.request("/v1/models")[1])["data"][0]["id"], "coder")
        self.assertEqual(self.request("/discard", {"id": "x"})[0], 403)
        self.assertEqual(self.request("/v1/chat/completions", {"model": "missing"})[0], 404)

    def test_raw_completion_is_incremental(self) -> None:
        self.output = "A🌍".encode()
        self.resume = threading.Event()
        addr = f"127.0.0.1:{self.server.server_port}"
        with patch("armada.common.state", return_value=dict(addr=addr, key="fleet", uid="node")):
            stream = stream_completion("coder", "raw prompt", 10)
            try:
                self.assertEqual(next(stream), "A")
                self.resume.set()
                self.assertEqual("".join(stream), "🌍")
            finally:
                self.resume.set()
                stream.close()
        self.assertEqual(self.prompts, ["raw prompt"])

    def test_run_diagnostics(self) -> None:
        import itertools

        addr = f"127.0.0.1:{self.server.server_port}"
        for output in (b"AB", b"A", b""):
            self.output = output
            for stream in (False, True):
                with self.subTest(output=output, stream=stream), patch(
                    "armada.runtime.monotonic", side_effect=itertools.count()
                ), patch("armada.common.state", return_value=dict(addr=addr, key="fleet")):
                    report: Fields = {}
                    if stream:
                        text = "".join(stream_completion("coder", "prompt", 8, report))
                    else:
                        status, body = self.request("/v1/completions", dict(
                            model="coder", prompt="prompt", max_tokens=8, diagnostics=True))
                        self.assertEqual(status, 200, body)
                        response = json.loads(body)
                        report = response["diagnostics"]
                        text = response["choices"][0]["text"]
                    self.assertEqual(text.encode(), output)
                    self.assertEqual(report["first_ms"], 5000 if output else None)
                    self.assertEqual((report["prompt_tokens"], report["setup_ms"]), (1, 1000))
                    self.assertEqual([(r["prefill_ms"], r["prefill_compute_ms"]) for r in report["owners"]],
                                     [(1000, 750), (1000, None)])
                    self.assertEqual(report["tokens_per_second"], 0.2 if len(output) == 2 else None)
                    self.assertEqual([(r["owner"], r["layers"]) for r in report["owners"]],
                                     [("a", "0"), ("b", "1")])
                    for row in report["owners"]:
                        self.assertEqual(row["avg_ms"], 1000 if len(output) == 2 else None)

    def test_tool_round_trip(self) -> None:
        self.output = b"<tool_call>\n<function=lookup>\n<parameter=n>\n42\n</parameter>\n</function>\n</tool_call>"
        body: Fields = dict(model="coder", messages=[dict(role="user", content="Look it up")],
                            tools=[dict(type="function", function=dict(name="lookup", parameters={
                                "properties": {"n": {"type": "integer"}}}))])
        status, data = self.request("/v1/chat/completions", body)
        self.assertEqual(status, 200, data)
        choice = json.loads(data)["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"n": 42})
        _, stream = self.request("/v1/chat/completions", dict(**body, stream=True))
        streamed = [json.loads(line[6:])["choices"][0] for line in stream.splitlines()
                    if line.startswith("data: {")]
        self.assertEqual(streamed[-1]["finish_reason"], "tool_calls")
        self.assertEqual(next(c["delta"]["tool_calls"][0]["function"] for c in streamed
                              if "tool_calls" in c["delta"]), call["function"])
        body["messages"].extend([choice["message"], dict(role="tool", tool_call_id=call["id"], content="found")])
        body.update(stream=True, stream_options={"include_usage": True})
        self.output = "Voilà!".encode()
        status, data = self.request("/v1/chat/completions", body)
        self.assertIn("found", self.prompts[-1])
        self.assertEqual(status, 200)
        events = data.strip().split("\n\n")
        self.assertEqual(events.pop(), "data: [DONE]")
        chunks = [json.loads(e.removeprefix("data: ")) for e in events]
        self.assertEqual("".join(c["delta"].get("content", "") for e in chunks for c in e["choices"]), "Voilà!")
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], len(self.output) + 1)
