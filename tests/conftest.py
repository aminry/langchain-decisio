# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the langchain-decisio project
"""A stdlib stand-in for a Decisio server. It is not a model: every answer is scripted by the test."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class FakeServer:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.paths: list[str] = []
        self.responder: Callable[[dict], tuple[int, dict, dict]] = self.default
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status: int, body: dict, headers: dict | None = None):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.paths.append(self.path)
                if self.path == "/health":
                    self._send(200, {"ok": True, "base": "gemma-4-31b", "engine": "fake"})
                else:
                    self._send(404, {})

            def do_POST(self):
                outer.paths.append(self.path)
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append(body)
                if self.path == "/v1/tasks":
                    return self._send(200, {"id": body.get("id"), "applied": True})
                status, payload, headers = outer.responder(body)
                self._send(status, payload, headers)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @staticmethod
    def default(body: dict) -> tuple[int, dict, dict]:
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.9}
            elif q["type"] == "choice":
                keys = list(q["criteria"])
                p = {k: 0.1 / (len(keys) - 1) for k in keys}
                p[keys[0]] = 0.9
                answers[name] = {"type": "choice", "choice": keys[0], "probabilities": p, "confidence": 0.8}
            else:
                n = len(q["criteria"])
                answers[name] = {
                    "type": "score",
                    "score": 1.0,
                    "probabilities": {str(i): 1 / n for i in range(n)},
                    "confidence": 0.1,
                    "legend": {str(i): lv for i, lv in enumerate(q["criteria"])},
                }
        return (
            200,
            {"model": "fake-model", "answers": answers, "usage": {"input_tokens": 7, "output_tokens": 1}},
            {
                "x-decisio-server-ms": "12.5",
                "x-decisio-tasks": "t1,t2",
            },
        )

    def always(self, **by_question: dict) -> None:
        """Answer every request with these answers (by question name), and nothing else."""
        self.responder = lambda body: (200, {"answers": by_question}, {})

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture()
def server():
    s = FakeServer()
    yield s
    s.close()
