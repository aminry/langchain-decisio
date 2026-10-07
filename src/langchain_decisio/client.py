# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the langchain-decisio project
"""The wire layer: Decisio's HTTP routes in plain httpx, with no LangChain in it.

It needs no API key and sends no model name unless asked to (Ollama's decision route wants one). It keeps everything
the server returns, and an error carries the server's own `detail`. It refuses TypeSafe's hosted service: this package
is for a Decisio server, and `langchain-typesafe` is the package for TypeSafe's.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import httpx

from .types import Choice, Decision, Noul, Score, parse_answer, questions_to_wire

DEFAULT_URL = "http://127.0.0.1:8000"
_HOSTED_SUFFIXES = ("typesafe.ai",)


class DecisioError(RuntimeError):
    """The server answered with an error status. `status` and `detail` are what it said."""

    def __init__(self, status: int, detail: Any):
        super().__init__(f"decisio answered {status}: {detail}")
        self.status = status
        self.detail = detail


class DecisioConnectionError(RuntimeError):
    """The server could not be reached, or did not answer in time."""


def resolve_base_url(base_url: str | None) -> str:
    url = (base_url or os.environ.get("DECISIO_URL") or DEFAULT_URL).rstrip("/")
    host = (urlparse(url).hostname or "").lower()
    if any(host == s or host.endswith("." + s) for s in _HOSTED_SUFFIXES):
        raise ValueError(
            f"{host} is TypeSafe's hosted service. langchain-decisio talks to a Decisio server; "
            "for TypeSafe's service use langchain-typesafe."
        )
    return url


class DecisioClient:
    """Synchronous and asynchronous calls to one Decisio server.

    Args:
        base_url: The server (default `$DECISIO_URL`, else http://127.0.0.1:8000).
        model: Sent as `model` in the request body. A Decisio server ignores it; Ollama's decision route needs it
            (for example `aminroudaki/decisio-gemma`). Default `$DECISIO_MODEL`, else none.
        timeout: Seconds per request.
        headers: Extra headers, for a proxy in front of the server.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        model: str | None = None,
        timeout: float = 30.0,
        headers: Mapping[str, str] | None = None,
    ):
        self.base_url = resolve_base_url(base_url)
        self.model = model or os.environ.get("DECISIO_MODEL") or None
        self.timeout = timeout
        self._headers = dict(headers or {})
        self._sync: httpx.Client | None = None
        self._async: httpx.AsyncClient | None = None

    # -- transport -----------------------------------------------------------------------------------------------

    def _sync_client(self) -> httpx.Client:
        if self._sync is None:
            self._sync = httpx.Client(base_url=self.base_url, timeout=self.timeout, headers=self._headers)
        return self._sync

    def _async_client(self) -> httpx.AsyncClient:
        if self._async is None:
            self._async = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, headers=self._headers)
        return self._async

    def close(self) -> None:
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        if self._async is not None:
            await self._async.aclose()
            self._async = None
        self.close()

    def __enter__(self) -> DecisioClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> DecisioClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- routes --------------------------------------------------------------------------------------------------

    def _body(self, state: Any, questions: Mapping[str, Noul | Choice | Score]) -> dict[str, Any]:
        body: dict[str, Any] = {"state": state, "questions": questions_to_wire(questions)}
        if self.model:
            body = {"model": self.model, **body}
        return body

    @staticmethod
    def _decision(response: httpx.Response) -> Decision:
        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise DecisioError(response.status_code, detail)
        data = response.json()
        server_ms = response.headers.get("x-decisio-server-ms")
        return Decision(
            answers={name: parse_answer(a) for name, a in data["answers"].items()},
            model=data.get("model"),
            usage=data.get("usage") or {},
            server_ms=float(server_ms) if server_ms else None,
            tasks=[t for t in response.headers.get("x-decisio-tasks", "").split(",") if t],
        )

    def ask(self, state: Any, questions: Mapping[str, Noul | Choice | Score]) -> Decision:
        """Ask typed questions about a state. `state` is a string, a dict or a list (already JSON-able)."""
        try:
            return self._decision(self._sync_client().post("/v1/systemone", json=self._body(state, questions)))
        except httpx.TransportError as e:
            raise DecisioConnectionError(f"could not reach {self.base_url}: {e}") from e

    async def aask(self, state: Any, questions: Mapping[str, Noul | Choice | Score]) -> Decision:
        try:
            return self._decision(await self._async_client().post("/v1/systemone", json=self._body(state, questions)))
        except httpx.TransportError as e:
            raise DecisioConnectionError(f"could not reach {self.base_url}: {e}") from e

    def health(self) -> dict[str, Any]:
        """The server's /health. Ollama's decision route has none, so a 404 gives `{"ok": None}`."""
        try:
            r = self._sync_client().get("/health")
        except httpx.TransportError as e:
            raise DecisioConnectionError(f"could not reach {self.base_url}: {e}") from e
        if r.status_code == 404:
            return {"ok": None, "engine": "no /health route"}
        r.raise_for_status()
        return r.json()

    def served_base(self) -> str | None:
        """The base the server reports (`gemma-4-31b`, for example), or None when it does not say."""
        return self.health().get("base")

    def register_task(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """POST /v1/tasks with the body decisio's docs/tasks.md describes; returns the task's record."""
        try:
            r = self._sync_client().post("/v1/tasks", json=dict(task))
        except httpx.TransportError as e:
            raise DecisioConnectionError(f"could not reach {self.base_url}: {e}") from e
        if r.status_code != 200:
            raise DecisioError(r.status_code, r.text)
        return r.json()
