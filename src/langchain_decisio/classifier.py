# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the langchain-decisio project
"""`DecisioClassifier`: typed questions about a state, as a LangChain Runnable."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig

from .client import DecisioClient
from .types import Choice, Decision, Noul, Score

QuestionSet = Mapping[str, Noul | Choice | Score]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def to_state(value: Any) -> Any:
    """A state for the wire: strings, dicts and lists pass through; LangChain messages become role and content."""
    if isinstance(value, PromptValue):
        value = value.to_messages()
    if isinstance(value, BaseMessage):
        return _text(value.content)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and value
        and all(isinstance(m, BaseMessage) for m in value)
    ):
        return [{"role": m.type, "content": _text(m.content)} for m in value]
    return value


class DecisioClassifier(Runnable[Any, Decision]):
    """Ask typed questions about a state and get a probability for every option.

    Two ways to call it:

    - With questions fixed at construction, pass the state: `DecisioClassifier(questions=Q).invoke("some text")`.
    - With questions per call, pass a mapping with a `questions` key: `invoke({"state": "...", "questions": Q})`.
      A mapping without a `questions` key is itself the state.

    The state is a string, a dict, a list, a LangChain message or a list of messages. It returns a `Decision`:
    every answer, the server's usage, its own time for the request, and the registered tasks it applied.

    Args:
        base_url: The Decisio server (default `$DECISIO_URL`, else http://127.0.0.1:8000).
        questions: Questions to ask about every state this instance sees.
        client: A `DecisioClient`, to share one connection between components.
        model: The model name, only for servers that need one (Ollama's decision route).
        timeout: Seconds per request.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        questions: QuestionSet | None = None,
        client: DecisioClient | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        self.client = client or DecisioClient(base_url, model=model, timeout=timeout)
        self.questions = dict(questions) if questions else None

    def _split(self, value: Any) -> tuple[Any, QuestionSet]:
        if isinstance(value, Mapping) and "questions" in value:
            extra = set(value) - {"state", "questions"}
            if extra:
                raise ValueError(f"unexpected keys {sorted(extra)}; a request holds `state` and `questions`")
            return to_state(value.get("state")), value["questions"]
        if self.questions is None:
            raise ValueError("no questions: pass them at construction, or invoke with {'state': ..., 'questions': ...}")
        return to_state(value), self.questions

    def _invoke(self, value: Any) -> Decision:
        state, questions = self._split(value)
        return self.client.ask(state, questions)

    async def _ainvoke(self, value: Any) -> Decision:
        state, questions = self._split(value)
        return await self.client.aask(state, questions)

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Decision:
        return self._call_with_config(self._invoke, input, config, run_type="chain")

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Decision:
        return await self._acall_with_config(self._ainvoke, input, config, run_type="chain")
