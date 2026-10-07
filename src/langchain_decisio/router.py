# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the langchain-decisio project
"""`DecisioRouter`: pick a branch by probability, and send what the model is unsure about somewhere safe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel

from .classifier import DecisioClassifier
from .client import DecisioClient
from .types import ChoiceAnswer, Decision

_QUESTION = "route"


class Route(BaseModel):
    """Where a state goes, and why."""

    name: str
    """The branch to take: the model's choice when it cleared the threshold, else the fallback."""
    chosen: str
    """What the model chose, whether or not it was accepted."""
    probability: float
    """The probability of `chosen`."""
    accepted: bool
    """True when `probability` reached the threshold, so `name == chosen`."""
    decision: Decision


class Routed(BaseModel):
    """The output of `run_branches`: the route taken and what the branch returned."""

    route: Route
    output: Any


class DecisioRouter(Runnable[Any, Route]):
    """Turn one `Choice` question into a route.

    The chosen option is accepted when its probability reaches `accept_at`, and anything less goes to `fallback`:
    a human queue, a larger model, a clarifying question. There is no default for `accept_at` tuned on anything.
    Measure it on labelled items of your own: sweep it and read accuracy against coverage, as example 1 of
    https://github.com/aminry/decisio-examples does.

    Args:
        question: The `Choice` to ask. Its option keys are the branch names.
        fallback: The branch for an unsure state. It may be an option key or not.
        accept_at: The probability the chosen option needs.
        base_url, client, model, timeout: As for `DecisioClassifier`.
    """

    def __init__(
        self,
        question: Any,
        *,
        fallback: str,
        accept_at: float,
        base_url: str | None = None,
        client: DecisioClient | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        if not 0.0 < accept_at <= 1.0:
            raise ValueError(f"accept_at must be in (0, 1], got {accept_at}")
        self.question = question
        self.fallback = fallback
        self.accept_at = accept_at
        self.classifier = DecisioClassifier(
            base_url, questions={_QUESTION: question}, client=client, model=model, timeout=timeout
        )

    def _route(self, decision: Decision) -> Route:
        answer = decision.answers[_QUESTION]
        if not isinstance(answer, ChoiceAnswer):
            raise TypeError("DecisioRouter needs a Choice question")
        accepted = answer.probability >= self.accept_at
        return Route(
            name=answer.choice if accepted else self.fallback,
            chosen=answer.choice,
            probability=answer.probability,
            accepted=accepted,
            decision=decision,
        )

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Route:
        return self._call_with_config(lambda v: self._route(self.classifier.invoke(v)), input, config, run_type="chain")

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Route:
        async def go(v: Any) -> Route:
            return self._route(await self.classifier.ainvoke(v))

        return await self._acall_with_config(go, input, config, run_type="chain")

    def run_branches(self, branches: Mapping[str, Runnable[Any, Any]]) -> Runnable[Any, Routed]:
        """A Runnable that routes its input and runs the chosen branch on that same input.

        `branches` must hold a Runnable for every option key and for the fallback.
        """
        needed = set(getattr(self.question, "options", {})) | {self.fallback}
        missing = needed - set(branches)
        if missing:
            raise ValueError(f"no branch for {sorted(missing)}")

        def go(value: Any, config: RunnableConfig | None = None) -> Routed:
            route = self.invoke(value, config)
            return Routed(route=route, output=branches[route.name].invoke(value, config))

        async def ago(value: Any, config: RunnableConfig | None = None) -> Routed:
            route = await self.ainvoke(value, config)
            return Routed(route=route, output=await branches[route.name].ainvoke(value, config))

        return RunnableLambda(go, afunc=ago)
