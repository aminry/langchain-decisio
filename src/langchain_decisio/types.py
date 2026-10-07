# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the langchain-decisio project
"""Questions and answers in Decisio's System One wire format.

A question is checked here, before a request is sent, with the rules the server applies: a choice needs at least two
options and at most 255, a score needs at least two levels. A malformed question fails with a message that names the
question, not with a 422 from the server.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_OPTIONS = 255


class Noul(BaseModel):
    """A yes/no question. Its answer is the probability of yes."""

    model_config = ConfigDict(frozen=True)

    type: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1)
    true: str | None = None
    false: str | None = None

    def __init__(self, instructions: str, true: str | None = None, false: str | None = None, **data: Any):
        super().__init__(instructions=instructions, true=true, false=false, **data)

    def to_wire(self) -> dict[str, Any]:
        q: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.true or self.false:
            q["criteria"] = {"true": self.true, "false": self.false}
        return q


class Choice(BaseModel):
    """A choice among named options, each with an optional description."""

    model_config = ConfigDict(frozen=True)

    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    options: dict[str, str | None]

    def __init__(self, instructions: str, options: Any, **data: Any):
        super().__init__(instructions=instructions, options=options, **data)

    @field_validator("options", mode="before")
    @classmethod
    def _options_from_list(cls, v: Any) -> Any:
        if isinstance(v, Sequence) and not isinstance(v, str | bytes):
            return {str(k): None for k in v}
        return v

    @field_validator("options")
    @classmethod
    def _count(cls, v: dict[str, str | None]) -> dict[str, str | None]:
        if not 2 <= len(v) <= MAX_OPTIONS:
            raise ValueError(f"a choice needs between 2 and {MAX_OPTIONS} options, got {len(v)}")
        return v

    def to_wire(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.options)}


class Score(BaseModel):
    """A score on an ordered scale, lowest level first. Its answer is the probability-weighted level."""

    model_config = ConfigDict(frozen=True)

    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1)
    levels: list[str]

    def __init__(self, instructions: str, levels: list[str], **data: Any):
        super().__init__(instructions=instructions, levels=levels, **data)

    @field_validator("levels")
    @classmethod
    def _count(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError(f"a score needs at least 2 levels, got {len(v)}")
        return v

    def to_wire(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.levels)}


Question = Noul | Choice | Score


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    probability: float
    """The probability of yes."""


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None
    unknown_probability: float | None = None
    abstained: bool | None = None
    """Set when the server was started with an abstain option: whether it abstained rather than chose."""

    @property
    def probability(self) -> float:
        """The probability of the chosen option."""
        return self.probabilities[self.choice]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, float]
    confidence: float | None = None
    legend: dict[str, str] = Field(default_factory=dict)


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Decision(BaseModel):
    """What a request returned, with everything the server said about it."""

    answers: dict[str, Answer]
    model: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    server_ms: float | None = None
    """The server's own time for the request (the `x-decisio-server-ms` header)."""
    tasks: list[str] = Field(default_factory=list)
    """The registered tasks the server applied (the `x-decisio-tasks` header)."""

    def __getitem__(self, name: str) -> Answer:
        return self.answers[name]

    def p_yes(self, name: str) -> float:
        a = self.answers[name]
        if not isinstance(a, NoulAnswer):
            raise TypeError(f"question {name!r} is a {a.type} question, not a yes/no question")
        return a.probability

    def top(self, name: str) -> tuple[str, float]:
        """The chosen option and its probability."""
        a = self.answers[name]
        if not isinstance(a, ChoiceAnswer):
            raise TypeError(f"question {name!r} is a {a.type} question, not a choice")
        return a.choice, a.probability


def parse_answer(raw: Mapping[str, Any]) -> Answer:
    kind = raw.get("type")
    if kind == "noul":
        return NoulAnswer(probability=raw["noul"])
    if kind == "choice":
        return ChoiceAnswer(
            choice=raw["choice"],
            probabilities=raw["probabilities"],
            confidence=raw.get("confidence"),
            unknown_probability=raw.get("unknown_probability"),
            abstained=raw.get("abstained"),
        )
    if kind == "score":
        return ScoreAnswer(
            score=raw["score"],
            probabilities=raw["probabilities"],
            confidence=raw.get("confidence"),
            legend=raw.get("legend", {}),
        )
    raise ValueError(f"the server returned an answer of unknown type {kind!r}")


def questions_to_wire(questions: Mapping[str, Noul | Choice | Score]) -> dict[str, dict[str, Any]]:
    """The wire form of a named set of questions; at least one is required."""
    if not questions:
        raise ValueError("a request needs at least one question")
    out = {}
    for name, q in questions.items():
        if not isinstance(q, Noul | Choice | Score):
            raise TypeError(f"question {name!r} is a {type(q).__name__}; use Noul, Choice or Score")
        out[name] = q.to_wire()
    return out
