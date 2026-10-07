# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the langchain-decisio project
"""LangChain components for Decisio, the open-source serving layer for decisions.

Decisio is an independent project, not affiliated with or endorsed by TypeSafe. It implements TypeSafe's published
System One wire format. This package is not affiliated with LangChain either.
"""

from __future__ import annotations

from typing import Any

from .classifier import DecisioClassifier, to_state
from .client import DecisioClient, DecisioConnectionError, DecisioError
from .router import DecisioRouter, Route, Routed
from .types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Decision,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "Decision",
    "DecisioClassifier",
    "DecisioClient",
    "DecisioConnectionError",
    "DecisioError",
    "DecisioRouter",
    "DecisioToolGate",
    "Noul",
    "NoulAnswer",
    "Route",
    "Routed",
    "Score",
    "ScoreAnswer",
    "to_state",
]


def __getattr__(name: str) -> Any:
    # DecisioToolGate needs the agent framework, so it loads on first use and tells you how to install it.
    if name == "DecisioToolGate":
        from .middleware import DecisioToolGate

        return DecisioToolGate
    raise AttributeError(f"module 'langchain_decisio' has no attribute {name!r}")
