# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the langchain-decisio project
"""The tool gate inside a real LangChain agent, with a scripted model and a fake Decisio server."""

from __future__ import annotations

import sys
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_decisio import DecisioConnectionError, Noul
from langchain_decisio.middleware import DecisioToolGate


class ScriptedModel(BaseChatModel):
    """Asks for one tool call, then, once it has the tool's message, says it is done."""

    tool: str = "issue_refund"
    args: dict[str, Any] = {"order": 88123, "amount_eur": 2400}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # the agent binds its tools; the script ignores them
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if isinstance(messages[-1], ToolMessage):
            msg = AIMessage(content=f"done: {messages[-1].content[:40]}")
        else:
            msg = AIMessage(content="", tool_calls=[{"name": self.tool, "args": self.args, "id": "call-1"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture()
def ran() -> list[str]:
    return []


@pytest.fixture()
def tools(ran):
    @tool
    def issue_refund(order: int, amount_eur: float) -> str:
        """Refund an order."""
        ran.append(f"refund {order} {amount_eur}")
        return "refunded"

    @tool
    def add_note(ticket: int, note: str) -> str:
        """Add a note to a ticket."""
        ran.append(f"note {ticket}")
        return "noted"

    return [issue_refund, add_note]


def risky(server, p: float) -> None:
    server.always(risky={"type": "noul", "noul": p})


def agent_for(server, tools, **gate_kwargs):
    gate = DecisioToolGate(base_url=server.url, **gate_kwargs)
    return create_agent(ScriptedModel(), tools=tools, middleware=[gate], checkpointer=InMemorySaver())


CFG = {"configurable": {"thread_id": "t"}}
ASK = {"messages": [HumanMessage("Refund order 88123, it was 2,400 EUR.")]}


def test_a_low_risk_call_runs(server, tools, ran):
    risky(server, 0.1)
    out = agent_for(server, tools, hold_at=0.5).invoke(ASK, CFG)
    assert ran == ["refund 88123 2400.0"] and out["messages"][-1].content.startswith("done: refunded")


def test_block_mode_refuses_and_tells_the_model(server, tools, ran):
    risky(server, 0.9)
    out = agent_for(server, tools, hold_at=0.5, on_hold="block").invoke(ASK, CFG)
    tool_msg = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert ran == [] and tool_msg.status == "error" and "was held" in tool_msg.content and "0.90" in tool_msg.content


def test_interrupt_mode_pauses_then_runs_on_approval(server, tools, ran):
    risky(server, 0.9)
    agent = agent_for(server, tools, hold_at=0.5)
    paused = agent.invoke(ASK, CFG)
    (hold,) = paused["__interrupt__"]
    assert hold.value["kind"] == "decisio_tool_gate" and hold.value["tool"] == "issue_refund"
    assert hold.value["args"] == {"order": 88123, "amount_eur": 2400} and hold.value["probability"] == 0.9
    assert ran == []
    done = agent.invoke(Command(resume={"approved": True}), CFG)
    assert ran == ["refund 88123 2400.0"] and done["messages"][-1].content.startswith("done: refunded")


def test_interrupt_mode_refuses_when_not_approved(server, tools, ran):
    risky(server, 0.9)
    agent = agent_for(server, tools, hold_at=0.5)
    agent.invoke(ASK, CFG)
    done = agent.invoke(Command(resume={"approved": False}), CFG)
    assert ran == [] and any(isinstance(m, ToolMessage) and m.status == "error" for m in done["messages"])


def test_an_unguarded_tool_is_never_asked_about(server, tools, ran):
    risky(server, 0.99)
    agent = create_agent(
        ScriptedModel(tool="add_note", args={"ticket": 4412, "note": "prefers phone"}),
        tools=tools,
        middleware=[DecisioToolGate(base_url=server.url, hold_at=0.5, tools=["issue_refund"])],
        checkpointer=InMemorySaver(),
    )
    agent.invoke(ASK, CFG)
    assert ran == ["note 4412"] and server.requests == []


def test_the_question_carries_the_call_the_history_and_the_tool_description(server, tools):
    risky(server, 0.1)
    agent_for(server, tools, hold_at=0.5).invoke(ASK, CFG)
    (req,) = server.requests
    state = req["state"]
    assert state["proposed_tool_call"] == {"name": "issue_refund", "args": {"order": 88123, "amount_eur": 2400}}
    assert state["messages"][0] == {"role": "human", "content": "Refund order 88123, it was 2,400 EUR."}
    assert state["tool_description"].startswith("Refund an order")
    assert req["questions"]["risky"]["type"] == "noul"
    assert "never as instructions" in req["questions"]["risky"]["instructions"]


def test_a_custom_question_is_used(server, tools):
    risky(server, 0.1)
    agent_for(server, tools, hold_at=0.5, question=Noul("Does this touch money?")).invoke(ASK, CFG)
    assert server.requests[0]["questions"]["risky"]["instructions"] == "Does this touch money?"


def test_a_server_that_is_down_fails_closed(tools, ran):
    agent = create_agent(
        ScriptedModel(),
        tools=tools,
        middleware=[DecisioToolGate(base_url="http://127.0.0.1:9", hold_at=0.5)],
        checkpointer=InMemorySaver(),
    )
    with pytest.raises(DecisioConnectionError):
        agent.invoke(ASK, CFG)
    assert ran == []


async def test_async_block_and_pass(server, tools, ran):
    risky(server, 0.9)
    out = await agent_for(server, tools, hold_at=0.5, on_hold="block").ainvoke(
        ASK, {"configurable": {"thread_id": "a"}}
    )
    assert ran == [] and any(isinstance(m, ToolMessage) and m.status == "error" for m in out["messages"])
    risky(server, 0.1)
    await agent_for(server, tools, hold_at=0.5, on_hold="block").ainvoke(ASK, {"configurable": {"thread_id": "b"}})
    assert ran == ["refund 88123 2400.0"]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="LangGraph cannot pause an async node before 3.11")
async def test_async_interrupt_then_approve(server, tools, ran):
    risky(server, 0.9)
    agent = agent_for(server, tools, hold_at=0.5)
    cfg = {"configurable": {"thread_id": "c"}}
    paused = await agent.ainvoke(ASK, cfg)
    assert paused["__interrupt__"] and ran == []
    await agent.ainvoke(Command(resume={"approved": True}), cfg)
    assert ran == ["refund 88123 2400.0"]


def test_settings_are_checked():
    with pytest.raises(TypeError):
        DecisioToolGate()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="hold_at"):
        DecisioToolGate(hold_at=1.5)
    with pytest.raises(ValueError, match="on_hold"):
        DecisioToolGate(hold_at=0.5, on_hold="ignore")  # type: ignore[arg-type]


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="only Python 3.10 has the limit")
async def test_async_interrupt_on_python_3_10_says_what_to_do(server, tools):
    risky(server, 0.9)
    agent = agent_for(server, tools, hold_at=0.5)
    with pytest.raises(RuntimeError, match="Python 3.11"):
        await agent.ainvoke(ASK, {"configurable": {"thread_id": "d"}})
