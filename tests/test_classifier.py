# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the langchain-decisio project
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from langchain_decisio import (
    Choice,
    ChoiceAnswer,
    DecisioClassifier,
    DecisioClient,
    DecisioConnectionError,
    DecisioError,
    Noul,
    Score,
)

QUESTIONS = {
    "urgent": Noul("Is it urgent?", true="Yes", false="No"),
    "queue": Choice("Which queue?", {"billing": "Invoices", "access": None, "bug": None}),
    "impact": Score("How bad?", ["none", "one user", "many users"]),
}


def test_all_three_types_round_trip(server):
    d = DecisioClassifier(server.url, questions=QUESTIONS).invoke("text")
    assert d.p_yes("urgent") == 0.9
    assert d.top("queue") == ("billing", 0.9)
    assert d["impact"].legend == {"0": "none", "1": "one user", "2": "many users"}
    assert d.usage == {"input_tokens": 7, "output_tokens": 1}
    assert d.server_ms == 12.5 and d.tasks == ["t1", "t2"] and d.model == "fake-model"


def test_wire_body_has_no_model_or_key_unless_asked(server):
    DecisioClassifier(server.url, questions=QUESTIONS).invoke("text")
    body = server.requests[0]
    assert set(body) == {"state", "questions"}
    assert body["questions"]["queue"] == {
        "type": "choice",
        "instructions": "Which queue?",
        "criteria": {"billing": "Invoices", "access": None, "bug": None},
    }
    assert body["questions"]["urgent"]["criteria"] == {"true": "Yes", "false": "No"}
    assert body["questions"]["impact"]["criteria"] == ["none", "one user", "many users"]
    DecisioClassifier(server.url, questions=QUESTIONS, model="aminroudaki/decisio-gemma").invoke("text")
    assert server.requests[1]["model"] == "aminroudaki/decisio-gemma"


def test_questions_per_call_and_mapping_without_questions_is_the_state(server):
    c = DecisioClassifier(server.url, questions=QUESTIONS)
    c.invoke({"state": "per call", "questions": {"u": Noul("Urgent?")}})
    assert server.requests[-1]["state"] == "per call" and list(server.requests[-1]["questions"]) == ["u"]
    c.invoke({"subject": "login", "body": "cannot log in"})
    assert server.requests[-1]["state"] == {"subject": "login", "body": "cannot log in"}


def test_messages_become_role_and_content(server):
    c = DecisioClassifier(server.url, questions={"u": Noul("Urgent?")})
    c.invoke([HumanMessage("hello"), AIMessage("hi, how can I help?")])
    assert server.requests[-1]["state"] == [
        {"role": "human", "content": "hello"},
        {"role": "ai", "content": "hi, how can I help?"},
    ]
    c.invoke(HumanMessage("one message"))
    assert server.requests[-1]["state"] == "one message"


def test_no_questions_is_an_error_before_any_request(server):
    with pytest.raises(ValueError, match="no questions"):
        DecisioClassifier(server.url).invoke("text")
    assert server.requests == []


def test_a_bad_question_fails_before_any_request(server):
    with pytest.raises(ValidationError, match="between 2 and 255"):
        Choice("Which?", ["only-one"])
    with pytest.raises(ValidationError, match="at least 2 levels"):
        Score("How?", ["one"])
    with pytest.raises(ValidationError):
        Noul("")
    assert server.requests == []


def test_error_message_carries_the_servers_detail(server):
    server.responder = lambda body: (422, {"detail": "question 'queue': options must be unique"}, {})
    with pytest.raises(DecisioError) as e:
        DecisioClassifier(server.url, questions=QUESTIONS).invoke("text")
    assert e.value.status == 422
    assert "options must be unique" in str(e.value)


def test_abstention_fields_are_kept(server):
    server.always(
        queue={
            "type": "choice",
            "choice": "bug",
            "probabilities": {"billing": 0.2, "access": 0.2, "bug": 0.6},
            "unknown_probability": 0.4,
            "abstained": True,
        }
    )
    a = DecisioClassifier(server.url, questions={"queue": QUESTIONS["queue"]}).invoke("text")["queue"]
    assert isinstance(a, ChoiceAnswer) and a.abstained is True and a.unknown_probability == 0.4
    assert a.probability == 0.6


async def test_ainvoke_and_batch(server):
    c = DecisioClassifier(server.url, questions={"u": Noul("Urgent?")})
    assert (await c.ainvoke("text")).p_yes("u") == 0.9
    out = c.batch(["a", "b", "c"])
    assert [d.p_yes("u") for d in out] == [0.9, 0.9, 0.9]
    assert sorted(r["state"] for r in server.requests[-3:]) == ["a", "b", "c"]


def test_the_wrong_answer_kind_is_a_type_error(server):
    d = DecisioClassifier(server.url, questions=QUESTIONS).invoke("text")
    with pytest.raises(TypeError, match="not a yes/no"):
        d.p_yes("queue")
    with pytest.raises(TypeError, match="not a choice"):
        d.top("urgent")


def test_hosted_typesafe_is_refused():
    with pytest.raises(ValueError, match="langchain-typesafe"):
        DecisioClient("https://api.typesafe.ai")


def test_an_unreachable_server_is_a_connection_error():
    c = DecisioClassifier("http://127.0.0.1:9", questions={"u": Noul("Urgent?")})
    with pytest.raises(DecisioConnectionError, match="could not reach"):
        c.invoke("text")


def test_client_routes(server):
    c = DecisioClient(server.url)
    assert c.served_base() == "gemma-4-31b"
    assert c.register_task({"id": "tickets", "examples": []}) == {"id": "tickets", "applied": True}
    assert server.paths[-2:] == ["/health", "/v1/tasks"]
