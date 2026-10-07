# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the langchain-decisio project
"""Against a real Decisio server, when DECISIO_URL names one. CI points it at decisio's CPU stand-in.

The stand-in runs a 0.6B model on the CPU, so these tests check that the steps work and the shapes are right.
They say nothing about how good the probabilities are.
"""

import os

import pytest

from langchain_decisio import Choice, DecisioClassifier, DecisioClient, DecisioError, DecisioRouter, Noul, Score

URL = os.environ.get("DECISIO_URL")
pytestmark = pytest.mark.skipif(not URL, reason="DECISIO_URL is not set")


def test_a_real_server_answers_all_three_types():
    c = DecisioClassifier(
        URL,
        questions={
            "urgent": Noul("Does this ticket need a response within the hour?"),
            "queue": Choice("Which team should handle this?", {"billing": None, "access": None, "bug": None}),
            "impact": Score("How bad is it?", ["none", "one user", "many users", "data loss"]),
        },
    )
    d = c.invoke("Since this morning none of our 40 staff can log in to the dashboard.")
    assert 0.0 <= d.p_yes("urgent") <= 1.0
    choice, p = d.top("queue")
    assert choice in {"billing", "access", "bug"} and abs(sum(d["queue"].probabilities.values()) - 1) < 1e-3
    assert 0.0 <= d["impact"].score <= 3.0 and d.server_ms is not None


def test_a_real_server_rejects_a_malformed_request_with_its_detail():
    # The client refuses an empty request before sending it, so send one by hand to see the server's own 422.
    client = DecisioClient(URL)
    response = client._sync_client().post("/v1/systemone", json={"state": "text", "questions": {}})
    with pytest.raises(DecisioError) as e:
        client._decision(response)
    assert e.value.status == 422 and e.value.detail


def test_router_and_health():
    r = DecisioRouter(Choice("Which?", ["a", "b"]), fallback="human", accept_at=0.99, base_url=URL)
    route = r.invoke("some text")
    assert route.chosen in {"a", "b"} and route.name in {"a", "b", "human"}
    assert DecisioClient(URL).health().get("ok") is True
