# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the langchain-decisio project
import pytest
from langchain_core.runnables import RunnableLambda

from langchain_decisio import Choice, DecisioRouter

Q = Choice("Which team?", ["billing", "access", "bug"])


def answer(billing: float, access: float, bug: float):
    p = {"billing": billing, "access": access, "bug": bug}
    top = max(p, key=p.get)
    return {"type": "choice", "choice": top, "probabilities": p}


def test_accepts_at_the_threshold_and_falls_back_below_it(server):
    router = DecisioRouter(Q, fallback="human", accept_at=0.8, base_url=server.url)
    server.always(route=answer(0.8, 0.1, 0.1))
    r = router.invoke("text")
    assert (r.name, r.chosen, r.accepted, r.probability) == ("billing", "billing", True, 0.8)
    server.always(route=answer(0.5, 0.3, 0.2))
    r = router.invoke("text")
    assert (r.name, r.chosen, r.accepted) == ("human", "billing", False)
    assert r.decision.top("route") == ("billing", 0.5)


def test_accept_at_is_required_and_checked():
    with pytest.raises(TypeError):
        DecisioRouter(Q, fallback="human")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="accept_at"):
        DecisioRouter(Q, fallback="human", accept_at=0)


async def test_async_route(server):
    server.always(route=answer(0.1, 0.1, 0.8))
    r = await DecisioRouter(Q, fallback="human", accept_at=0.6, base_url=server.url).ainvoke("text")
    assert r.name == "bug"


def test_run_branches_runs_the_chosen_branch_on_the_same_input(server):
    router = DecisioRouter(Q, fallback="human", accept_at=0.6, base_url=server.url)
    chain = router.run_branches(
        {n: RunnableLambda(lambda x, n=n: f"{n}:{x}") for n in ("billing", "access", "bug", "human")}
    )
    server.always(route=answer(0.1, 0.8, 0.1))
    out = chain.invoke("login problem")
    assert out.route.name == "access" and out.output == "access:login problem"
    server.always(route=answer(0.4, 0.3, 0.3))
    assert chain.invoke("odd").output == "human:odd"


def test_run_branches_names_the_missing_branch(server):
    router = DecisioRouter(Q, fallback="human", accept_at=0.6, base_url=server.url)
    with pytest.raises(ValueError, match="human"):
        router.run_branches({"billing": RunnableLambda(str), "access": RunnableLambda(str), "bug": RunnableLambda(str)})


async def test_run_branches_async(server):
    router = DecisioRouter(Q, fallback="human", accept_at=0.6, base_url=server.url)
    chain = router.run_branches(
        {n: RunnableLambda(lambda x, n=n: f"{n}!") for n in ("billing", "access", "bug", "human")}
    )
    server.always(route=answer(0.9, 0.05, 0.05))
    assert (await chain.ainvoke("x")).output == "billing!"
