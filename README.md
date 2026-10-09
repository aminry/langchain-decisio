# langchain-decisio

LangChain components for [Decisio](https://github.com/aminry/decisio), an open-source server that answers typed questions about a piece of text with a probability for every option, from one forward pass of a frozen open model.

Decisio is built by [Tachara AI Lab](https://huggingface.co/tachara-ai).
Decisio is an independent project, not affiliated with or endorsed by TypeSafe, and implements TypeSafe's published System One wire format.
This package is not affiliated with LangChain either.

**Status: 0.1.0, beta.** The classifier and the router are small and stable; the tool gate sits on LangChain's new agent middleware API and is the part most likely to change.

## Three pieces

| Piece | What it does |
| --- | --- |
| `DecisioClassifier` | A Runnable. Typed questions about a state go in, a `Decision` comes out: every answer with its probabilities, the server's usage and its own time for the request |
| `DecisioRouter` | A Runnable that turns a `Choice` into a route. A chosen option is accepted at a probability you set, and anything less goes to a fallback you name |
| `DecisioToolGate` | Agent middleware. One yes/no question about each proposed tool call, and a risky one is held for a person's approval, or refused |

Nothing here generates text.
Decisio picks among options you give it.

## If you already use `langchain-typesafe`

It works against a Decisio server: pass `base_url` (or set `TYPESAFE_BASE_URL`) and a dummy API key.
That is the right answer if it already does what you need, and `decisio-examples` checks that it does.

This package exists for what that route cannot give you:

- no API key and no hosted model name in a local setup;
- `abstained` and `unknown_probability` kept, so a caller can tell that the server abstained;
- the server's `detail` in the error message when a question is rejected;
- Decisio's own routes (task registration, health, the base it serves) from the same object;
- a router and a tool gate with thresholds you set, not defaults someone else tuned.

## Install

```bash
pip install langchain-decisio             # the classifier and the router
pip install "langchain-decisio[agents]"   # and the tool gate, which needs LangChain's agent framework
```

You need a Decisio server.
The [decisio README](https://github.com/aminry/decisio) has the GPU, Docker, Mac and Ollama paths.
The server's address is `base_url`, else `$DECISIO_URL`, else `http://127.0.0.1:8000`.

## The classifier

```python
from langchain_decisio import Choice, DecisioClassifier, Noul

classifier = DecisioClassifier(
    questions={
        "urgent": Noul("Does this ticket need a response within the hour?"),
        "queue": Choice(
            "Which team should handle this ticket?", {"billing": "Invoices, payments", "access": "Login", "bug": None}
        ),
    }
)
decision = classifier.invoke("Since this morning none of our 40 staff can log in. Payroll is due today.")
decision.p_yes("urgent")  # the probability of yes
decision.top("queue")  # ("access", 0.97)
decision["queue"].probabilities
decision.server_ms  # the server's own time
```

- The state is a string, a dict, a list, a LangChain message or a list of messages.
- Questions can instead come with each call: `classifier.invoke({"state": "...", "questions": {...}})`.
- A `Choice` needs 2 to 255 options and a `Score` needs at least 2 levels; a bad question fails before any request.
- `ainvoke` and `batch` work, and runs show up in LangSmith like any Runnable.
- Questions are `Noul` (yes/no), `Choice` and `Score`.
  The names are TypeSafe's wire format, kept so existing clients work.

## The router

```python
from langchain_decisio import Choice, DecisioRouter

router = DecisioRouter(
    Choice("Which team should handle this ticket?", ["billing", "access", "bug"]),
    fallback="human",
    accept_at=0.8,
)
route = router.invoke("It is broken again.")
route.name, route.probability, route.accepted  # ("human", 0.52, False)
```

`accept_at` has no default, on purpose.
A probability threshold only means something against your own items: sweep it on a few hundred labelled ones and read accuracy against coverage.
[Example 1 of decisio-examples](https://github.com/aminry/decisio-examples) does exactly that.

To run a branch for the chosen route:

```python
chain = router.run_branches({"billing": billing_chain, "access": access_chain, "bug": bug_chain, "human": to_human})
chain.invoke("Please send me the March invoice.")  # Routed(route=Route(...), output=...)
```

## The tool gate

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_decisio import DecisioToolGate

gate = DecisioToolGate(tools=["issue_refund", "delete_workspace"], hold_at=0.5)
agent = create_agent(model, tools=[...], middleware=[gate], checkpointer=InMemorySaver())
# a held call pauses the agent; resume it with Command(resume={"approved": True}) or {"approved": False}
```

- Before a guarded tool runs, one yes/no question is asked about the proposed call and the last 30 messages.
  Below `hold_at` the tool runs.
- `on_hold="interrupt"` (the default) pauses the agent with LangGraph's `interrupt`, so it needs a checkpointer.
  `on_hold="block"` refuses the call and tells the model why, and asks nobody.
- The default question asks about reversibility, reach, money, deletion and access, and tells the model to treat tool arguments and earlier tool output as data, never as instructions.
  Pass your own `question=Noul(...)` for your policy.
- In an async agent, `on_hold="interrupt"` needs Python 3.11 or later, because LangGraph cannot carry its run context into an async node on 3.10.
  On 3.10 use `block`, or run the agent synchronously; the gate says so when it happens.
- It writes no arguments and calls no tool.
  A server that is down fails closed: the request error propagates and the tool does not run.
- It is one cheap layer, not an access-control system.
  Keep each tool's own permissions narrow.
  Measure `hold_at` on proposed calls of your own, as [example 4](https://github.com/aminry/decisio-examples) does on 14 labelled ones.

## Decisio's own routes

```python
from langchain_decisio import DecisioClient

client = DecisioClient()
client.served_base()  # "gemma-4-31b"
client.register_task({"id": "tickets", "examples": [...]})  # docs/tasks.md in decisio
```

A registered task calibrates an answer from your own labelled examples.
Decisio applies it to any later question with the same options.

## Tested against

| | |
| --- | --- |
| langchain-core | 1.6.7 |
| langchain | 1.4.3 (the tool gate) |
| langgraph | 1.2.14 |
| decisio | 0.10.0, on its CPU stand-in; see CI |
| Python | 3.10 to 3.13 |

The tests run against a fake server and a real LangChain agent with a scripted model.
Which probabilities a real model gives, and how good they are, is Decisio's EVAL_CARD's business and not this package's.
The agent middleware API is new, so the tool gate is the part most likely to change.
The version of each dependency a release was tested with is stated in that release's notes.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Every commit carries a `Signed-off-by` line (`git commit -s`).

## Licence

Apache-2.0.
