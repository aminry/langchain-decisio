# Contributing

Open an issue first for anything beyond a small fix.
Every commit carries a `Signed-off-by` line with your real name and an email you control (`git commit -s`), the Developer Certificate of Origin 1.1.
Pull requests pass `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest`.
A change to the tool gate is tested inside a real LangChain agent, as `tests/test_tool_gate.py` does.
