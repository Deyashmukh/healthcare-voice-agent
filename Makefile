.PHONY: agent test lint format typecheck install eval-spike

install:
	uv sync

agent:
	uv run uvicorn agent.main:app --host 0.0.0.0 --port 8000

test:
	uv run pytest tests/unit

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run pyright

# M-eval/0 measurement spike — LIVE LLMs (~200 calls). Needs GROQ_API_KEY +
# ANTHROPIC_API_KEY. Prints run-to-run agreement to set Plan 2 thresholds.
eval-spike:
	uv run python -m agent.eval._spike
