# pyright: strict
"""Run the component evals and write a report. Live LLMs; coverage-omitted.

Usage: `python -m agent.eval.cli {ivr,rep,all}` (default: all), or `make evals`.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from agent.eval._report import render_table, write_report
from agent.eval._types import ScoreReport
from agent.eval.e2e_trajectory import eval as e2e_eval
from agent.eval.ivr_tool_choice import eval as ivr_eval
from agent.eval.rep_extraction import eval as rep_eval

_RESULTS_DIR = Path("eval_results")
_HISTORY_PATH = Path("eval_history.jsonl")

_LAYERS: dict[str, Callable[[], Awaitable[ScoreReport]]] = {
    "ivr": ivr_eval.run,
    "rep": rep_eval.run,
    "e2e": e2e_eval.run,
}


async def _run_and_report(layers: list[str], timestamp: str) -> None:
    # Write each layer's report the moment its run completes, so a later layer
    # failing (e.g. a corpus or config error) can't discard an earlier layer's
    # already-computed results — and its already-spent API calls.
    for name in layers:
        report = await _LAYERS[name]()
        print(render_table(report))
        out = write_report(
            report, results_dir=_RESULTS_DIR, history_path=_HISTORY_PATH, timestamp=timestamp
        )
        print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run component evals.")
    parser.add_argument("layer", nargs="?", default="all", choices=["ivr", "rep", "e2e", "all"])
    args = parser.parse_args()
    layers = ["ivr", "rep", "e2e"] if args.layer == "all" else [args.layer]

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    asyncio.run(_run_and_report(layers, timestamp))


if __name__ == "__main__":
    main()
