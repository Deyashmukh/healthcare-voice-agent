# pyright: strict
"""Run the component evals and write a report. Live LLMs; coverage-omitted.

Usage: `python -m agent.eval.cli {ivr,rep,all}` (default: all), or `make evals`.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from agent.eval._report import render_table, write_report
from agent.eval._types import ScoreReport
from agent.eval.ivr_tool_choice import eval as ivr_eval
from agent.eval.rep_extraction import eval as rep_eval

_RESULTS_DIR = Path("eval_results")
_HISTORY_PATH = Path("eval_history.jsonl")


async def _run_selected(layers: list[str]) -> list[ScoreReport]:
    reports: list[ScoreReport] = []
    if "ivr" in layers:
        reports.append(await ivr_eval.run())
    if "rep" in layers:
        reports.append(await rep_eval.run())
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Run component evals.")
    parser.add_argument("layer", nargs="?", default="all", choices=["ivr", "rep", "all"])
    args = parser.parse_args()
    layers = ["ivr", "rep"] if args.layer == "all" else [args.layer]

    reports = asyncio.run(_run_selected(layers))
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for report in reports:
        print(render_table(report))
        out = write_report(
            report, results_dir=_RESULTS_DIR, history_path=_HISTORY_PATH, timestamp=timestamp
        )
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
