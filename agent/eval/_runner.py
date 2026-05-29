# pyright: strict
"""Run an injected per-case scorer over a corpus and aggregate a ScoreReport.

The scorer returns a `CaseResult` for a scoring decision (PASS/FAIL) and RAISES
for a transient/unexpected failure (API error, malformed response). The runner
centralizes ERROR handling: a raise is retried up to `per_case_retries`, drawing
from a shared `total_retry_budget`, before being recorded as `EvalOutcome.ERROR`.
This keeps provider noise (ERROR) from masquerading as a behavioral FAIL. Note:
only RAISES are retried — a scorer that returns a CaseResult with
`outcome=ERROR` directly is treated as a final scoring decision and not retried.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from agent.eval._types import CaseResult, EvalOutcome, ScoreReport
from agent.logging_config import log

_DEFAULT_MAX_CASES = 500
"""Cost guard: hard cap on cases per run so a runaway corpus can't silently
burn the account. Truncation is logged loudly, never silent."""


async def run_eval[CaseT: BaseModel](
    cases: list[CaseT],
    scorer: Callable[[CaseT], Awaitable[CaseResult]],
    *,
    layer: str,
    max_cases: int | None = _DEFAULT_MAX_CASES,
    per_case_retries: int = 1,
    total_retry_budget: int = 50,
) -> ScoreReport:
    selected = cases
    if max_cases is not None and len(cases) > max_cases:
        log.warning("eval_corpus_truncated", layer=layer, total=len(cases), kept=max_cases)
        selected = cases[:max_cases]

    retry_budget = total_retry_budget
    results: list[CaseResult] = []
    for case in selected:
        case_id = getattr(case, "id", "<unknown>")
        attempts_left = per_case_retries
        last_error: Exception | None = None
        result: CaseResult | None = None
        while True:
            try:
                result = await scorer(case)
                break
            except Exception as exc:  # any scorer raise is an ERROR candidate
                last_error = exc
                if attempts_left > 0 and retry_budget > 0:
                    attempts_left -= 1
                    retry_budget -= 1
                    log.warning("eval_case_retry", layer=layer, case_id=case_id, error=str(exc))
                    continue
                break
        if result is None:
            result = CaseResult(
                case_id=str(case_id),
                outcome=EvalOutcome.ERROR,
                error=str(last_error),
            )
        results.append(result)

    return ScoreReport.from_results(layer=layer, results=results)
