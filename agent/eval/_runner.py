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

from agent.eval._types import CaseResult, EvalCase, EvalOutcome, ScoreReport
from agent.logging_config import log

_DEFAULT_MAX_CASES = 500
"""Cost guard: hard cap on cases per run so a runaway corpus can't silently
burn the account. Truncation is logged loudly, never silent."""


async def run_eval[CaseT: EvalCase](
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
    if not selected:
        # An empty run reports pass_rate=0.0, which reads like a catastrophic
        # regression rather than "nothing ran" — say so loudly.
        log.warning("eval_corpus_empty", layer=layer)

    retry_budget = total_retry_budget
    results: list[CaseResult] = []
    for case in selected:
        case_id = case.id
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
            # Terminal ERROR (retries exhausted, or per_case_retries=0). Log it
            # loudly — a model/provider failure that exists only as a number in
            # the aggregate report is the silent failure we're guarding against.
            # Prefix the exception type so a real bug (AttributeError) is visibly
            # distinct from a provider blip (ConnectionError) in the report.
            error_text = f"{type(last_error).__name__}: {last_error}"
            log.warning("eval_case_error", layer=layer, case_id=case_id, error=error_text)
            result = CaseResult(
                case_id=case_id,
                outcome=EvalOutcome.ERROR,
                error=error_text,
            )
        results.append(result)

    return ScoreReport.from_results(layer=layer, results=results)
