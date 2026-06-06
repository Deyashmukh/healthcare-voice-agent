# pyright: strict
"""Live rep-extraction eval — runs the real Anthropic client over the seed
corpus, routing history through the production projection so the eval sees what
`_rep_turn` sends. Not a pytest test; run via `make evals`. Coverage-omitted.
"""

from __future__ import annotations

from agent.call_session import _history_to_anthropic_messages  # pyright: ignore[reportPrivateUsage]
from agent.eval._loader import load_cases
from agent.eval._runner import run_eval
from agent.eval._types import CaseResult, RepEvalCase, ScoreReport
from agent.eval.rep_extraction._score import CORPUS, score_rep
from agent.llm_client import AnthropicRepClient
from agent.main import _default_patient, _rep_system_prompt  # pyright: ignore[reportPrivateUsage]
from agent.schemas import RepTurnOutput

LAYER = "rep_extraction"


async def run() -> ScoreReport:
    client = AnthropicRepClient()
    system = _rep_system_prompt(_default_patient())
    cases = load_cases(CORPUS, RepEvalCase)

    async def scorer(case: RepEvalCase) -> CaseResult:
        output = await client.complete_structured(
            system=system,
            history=_history_to_anthropic_messages(case.history),
            schema=RepTurnOutput,
        )
        return score_rep(case, output)

    return await run_eval(cases, scorer, layer=LAYER)
