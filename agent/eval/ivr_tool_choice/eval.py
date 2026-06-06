# pyright: strict
"""Live IVR tool-choice eval — runs the real Groq client over the seed corpus.
Not a pytest test; run via `make evals`. Coverage-omitted.
"""

from __future__ import annotations

from agent.eval._loader import load_cases
from agent.eval._runner import run_eval
from agent.eval._types import CaseResult, IVREvalCase, ScoreReport
from agent.eval.ivr_tool_choice._score import CORPUS, score_ivr
from agent.llm_client import GroqToolCallingClient
from agent.main import _default_patient, _ivr_system_prompt  # pyright: ignore[reportPrivateUsage]
from agent.tools import groq_tool_schemas

LAYER = "ivr_tool_choice"


async def run() -> ScoreReport:
    client = GroqToolCallingClient()
    tools = groq_tool_schemas()
    system = _ivr_system_prompt(_default_patient())
    cases = load_cases(CORPUS, IVREvalCase)

    async def scorer(case: IVREvalCase) -> CaseResult:
        response = await client.complete_with_tools(
            system=system, history=case.history, tools=tools, temperature=0.1
        )
        return score_ivr(case, response)

    return await run_eval(cases, scorer, layer=LAYER)
