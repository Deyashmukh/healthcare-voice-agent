# pyright: strict
"""Live E2E trajectory eval — drives the real runner against each scripted payer
scenario. Not a pytest test; run via `make evals`. Coverage-omitted.
"""

from __future__ import annotations

from agent.actuator import Actuator
from agent.call_session import CallSessionRunner
from agent.eval._runner import run_eval
from agent.eval._types import CaseResult, ScoreReport
from agent.eval.e2e_trajectory._driver import RecordingActuator, run_scenario, score_scenario
from agent.eval.e2e_trajectory._scripted_payer import SCENARIOS, Scenario
from agent.llm_client import AnthropicRepClient, GroqToolCallingClient
from agent.main import (
    _default_patient,  # pyright: ignore[reportPrivateUsage]
    _ivr_system_prompt,  # pyright: ignore[reportPrivateUsage]
    _rep_system_prompt,  # pyright: ignore[reportPrivateUsage]
)
from agent.schemas import CallSession
from agent.tools import dispatch, groq_tool_schemas

LAYER = "e2e_trajectory"


async def run() -> ScoreReport:
    patient = _default_patient()
    tools = groq_tool_schemas()
    ivr_system = _ivr_system_prompt(patient)
    rep_system = _rep_system_prompt(patient)

    async def scorer(scenario: Scenario) -> CaseResult:
        session = CallSession(call_sid=f"E2E-{scenario.id}", patient=patient)
        actuator: Actuator = RecordingActuator()
        runner = CallSessionRunner(
            session=session,
            ivr_llm=GroqToolCallingClient(),
            rep_llm=AnthropicRepClient(),
            tool_dispatcher=dispatch,
            ivr_system_prompt=ivr_system,
            rep_system_prompt=rep_system,
            tools=tools,
            actuator=actuator,
        )
        await runner.start()
        try:
            await run_scenario(runner, scenario)
        finally:
            await runner.stop()
        return score_scenario(scenario, runner)

    return await run_eval(list(SCENARIOS), scorer, layer=LAYER)
