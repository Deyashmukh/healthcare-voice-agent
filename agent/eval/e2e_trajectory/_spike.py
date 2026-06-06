# pyright: strict
"""M-eval/D reachability spike — LIVE LLMs, not a pytest test.

Run: `uv run python -m agent.eval.e2e_trajectory._spike` (needs GROQ_API_KEY +
ANTHROPIC_API_KEY). Proves one happy-path call drives the real CallSessionRunner
through IVR nav -> rep handoff -> extraction -> completion. Prints the terminal
state. Nothing asserted; this de-risks the driver wiring for Tasks 2-6.
"""

from __future__ import annotations

import asyncio

from agent.actuator import Actuator
from agent.call_session import CallSessionRunner
from agent.llm_client import AnthropicRepClient, GroqToolCallingClient
from agent.main import (
    _default_patient,  # pyright: ignore[reportPrivateUsage]
    _ivr_system_prompt,  # pyright: ignore[reportPrivateUsage]
    _rep_system_prompt,  # pyright: ignore[reportPrivateUsage]
)
from agent.schemas import CallSession, SideEffectIntent
from agent.tools import dispatch, groq_tool_schemas

_SCRIPT = [
    "Thank you for calling provider services. For claims press 1, to speak with a representative press 0.",
    "One moment, connecting you to a representative.",
    "Hi, this is Jamie in benefits, who am I speaking with?",
    "Sure, I can help. Her plan is active. The specialist copay is forty dollars.",
    "She has three hundred dollars left on her deductible, and it's twenty percent coinsurance.",
    "Out of network is not covered. Is there anything else?",
    "You're welcome, have a great day. Goodbye.",
]


class _RecordingActuator:  # satisfies Actuator
    def __init__(self) -> None:
        self.intents: list[SideEffectIntent] = []

    async def execute(self, intent: SideEffectIntent) -> None:
        self.intents.append(intent)


async def main() -> None:
    session = CallSession(call_sid="E2E-spike", patient=_default_patient())
    actuator: Actuator = _RecordingActuator()
    runner = CallSessionRunner(
        session=session,
        ivr_llm=GroqToolCallingClient(),
        rep_llm=AnthropicRepClient(),
        tool_dispatcher=dispatch,
        ivr_system_prompt=_ivr_system_prompt(session.patient),
        rep_system_prompt=_rep_system_prompt(session.patient),
        tools=groq_tool_schemas(),
        actuator=actuator,
    )
    await runner.start()
    try:
        runner.submit_transcript(_SCRIPT[0])
        line = 1
        for _ in range(30):  # turn cap
            await _wait_turn(runner)
            if session.done:
                break
            runner.submit_transcript(_SCRIPT[line] if line < len(_SCRIPT) else _SCRIPT[-1])
            line += 1
    finally:
        await runner.stop()
    print(f"completion_reason={session.completion_reason}")
    print(f"mode_at_end={session.mode}  turns={session.turn_count}")
    print(f"benefits={session.benefits.model_dump()}")


async def _wait_turn(runner: CallSessionRunner) -> None:
    start = runner.session.turn_count
    for _ in range(2000):  # ~ up to 20s at 10ms; live LLM turns are seconds
        if runner.session.turn_count > start or runner.session.done:
            return
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(main())
