# pyright: strict
"""Scripted-payer scenarios for the E2E trajectory eval.

A Scenario is an ordered list of payer utterances (`script`) plus the terminal
state we expect the agent to reach. The driver feeds one line per agent turn;
write each line so the agent takes a single action in response, and make the
last line a clear closing that triggers `complete_call`.
"""

from __future__ import annotations

from agent.eval._types import EvalCase
from agent.schemas import Benefits, CompletionReason


class Scenario(EvalCase):
    # Subclass EvalCase (provides `id` + extra="forbid") so it satisfies the
    # `run_eval[CaseT: EvalCase]` bound — passing a bare BaseModel fails pyright.
    script: tuple[str, ...]  # payer utterances, one fed per agent turn
    expected_completion_reason: CompletionReason
    expected_benefits: Benefits | None = None
    max_turns: int = 25


_HAPPY_PATH = Scenario(
    id="happy-path-full-extraction",
    script=(
        "Thank you for calling provider services. For claims press 1, "
        "to speak with a representative press 0.",
        "One moment, connecting you to a representative.",
        "Hi, this is Jamie in benefits. Who am I speaking with today?",
        "Sure, I can help with that. Her plan is active and effective. "
        "The specialist copay is forty dollars.",
        "She has three hundred dollars remaining on her deductible, "
        "and after that it's twenty percent coinsurance.",
        "Out of network services are not covered under this plan. "
        "Is there anything else I can help with?",
        "You're very welcome. Have a great day. Goodbye.",
    ),
    expected_completion_reason="rep_complete",
    expected_benefits=Benefits(
        active=True,
        copay=40.0,
        deductible_remaining=300.0,
        coinsurance=20.0,
        out_of_network_coverage=False,
    ),
)

_REP_STUCK = Scenario(
    id="rep-stuck-no-info",
    script=(
        "Thank you for calling. To speak with a representative press 0.",
        "Connecting you now.",
        "Hi, this is Sam. I'm sorry, our benefits system is down right now, "
        "I can't pull up any of that information.",
        "I really can't access it, I'm sorry. There's nothing I can tell you today.",
        "No, still nothing, I apologize.",
    ),
    expected_completion_reason="rep_stuck",
)

SCENARIOS: tuple[Scenario, ...] = (_HAPPY_PATH, _REP_STUCK)
