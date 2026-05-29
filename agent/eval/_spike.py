# pyright: strict
"""M-eval/0 measurement spike — LIVE LLMs, not a pytest test.

Run: `make eval-spike`. Requires ONLY GROQ_API_KEY + ANTHROPIC_API_KEY —
importing `agent.main` is side-effect-safe (it builds a FastAPI app + reads
.env at import, but every credentialed client — Twilio/Deepgram/ElevenLabs —
is constructed lazily inside request handlers, not at module scope).

Outputs (to stdout): per-probe run-to-run agreement for the IVR tool-choice and
rep-extraction layers. These numbers set the pass thresholds for Plan 2
(M-eval/B/C/D). Nothing here is asserted; the spike informs, it does not gate.

Caveat (matches spec §3): the rep probes pass a single raw user-turn dict to
`complete_structured`, not the production `_history_to_anthropic_messages`
projection. For these single-turn probes the projection is near-identity, so the
agreement numbers transfer; multi-turn corpus cases in Plan 2's M-eval/C MUST
route through the real projection.
"""

from __future__ import annotations

import asyncio
from collections import Counter

from agent.llm_client import AnthropicRepClient, GroqToolCallingClient

# `_default_patient` / `_ivr_system_prompt` / `_rep_system_prompt` are module-
# private in agent.main; the spike is a throwaway measurement tool that
# legitimately reuses them rather than duplicating prompt-format logic. The
# per-name pyright ignores are required because the error is reported on each
# imported name's line, not the `from` line.
from agent.main import (
    _default_patient,  # pyright: ignore[reportPrivateUsage]
    _ivr_system_prompt,  # pyright: ignore[reportPrivateUsage]
    _rep_system_prompt,  # pyright: ignore[reportPrivateUsage]
)
from agent.schemas import RepTurnOutput, Turn
from agent.tools import groq_tool_schemas

_RUNS = 20  # 20 runs x 10 probes = ~200 live LLM calls per spike invocation.

# A handful of hand-written probes. Each is a (label, history-as-text).
_IVR_PROBES: list[tuple[str, str]] = [
    ("aetna-billing", "Thank you for calling. For billing press 1, for eligibility press 2."),
    ("rep-option", "For benefits press 2, to speak to a representative press 0."),
    ("greeting", "Welcome to the provider line. Please hold while we connect you."),
    ("member-id", "Please enter your member ID followed by the pound key."),
    ("repeat", "Press 9 at any time to repeat this menu."),
]

_REP_PROBES: list[tuple[str, str]] = [
    ("copay", "Sure, her copay is thirty dollars for a specialist visit."),
    ("active", "Yes, the policy is active and in good standing."),
    ("deductible", "She has four hundred dollars left on her deductible this year."),
    ("greeting", "Thanks for holding, this is Jamie, how can I help you?"),
    ("coinsurance", "After the deductible it's twenty percent coinsurance."),
]


async def _measure_ivr(ivr_system_prompt: str) -> None:
    client = GroqToolCallingClient()
    tools = groq_tool_schemas()
    print(f"\n=== IVR tool-choice stability (n={_RUNS}) ===")
    for label, text in _IVR_PROBES:
        history = [Turn(role="user", content=text)]
        names: Counter[str] = Counter()
        for _ in range(_RUNS):
            resp = await client.complete_with_tools(
                system=ivr_system_prompt, history=history, tools=tools, temperature=0.1
            )
            chosen = resp.tool_calls[0].name if resp.tool_calls else "<none>"
            names[chosen] += 1
        top, top_n = names.most_common(1)[0]
        print(f"  {label:14s} top={top:16s} agreement={top_n}/{_RUNS}  dist={dict(names)}")


async def _measure_rep(rep_system_prompt: str) -> None:
    client = AnthropicRepClient()
    print(f"\n=== Rep extraction stability (n={_RUNS}) ===")
    for label, text in _REP_PROBES:
        history: list[dict[str, object]] = [{"role": "user", "content": text}]
        phases: Counter[str] = Counter()
        extracted: Counter[str] = Counter()
        for _ in range(_RUNS):
            out = await client.complete_structured(
                system=rep_system_prompt, history=history, schema=RepTurnOutput
            )
            phases[out.phase] += 1
            extracted[out.extracted.model_dump_json(exclude_none=True)] += 1
        top_phase, top_phase_n = phases.most_common(1)[0]
        top_ex, top_ex_n = extracted.most_common(1)[0]
        print(
            f"  {label:14s} phase_top={top_phase}({top_phase_n}/{_RUNS}) "
            f"extract_top={top_ex}({top_ex_n}/{_RUNS})"
        )


async def main() -> None:
    total_calls = _RUNS * (len(_IVR_PROBES) + len(_REP_PROBES))
    print(
        f"eval spike: ~{total_calls} live LLM calls ({_RUNS} runs x {total_calls // _RUNS} probes)"
    )
    patient = _default_patient()
    await _measure_ivr(_ivr_system_prompt(patient))
    await _measure_rep(_rep_system_prompt(patient))
    print("\n(Spike complete. Use these agreement rates to set Plan 2 thresholds.)")


if __name__ == "__main__":
    asyncio.run(main())
