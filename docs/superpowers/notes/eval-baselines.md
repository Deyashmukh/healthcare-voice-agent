# Eval Stability Baselines (M-eval/0 spike)

Run: 2026-06-05, `make eval-spike` — `agent/eval/_spike.py`, n=20 runs per probe,
5 IVR + 5 rep hand-written probes, real Groq (Llama-4-Scout, temp 0.1) + Anthropic
(Haiku-4.5) clients. ~200 live calls.

## Raw results

IVR tool-choice stability (n=20):

| probe          | top tool   | agreement |
|----------------|------------|-----------|
| aetna-billing  | send_dtmf  | 20/20     |
| rep-option     | send_dtmf  | 20/20     |
| greeting       | wait       | 20/20     |
| member-id      | speak      | 20/20     |
| repeat         | wait       | 20/20     |

Rep extraction stability (n=20):

| probe        | phase           | extracted (top)                  |
|--------------|-----------------|----------------------------------|
| copay        | extracting 20/20| `{"copay":30.0}` 20/20           |
| active       | extracting 20/20| `{"active":true}` 20/20          |
| deductible   | extracting 20/20| `{"deductible_remaining":400.0}` 20/20 |
| greeting     | extracting 20/20| `{}` 20/20                       |
| coinsurance  | extracting 20/20| `{"coinsurance":20.0}` 20/20     |

## Interpretation

Perfect run-to-run agreement on every probe, both layers — including exact float
values on the rep side. At temp 0.1 on clear, unambiguous inputs the models are
effectively deterministic.

**Go/no-go: GREEN.** Single-run exact-match scoring is viable for both component
layers. The majority-vote / N-run contingency the spec hedged on is NOT needed —
drop it from Plan 2. This simplifies B/C (one call per case, exact assert).

## Caveats (do not over-read 20/20)

- These are 5 hand-picked CLEAN probes per layer (unambiguous menus, clear value
  statements). They establish that the EASY cases are rock-solid, not that
  everything is. Genuinely ambiguous menus and fuzzy multi-turn rep utterances
  will have real variance.
- n=20 gives high confidence the per-probe agreement is very high, but is a
  coarse variance estimate for marginal cases.
- The rep probes used a single raw user-turn dict, not the production
  `_history_to_anthropic_messages` projection (near-identity for single-turn).
  Plan 2's M-eval/C multi-turn cases MUST route through the real projection.

## Plan 2 threshold guidance (derived)

- **IVR tool-name + deterministic args:** single-run exact-match. Seed corpus of
  clear cases should score ~100%; set the CI gate with headroom (e.g. allow a
  small number of genuinely-ambiguous cases to be tracked separately rather than
  hard-failing the build). Measure the real corpus before pinning the number.
- **Rep extraction + phase + reply-presence:** single-run exact-match, floats
  compared exactly (confirmed stable). Same headroom posture.
- **E2E (M-eval/D):** per-turn determinism is high, so terminal state should be
  reachable reliably; D can start with a single-run terminal-state assertion plus
  a small stability margin rather than a large N. Re-measure once scenarios exist.
- **Net:** Plan 2 can drop the N-run voting machinery and is simpler than the
  spec's worst-case hedge assumed.
