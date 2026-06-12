# M-voice/A: Noisy-Transcript Corpus + RLVR Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harvest ASR-realistic noisy variants of the 12 component eval cases (TTS → mu-law degrade → Deepgram), freeze them as an audited corpus, register `ivr-noisy`/`rep-noisy` eval layers, and produce the K=3 clean-vs-noisy measurement that decides the pre-committed RLVR gate.

**Architecture:** A new `agent/eval/audio/` package: pure offline-testable modules (`_degrade.py` mu-law codec, `_wer.py` word error rate, `_wav.py` WAV writing, `_noisy.py` case transform) plus thin live wrappers (`synthesize.py` ElevenLabs, `transcribe.py` Deepgram REST, `harvest.py` orchestrator — all coverage-omitted per the existing live-file convention). A `DEEPGRAM_MODEL` constant in new `agent/config.py` is consumed by both `main.py` and the toolkit so they can't drift. `source_id` is added to `EvalCase`; the loader rejects `UNAUDITED:` rationales so an unaudited corpus is mechanically unusable.

**Tech Stack:** Python 3.12, pydantic v2, httpx 0.28 (Deepgram REST), `elevenlabs` SDK (already a dep), stdlib `wave`/`struct` (NO `audioop` — deprecated in 3.12, removed in 3.13; the mu-law codec is ~30 lines of pure Python we own and test).

**Spec:** `docs/superpowers/specs/2026-06-10-voice-eval-track-design.md`. One deliberate simplification vs the spec's wording: `synthesize()` requests `pcm_8000` output directly from ElevenLabs, so `degrade()` is a pure mu-law round-trip with **no resampler** — the 8kHz bandwidth limit is applied by ElevenLabs' own downsampler, which matches the production path (ElevenLabs → Twilio is natively 8kHz). If the account tier rejects `pcm_8000`, Task 7's spike step says what to do.

**Conventions that bind every task:** every new `.py` file starts with `# pyright: strict` (except where a task says otherwise). Run `uv run ruff check . && uv run ruff format --check . && uv run pyright` before each commit (pre-commit runs them anyway; don't fight it). Tests are offline with zero network. Worktree: branch `m-voice-a-noisy-corpus` via `superpowers:using-git-worktrees`.

---

### Task 1: `DEEPGRAM_MODEL` shared constant

**Files:**
- Create: `agent/config.py`
- Modify: `agent/main.py:243`

- [ ] **Step 1: Create the config module**

```python
# pyright: strict
"""Shared configuration constants.

`DEEPGRAM_MODEL` exists so production (`main.py`) and the eval audio toolkit
(`agent/eval/audio/transcribe.py`) transcribe with the SAME model and cannot
drift apart silently. Production previously used pipecat's implicit default
("nova-3-general"); this pins the same value explicitly.
"""

from __future__ import annotations

DEEPGRAM_MODEL = "nova-3-general"
```

- [ ] **Step 2: Verify pipecat's default model and live-options merge behavior**

Read `.venv/lib/python3.12/site-packages/pipecat/services/deepgram/stt.py` (the installed source). Confirm: (a) the default `LiveOptions` model is `nova-3-general` — if it differs, change `agent/config.py` to match the actual default, since the constant must describe production, not assume it; (b) whether a user-supplied `live_options` is merged over the defaults or replaces them. If merged, Step 3 passes a minimal `LiveOptions(model=DEEPGRAM_MODEL)`. If it REPLACES the defaults, Step 3 must instead copy the full default `LiveOptions` construction from the installed source and override only `model`.

- [ ] **Step 3: Wire main.py to the constant**

In `agent/main.py`, add imports `from deepgram import LiveOptions` and `from agent.config import DEEPGRAM_MODEL`, and change line 243:

```python
stt = DeepgramSTTService(
    api_key=os.environ["DEEPGRAM_API_KEY"],
    live_options=LiveOptions(model=DEEPGRAM_MODEL),
)
```

(adjusted per Step 2's merge finding).

- [ ] **Step 4: Lint + typecheck**

Run: `uv run ruff check . && uv run pyright`
Expected: clean. (`main.py` is coverage-omitted; no test for this wiring.)

- [ ] **Step 5: Commit**

```bash
git add agent/config.py agent/main.py
git commit -m "feat(eval-audio): pin DEEPGRAM_MODEL in shared config, consumed by main.py"
```

---

### Task 2: mu-law codec + `degrade()` (pure, TDD)

**Files:**
- Create: `agent/eval/audio/__init__.py` (empty, with `# pyright: strict` header)
- Create: `agent/eval/audio/_degrade.py`
- Test: `tests/eval/test_degrade.py`

- [ ] **Step 1: Write the failing tests**

```python
# pyright: strict
"""G.711 mu-law codec properties. The codec is the entire telephony
degradation (input is already 8kHz from ElevenLabs), so these tests pin its
correctness against known G.711 anchor values and round-trip error bounds."""

from __future__ import annotations

import struct

import pytest

from agent.eval.audio._degrade import degrade, mulaw_decode, mulaw_encode


def test_encode_known_anchors() -> None:
    # G.711: linear 0 encodes to 0xFF; the most negative values hit 0x00.
    assert mulaw_encode(0) == 0xFF
    assert mulaw_encode(-32768) == 0x00
    assert mulaw_encode(32767) == 0x80


def test_decode_zero_byte_is_loud_negative() -> None:
    # 0x00 decodes to the loudest negative magnitude (-32124 in G.711).
    assert mulaw_decode(0x00) == -32124


def test_round_trip_error_bounded() -> None:
    # mu-law is logarithmic: quantization error grows with magnitude but is
    # bounded by half the step size of the segment (~ sample/16 + 16).
    for sample in [0, 1, -1, 100, -100, 1000, -1000, 8000, -8000, 30000, -30000]:
        decoded = mulaw_decode(mulaw_encode(sample))
        assert abs(decoded - sample) <= abs(sample) // 16 + 16


def test_decode_output_is_int16() -> None:
    for byte in range(256):
        assert -32768 <= mulaw_decode(byte) <= 32767


def test_degrade_preserves_length_and_silence() -> None:
    silence = struct.pack("<4h", 0, 0, 0, 0)
    out = degrade(silence)
    assert len(out) == len(silence)
    assert struct.unpack("<4h", out) == (0, 0, 0, 0)


def test_degrade_rejects_odd_length() -> None:
    with pytest.raises(ValueError, match="even"):
        degrade(b"\x00\x01\x02")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_degrade.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.eval.audio'`

- [ ] **Step 3: Implement the codec**

`agent/eval/audio/_degrade.py`:

```python
# pyright: strict
"""Telephony degradation: G.711 mu-law round-trip over 8kHz s16le PCM.

Pure Python on purpose — stdlib `audioop` is deprecated in 3.12 and removed in
3.13, and 30 lines we own and test beats a dependency on a dying module. Input
is already 8kHz (ElevenLabs synthesizes pcm_8000 directly), so the codec loss
IS the whole degradation; no resampler. Background-noise overlay is deferred
behind the spec's sub-5%-WER re-harvest trigger.
"""

from __future__ import annotations

import struct

_BIAS = 0x84  # 132, G.711 standard bias
_CLIP = 32635


def mulaw_encode(sample: int) -> int:
    """Linear s16 sample -> one mu-law byte (G.711)."""
    sign = 0x80 if sample < 0 else 0x00
    magnitude = min(-sample if sample < 0 else sample, _CLIP) + _BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not magnitude & mask:
        mask >>= 1
        exponent -= 1
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def mulaw_decode(byte: int) -> int:
    """One mu-law byte -> linear s16 sample (G.711)."""
    byte = ~byte & 0xFF
    sign = byte & 0x80
    exponent = (byte >> 4) & 0x07
    mantissa = byte & 0x0F
    magnitude = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return -magnitude if sign else magnitude


def degrade(pcm: bytes) -> bytes:
    """Round-trip s16le PCM through the mu-law codec (phone-line loss)."""
    if len(pcm) % 2:
        raise ValueError("PCM byte length must be even (s16le)")
    count = len(pcm) // 2
    samples = struct.unpack(f"<{count}h", pcm)
    return struct.pack(f"<{count}h", *(mulaw_decode(mulaw_encode(s)) for s in samples))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_degrade.py -v --no-cov`
Expected: 6 passed. If `test_encode_known_anchors` or `test_decode_zero_byte_is_loud_negative` fails on the exact anchor values, do NOT bend the anchors to the implementation — cross-check against G.711 (e.g. `mulaw_decode(0x00)` is -32124 in every reference table) and fix the code.

- [ ] **Step 5: Commit**

```bash
git add agent/eval/audio/__init__.py agent/eval/audio/_degrade.py tests/eval/test_degrade.py
git commit -m "feat(eval-audio): pure-python G.711 mu-law codec as telephony degrade()"
```

---

### Task 3: word error rate (pure, TDD)

**Files:**
- Create: `agent/eval/audio/_wer.py`
- Test: `tests/eval/test_wer.py`

- [ ] **Step 1: Write the failing tests**

```python
# pyright: strict
"""WER feeds the spec's pre-committed re-harvest trigger (median WER < 5% =>
codec-only too gentle), so it must be normalization-aware: Deepgram's
smart_format adds casing/punctuation that is NOT transcription noise."""

from __future__ import annotations

from agent.eval.audio._wer import normalize_words, wer


def test_identical_after_normalization_is_zero() -> None:
    assert wer("Press two for claims.", "press two, for claims") == 0.0


def test_total_mismatch_is_one() -> None:
    assert wer("alpha beta", "gamma delta") == 1.0


def test_substitution() -> None:
    # 1 substitution over 4 reference words.
    assert wer("press one for claims", "press won for claims") == 0.25


def test_deletion_and_insertion() -> None:
    assert wer("a b c d", "a b d") == 0.25  # one deletion
    assert wer("a b d", "a b c d") == 1 / 3  # one insertion, 3 ref words


def test_empty_reference() -> None:
    assert wer("", "") == 0.0
    assert wer("", "anything") == 1.0


def test_normalize_words() -> None:
    assert normalize_words("Press 2, for Claims!") == ["press", "2", "for", "claims"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_wer.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`agent/eval/audio/_wer.py`:

```python
# pyright: strict
"""Word error rate over normalized words (Levenshtein / reference length)."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9']+")


def normalize_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1] / len(ref)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_wer.py -v --no-cov`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add agent/eval/audio/_wer.py tests/eval/test_wer.py
git commit -m "feat(eval-audio): normalized word-error-rate for the re-harvest trigger"
```

---

### Task 4: `source_id` schema change + `UNAUDITED:` loader rejection (TDD)

**Files:**
- Modify: `agent/eval/_types.py:48` (EvalCase)
- Modify: `agent/eval/_loader.py:23-36` (load_cases)
- Test: `tests/eval/test_loader.py` (extend the existing file)

- [ ] **Step 1: Write the failing tests** (append to `tests/eval/test_loader.py`, matching its existing style — read it first and reuse its fixtures/helpers if it has any)

```python
def test_source_id_loads_and_defaults_none(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "c1-noisy", "source_id": "c1", "payer": "p", "history": [], '
        '"expected_tool": "wait", "rationale": "ok"}\n'
        '{"id": "c2", "payer": "p", "history": [], '
        '"expected_tool": "wait", "rationale": "ok"}\n',
        encoding="utf-8",
    )
    cases = load_cases(path, IVREvalCase)
    assert cases[0].source_id == "c1"
    assert cases[1].source_id is None


def test_unaudited_rationale_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "c1-noisy", "source_id": "c1", "payer": "p", "history": [], '
        '"expected_tool": "wait", "rationale": "UNAUDITED: ok"}\n',
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="UNAUDITED"):
        load_cases(path, IVREvalCase)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_loader.py -v --no-cov`
Expected: the two new tests FAIL (`source_id` rejected by `extra="forbid"`; no UNAUDITED check); existing tests still PASS.

- [ ] **Step 3: Implement both changes**

In `agent/eval/_types.py`, add to `EvalCase` below `id: str`:

```python
    # Links a harvested noisy variant back to its clean source case (None for
    # original cases). On the BASE model deliberately: extra="forbid" means a
    # subclass-only field would make the other corpus reject it.
    source_id: str | None = None
```

In `agent/eval/_loader.py`, inside the `for` loop after the case is parsed (before `cases.append`):

```python
            rationale = getattr(case, "rationale", "")
            if isinstance(rationale, str) and rationale.startswith("UNAUDITED:"):
                raise CorpusError(
                    f"{path} line {line_no}: case '{case.id}' is UNAUDITED — "
                    "finish the human audit (remove the prefix) before this corpus is usable"
                )
```

(`getattr` because `rationale` lives on the subclasses, not `EvalCase`; `Scenario` has none and passes through.) Restructure the try/except so the parsed `case` is available — parse into a local first:

```python
        try:
            case = model.model_validate_json(line)
        except ValidationError as exc:
            raise CorpusError(f"{path} line {line_no}: invalid {model.__name__}: {exc}") from exc
        rationale = getattr(case, "rationale", "")
        if isinstance(rationale, str) and rationale.startswith("UNAUDITED:"):
            raise CorpusError(
                f"{path} line {line_no}: case '{case.id}' is UNAUDITED — "
                "finish the human audit (remove the prefix) before this corpus is usable"
            )
        cases.append(case)
```

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest --no-cov -q`
Expected: all pass (the schema change is additive-with-default, so existing corpora load unchanged).

- [ ] **Step 5: Commit**

```bash
git add agent/eval/_types.py agent/eval/_loader.py tests/eval/test_loader.py
git commit -m "feat(eval): source_id on EvalCase + loader rejects UNAUDITED corpora"
```

---

### Task 5: WAV writer + noisy-case transform (pure, TDD)

**Files:**
- Create: `agent/eval/audio/_wav.py`
- Create: `agent/eval/audio/_noisy.py`
- Test: `tests/eval/test_noisy_transform.py`

- [ ] **Step 1: Write the failing tests**

```python
# pyright: strict
"""The harvest's pure core: WAV persistence and the clean->noisy case
transform. Live API calls are NOT here — harvest.py wires those."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from agent.eval._types import IVREvalCase, RepEvalCase
from agent.eval.audio._noisy import make_noisy_case
from agent.eval.audio._wav import write_wav
from agent.schemas import Benefits, Turn


def _ivr_case() -> IVREvalCase:
    return IVREvalCase(
        id="c1",
        payer="generic",
        history=[
            Turn(role="user", content="For claims press 1, for eligibility press 2."),
        ],
        expected_tool="send_dtmf",
        expected_args={"digits": "2"},
        rationale="Eligibility is option 2.",
    )


def test_write_wav_8k_mono_s16(tmp_path: Path) -> None:
    pcm = struct.pack("<4h", 0, 1000, -1000, 0)
    out = tmp_path / "x.wav"
    write_wav(out, pcm)
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 8000
        assert w.readframes(4) == pcm


def test_make_noisy_case_ivr() -> None:
    noisy = make_noisy_case(_ivr_case(), ["four claims press one for eligibility press to"])
    assert noisy.id == "c1-noisy"
    assert noisy.source_id == "c1"
    assert noisy.rationale.startswith("UNAUDITED: ")
    assert noisy.history[0].content == "four claims press one for eligibility press to"
    assert noisy.expected_tool == "send_dtmf"  # labels untouched: the audit owns relabeling
    original = _ivr_case()
    assert original.history[0].content.startswith("For claims")  # input not mutated


def test_make_noisy_case_rep_replaces_only_user_turns() -> None:
    case = RepEvalCase(
        id="r1",
        history=[
            Turn(role="assistant", content="What is the copay?"),
            Turn(role="user", content="The copay is forty dollars."),
        ],
        expected_extracted=Benefits(copay=40.0),
        expected_phase="extracting",
        rationale="Copay stated.",
    )
    noisy = make_noisy_case(case, ["the co pay is fourteen dollars"])
    assert noisy.history[0].content == "What is the copay?"  # assistant untouched
    assert noisy.history[1].content == "the co pay is fourteen dollars"


def test_make_noisy_case_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="user turns"):
        make_noisy_case(_ivr_case(), ["one", "two"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_noisy_transform.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`agent/eval/audio/_wav.py`:

```python
# pyright: strict
"""Persist 8kHz mono s16le PCM as WAV — the audit listens to these when text
alone can't say whether TTS or STT mangled a digit."""

from __future__ import annotations

import wave
from pathlib import Path

SAMPLE_RATE = 8000


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
```

`agent/eval/audio/_noisy.py`:

```python
# pyright: strict
"""Clean case -> noisy variant: replace user-turn text with harvested
transcripts, suffix the id, link source_id, and mark UNAUDITED (the loader
refuses the corpus until the human audit removes the prefix)."""

from __future__ import annotations

from collections.abc import Sequence

from agent.eval._types import IVREvalCase, RepEvalCase
from agent.schemas import Turn


def make_noisy_case[CaseT: (IVREvalCase, RepEvalCase)](
    case: CaseT, noisy_user_texts: Sequence[str]
) -> CaseT:
    user_indices = [i for i, turn in enumerate(case.history) if turn.role == "user"]
    if len(user_indices) != len(noisy_user_texts):
        raise ValueError(
            f"case {case.id}: {len(user_indices)} user turns but "
            f"{len(noisy_user_texts)} noisy texts"
        )
    history: list[Turn] = list(case.history)
    for idx, text in zip(user_indices, noisy_user_texts, strict=True):
        history[idx] = history[idx].model_copy(update={"content": text})
    return case.model_copy(
        update={
            "id": f"{case.id}-noisy",
            "source_id": case.id,
            "history": history,
            "rationale": f"UNAUDITED: {case.rationale}",
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_noisy_transform.py -v --no-cov`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add agent/eval/audio/_wav.py agent/eval/audio/_noisy.py tests/eval/test_noisy_transform.py
git commit -m "feat(eval-audio): WAV writer + clean-to-noisy case transform"
```

---

### Task 6: live wrappers (`synthesize`, `transcribe`) + housekeeping

**Files:**
- Create: `agent/eval/audio/synthesize.py`
- Create: `agent/eval/audio/transcribe.py`
- Modify: `pyproject.toml` (httpx dep + coverage omit)
- Modify: `.gitignore`

No TDD here — these are thin live wrappers under the existing coverage-omit convention; they are verified live in Task 8's spike.

- [ ] **Step 1: Write `synthesize.py`**

```python
# pyright: strict
"""ElevenLabs TTS -> 8kHz s16le PCM. Live; coverage-omitted; verified by the
harvest spike. pcm_8000 is requested directly so the toolkit needs no
resampler — same native-8kHz path production uses toward Twilio."""

from __future__ import annotations

import os

from elevenlabs.client import ElevenLabs

_DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # matches agent/main.py default


def synthesize(text: str) -> bytes:
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", _DEFAULT_VOICE_ID)
    chunks = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        output_format="pcm_8000",
    )
    return b"".join(chunks)
```

Check the installed SDK first (`uv run python -c "from elevenlabs.client import ElevenLabs; help(ElevenLabs.__init__)"` and the `text_to_speech.convert` signature). If `convert`'s parameter names differ (e.g. `model_id` required), adapt — the contract is text in, raw `pcm_8000` bytes out.

- [ ] **Step 2: Write `transcribe.py`**

```python
# pyright: strict
"""Deepgram prerecorded transcription over raw 8kHz s16le PCM. Live;
coverage-omitted; verified by the harvest spike.

Deliberate divergence from production, documented per the spec: production
uses Deepgram's STREAMING API via pipecat (endpointing, interim results);
this uses the prerecorded REST API. Same model (DEEPGRAM_MODEL) — the
divergence is the API mode, not the acoustic model.
"""

from __future__ import annotations

import os

import httpx

from agent.config import DEEPGRAM_MODEL

_URL = "https://api.deepgram.com/v1/listen"


def transcribe(pcm_8k: bytes) -> str:
    response = httpx.post(
        _URL,
        params={
            "model": DEEPGRAM_MODEL,
            "smart_format": "true",
            "encoding": "linear16",
            "sample_rate": "8000",
        },
        headers={
            "Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
            "Content-Type": "application/octet-stream",
        },
        content=pcm_8k,
        timeout=60.0,
    )
    response.raise_for_status()
    body = response.json()
    transcript = body["results"]["channels"][0]["alternatives"][0]["transcript"]
    assert isinstance(transcript, str)
    return transcript
```

- [ ] **Step 3: Declare httpx as a direct dependency**

It's already in `uv.lock` transitively at 0.28.1; using it directly without declaring it is how transitive bumps break you. In `pyproject.toml` `dependencies`, add (alphabetical position):

```toml
    "httpx==0.28.1",
```

Run: `uv sync` — expected: no resolution change (same pinned version).

- [ ] **Step 4: Coverage omit + gitignore**

In `pyproject.toml` `[tool.coverage.run] omit`, add:

```toml
    # Live audio toolkit: TTS/STT API wrappers + harvest orchestrator, verified
    # by the M-voice/A spike + harvest runs. Pure pieces (_degrade, _wer, _wav,
    # _noisy) are NOT omitted.
    "agent/eval/audio/synthesize.py",
    "agent/eval/audio/transcribe.py",
    "agent/eval/audio/harvest.py",
```

In `.gitignore`, add:

```
# Harvest staging output — only the human-audited cases_noisy.jsonl is committed
agent/eval/*/corpus/cases_noisy.staging.jsonl
agent/eval/*/corpus/wavs/
```

- [ ] **Step 5: Lint, typecheck, full suite**

Run: `uv run ruff check . && uv run pyright && uv run pytest -q`
Expected: clean, coverage floor holds (new omitted files don't count; new pure files are covered).

- [ ] **Step 6: Commit**

```bash
git add agent/eval/audio/synthesize.py agent/eval/audio/transcribe.py pyproject.toml uv.lock .gitignore
git commit -m "feat(eval-audio): live ElevenLabs/Deepgram wrappers + httpx pin + omit/ignore housekeeping"
```

---

### Task 7: harvest orchestrator

**Files:**
- Create: `agent/eval/audio/harvest.py`

Live, coverage-omitted; its pure logic was extracted into Tasks 2/3/5, so what remains is wiring.

- [ ] **Step 1: Write `harvest.py`**

```python
# pyright: strict
"""Harvest noisy variants of the component eval corpora.

For every user turn of every case: ElevenLabs TTS (pcm_8000) -> mu-law
degrade -> Deepgram prerecorded -> noisy text. Writes per-corpus:

- `cases_noisy.staging.jsonl` (gitignored; every rationale UNAUDITED:)
- `wavs/<case-id>-turn<i>.wav` (gitignored; the audit listens when text lies)

Prints per-corpus and overall median WER, and the spec's pre-committed
re-harvest verdict (median WER < 5% => codec-only too gentle => add the noise
overlay before the RLVR gate may be decided).

Usage: `uv run python -m agent.eval.audio.harvest [--force]`
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from agent.eval._loader import load_cases
from agent.eval._types import EvalCase, IVREvalCase, RepEvalCase
from agent.eval.audio._degrade import degrade
from agent.eval.audio._noisy import make_noisy_case
from agent.eval.audio._wav import write_wav
from agent.eval.audio._wer import wer
from agent.eval.audio.synthesize import synthesize
from agent.eval.audio.transcribe import transcribe
from agent.eval.ivr_tool_choice._score import CORPUS as IVR_CORPUS
from agent.eval.rep_extraction._score import CORPUS as REP_CORPUS

WER_TRIGGER = 0.05  # spec: median below this => degradation too gentle


def _harvest_corpus[CaseT: (IVREvalCase, RepEvalCase)](
    corpus: Path, model: type[CaseT], *, force: bool
) -> list[float]:
    staging = corpus.with_name("cases_noisy.staging.jsonl")
    if staging.exists() and not force:
        raise SystemExit(f"{staging} exists; pass --force to re-harvest (audit work would be lost)")
    wav_dir = corpus.parent / "wavs"
    wers: list[float] = []
    lines: list[str] = []
    for case in load_cases(corpus, model):
        noisy_texts: list[str] = []
        user_turn_no = 0
        for turn in case.history:
            if turn.role != "user":
                continue
            pcm = degrade(synthesize(turn.content))
            write_wav(wav_dir / f"{case.id}-turn{user_turn_no}.wav", pcm)
            noisy = transcribe(pcm)
            noisy_texts.append(noisy)
            turn_wer = wer(turn.content, noisy)
            wers.append(turn_wer)
            print(f"  {case.id} turn {user_turn_no}: WER={turn_wer:.1%}")
            print(f"    clean: {turn.content}")
            print(f"    noisy: {noisy}")
            user_turn_no += 1
        lines.append(make_noisy_case(case, noisy_texts).model_dump_json(exclude_none=True))
    staging.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {staging}")
    return wers


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest noisy eval corpora.")
    parser.add_argument("--force", action="store_true", help="overwrite existing staging output")
    args = parser.parse_args()

    all_wers: list[float] = []
    print("=== ivr_tool_choice ===")
    all_wers += _harvest_corpus(IVR_CORPUS, IVREvalCase, force=bool(args.force))
    print("=== rep_extraction ===")
    all_wers += _harvest_corpus(REP_CORPUS, RepEvalCase, force=bool(args.force))

    median = statistics.median(all_wers)
    print(f"\noverall: {len(all_wers)} turns, median WER={median:.1%}")
    if median < WER_TRIGGER:
        print(
            "VERDICT: below the 5% trigger — codec-only degradation is too gentle. "
            "Add the background-noise overlay to degrade() and re-harvest before "
            "the RLVR gate may be declared decided (spec, M-voice/A)."
        )
    else:
        print("VERDICT: at or above the 5% trigger — corpus is noisy enough to audit.")


if __name__ == "__main__":
    main()
```

Note `exclude_none=True` in the dump: keeps `source_id: null` and unset `Benefits` fields out of the JSONL, matching the existing corpora's compact style. Check that the loader round-trips one staged line (it will — all dropped fields have defaults). Note for the spec-reviewer: `EvalCase` (Task 4) means `load_cases` here rejects an UNAUDITED *staging* file too — which is why harvest reads the CLEAN corpus and only ever WRITES the staging file; nothing loads staging through `load_cases`.

- [ ] **Step 2: Lint + typecheck**

Run: `uv run ruff check . && uv run pyright`
Expected: clean. Do NOT run the harvest — it spends API money; Task 8 owns live execution.

- [ ] **Step 3: Commit**

```bash
git add agent/eval/audio/harvest.py
git commit -m "feat(eval-audio): harvest orchestrator with WER trigger + staging/WAV output"
```

---

### Task 8: LIVE — spike, harvest, PAUSE for audit

This task spends API money and ends in a human gate. It is run by the controller (not a subagent), with the user present.

- [ ] **Step 1: One-line spike (validates pcm_8000 + the whole round trip)**

Run:

```bash
uv run python -c "
from agent.eval.audio._degrade import degrade
from agent.eval.audio._wer import wer
from agent.eval.audio.synthesize import synthesize
from agent.eval.audio.transcribe import transcribe
text = 'For claim status press 1, for eligibility and benefits press 2.'
pcm = synthesize(text)
print('synthesized bytes:', len(pcm))
noisy = transcribe(degrade(pcm))
print('transcript:', noisy)
print('WER:', f'{wer(text, noisy):.1%}')
"
```

Expected: a transcript resembling the input, a few thousand+ bytes of PCM. **If ElevenLabs rejects `output_format=pcm_8000`** (tier restriction), switch `synthesize.py` to `output_format="ulaw_8000"` and decode it to PCM with `mulaw_decode` before returning (the audio is then already mu-law-degraded once; `degrade()` becomes a second pass — acceptable, note it in the module docstring), and re-run the spike.

- [ ] **Step 2: Run the harvest (~14 TTS + 14 STT calls, low single-digit dollars)**

Run: `uv run python -m agent.eval.audio.harvest`
Expected: per-turn clean/noisy printouts, two staging files, WAVs, and the median-WER verdict line.

- [ ] **Step 3: If the verdict is "too gentle" — STOP and report**

Do not improvise a noise overlay. Report the measured WER distribution to the user; the noise-overlay design (noise source, SNR level, new `degrade()` tests) is a small follow-up to plan deliberately, per the spec.

- [ ] **Step 4: PAUSE — human audit**

Present every clean/noisy pair (the harvest printout) to the user case by case, with the WAV paths for listening where text is ambiguous. For each case the user decides: **keep** (label still right; remove the `UNAUDITED: ` prefix), **relabel** (corruption changed the right answer; update `expected_*` and note why in the rationale), or **drop** (case unanswerable; delete the line, note it in the PR body). Write the audited result to `corpus/cases_noisy.jsonl` (both corpora), delete the staging files.

- [ ] **Step 5: Verify the audited corpora load**

Run:

```bash
uv run python -c "
from agent.eval._loader import load_cases
from agent.eval._types import IVREvalCase, RepEvalCase
from agent.eval.ivr_tool_choice._score import CORPUS as IVR
from agent.eval.rep_extraction._score import CORPUS as REP
ivr = load_cases(IVR.with_name('cases_noisy.jsonl'), IVREvalCase)
rep = load_cases(REP.with_name('cases_noisy.jsonl'), RepEvalCase)
print(f'ivr-noisy: {len(ivr)} cases; rep-noisy: {len(rep)} cases')
assert all(c.source_id for c in ivr + rep)
"
```

Expected: counts print; no `CorpusError` (any leftover `UNAUDITED:` fails here, by design).

- [ ] **Step 6: Commit**

```bash
git add agent/eval/ivr_tool_choice/corpus/cases_noisy.jsonl agent/eval/rep_extraction/corpus/cases_noisy.jsonl
git commit -m "feat(eval): audited noisy corpora (harvested TTS->mulaw->Deepgram)"
```

---

### Task 9: `ivr-noisy` / `rep-noisy` eval layers

**Files:**
- Modify: `agent/eval/ivr_tool_choice/_score.py:35` (add NOISY_CORPUS)
- Modify: `agent/eval/rep_extraction/_score.py:17` (add NOISY_CORPUS)
- Modify: `agent/eval/ivr_tool_choice/eval.py`
- Modify: `agent/eval/rep_extraction/eval.py`
- Modify: `agent/eval/cli.py`

- [ ] **Step 1: Add NOISY_CORPUS constants**

In each `_score.py`, directly below the existing `CORPUS` line:

```python
NOISY_CORPUS = Path(__file__).parent / "corpus" / "cases_noisy.jsonl"
```

- [ ] **Step 2: Parametrize both eval entrypoints**

`agent/eval/ivr_tool_choice/eval.py` — change the signature and the two lines that use it:

```python
async def run(*, corpus: Path | None = None, layer: str = LAYER) -> ScoreReport:
    client = GroqToolCallingClient()
    tools = groq_tool_schemas()
    system = _ivr_system_prompt(_default_patient())
    cases = load_cases(corpus or CORPUS, IVREvalCase)

    async def scorer(case: IVREvalCase) -> CaseResult:
        response = await client.complete_with_tools(
            system=system, history=case.history, tools=tools, temperature=0.1
        )
        return score_ivr(case, response)

    return await run_eval(cases, scorer, layer=layer)
```

Add `from pathlib import Path` to its imports. Mirror the same change in `agent/eval/rep_extraction/eval.py` (same `corpus`/`layer` kwargs, its own client/scorer body unchanged).

- [ ] **Step 3: Register the layers in the CLI**

In `agent/eval/cli.py`, add imports for the noisy corpora and extend `_LAYERS`, the argparse `choices`, and the `all` expansion:

```python
from agent.eval.ivr_tool_choice._score import NOISY_CORPUS as IVR_NOISY_CORPUS
from agent.eval.rep_extraction._score import NOISY_CORPUS as REP_NOISY_CORPUS

_LAYERS: dict[str, Callable[[], Awaitable[ScoreReport]]] = {
    "ivr": ivr_eval.run,
    "rep": rep_eval.run,
    "e2e": e2e_eval.run,
    "ivr-noisy": lambda: ivr_eval.run(corpus=IVR_NOISY_CORPUS, layer="ivr_tool_choice_noisy"),
    "rep-noisy": lambda: rep_eval.run(corpus=REP_NOISY_CORPUS, layer="rep_extraction_noisy"),
}
```

```python
    parser.add_argument(
        "layer",
        nargs="?",
        default="all",
        choices=["ivr", "rep", "e2e", "ivr-noisy", "rep-noisy", "all"],
    )
    args = parser.parse_args()
    layers = list(_LAYERS) if args.layer == "all" else [args.layer]
```

(`list(_LAYERS)` replaces the hand-maintained `["ivr", "rep", "e2e"]` so `all` can never silently miss a registered layer again.)

- [ ] **Step 4: Lint, typecheck, suite**

Run: `uv run ruff check . && uv run pyright && uv run pytest -q`
Expected: clean (eval.py/cli.py are coverage-omitted; the suite guards against import-time breakage).

- [ ] **Step 5: Commit**

```bash
git add agent/eval/ivr_tool_choice/_score.py agent/eval/rep_extraction/_score.py agent/eval/ivr_tool_choice/eval.py agent/eval/rep_extraction/eval.py agent/eval/cli.py
git commit -m "feat(eval): register ivr-noisy/rep-noisy layers over the audited corpora"
```

---

### Task 10: RLVR demotion headers

**Files:**
- Modify: `docs/superpowers/specs/2026-06-07-rlvr-phone-agent-design.md:4` (Status line)
- Modify: `docs/superpowers/plans/2026-06-08-rlvr-vertical-slice.md` (its Status/header area)

- [ ] **Step 1: Update both status headers**

In the RLVR spec, replace the `Status:` line with:

```markdown
Status: PAUSED — pending noisy-eval headroom evidence (2026-06-12).
The decision gate lives in `docs/superpowers/specs/2026-06-10-voice-eval-track-design.md`
(M-voice/A): pooled 12 cases, K=3 runs per surface, ERRORs re-run to resolution;
mean net case-level delta ≤ 1 → this track is ARCHIVED; ≥ 3 → re-scoped around
noise robustness; exactly 2 → explicit user call recorded here.
```

In the RLVR plan, add the equivalent `> **Status: PAUSED** ...` blockquote directly under its title with the same gate summary and pointer.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-07-rlvr-phone-agent-design.md docs/superpowers/plans/2026-06-08-rlvr-vertical-slice.md
git commit -m "docs: pause RLVR track behind the M-voice/A noisy-eval gate"
```

---

### Task 11: LIVE — K=3 measurement + gate input

Controller-run, with the user. ~12 cases × 4 layers × 3 runs ≈ 144 LLM calls (Groq + Anthropic; same per-run cost as three `make evals`).

- [ ] **Step 1: Three runs of each surface**

Run, three times each (sequentially; each appends to `eval_history.jsonl`):

```bash
uv run python -m agent.eval.cli ivr
uv run python -m agent.eval.cli rep
uv run python -m agent.eval.cli ivr-noisy
uv run python -m agent.eval.cli rep-noisy
```

If any run reports `error > 0`: re-run that surface until a run completes with zero ERRORs (the gate's denominator excludes ERRORs by re-running, per the spec). Keep all completed-clean runs; discard errored runs from the gate arithmetic (they stay in eval_history.jsonl, which is fine — it's a trend log, not the gate input).

- [ ] **Step 2: Compute the gate metric**

From the three clean runs per surface: mean failed-case count, pooled across ivr+rep (clean) and ivr-noisy+rep-noisy (noisy). The gate metric is `mean_noisy_failures - mean_clean_failures`. Bands (pre-committed in the spec): ≤ 1 → RLVR archived; ≥ 3 → RLVR re-scoped around noise robustness; exactly 2 → user judgment call. Tabulate per-run pass rates and the delta; this table goes in the PR body verbatim.

- [ ] **Step 3: PAUSE — present the measurement and the band to the user**

The gate decision is the user's. Record the decision (and reasoning if band 2) by updating the RLVR spec's Status header accordingly in a follow-up commit (`ARCHIVED — gate decided YYYY-MM-DD: <numbers>` or `RE-SCOPED — ...`).

- [ ] **Step 4: Commit the evidence**

```bash
git add eval_history.jsonl docs/superpowers/specs/2026-06-07-rlvr-phone-agent-design.md
git commit -m "eval: K=3 clean-vs-noisy measurement + RLVR gate decision"
```

---

### Task 12: finish the branch

- [ ] Run the full verification gauntlet: `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q` — all clean, coverage ≥ 90%.
- [ ] Use `superpowers:finishing-a-development-branch`: PR from `m-voice-a-noisy-corpus` with the measurement table, audit notes (kept/relabeled/dropped counts), and WER verdict in the body; merge after self-review (`/pr-review-toolkit:review-pr <N>`).

---

## Self-review notes

- **Spec coverage:** toolkit (T2/3/5/6), DEEPGRAM_MODEL (T1), harvest+staging+UNAUDITED+WAVs+WER trigger (T7/8), source_id (T4), CLI layers incl. choices-list pitfall (T9), RLVR demotion (T10), audit (T8), K=3 measurement + gate (T11). Spec's `synthesize`-returns-PCM contract kept; resampler dropped with rationale (header note).
- **Sequencing constraint:** Task 9 cannot run its layers before Task 8 produces `cases_noisy.jsonl`, but its code changes are independent — only `all` would fail at runtime on the missing file, and nothing in CI invokes the live CLI. Safe to build T9 before T8's live run if parallelizing; T11 needs both.
- **Type consistency check:** `make_noisy_case` constrained-TypeVar matches its T5 tests; `run(*, corpus, layer)` kwargs match T9's CLI lambdas; `NOISY_CORPUS` names match between `_score.py` and `cli.py` imports; `wer`/`normalize_words` names consistent across T3/T7.
