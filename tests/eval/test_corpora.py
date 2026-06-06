# pyright: strict
"""Hermetic integrity checks for the seed corpora — they parse, ids are unique,
and the deterministic-arg promise holds. Catches a malformed seed case at test
time instead of mid live-run."""

from __future__ import annotations

from agent.eval._loader import load_cases
from agent.eval._types import IVREvalCase, RepEvalCase
from agent.eval.ivr_tool_choice._score import CORPUS as IVR_CORPUS
from agent.eval.rep_extraction._score import CORPUS as REP_CORPUS


def test_ivr_corpus_loads_and_ids_unique() -> None:
    cases = load_cases(IVR_CORPUS, IVREvalCase)
    assert len(cases) >= 6
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_rep_corpus_loads_and_ids_unique() -> None:
    cases = load_cases(REP_CORPUS, RepEvalCase)
    assert len(cases) >= 6
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_ivr_expected_args_only_for_send_dtmf() -> None:
    # The deterministic-arg assertion only makes sense where we list args; today
    # that is send_dtmf. A case with expected_args on another tool is a mistake.
    for case in load_cases(IVR_CORPUS, IVREvalCase):
        if case.expected_args:
            assert case.expected_tool == "send_dtmf", f"{case.id}: args on {case.expected_tool}"
