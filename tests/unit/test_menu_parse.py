# pyright: strict
"""Unit tests for `tools.parse_menu_options` — the heuristic menu-option
extractor that populates `CallSession.recent_menu_options` so the `send_dtmf`
digit-allowlist validator has an independent source to check against.

The extractor biases toward capture: if the transcript contains any
press/dial/select-style cue, every single key in it is captured (count-phrase
digits like '9 digit' excluded). A cue-less or non-menu transcript yields an
empty list, leaving the validator permissive. Over-capture is the safe failure
direction; under-capture would false-reject a real option."""

from __future__ import annotations

import pytest

from agent.tools import parse_menu_options


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        # --- bare-digit menus, cue-gated --------------------------------------
        ("For billing, press 2. For claims, press 3.", ["2", "3"]),
        ("Press 1 for English.", ["1"]),
        ("To reach a representative, press 0.", ["0"]),
        ("Dial 4 for pharmacy, or enter 5 for nursing.", ["4", "5"]),
        ("Select 7 to repeat these options.", ["7"]),
        # --- number-word menus ------------------------------------------------
        ("For eligibility, press two. For benefits, press three.", ["2", "3"]),
        ("Press one for English, press two for Spanish.", ["1", "2"]),
        ("To hear this again, press nine.", ["9"]),
        ("Press zero to speak with an agent.", ["0"]),
        # --- star / pound -----------------------------------------------------
        ("Press star to return to the main menu.", ["*"]),
        ("Press pound when you are finished.", ["#"]),
        ("Press the asterisk key to repeat.", ["*"]),
        ("Press the hash key to continue.", ["#"]),
        # --- multiple options after one cue -----------------------------------
        ("Press one or two to continue.", ["1", "2"]),
        ("Press 9 at any time to repeat this menu.", ["9"]),
        # --- 'press X for A or Y for B' (the false-reject regression) ----------
        # The second option's digit is far from the cue and after a purpose
        # clause; the cue-SENTENCE design captures it where a cue-window did not.
        ("Press 1 for sales or 2 for billing.", ["1", "2"]),
        ("For sales press 1, for billing press 2, for support press 3.", ["1", "2", "3"]),
        ("Press 1, 2, or 3 to choose a department.", ["1", "2", "3"]),
        # --- count-phrase exclusion (digit is a length, not an option) --------
        ("Please enter your 9 digit member ID.", []),
        ("Enter your 10 digit phone number.", []),
        # exclusion fires per-key: a real option survives alongside an excluded
        # count-digit (isolates the count-noun branch from the cue branch)
        ("Press 1 for billing, or enter your 9 digit ID.", ["1"]),
        # 'number'/'numbers' are count nouns too, not just 'digit'
        ("Press 1 for sales, or press 2 numbers for the directory.", ["1"]),
        ("Press 3 numbers to confirm.", []),
        # STT fusion ('press2' as one token) → no cue detected → []. Pins the
        # accepted-loss tradeoff documented in the tools.py module comment.
        ("Press2 for billing.", []),
        # --- de-duplication, order preserved ----------------------------------
        ("Press 1 for sales. Press 1 again to confirm. Press 2 to cancel.", ["1", "2"]),
        # --- NON-menu input → empty (validator stays permissive) --------------
        ("", []),
        ("Please continue to hold while we connect you.", []),
        ("Thank you for calling Aetna. Your call may be recorded.", []),
        # bare digits with NO cue must be ignored (member IDs, DOBs, years)
        ("Your member ID is 1 2 3 4 5 6.", []),
        ("You were born in 1980, correct?", []),
        ("You have three options available today.", []),  # number-word, no cue
        # cue with no following digit captures nothing
        ("Press the button when ready.", []),
    ],
)
def test_parse_menu_options(transcript: str, expected: list[str]) -> None:
    assert parse_menu_options(transcript) == expected


def test_parse_menu_options_is_case_insensitive() -> None:
    assert parse_menu_options("PRESS 2 FOR BILLING") == ["2"]
    assert parse_menu_options("Press TWO for billing") == ["2"]


def test_parse_menu_options_captures_options_phrased_apart_from_cue() -> None:
    """The whole-transcript scan captures every real option even when the cue
    and the option list are in different sentences. A per-sentence scan would
    miss '2'/'3' here (their sentences carry no cue) and false-reject a correct
    press — the exact failure the whole-transcript design exists to prevent."""
    assert parse_menu_options("Press one of the following. Billing, 2. Claims, 3.") == [
        "1",  # 'one' (in 'one of the following') rides along — safe over-capture
        "2",
        "3",
    ]


def test_parse_menu_options_over_captures_non_option_digits() -> None:
    """Documents the deliberate over-capture: a non-option digit anywhere in a
    cue-bearing transcript (a time, here) IS captured. This is the safe failure
    direction — an over-broad allowlist only loses some validator teeth; it
    never false-rejects a real option. A hallucinated digit absent from the
    transcript text is still rejected, so the validator keeps working."""
    assert parse_menu_options("Press 1 for sales. We're open until 9.") == ["1", "9"]
