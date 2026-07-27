"""Regression tests for the OSpan blind-adjudication protocol.

Covers the masking/redaction rules, the frozen machine/blind classifier,
the adjudication schema validation, the adjudicated scoring path, and the
COGARENA_REQUIRE_ADJ formal gate.
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'reanalysis'))

from aplus_rescore_20260718 import classify_response, adjudicated_credit  # noqa: E402
from build_ospan_blind_package import (  # noqa: E402
    mask_gold_echoes, semantic_redactions, apply_redactions, redact_line, MASK_CHAR)
import compute_agreement as ca  # noqa: E402


# ---- narrow verbatim-echo mask -------------------------------------------- #

def test_narrow_mask_verbatim_echo():
    assert mask_gold_echoes("Remember the letter: X") == f"Remember the letter: {MASK_CHAR}"
    assert mask_gold_echoes("Remembering the letter: Y") == f"Remembering the letter: {MASK_CHAR}"


def test_narrow_mask_leaves_claims():
    assert mask_gold_echoes("M, J, F") == "M, J, F"


# ---- semantic redaction --------------------------------------------------- #

def _masked(text):
    return apply_redactions(text, semantic_redactions(text))


def test_transcript_walkthrough_masked():
    out = _masked("Trial 2: Is (6 x 1) - 1 = 7? YES, B")
    assert 'B' not in out and 'YES' in out and MASK_CHAR in out


def test_usage_example_masked():
    out = _masked('Usage: Item 1: Remember the letter H. The user should remember it.')
    assert ' H' not in out and MASK_CHAR in out


def test_letter_shown_masked():
    out = _masked("2. The letter shown in Trial 3 is H.")
    assert ' H' not in out


def test_first_person_claim_kept():
    t = "I have now recalled all the letters in the order they were presented: T, F, K."
    assert _masked(t) == t


def test_we_narrative_claim_kept():
    t = "In summary, we verified the arithmetic and remembered the corresponding letter. Final: X Y Z"
    assert _masked(t) == t


def test_bare_sequence_kept():
    t = "Z R A M A"
    assert _masked(t) == t


def test_claim_head_kept():
    t = "The sequence of remembered letters is: W, S, F, X."
    assert _masked(t) == t


def test_redact_line_preserves_yes_no():
    assert redact_line("YES, B then NO, K") == f"YES, {MASK_CHAR} then NO, {MASK_CHAR}"


def test_redact_line_masks_compact_runs():
    assert MASK_CHAR * 4 in redact_line('respond with "HXPM"')


# ---- production counterexamples from the sixth re-verification round ------ #
# (structurally identical to the named b04xx/b00xx cases, letters substituted)

def test_plural_per_trial_echo_masked():
    # b0417/b0288 family: plural letter-presentation echo on a math line
    out = _masked("Trials 1 and 2: Is (4 x 5) + 10 = 35? YES or NO. Remember the letters: Q, W")
    assert 'Q' not in out and 'W' not in out and 'YES' in out


def test_bare_plural_echo_masked():
    # b0070 family: plural echo without a math prefix
    out = _masked("5. Remember the letters: Q, W")
    assert 'Q' not in out and MASK_CHAR in out


def test_imperative_full_sequence_masked():
    # b0283 family: instruction-voice restatement, long run does NOT exempt
    out = _masked("Remember the letters Q, W, E, R, and T in that order.")
    assert 'Q' not in out and 'T' not in out and MASK_CHAR in out


def test_enumerated_math_echo_masked():
    # b0384 family: enumerated math echo without the words Item/Trial
    out = _masked('3. Is (180 / 3) + 4 = 172? YES or NO: Remember the letter "Q"')
    assert 'Q' not in out and 'YES' in out


def test_recall_attempt_with_run_kept():
    # b0054 family: sole claimed sequence in recall-phase phrasing stays
    t = ("1. For Trial 4, you are asked to recall all the letters in the order "
         "they were presented: Q, Q, W, and E. So the answer is QQWE.")
    assert _masked(t) == t


def test_claim_with_provenance_kept():
    # b0147 family: claimed sequence with per-item provenance stays
    t = ("The letters are recalled in the order they were presented: "
         "Q (from Item 1), W (from Item 2), E (from Item 3).")
    assert _masked(t) == t


def test_next_item_echo_letter_masked():
    # b0283 head family: alternating math/letter echo list
    out = _masked("4. Remember the letter Q")
    assert 'Q' not in out and MASK_CHAR in out


# ---- frozen classifier ---------------------------------------------------- #

def test_classifier_pure_sequence_is_machine():
    cls, toks = classify_response("M, J, F")
    assert cls == 'machine_sequence' and toks == ['M', 'J', 'F']


def test_classifier_empty_is_machine():
    assert classify_response("")[0] == 'machine_empty'


def test_classifier_prose_is_blind():
    assert classify_response("The letters are N, K, G.")[0] == 'blind'


# ---- adjudicated scoring path --------------------------------------------- #

def test_adjudicated_sequence_scores_positionally():
    assert adjudicated_credit('sequence', ['M', 'J', 'F'], ['M', 'J', 'F']) == 1.0
    assert adjudicated_credit('sequence', ['M', 'X', 'F'], ['M', 'J', 'F']) == pytest.approx(2 / 3)


def test_adjudicated_non_sequence_scores_zero():
    for lab in ('ambiguous', 'refusal', 'echo', 'non-answer'):
        assert adjudicated_credit(lab, [], ['M', 'J', 'F']) == 0.0


# ---- adjudication schema (single source: compute_agreement.load_annotation) - #

def _write(tmp_path, obj):
    p = tmp_path / "ann.json"
    p.write_text(json.dumps(obj))
    return str(p)


def test_schema_rejects_lowercase_tokens(tmp_path):
    p = _write(tmp_path, {"b0000": {"label": "sequence", "tokens": ["m"]}})
    with pytest.raises(RuntimeError):
        ca.load_annotation(p, {"b0000"})


def test_schema_rejects_unicode_tokens(tmp_path):
    p = _write(tmp_path, {"b0000": {"label": "sequence", "tokens": ["Ａ"]}})
    with pytest.raises(RuntimeError):
        ca.load_annotation(p, {"b0000"})


def test_schema_rejects_tokens_on_non_sequence(tmp_path):
    p = _write(tmp_path, {"b0000": {"label": "refusal", "tokens": ["M"]}})
    with pytest.raises(RuntimeError):
        ca.load_annotation(p, {"b0000"})


def test_schema_rejects_incomplete_coverage(tmp_path):
    p = _write(tmp_path, {"b0000": {"label": "refusal"}})
    with pytest.raises(RuntimeError):
        ca.load_annotation(p, {"b0000", "b0001"})


def test_schema_accepts_valid(tmp_path):
    p = _write(tmp_path, {"b0000": {"label": "sequence", "tokens": ["M", "J"]},
                          "b0001": {"label": "echo"}})
    assert set(ca.load_annotation(p, {"b0000", "b0001"})) == {"b0000", "b0001"}


# ---- formal require gate -------------------------------------------------- #

@pytest.mark.slow
def test_require_adj_fails_without_adjudications():
    env = dict(os.environ, COGARENA_REQUIRE_ADJ='1', COGARENA_ROOT=ROOT)
    adj = os.path.join(ROOT, 'results', 'reanalysis', 'aplus_20260718', 'adjudications.json')
    assert not os.path.exists(adj), "test assumes no adjudications file yet"
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'scripts', 'reanalysis', 'aplus_rescore_20260718.py')],
        env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0
    assert 'COGARENA_REQUIRE_ADJ' in (r.stderr + r.stdout)
