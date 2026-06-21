"""Regression tests for run_unified.py's inline aggregation.

Guards against the bug where paradigms whose per-item ``score`` is a dict
(e.g. digit_span's ``{"accuracy": 1.0, ...}``) were counted as 0 by the inline
aggregator even though ``details.json`` showed correct per-item scores. Scalar/
bool-score paradigms (stroop, flanker, ...) were unaffected.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Importing run_unified pulls in the cogarena package; skip cleanly if the
# runtime deps are unavailable rather than erroring the whole suite.
ru = pytest.importorskip("run_unified")


def test_item_accuracy_dict_scores():
    # dict score with `accuracy` key (digit_span, false_belief, epitome_tom,
    # confidence_calibration, post_decision_wagering)
    assert ru._item_accuracy({"score": {"accuracy": 1.0, "order": 1.0}}) == 1.0
    assert ru._item_accuracy({"score": {"accuracy": 0.0, "order": 1.0}}) == 0.0
    assert ru._item_accuracy({"score": {"accuracy": 0.75}}) == 0.75
    # dict score falling back to `score`, then to `correct`
    assert ru._item_accuracy({"score": {"score": 0.6}}) == 0.6
    assert ru._item_accuracy({"score": {"correct": True}}) == 1.0
    assert ru._item_accuracy({"score": {"correct": False}}) == 0.0


def test_item_accuracy_scalar_and_toplevel():
    # scalar / bool score
    assert ru._item_accuracy({"score": 1.0}) == 1.0
    assert ru._item_accuracy({"score": 0}) == 0.0
    assert ru._item_accuracy({"score": True}) == 1.0
    assert ru._item_accuracy({"score": False}) == 0.0
    # image mode: top-level `correct` bool, no `score`
    assert ru._item_accuracy({"correct": True}) == 1.0
    assert ru._item_accuracy({"correct": False}) == 0.0
    # agent mode: top-level `accuracy`
    assert ru._item_accuracy({"accuracy": 1.0}) == 1.0
    # nothing scorable -> 0.0
    assert ru._item_accuracy({}) == 0.0


def test_aggregate_mixes_dict_and_scalar_scores():
    # Synthetic details list mixing dict-score and scalar/bool-score paradigms.
    details = [
        # digit_span: dict scores -> must count, NOT be silently 0
        {"paradigm": "digit_span", "score": {"accuracy": 1.0, "span_correct": 1.0}},
        {"paradigm": "digit_span", "score": {"accuracy": 1.0, "span_correct": 1.0}},
        {"paradigm": "digit_span", "score": {"accuracy": 0.0, "span_correct": 0.0}},
        # stroop: bool-style dict score (unaffected by the bug, must stay correct)
        {"paradigm": "stroop", "score": {"correct": True}},
        {"paradigm": "stroop", "score": {"correct": False}},
        # source_monitoring: partial-credit -> mean(accuracy)
        {"paradigm": "source_monitoring", "score": {"accuracy": 0.5}},
        {"paradigm": "source_monitoring", "score": {"accuracy": 1.0}},
    ]
    agg = ru.aggregate_by_paradigm(details)

    # Regression: dict-score paradigm is counted (was 0/3 before the fix).
    assert agg["digit_span"]["count"] == 3
    assert agg["digit_span"]["correct"] == 2
    assert agg["digit_span"]["accuracy"] == pytest.approx(2 / 3)

    # Scalar/bool paradigm still aggregates correctly.
    assert agg["stroop"]["correct"] == 1
    assert agg["stroop"]["accuracy"] == 0.5

    # Partial-credit paradigm uses mean(accuracy).
    assert agg["source_monitoring"]["accuracy"] == pytest.approx(0.75)
