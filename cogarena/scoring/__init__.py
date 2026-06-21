"""Canonical per-item scoring for the CogArena battery.

Each paradigm has a validated scorer exposed as a ``<Generator>.score(item, response)``
static method under ``cogarena.dimensions.*``. ``score_static()`` routes an item to its
paradigm scorer (the same mapping used to produce the paper's results) and returns the
score dict; ``item_accuracy()`` extracts a single [0, 1] accuracy from any score dict.

This is the scoring entry point the CLI and third-party users should call, rather than
``TaskInstance.score()`` (whose generic ``custom`` path uses a different argument
convention than these paradigm scorers).
"""
from __future__ import annotations

import importlib
from typing import Any, Dict

# paradigm name -> "module:Class.method" scorer (matches scripts/run_eval.py)
PARADIGM_SCORERS = {
    # cognitive_control
    "stroop": "cogarena.dimensions.cognitive_control:StroopParadigm.score",
    "flanker": "cogarena.dimensions.cognitive_control:FlankerParadigm.score",
    "go_nogo": "cogarena.dimensions.cognitive_control:GoNoGoParadigm.score",
    # working_memory
    "n_back": "cogarena.dimensions.working_memory:NBackGenerator.score",
    "digit_span": "cogarena.dimensions.working_memory:DigitSpanGenerator.score",
    "operation_span": "cogarena.dimensions.working_memory:OperationSpanGenerator.score",
    # set_shifting
    "wcst": "cogarena.dimensions.set_shifting:WCSTGenerator.score",
    "reversal_learning": "cogarena.dimensions.set_shifting:ReversalLearningGenerator.score",
    # episodic_memory
    "cvlt_word_list": "cogarena.dimensions.episodic_memory:CVLTGenerator.score",
    "drm_false_memory": "cogarena.dimensions.episodic_memory:DRMGenerator.score",
    "source_monitoring": "cogarena.dimensions.episodic_memory:SourceMonitoringGenerator.score",
    # theory_of_mind
    "false_belief": "cogarena.dimensions.theory_of_mind:FalseBeliefGenerator.score",
    "epitome_tom": "cogarena.dimensions.theory_of_mind:EpitomeToMGenerator.score",
    # metacognition
    "confidence_calibration": "cogarena.dimensions.metacognition:ConfidenceCalibrationGenerator.score",
    "post_decision_wagering": "cogarena.dimensions.metacognition:PostDecisionWageringGenerator.score",
}


def _resolve_scorer(dotted_path: str):
    """Import ``'module.path:Class.method'`` and return the callable."""
    mod_path, attr_path = dotted_path.split(":", 1)
    obj: Any = importlib.import_module(mod_path)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def _paradigm_of(item) -> str:
    meta = getattr(item, "metadata", None)
    if meta is not None:
        return getattr(meta, "paradigm", "")
    return getattr(item, "paradigm", "")


def score_static(item, response: str) -> Dict[str, Any]:
    """Score one static item with its paradigm-specific scorer.

    Falls back to a safe exact-match if no paradigm scorer is registered or the
    scorer raises. The returned dict always lets ``item_accuracy()`` derive a
    [0, 1] accuracy.
    """
    paradigm = _paradigm_of(item)
    path = PARADIGM_SCORERS.get(paradigm)
    if path:
        try:
            return _resolve_scorer(path)(item, response)
        except Exception:
            pass  # fall through to the generic exact-match below
    expected = getattr(item, "expected_response", None)
    exp = str(expected).strip().lower() if expected is not None else ""
    act = str(response).strip().lower()
    if not exp:
        return {"scored": False}
    correct = exp == act
    return {"accuracy": 1.0 if correct else 0.0, "correct": correct}


def item_accuracy(score) -> float:
    """Extract a single [0, 1] accuracy from a score dict (or numeric score)."""
    if isinstance(score, dict):
        if "accuracy" in score:
            return float(score["accuracy"])
        if "score" in score:
            return float(score["score"])
        if "correct" in score:
            c = score["correct"]
            if isinstance(c, bool):
                return 1.0 if c else 0.0
            try:
                return 1.0 if float(c) > 0 else 0.0
            except (TypeError, ValueError):
                return 0.0
        return 0.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0
