"""Frozen scorer adapter for new causal-selectivity responses.

Operation span and CVLT use the paper's corrected primary estimands. Every
other paradigm routes directly to its current paradigm-native scorer. This
module intentionally raises instead of falling back to substring matching.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

from cogarena.dimensions.episodic_memory import _parse_word_list
from scripts.reanalysis.aplus_rescore_20260718 import (
    canonical_parse,
    positional_credit,
    strict_parse,
)

from .common import (
    TRANSPORT_INCOMPLETE_FINISH_REASON,
    ensure_finite_accuracy,
    is_protocol_invalid_finish_reason,
    jsonable,
    require,
)


SCORER_CONTRACT_VERSION = (
    "causal-selectivity-primary-v3-strict4-cvlt-fixed-protocol-invalid-zero"
)

STATIC_SCORERS = {
    "digit_span": "cogarena.dimensions.working_memory:DigitSpanGenerator.score",
    "stroop": "cogarena.dimensions.cognitive_control:StroopParadigm.score",
    "flanker": "cogarena.dimensions.cognitive_control:FlankerParadigm.score",
    "go_nogo": "cogarena.dimensions.cognitive_control:GoNoGoParadigm.score",
    "drm_false_memory": "cogarena.dimensions.episodic_memory:DRMGenerator.score",
    "source_monitoring": "cogarena.dimensions.episodic_memory:SourceMonitoringGenerator.score",
    "false_belief": "cogarena.dimensions.theory_of_mind:FalseBeliefGenerator.score",
    "epitome_tom": "cogarena.dimensions.theory_of_mind:EpitomeToMGenerator.score",
    "confidence_calibration": "cogarena.dimensions.metacognition:ConfidenceCalibrationGenerator.score",
    "post_decision_wagering": "cogarena.dimensions.metacognition:PostDecisionWageringGenerator.score",
}


def _resolve(path: str):
    module_name, attr_path = path.split(":", 1)
    obj: Any = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def _extract_accuracy(metrics: dict[str, Any]) -> float:
    for key in ("accuracy", "score", "correct"):
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, bool):
            return float(value)
        return ensure_finite_accuracy(value, key)
    raise RuntimeError(f"paradigm scorer returned no accuracy-like field: {sorted(metrics)}")


def _score_ospan(item: Any, responses: list[str]) -> dict[str, Any]:
    turns = item.metadata.parameters["turns"]
    metadata_letters = item.metadata.parameters.get("letters")
    turn_letters = []
    for turn in turns:
        if turn.get("type") != "operation_letter":
            continue
        match = re.search(r"Remember the letter:\s*([A-Za-z])", str(turn.get("stimulus", "")))
        require(match is not None, "operation-span turn has no recoverable recall letter")
        letter_from_stimulus = match.group(1).upper()
        if turn.get("recall_letter") is not None:
            require(str(turn["recall_letter"]).upper() == letter_from_stimulus,
                    "operation-span turn gold disagrees with presented stimulus")
        turn_letters.append(letter_from_stimulus)
    require(turn_letters, "operation-span has no recall-letter turns")
    if metadata_letters is not None:
        metadata_letters = [str(x).upper() for x in metadata_letters]
        require(metadata_letters == turn_letters,
                "operation-span metadata letters disagree with presented turn stimuli")
    expected = turn_letters
    recall_text = responses[-1] if responses else ""
    strict_tokens, status, line_index = strict_parse(recall_text)
    strict = positional_credit(strict_tokens, expected)
    canonical = positional_credit(canonical_parse(recall_text), expected)

    math_correct = 0
    math_total = 0
    for turn, response in zip(turns, responses):
        if turn.get("type") != "operation_letter":
            continue
        math_total += 1
        answer = (response or "").upper()
        expected_math = turn["math_expected"]
        if expected_math == "YES" and "YES" in answer:
            math_correct += 1
        elif expected_math == "NO" and "NO" in answer:
            math_correct += 1
    return {
        "primary_accuracy": ensure_finite_accuracy(strict),
        "scorer": "operation_span_strict_v4_positional",
        "metrics": {
            "accuracy": strict,
            "strict_tokens": strict_tokens,
            "strict_parse_status": status,
            "strict_parse_line": line_index,
            "canonical_accuracy": canonical,
            "math_accuracy": math_correct / max(math_total, 1),
            "math_correct": math_correct,
            "math_total": math_total,
        },
    }


def _score_cvlt(item: Any, responses: list[str]) -> dict[str, Any]:
    turns = item.metadata.parameters["turns"]
    require(len(responses) == len(turns), "CVLT response/turn length mismatch")
    turn_metrics: list[dict[str, Any]] = []
    binary_scores: list[float] = []
    for index, (turn, response) in enumerate(zip(turns, responses)):
        if turn.get("type") == "filler_task":
            continue
        expected_words = turn.get("expected_words")
        if not isinstance(expected_words, list) or not expected_words:
            continue
        target = {str(word).strip().lower() for word in expected_words}
        recalled = set(_parse_word_list(response or ""))
        recall = len(recalled & target) / len(target)
        binary = float(recall >= 0.5)
        binary_scores.append(binary)
        turn_metrics.append(
            {
                "turn_index": index,
                "turn_type": turn.get("type"),
                "unique_hits": len(recalled & target),
                "target_total": len(target),
                "recall": recall,
                "thresholded_accuracy": binary,
            }
        )
    require(binary_scores, "CVLT has no designated recall turns")
    accuracy = sum(binary_scores) / len(binary_scores)
    return {
        "primary_accuracy": ensure_finite_accuracy(accuracy),
        "scorer": "cvlt_unique_hit_thresholded_recall",
        "metrics": {"accuracy": accuracy, "n_scored_turns": len(binary_scores), "turns": turn_metrics},
    }


def _score_nback(item: Any, responses: list[str]) -> dict[str, Any]:
    turns = item.metadata.parameters["turns"]
    require(len(responses) == len(turns), "n-back response/turn length mismatch")
    correct = hits = misses = false_alarms = correct_rejections = parseable = 0
    for turn, response in zip(turns, responses):
        expected = str(turn["expected"]).strip().lower()
        actual = (response or "").strip().lower()
        core = actual.strip().strip('."\'!').strip()
        parsed = None
        if core == "no match" or core.startswith("no match "):
            parsed = "no match"
        elif core == "match" or core.startswith("match "):
            parsed = "match"
        if parsed is not None:
            parseable += 1
        is_correct = parsed == expected
        if expected == "match" and "no" in actual:
            # Mirrors scripts/run_eval.py: an explicit NO must never be
            # credited as MATCH even when verbose text also contains MATCH.
            is_correct = False
        correct += int(is_correct)
        if expected == "match":
            if is_correct:
                hits += 1
            else:
                misses += 1
        else:
            if parsed == "match":
                false_alarms += 1
            elif is_correct:
                correct_rejections += 1
    accuracy = correct / max(len(turns), 1)
    return {
        "primary_accuracy": ensure_finite_accuracy(accuracy),
        "scorer": "n_back_strict_production_turn_v1",
        "metrics": {
            "accuracy": accuracy,
            "parseable_turns": parseable,
            "unparseable_turns": len(turns) - parseable,
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_rejections": correct_rejections,
            "n_targets": hits + misses,
            "n_non_targets": len(turns) - hits - misses,
        },
    }


def score_response(item: Any, response_payload: str | list[dict[str, Any]]) -> dict[str, Any]:
    paradigm = item.metadata.paradigm
    if isinstance(response_payload, list):
        responses = [entry["response"] for entry in response_payload]
        require(all(isinstance(x, str) for x in responses), "non-string multi-turn response")
    else:
        require(isinstance(response_payload, str), "non-string static response")
        responses = []

    if paradigm == "operation_span":
        return _score_ospan(item, responses)
    if paradigm == "cvlt_word_list":
        return _score_cvlt(item, responses)
    if paradigm == "n_back":
        return _score_nback(item, responses)

    require(not isinstance(response_payload, list), f"unexpected multi-turn payload for {paradigm}")
    scorer_path = STATIC_SCORERS.get(paradigm)
    require(scorer_path is not None, f"no frozen scorer for {paradigm}")
    metrics = _resolve(scorer_path)(item, response_payload)
    require(isinstance(metrics, dict), f"{paradigm} scorer did not return a dict")
    accuracy = _extract_accuracy(metrics)
    return {
        "primary_accuracy": accuracy,
        "scorer": scorer_path,
        "metrics": jsonable(metrics),
    }


def score_response_with_completion_contract(
    item: Any,
    response_payload: str | list[dict[str, Any]],
    api_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen response-completion rule after native scoring.

    A completion that reaches the frozen token ceiling, or a logical call whose
    terminal metadata remains incomplete after three attempts sharing one
    canonical request-payload SHA-256,
    is an observable protocol failure rather than missing data. The body remains
    in the private raw record for audit, but never reaches a native parser; any
    task record containing either failure receives primary accuracy zero.
    """
    require(isinstance(api_calls, list) and api_calls, "missing API-call completion log")
    finish_reasons = [str(call.get("finish_reason", "")) for call in api_calls]
    truncated_calls = sum(reason == "length" for reason in finish_reasons)
    transport_invalid_calls = sum(
        reason == TRANSPORT_INCOMPLETE_FINISH_REASON for reason in finish_reasons
    )
    invalid_calls = sum(is_protocol_invalid_finish_reason(reason) for reason in finish_reasons)
    policy = (
        "any finish_reason=length or exhausted terminal-metadata fault "
        "invalidates the complete task record"
    )
    if invalid_calls:
        metrics: dict[str, Any] = {
            "accuracy": 0.0,
            "completion_contract": {
                "policy": policy,
                "invalid_call_count": invalid_calls,
                "truncated_call_count": truncated_calls,
                "transport_protocol_invalid_call_count": transport_invalid_calls,
                "task_invalidated": True,
                "native_scorer_evaluated": False,
            },
        }
        if item.metadata.paradigm == "operation_span":
            # Keep OSpan fields structurally present without falsely labeling a
            # parser failure: the native parser was deliberately never called.
            metrics.update(
                {
                    "strict_tokens": [],
                    "strict_parse_status": "not_evaluated_protocol_invalid",
                    "strict_parse_line": None,
                    "canonical_accuracy": 0.0,
                    "math_accuracy": 0.0,
                    "math_correct": 0,
                    "math_total": 0,
                }
            )
        return {
            "primary_accuracy": 0.0,
            "scorer": "protocol_completion_contract_task_invalid_zero",
            "metrics": metrics,
        }

    native = jsonable(score_response(item, response_payload))
    require(isinstance(native, dict), "native scorer did not return a dict")
    native_accuracy = ensure_finite_accuracy(native.get("primary_accuracy"))
    metrics = native.get("metrics")
    require(isinstance(metrics, dict), "native scorer metrics are not a dict")
    metrics = dict(metrics)
    metrics["completion_contract"] = {
        "policy": policy,
        "invalid_call_count": invalid_calls,
        "truncated_call_count": truncated_calls,
        "transport_protocol_invalid_call_count": transport_invalid_calls,
        "task_invalidated": False,
        "native_scorer_evaluated": True,
        "native_primary_accuracy_before_completion_contract": native_accuracy,
    }
    native["metrics"] = metrics
    native["primary_accuracy"] = native_accuracy
    return native
