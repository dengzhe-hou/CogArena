"""
Cognitive Control / Inhibition dimension for CogArena.

Implements three paradigms with procedural generation:
  1. Stroop Task (Text Version) -- multiple conflict types
  2. Flanker Task (Text Version) -- target flanked by distractors
  3. Go/No-Go Task -- response inhibition with prepotent tendency

All paradigms produce TaskInstance dataclasses and companion scoring functions.
"""

from __future__ import annotations

import math
import random
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cogarena.core import (
    AdaptationDistance,
    DifficultyLevel,
    EvalMode,
    ScoringConfig,
    TaskInstance,
    TaskMetadata,
)


def _make_cc_item(
    task_id: str,
    paradigm: str,
    mode: str,
    parameters: Dict[str, Any],
    stimulus: str,
    expected_response: str,
    scoring_config: Dict[str, Any],
    difficulty: str = "medium",
    adaptation_distance: str = "medium",
) -> TaskInstance:
    """Helper to create a TaskInstance with nested TaskMetadata (core schema)."""
    _diff_map = {"easy": DifficultyLevel.EASY, "medium": DifficultyLevel.MEDIUM, "hard": DifficultyLevel.HARD}
    _adapt_map = {"low": AdaptationDistance.LOW, "medium": AdaptationDistance.MEDIUM, "high": AdaptationDistance.HIGH}
    _mode_map = {"static": EvalMode.LLM_STATIC, "multi_turn": EvalMode.AGENT_INTERACTIVE}

    meta = TaskMetadata(
        dimension="cognitive_control",
        paradigm=paradigm,
        mode=_mode_map.get(mode, EvalMode.LLM_STATIC),
        parameters={**parameters, "_scoring_config": scoring_config},
        scoring=ScoringConfig(method="exact_match"),
        difficulty=_diff_map.get(difficulty, DifficultyLevel.MEDIUM),
        adaptation_distance=_adapt_map.get(adaptation_distance, AdaptationDistance.MEDIUM),
    )

    # Lazy-bind the paradigm scorer. It is resolved at first call via scoring_fn.
    _scorer_map = {
        "stroop": lambda r, e, m: StroopParadigm.score.__func__(None, _current_task[0], r),
        "flanker": lambda r, e, m: FlankerParadigm.score.__func__(None, _current_task[0], r),
        "go_nogo": lambda r, e, m: GoNoGoParadigm.score.__func__(None, _current_task[0], r),
    }

    def _make_scorer(para, sc_config):
        """Return a scoring_fn(response, expected, metadata) -> dict."""
        def _score(response, expected, metadata):
            # Build a lightweight proxy for the scorer
            class _Proxy:
                pass
            proxy = _Proxy()
            proxy.expected_response = expected
            proxy.metadata = metadata
            proxy.metadata_params = metadata.parameters if metadata else {}
            # Inject _scoring_config into metadata.parameters for _get_scoring_config
            return _resolve_paradigm_score(para, proxy, response, sc_config)
        return _score

    inst = TaskInstance(
        task_id=task_id,
        metadata=meta,
        stimulus=stimulus,
        expected_response=expected_response,
        scoring_fn=_make_scorer(paradigm, scoring_config),
    )
    return inst


def _get_scoring_config(task: TaskInstance) -> Dict[str, Any]:
    """Retrieve the scoring_config dict from a (nested) TaskInstance."""
    return task.metadata.parameters.get("_scoring_config", {})


def _match_option_token(expected: str, response: str, stimulus: str) -> bool:
    """Punctuation-tolerant token match, strict on two-option ambiguity.

    'Left.' or 'The center letter is "K".' must match the answer token; a
    response naming BOTH options of a forced choice earns no credit. Items
    without a parseable two-option line (counts, letters) keep plain
    token-presence semantics.
    """
    expected = str(expected).strip().lower()
    given = response.strip().lower()
    if expected == given:
        return True
    tokens = re.findall(r"[a-z0-9]+", given)
    if expected not in tokens:
        return False
    m = re.search(r"answer with exactly one word:\s*([a-z0-9]+)\s+or\s+([a-z0-9]+)",
                  stimulus.lower())
    if m:
        alt = m.group(2) if m.group(1) == expected else m.group(1)
        if alt != expected and alt in tokens:
            return False
    return True


def _resolve_paradigm_score(paradigm: str, task, response: str, sc_config: Dict[str, Any]) -> Dict[str, Any]:
    """Score using the paradigm-specific scorer with injected scoring_config."""
    expected = str(task.expected_response).strip().lower() if task.expected_response else ""
    given = response.strip().lower()

    if not expected:
        return {"correct": False, "response": response}

    if paradigm == "stroop":
        correct = expected == given or expected in given.split()
        return {
            "correct": correct,
            "condition": sc_config.get("condition", "unknown"),
            "conflict_type": sc_config.get("conflict_type", "unknown"),
            "is_contamination_probe": sc_config.get("is_contamination_probe", False),
        }
    elif paradigm == "flanker":
        correct = expected == given or expected in given.split()
        return {
            "correct": correct,
            "condition": sc_config.get("condition", "unknown"),
            "symbol_set": sc_config.get("symbol_set", "unknown"),
            "is_contamination_probe": sc_config.get("is_contamination_probe", False),
        }
    elif paradigm == "go_nogo":
        given_upper = response.strip().upper()
        if "NO-GO" in given_upper or "NOGO" in given_upper:
            given_clean = "NO-GO"
        elif "GO" in given_upper:
            given_clean = "GO"
        else:
            given_clean = given_upper
        condition = sc_config.get("condition", "go")
        correct = (
            (condition == "go" and given_clean == "GO")
            or (condition == "nogo" and given_clean == "NO-GO")
        )
        if condition == "go":
            response_type = "hit" if given_clean == "GO" else "miss"
        else:
            response_type = "correct_rejection" if given_clean == "NO-GO" else "false_alarm"
        return {
            "correct": correct,
            "condition": condition,
            "response_type": response_type,
            "trial_index": sc_config.get("trial_index", 0),
            "is_contamination_probe": sc_config.get("is_contamination_probe", False),
        }
    else:
        correct = expected == given
        return {"correct": correct, "response": response}


# Sentinel for lazy scorer binding (unused, kept for cleanup)
_current_task = [None]


# ===================================================================
# Paradigm 1 -- Stroop Task (Text Version)
# ===================================================================

# --- attribute pools for procedural generation ---------------------------

_SIZE_WORDS: List[str] = ["LARGE", "BIG", "HUGE", "ENORMOUS", "GIANT",
                          "SMALL", "TINY", "LITTLE", "MINIATURE", "PETITE"]
_SIZE_FONT_OPTIONS: List[str] = ["large", "small"]

_DIRECTION_WORDS: List[str] = ["LEFT", "RIGHT", "UP", "DOWN"]
_DIRECTION_POSITIONS: List[str] = ["left", "right", "top", "bottom"]

_NUMBER_WORDS: Dict[str, int] = {
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
    "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10,
}

_COLOR_WORDS: List[str] = ["RED", "BLUE", "GREEN", "YELLOW", "ORANGE",
                           "PURPLE", "PINK", "BROWN", "BLACK", "WHITE"]


def _stroop_size_trial(rng: random.Random, congruent: bool) -> Tuple[str, str, str]:
    """Return (stimulus, expected_answer, conflict_type) for size-word Stroop."""
    word = rng.choice(_SIZE_WORDS)
    word_implies_big = word in _SIZE_WORDS[:5]  # first 5 are "big" words

    if congruent:
        font_size = "large" if word_implies_big else "small"
    else:
        font_size = "small" if word_implies_big else "large"

    stimulus = (
        f'The word "{word}" is written in a {font_size} font.\n'
        f"Question: What is the font size? Answer with exactly one word: large or small."
    )
    return stimulus, font_size, "size_word"


def _stroop_direction_trial(rng: random.Random, congruent: bool) -> Tuple[str, str, str]:
    """Return (stimulus, expected_answer, conflict_type) for direction-word Stroop."""
    horizontal = ["LEFT", "RIGHT"]
    vertical = ["UP", "DOWN"]
    axis = rng.choice(["horizontal", "vertical"])
    if axis == "horizontal":
        word = rng.choice(horizontal)
        if congruent:
            position = word.lower()
        else:
            position = "right" if word == "LEFT" else "left"
        stimulus = (
            f'The word "{word}" is positioned on the {position} side of the screen.\n'
            f"Question: What side of the screen is the word positioned on? "
            f"Answer with exactly one word: left or right."
        )
    else:
        word = rng.choice(vertical)
        if congruent:
            position = "top" if word == "UP" else "bottom"
        else:
            position = "bottom" if word == "UP" else "top"
        stimulus = (
            f'The word "{word}" is positioned at the {position} of the screen.\n'
            f"Question: What part of the screen is the word positioned at? "
            f"Answer with exactly one word: top or bottom."
        )
    return stimulus, position, "direction_word"


def _stroop_number_trial(rng: random.Random, congruent: bool) -> Tuple[str, str, str]:
    """Return (stimulus, expected_answer, conflict_type) for number-quantity Stroop."""
    word = rng.choice(list(_NUMBER_WORDS.keys()))
    word_value = _NUMBER_WORDS[word]

    if congruent:
        repeat_count = word_value
    else:
        # pick a different count
        candidates = [v for v in range(1, 11) if v != word_value]
        repeat_count = rng.choice(candidates)

    repeated = " ".join([word] * repeat_count)
    stimulus = (
        f'The word "{word}" appears {repeat_count} time{"s" if repeat_count > 1 else ""}: '
        f"{repeated}\n"
        f"Question: How many times does the word appear? "
        f"Answer with exactly one number."
    )
    return stimulus, str(repeat_count), "number_quantity"


def _stroop_color_trial(rng: random.Random, congruent: bool) -> Tuple[str, str, str]:
    """Classic color-word Stroop (contamination probe variant)."""
    word = rng.choice(_COLOR_WORDS)
    if congruent:
        ink_color = word.lower()
    else:
        others = [c for c in _COLOR_WORDS if c != word]
        ink_color = rng.choice(others).lower()

    stimulus = (
        f'The word "{word}" is printed in {ink_color} ink.\n'
        f"Question: What color is the ink the word is printed in? "
        f"Answer with exactly one word (the ink color)."
    )
    return stimulus, ink_color, "color_word"


_STROOP_GENERATORS = {
    "size_word": _stroop_size_trial,
    "direction_word": _stroop_direction_trial,
    "number_quantity": _stroop_number_trial,
    "color_word": _stroop_color_trial,
}


class StroopParadigm:
    """Stroop task with multiple conflict types for reduced contamination."""

    DIMENSION = "cognitive_control"
    PARADIGM = "stroop"
    MODE = "static"
    ADAPTATION_DISTANCE = "medium"

    @staticmethod
    def generate(
        seed: int = 42,
        n_congruent: int = 25,
        n_incongruent: int = 25,
        conflict_type: str = "mixed",
        difficulty: str = "medium",
        contamination_probe: bool = False,
    ) -> List[TaskInstance]:
        """Generate Stroop task instances.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        n_congruent : int
            Number of congruent trials.
        n_incongruent : int
            Number of incongruent trials.
        conflict_type : str
            One of "size_word", "direction_word", "number_quantity",
            "color_word", or "mixed" (all non-color types equally).
            "color_word" is used for contamination probes.
        difficulty : str
            "easy" (more congruent), "medium", "hard" (more incongruent).
        contamination_probe : bool
            If True, include classic color-word items as contamination probes.
        """
        rng = random.Random(seed)
        items: List[TaskInstance] = []
        idx = 0

        # Resolve which generators to use
        if conflict_type == "mixed":
            gen_keys = ["size_word", "direction_word", "number_quantity"]
        elif conflict_type in _STROOP_GENERATORS:
            gen_keys = [conflict_type]
        else:
            raise ValueError(f"Unknown conflict_type: {conflict_type}")

        # Add contamination probes if requested
        if contamination_probe and "color_word" not in gen_keys:
            gen_keys_probe = ["color_word"]
        else:
            gen_keys_probe = []

        def _make_items(n: int, congruent: bool, keys: List[str]) -> None:
            nonlocal idx
            per_key = max(1, n // len(keys))
            remainder = n - per_key * len(keys)
            for ki, key in enumerate(keys):
                count = per_key + (1 if ki < remainder else 0)
                gen_fn = _STROOP_GENERATORS[key]
                for _ in range(count):
                    stimulus, expected, ctype = gen_fn(rng, congruent)
                    condition = "congruent" if congruent else "incongruent"
                    items.append(_make_cc_item(
                        task_id=f"stroop_{ctype}_{condition}_{idx:04d}",
                        paradigm="stroop",
                        mode="static",
                        parameters={
                            "conflict_type": ctype,
                            "congruent": congruent,
                            "seed": seed,
                        },
                        stimulus=stimulus,
                        expected_response=expected,
                        scoring_config={
                            "condition": condition,
                            "conflict_type": ctype,
                            "is_contamination_probe": ctype == "color_word",
                        },
                        difficulty=difficulty,
                        adaptation_distance="high" if ctype == "color_word" else "medium",
                    ))
                    idx += 1

        _make_items(n_congruent, True, gen_keys)
        _make_items(n_incongruent, False, gen_keys)

        # contamination probes (small set)
        if gen_keys_probe:
            n_probe = max(5, (n_congruent + n_incongruent) // 10)
            _make_items(n_probe // 2, True, gen_keys_probe)
            _make_items(n_probe - n_probe // 2, False, gen_keys_probe)

        rng.shuffle(items)
        return items

    @staticmethod
    def score(task: TaskInstance, response: str) -> Dict[str, Any]:
        """Score a single Stroop trial.

        Returns dict with keys: correct (bool), condition, conflict_type,
        is_contamination_probe.
        """
        correct = _match_option_token(task.expected_response, response, task.stimulus)
        sc = _get_scoring_config(task)

        return {
            "correct": correct,
            "condition": sc.get("condition", "unknown"),
            "conflict_type": sc.get("conflict_type", "unknown"),
            "is_contamination_probe": sc.get("is_contamination_probe", False),
        }

    @staticmethod
    def aggregate(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate Stroop metrics from a list of scored trials.

        Returns dict with:
          acc_congruent, acc_incongruent, congruency_effect,
          per-conflict-type breakdowns, contamination analysis.
        """
        cong_correct, cong_total = 0, 0
        incong_correct, incong_total = 0, 0
        by_type: Dict[str, Dict[str, List[bool]]] = {}
        contam_correct, contam_total = 0, 0
        non_contam_correct, non_contam_total = 0, 0

        for s in scored:
            ct = s["conflict_type"]
            cond = s["condition"]
            c = s["correct"]

            if ct not in by_type:
                by_type[ct] = {"congruent": [], "incongruent": []}
            by_type[ct][cond].append(c)

            if cond == "congruent":
                cong_total += 1
                cong_correct += int(c)
            else:
                incong_total += 1
                incong_correct += int(c)

            if s.get("is_contamination_probe"):
                contam_total += 1
                contam_correct += int(c)
            else:
                non_contam_total += 1
                non_contam_correct += int(c)

        acc_cong = cong_correct / cong_total if cong_total else 0.0
        acc_incong = incong_correct / incong_total if incong_total else 0.0
        congruency_effect = acc_cong - acc_incong

        # Behavioral signature check:
        # For valid Stroop effect, incongruent accuracy should be lower
        signature_valid = acc_incong < acc_cong if (cong_total and incong_total) else None

        # Per-type breakdown
        type_breakdown: Dict[str, Dict[str, Any]] = {}
        for ct, cond_map in by_type.items():
            tb: Dict[str, Any] = {}
            for cond_name in ("congruent", "incongruent"):
                trials = cond_map.get(cond_name, [])
                tb[f"acc_{cond_name}"] = (
                    sum(trials) / len(trials) if trials else 0.0
                )
                tb[f"n_{cond_name}"] = len(trials)
            tb["congruency_effect"] = (
                tb["acc_congruent"] - tb["acc_incongruent"]
            )
            type_breakdown[ct] = tb

        # Contamination analysis
        contam_acc = contam_correct / contam_total if contam_total else None
        non_contam_acc = (
            non_contam_correct / non_contam_total if non_contam_total else None
        )
        contamination_gap = (
            (contam_acc - non_contam_acc)
            if contam_acc is not None and non_contam_acc is not None
            else None
        )

        return {
            "acc_congruent": acc_cong,
            "acc_incongruent": acc_incong,
            "congruency_effect": congruency_effect,
            "n_congruent": cong_total,
            "n_incongruent": incong_total,
            "behavioral_signature_valid": signature_valid,
            "per_conflict_type": type_breakdown,
            "contamination_probe_acc": contam_acc,
            "non_contamination_acc": non_contam_acc,
            "contamination_gap": contamination_gap,
        }


# ===================================================================
# Paradigm 2 -- Flanker Task (Text Version)
# ===================================================================

# --- symbol sets for procedural generation -------------------------------

_FLANKER_SYMBOL_SETS: Dict[str, Dict[str, Any]] = {
    "arrows": {
        "symbols": {">": "RIGHT", "<": "LEFT"},
        "question_template": "What direction does the CENTER arrow point?",
        "answer_options": "Answer with exactly one word: LEFT or RIGHT.",
    },
    "letters": {
        # Use pairs of letters that map to categories
        "symbols": {"H": "H", "S": "S", "K": "K", "C": "C"},
        "question_template": "What is the CENTER letter?",
        "answer_options": "Answer with exactly one letter.",
    },
    "numbers_parity": {
        "symbols": {
            "2": "even", "4": "even", "6": "even", "8": "even",
            "3": "odd", "5": "odd", "7": "odd", "9": "odd",
        },
        "question_template": (
            "Is the CENTER number even or odd?"
        ),
        "answer_options": "Answer with exactly one word: even or odd.",
    },
    "numbers_magnitude": {
        "symbols": {
            "1": "low", "2": "low", "3": "low", "4": "low",
            "6": "high", "7": "high", "8": "high", "9": "high",
        },
        "question_template": (
            "Is the CENTER number low (1-4) or high (6-9)?"
        ),
        "answer_options": "Answer with exactly one word: low or high.",
    },
}


def _flanker_trial_arrows(
    rng: random.Random, congruent: bool, n_flankers: int = 3,
) -> Tuple[str, str]:
    """Generate an arrow flanker trial.

    Returns (display_string, correct_answer).
    n_flankers: number of flanker symbols on each side of the target.
    """
    target = rng.choice([">", "<"])
    if congruent:
        flanker = target
    else:
        flanker = "<" if target == ">" else ">"

    display = flanker * n_flankers + target + flanker * n_flankers
    answer = "RIGHT" if target == ">" else "LEFT"
    return display, answer


def _flanker_trial_letters(
    rng: random.Random, congruent: bool, n_flankers: int = 3,
) -> Tuple[str, str]:
    """Generate a letter flanker trial."""
    symbols = list(_FLANKER_SYMBOL_SETS["letters"]["symbols"].keys())
    target = rng.choice(symbols)
    if congruent:
        flanker = target
    else:
        others = [s for s in symbols if s != target]
        flanker = rng.choice(others)
    # Add spaces for clarity
    parts = [flanker] * n_flankers + [target] + [flanker] * n_flankers
    display = " ".join(parts)
    answer = target
    return display, answer


def _flanker_trial_numbers(
    rng: random.Random,
    congruent: bool,
    n_flankers: int = 3,
    variant: str = "numbers_parity",
) -> Tuple[str, str]:
    """Generate a number flanker trial (parity or magnitude)."""
    sym_set = _FLANKER_SYMBOL_SETS[variant]
    symbols = list(sym_set["symbols"].keys())
    target = rng.choice(symbols)
    target_category = sym_set["symbols"][target]

    if congruent:
        same_cat = [s for s in symbols if sym_set["symbols"][s] == target_category and s != target]
        if same_cat:
            flanker = rng.choice(same_cat)
        else:
            flanker = target
    else:
        diff_cat = [s for s in symbols if sym_set["symbols"][s] != target_category]
        flanker = rng.choice(diff_cat)

    parts = [flanker] * n_flankers + [target] + [flanker] * n_flankers
    display = " ".join(parts)
    answer = target_category
    return display, answer


class FlankerParadigm:
    """Flanker task with various symbol sets."""

    DIMENSION = "cognitive_control"
    PARADIGM = "flanker"
    MODE = "static"
    ADAPTATION_DISTANCE = "medium"

    @staticmethod
    def generate(
        seed: int = 42,
        n_congruent: int = 25,
        n_incongruent: int = 25,
        symbol_set: str = "mixed",
        difficulty: str = "medium",
        contamination_probe: bool = False,
    ) -> List[TaskInstance]:
        """Generate Flanker task instances.

        Parameters
        ----------
        seed : int
            Random seed.
        n_congruent, n_incongruent : int
            Number of trials per condition.
        symbol_set : str
            "arrows", "letters", "numbers_parity", "numbers_magnitude", or "mixed".
        difficulty : str
            "easy" = 1 flanker each side, "medium" = 3, "hard" = 5.
        contamination_probe : bool
            If True, include classic arrow flanker items as contamination probes.
        """
        rng = random.Random(seed)
        n_flankers = {"easy": 1, "medium": 3, "hard": 5}.get(difficulty, 3)

        if symbol_set == "mixed":
            active_sets = ["arrows", "letters", "numbers_parity", "numbers_magnitude"]
        elif symbol_set in _FLANKER_SYMBOL_SETS:
            active_sets = [symbol_set]
        else:
            raise ValueError(f"Unknown symbol_set: {symbol_set}")

        items: List[TaskInstance] = []
        idx = 0

        def _gen_one(ss: str, congruent: bool) -> TaskInstance:
            nonlocal idx
            if ss == "arrows":
                display, answer = _flanker_trial_arrows(rng, congruent, n_flankers)
                question = _FLANKER_SYMBOL_SETS["arrows"]["question_template"]
                opts = _FLANKER_SYMBOL_SETS["arrows"]["answer_options"]
            elif ss == "letters":
                display, answer = _flanker_trial_letters(rng, congruent, n_flankers)
                question = _FLANKER_SYMBOL_SETS["letters"]["question_template"]
                opts = _FLANKER_SYMBOL_SETS["letters"]["answer_options"]
            else:
                display, answer = _flanker_trial_numbers(
                    rng, congruent, n_flankers, variant=ss,
                )
                question = _FLANKER_SYMBOL_SETS[ss]["question_template"]
                opts = _FLANKER_SYMBOL_SETS[ss]["answer_options"]

            condition = "congruent" if congruent else "incongruent"
            stimulus = (
                f"Stimulus: {display}\n"
                f"{question}\n"
                f"{opts}"
            )
            is_probe = ss == "arrows" and contamination_probe
            inst = _make_cc_item(
                task_id=f"flanker_{ss}_{condition}_{idx:04d}",
                paradigm="flanker",
                mode="static",
                parameters={
                    "symbol_set": ss,
                    "congruent": congruent,
                    "n_flankers": n_flankers,
                    "seed": seed,
                },
                stimulus=stimulus,
                expected_response=answer,
                scoring_config={
                    "condition": condition,
                    "symbol_set": ss,
                    "is_contamination_probe": is_probe,
                },
                difficulty=difficulty,
                adaptation_distance="medium",
            )
            idx += 1
            return inst

        # Distribute evenly across symbol sets
        for congruent, n_total in [(True, n_congruent), (False, n_incongruent)]:
            per_set = max(1, n_total // len(active_sets))
            remainder = n_total - per_set * len(active_sets)
            for si, ss in enumerate(active_sets):
                count = per_set + (1 if si < remainder else 0)
                for _ in range(count):
                    items.append(_gen_one(ss, congruent))

        rng.shuffle(items)
        return items

    @staticmethod
    def score(task: TaskInstance, response: str) -> Dict[str, Any]:
        """Score a single Flanker trial."""
        correct = _match_option_token(task.expected_response, response, task.stimulus)
        sc = _get_scoring_config(task)

        return {
            "correct": correct,
            "condition": sc.get("condition", "unknown"),
            "symbol_set": sc.get("symbol_set", "unknown"),
            "is_contamination_probe": sc.get("is_contamination_probe", False),
        }

    @staticmethod
    def aggregate(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate Flanker metrics."""
        cong_correct, cong_total = 0, 0
        incong_correct, incong_total = 0, 0
        by_set: Dict[str, Dict[str, List[bool]]] = {}

        for s in scored:
            ss = s["symbol_set"]
            cond = s["condition"]
            c = s["correct"]

            if ss not in by_set:
                by_set[ss] = {"congruent": [], "incongruent": []}
            by_set[ss][cond].append(c)

            if cond == "congruent":
                cong_total += 1
                cong_correct += int(c)
            else:
                incong_total += 1
                incong_correct += int(c)

        acc_cong = cong_correct / cong_total if cong_total else 0.0
        acc_incong = incong_correct / incong_total if incong_total else 0.0
        flanker_effect = acc_cong - acc_incong

        # Behavioral signature check
        signature_valid = acc_incong < acc_cong if (cong_total and incong_total) else None

        set_breakdown: Dict[str, Dict[str, Any]] = {}
        for ss, cond_map in by_set.items():
            tb: Dict[str, Any] = {}
            for cond_name in ("congruent", "incongruent"):
                trials = cond_map.get(cond_name, [])
                tb[f"acc_{cond_name}"] = (
                    sum(trials) / len(trials) if trials else 0.0
                )
                tb[f"n_{cond_name}"] = len(trials)
            tb["flanker_effect"] = tb["acc_congruent"] - tb["acc_incongruent"]
            set_breakdown[ss] = tb

        return {
            "acc_congruent": acc_cong,
            "acc_incongruent": acc_incong,
            "flanker_effect": flanker_effect,
            "n_congruent": cong_total,
            "n_incongruent": incong_total,
            "behavioral_signature_valid": signature_valid,
            "per_symbol_set": set_breakdown,
        }


# ===================================================================
# Paradigm 3 -- Go/No-Go Task
# ===================================================================

# --- category pools for procedural generation ----------------------------

_CATEGORY_POOLS: Dict[str, List[str]] = {
    "animals": [
        "dog", "cat", "horse", "elephant", "tiger", "lion", "bear", "wolf",
        "eagle", "dolphin", "whale", "rabbit", "deer", "fox", "owl",
        "penguin", "parrot", "shark", "snake", "frog", "turtle", "hawk",
        "salmon", "crab", "octopus", "bee", "ant", "butterfly", "sparrow",
        "goat",
    ],
    "plants": [
        "oak", "rose", "tulip", "maple", "daisy", "fern", "cactus", "ivy",
        "lily", "orchid", "bamboo", "willow", "pine", "birch", "cedar",
        "moss", "algae", "sunflower", "lavender", "basil", "mint", "sage",
        "thyme", "clover", "daffodil", "poppy", "violet", "iris", "lotus",
        "palm",
    ],
    "fruits": [
        "apple", "banana", "cherry", "grape", "lemon", "mango", "orange",
        "peach", "pear", "plum", "kiwi", "melon", "fig", "lime", "coconut",
        "papaya", "guava", "apricot", "berry", "date", "pomegranate",
        "tangerine", "nectarine", "grapefruit", "pineapple", "watermelon",
        "strawberry", "blueberry", "raspberry", "cranberry",
    ],
    "vegetables": [
        "carrot", "potato", "tomato", "onion", "pepper", "garlic", "corn",
        "spinach", "broccoli", "lettuce", "celery", "cucumber", "pea",
        "bean", "radish", "beet", "turnip", "squash", "zucchini", "cabbage",
        "kale", "leek", "asparagus", "artichoke", "eggplant", "mushroom",
        "pumpkin", "yam", "parsnip", "okra",
    ],
    "tools": [
        "hammer", "wrench", "screwdriver", "pliers", "saw", "drill",
        "chisel", "clamp", "file", "level", "tape", "ruler", "compass",
        "anvil", "vise", "mallet", "awl", "lathe", "trowel", "shovel",
        "rake", "hoe", "axe", "pickaxe", "crowbar", "bolt", "nail",
        "screw", "sandpaper", "jack",
    ],
    "instruments": [
        "piano", "guitar", "violin", "drums", "flute", "trumpet", "cello",
        "harp", "clarinet", "saxophone", "tuba", "oboe", "banjo",
        "accordion", "harmonica", "ukulele", "mandolin", "bassoon",
        "trombone", "xylophone", "tambourine", "cymbal", "maracas", "sitar",
        "lute", "organ", "fiddle", "kazoo", "bongo", "gong",
    ],
    "clothing": [
        "shirt", "pants", "jacket", "coat", "dress", "skirt", "sweater",
        "vest", "scarf", "hat", "gloves", "socks", "boots", "sandals",
        "tie", "belt", "cape", "robe", "shorts", "blouse", "cardigan",
        "hoodie", "parka", "raincoat", "suit", "apron", "poncho",
        "overalls", "gown", "uniform",
    ],
    "furniture": [
        "chair", "table", "desk", "sofa", "bed", "shelf", "cabinet",
        "dresser", "bench", "stool", "lamp", "mirror", "rug", "curtain",
        "pillow", "blanket", "wardrobe", "bookcase", "ottoman", "cradle",
        "hammock", "futon", "couch", "recliner", "bureau", "chest",
        "nightstand", "armchair", "cupboard", "rack",
    ],
}

# Pre-defined category pairs with balanced pool sizes
_CATEGORY_PAIRS: List[Tuple[str, str]] = [
    ("animals", "plants"),
    ("fruits", "vegetables"),
    ("tools", "instruments"),
    ("clothing", "furniture"),
    ("animals", "furniture"),
    ("fruits", "tools"),
    ("plants", "instruments"),
    ("vegetables", "clothing"),
]


class GoNoGoParadigm:
    """Go/No-Go task with procedurally generated category rules.

    Multi-turn paradigm: stimuli presented one at a time. The model must
    respond "GO" or "NO-GO" for each stimulus.
    """

    DIMENSION = "cognitive_control"
    PARADIGM = "go_nogo"
    MODE = "multi_turn"
    ADAPTATION_DISTANCE = "medium"

    @staticmethod
    def generate(
        seed: int = 42,
        n_trials: int = 60,
        go_ratio: float = 0.75,
        category_pair: Optional[Tuple[str, str]] = None,
        difficulty: str = "medium",
        contamination_probe: bool = False,
        n_items: int = 0,  # alias kept for unified interface; uses n_trials
    ) -> List[TaskInstance]:
        """Generate a Go/No-Go task sequence.

        Parameters
        ----------
        seed : int
            Random seed.
        n_trials : int
            Total number of trials (50-100 recommended).
        go_ratio : float
            Fraction of go trials (default 0.75 to create prepotent response).
        category_pair : tuple of str, optional
            (go_category, nogo_category). If None, randomly selected.
        difficulty : str
            "easy" = 0.85 go ratio, "medium" = 0.75, "hard" = 0.65.
        contamination_probe : bool
            If True, uses classic "animals vs plants" pair.
        """
        if n_items > 0:
            n_trials = n_items

        rng = random.Random(seed)

        # Adjust go_ratio by difficulty
        difficulty_ratios = {"easy": 0.85, "medium": 0.75, "hard": 0.65}
        go_ratio = difficulty_ratios.get(difficulty, go_ratio)

        # Select category pair
        if contamination_probe:
            go_cat, nogo_cat = "animals", "plants"
        elif category_pair is not None:
            go_cat, nogo_cat = category_pair
        else:
            go_cat, nogo_cat = rng.choice(_CATEGORY_PAIRS)

        go_pool = list(_CATEGORY_POOLS[go_cat])
        nogo_pool = list(_CATEGORY_POOLS[nogo_cat])
        rng.shuffle(go_pool)
        rng.shuffle(nogo_pool)

        n_go = round(n_trials * go_ratio)
        n_nogo = n_trials - n_go

        # Build trial sequence (with replacement from pool if needed)
        trial_types: List[str] = ["go"] * n_go + ["nogo"] * n_nogo
        rng.shuffle(trial_types)

        # Ensure no more than 4 consecutive go trials (to add variability)
        # Simple repair pass
        for i in range(4, len(trial_types)):
            if all(t == "go" for t in trial_types[i - 4 : i + 1]):
                # Find nearest nogo to swap
                for j in range(i + 1, len(trial_types)):
                    if trial_types[j] == "nogo":
                        trial_types[i], trial_types[j] = trial_types[j], trial_types[i]
                        break

        go_idx, nogo_idx = 0, 0
        items: List[TaskInstance] = []

        # Preamble instruction (first item carries the rule description)
        # Build grammatically correct category descriptors
        _article = lambda w: "an" if w[0].lower() in "aeiou" else "a"
        _singular = lambda w: w.rstrip("s") if w.endswith("s") else w

        go_sing = _singular(go_cat)
        nogo_sing = _singular(nogo_cat)

        rule_instruction = (
            f"In this task, you will see a series of words one at a time.\n"
            f"Rule: If the word belongs to the category '{go_cat}', respond with exactly: GO\n"
            f"If the word belongs to the category '{nogo_cat}', respond with exactly: NO-GO\n\n"
            f"IMPORTANT: Your response must be ONLY the word GO or the word NO-GO. "
            f"Do not repeat the stimulus word. Do not explain your reasoning.\n"
        )

        for trial_idx, ttype in enumerate(trial_types):
            if ttype == "go":
                word = go_pool[go_idx % len(go_pool)]
                go_idx += 1
                expected = "GO"
                condition = "go"
            else:
                word = nogo_pool[nogo_idx % len(nogo_pool)]
                nogo_idx += 1
                expected = "NO-GO"
                condition = "nogo"

            # Every trial carries the rule: items are dispatched as isolated
            # single-turn prompts, so a rule shown only on trial 0 never
            # reaches the model on later trials.
            stimulus = f"{rule_instruction}\nTrial {trial_idx + 1}: {word}\nYour response (GO or NO-GO):"

            items.append(_make_cc_item(
                task_id=f"gonogo_{go_cat}_{nogo_cat}_{trial_idx:04d}",
                paradigm="go_nogo",
                mode="multi_turn",
                parameters={
                    "go_category": go_cat,
                    "nogo_category": nogo_cat,
                    "go_ratio": go_ratio,
                    "trial_index": trial_idx,
                    "n_trials": n_trials,
                    "seed": seed,
                },
                stimulus=stimulus,
                expected_response=expected,
                scoring_config={
                    "condition": condition,
                    "go_category": go_cat,
                    "nogo_category": nogo_cat,
                    "is_contamination_probe": contamination_probe,
                    "trial_index": trial_idx,
                },
                difficulty=difficulty,
                adaptation_distance="medium",
            ))

        return items

    @staticmethod
    def score(task: TaskInstance, response: str) -> Dict[str, Any]:
        """Score a single Go/No-Go trial.

        Returns dict with: correct, condition (go/nogo), response_type
        (hit, miss, correct_rejection, false_alarm).
        """
        expected = task.expected_response.strip().upper()
        # The rule text shown on every trial contains both GO and NO-GO, so
        # prompt echoes must not count as answers: strip them, then grade the
        # first standalone GO / NO-GO token. A substring test biased toward
        # NO-GO flips "GO" answers that go on to mention NO-GO.
        text = response.upper()
        text = re.sub(r"RESPOND WITH EXACTLY:?\s*NO[\s\-_]?GO", " ", text)
        text = re.sub(r"RESPOND WITH EXACTLY:?\s*GO", " ", text)
        text = re.sub(r"(?:THE\s+WORD\s+)?GO\s+OR\s+(?:THE\s+WORD\s+)?NO[\s\-_]?GO", " ", text)
        m = re.search(r"\b(NO[\s\-_]?GO|GO)\b", text)
        if m:
            given_clean = "GO" if m.group(1) == "GO" else "NO-GO"
        elif text.strip().rstrip(".!") == "NO":
            given_clean = "NO-GO"
        else:
            given_clean = response.strip().upper()

        sc = _get_scoring_config(task)
        condition = sc.get("condition", "go")
        correct = (
            (condition == "go" and given_clean == "GO")
            or (condition == "nogo" and given_clean == "NO-GO")
        )

        if condition == "go":
            response_type = "hit" if given_clean == "GO" else "miss"
        else:
            response_type = (
                "correct_rejection" if given_clean == "NO-GO" else "false_alarm"
            )

        return {
            "correct": correct,
            "condition": condition,
            "response_type": response_type,
            "trial_index": sc.get("trial_index", 0),
            "is_contamination_probe": sc.get("is_contamination_probe", False),
        }

    @staticmethod
    def aggregate(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate Go/No-Go metrics.

        Returns: hit_rate, false_alarm_rate, d_prime, commission_errors,
        omission_errors, overall_accuracy.
        """
        hits, misses = 0, 0
        correct_rejections, false_alarms = 0, 0
        n_go, n_nogo = 0, 0

        for s in scored:
            rt = s["response_type"]
            if s["condition"] == "go":
                n_go += 1
                if rt == "hit":
                    hits += 1
                else:
                    misses += 1
            else:
                n_nogo += 1
                if rt == "correct_rejection":
                    correct_rejections += 1
                else:
                    false_alarms += 1

        hit_rate = hits / n_go if n_go else 0.0
        false_alarm_rate = false_alarms / n_nogo if n_nogo else 0.0

        # d-prime calculation with log-linear correction to avoid infinite values
        # Hautus (1995) correction: add 0.5 to both hits and false alarms,
        # add 1 to totals
        if n_go > 0 and n_nogo > 0:
            hr_adj = (hits + 0.5) / (n_go + 1)
            far_adj = (false_alarms + 0.5) / (n_nogo + 1)
            # Clamp to avoid domain errors (should not happen with correction)
            hr_adj = max(0.001, min(0.999, hr_adj))
            far_adj = max(0.001, min(0.999, far_adj))
            d_prime = _z(hr_adj) - _z(far_adj)
        else:
            d_prime = 0.0

        total_correct = hits + correct_rejections
        total_trials = n_go + n_nogo
        overall_accuracy = total_correct / total_trials if total_trials else 0.0

        return {
            "hit_rate": hit_rate,
            "false_alarm_rate": false_alarm_rate,
            "d_prime": d_prime,
            "commission_errors": false_alarms,
            "omission_errors": misses,
            "correct_rejections": correct_rejections,
            "hits": hits,
            "n_go": n_go,
            "n_nogo": n_nogo,
            "overall_accuracy": overall_accuracy,
        }


def _z(p: float) -> float:
    """Inverse of the standard normal CDF (probit).

    Uses the rational approximation from Abramowitz & Stegun (1964),
    formula 26.2.23.  Accurate to ~4.5e-4.
    """
    if p <= 0.0:
        return -5.0
    if p >= 1.0:
        return 5.0
    # Use symmetry: if p > 0.5, compute for 1-p and negate
    if p > 0.5:
        return -_z(1.0 - p)
    # Rational approximation for 0 < p <= 0.5
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))


# ===================================================================
# Convenience: unified interface
# ===================================================================

PARADIGMS = {
    "stroop": StroopParadigm,
    "flanker": FlankerParadigm,
    "go_nogo": GoNoGoParadigm,
}


def generate_all(
    seed: int = 42,
    n_per_paradigm: int = 50,
    difficulty: str = "medium",
    contamination_probe: bool = False,
) -> List[TaskInstance]:
    """Generate items for all three cognitive control paradigms.

    Parameters
    ----------
    seed : int
        Base random seed (each paradigm offsets by 1000).
    n_per_paradigm : int
        Approximate number of items per paradigm.
    difficulty : str
        Difficulty level for all paradigms.
    contamination_probe : bool
        Whether to include contamination probes.

    Returns
    -------
    list of TaskInstance
    """
    items: List[TaskInstance] = []

    # Stroop: split into congruent / incongruent
    n_cong = n_per_paradigm // 2
    n_incong = n_per_paradigm - n_cong
    items.extend(
        StroopParadigm.generate(
            seed=seed,
            n_congruent=n_cong,
            n_incongruent=n_incong,
            conflict_type="mixed",
            difficulty=difficulty,
            contamination_probe=contamination_probe,
        )
    )

    # Flanker
    items.extend(
        FlankerParadigm.generate(
            seed=seed + 1000,
            n_congruent=n_cong,
            n_incongruent=n_incong,
            symbol_set="mixed",
            difficulty=difficulty,
            contamination_probe=contamination_probe,
        )
    )

    # Go/No-Go
    items.extend(
        GoNoGoParadigm.generate(
            seed=seed + 2000,
            n_trials=n_per_paradigm,
            difficulty=difficulty,
            contamination_probe=contamination_probe,
        )
    )

    return items
