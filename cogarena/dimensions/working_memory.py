"""Working Memory dimension for CogArena.

Implements three paradigms with procedural generation:
  1. N-Back Task        -- updating / maintenance
  2. Digit Span         -- capacity (forward / backward / sequencing)
  3. Operation Span     -- complex span (dual-task processing + recall)

All items are procedurally generated from random seeds to minimise
contamination from training corpora.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

from cogarena.core import (
    AdaptationDistance,
    DifficultyLevel,
    EvalMode,
    ScoringConfig,
    TaskInstance,
    TaskMetadata,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Non-standard synthetic two-character tokens (avoids classic A,B,C...)
_SYNTHETIC_TOKENS: list[str] = [
    "ZQ", "MV", "PL", "KW", "HN", "DX", "FR", "GT", "BJ", "CY",
    "LT", "NR", "QS", "WP", "XK", "VH", "JM", "TG", "RD", "YF",
]

# Classic single letters -- only used for contamination probes
_CLASSIC_LETTERS: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _make_rng(seed: int) -> random.Random:
    """Return a seeded Random instance (reproducible, thread-safe)."""
    return random.Random(seed)


def _dprime(hit_rate: float, fa_rate: float) -> float:
    """Compute d' (d-prime) from hit rate and false-alarm rate.

    Rates are clipped to (0.01, 0.99) to avoid infinite z-scores.
    """

    def _z(p: float) -> float:
        """Inverse of the standard normal CDF (probit) via rational approx."""
        p = max(0.01, min(0.99, p))
        # Rational approximation (Abramowitz & Stegun 26.2.23)
        if p < 0.5:
            t = math.sqrt(-2.0 * math.log(p))
        else:
            t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        val = t - (c0 + c1 * t + c2 * t * t) / (
            1.0 + d1 * t + d2 * t * t + d3 * t * t * t
        )
        return val if p >= 0.5 else -val

    return _z(hit_rate) - _z(fa_rate)


def _difficulty_enum(s: str) -> DifficultyLevel:
    return DifficultyLevel(s.lower())


# ===================================================================
# PARADIGM 1 -- N-BACK
# ===================================================================

class NBackGenerator:
    """Procedural generator for the N-Back task.

    Produces multi-turn sequences where, at each position, the model
    must respond "MATCH" if the current stimulus equals the stimulus
    presented *n* positions earlier, or "NO MATCH" otherwise.
    """

    PARADIGM = "n_back"
    DIMENSION = "working_memory"

    # Difficulty mapping
    DIFFICULTY_MAP: Dict[str, Dict[str, Any]] = {
        "easy":   {"n": 1, "sequence_length": 16, "lure_rate": 0.0,  "target_rate": 0.30},
        "medium": {"n": 2, "sequence_length": 24, "lure_rate": 0.15, "target_rate": 0.30},
        "hard":   {"n": 3, "sequence_length": 32, "lure_rate": 0.25, "target_rate": 0.25},
    }

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_sequence(
        rng: random.Random,
        n: int,
        length: int,
        target_rate: float,
        lure_rate: float,
        token_pool: list[str],
    ) -> tuple[list[str], list[bool]]:
        """Build a stimulus sequence and corresponding target flags.

        A *target* trial repeats the token from *n* positions back.
        A *lure* trial repeats a token from *n-1* or *n+1* positions back
        (only when n >= 2).

        Returns
        -------
        sequence : list[str]
        is_target : list[bool]   (True where the correct answer is MATCH)
        """
        sequence: list[str] = []
        is_target: list[bool] = []

        for i in range(length):
            roll = rng.random()

            # Can we place a target?
            if i >= n and roll < target_rate:
                token = sequence[i - n]
                sequence.append(token)
                is_target.append(True)
                continue

            # Can we place a lure?  (only when n >= 2 and there is a valid
            # offset that is != n)
            if i >= 2 and n >= 2 and roll < target_rate + lure_rate:
                offsets = [o for o in (n - 1, n + 1) if 0 < o <= i]
                if offsets:
                    offset = rng.choice(offsets)
                    lure_token = sequence[i - offset]
                    # Make sure the lure does NOT accidentally equal the
                    # n-back token (which would make it a target).
                    if i >= n and lure_token == sequence[i - n]:
                        pass  # fall through to random
                    else:
                        sequence.append(lure_token)
                        is_target.append(False)
                        continue

            # Default: pick a random token that is NOT the n-back token
            available = list(token_pool)
            if i >= n:
                try:
                    available.remove(sequence[i - n])
                except ValueError:
                    pass
            token = rng.choice(available)
            sequence.append(token)
            is_target.append(False)

        return sequence, is_target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def generate(
        cls,
        seed: int,
        n_items: int = 10,
        difficulty: str = "medium",
        contamination_probe: bool = False,
    ) -> list[TaskInstance]:
        """Generate *n_items* N-Back episodes.

        Each episode is a multi-turn sequence.  Trial-level data is stored
        in ``metadata.parameters["turns"]``, with each element::

            {"position": int, "stimulus": str, "expected": "MATCH"|"NO MATCH"}

        Parameters
        ----------
        seed : int
            Base random seed.
        n_items : int
            Number of independent episodes to generate.
        difficulty : "easy" | "medium" | "hard"
            Selects n, sequence_length, lure_rate, target_rate.
        contamination_probe : bool
            If True, use classic single letters instead of synthetic tokens.
        """
        params = dict(cls.DIFFICULTY_MAP.get(difficulty, cls.DIFFICULTY_MAP["medium"]))
        n: int = params["n"]
        seq_len: int = params["sequence_length"]
        target_rate: float = params["target_rate"]
        lure_rate: float = params["lure_rate"]

        token_pool = (
            list(_CLASSIC_LETTERS[:10]) if contamination_probe
            else list(_SYNTHETIC_TOKENS)
        )

        rng = _make_rng(seed)
        items: list[TaskInstance] = []

        for idx in range(n_items):
            episode_seed = rng.randint(0, 2**31)
            ep_rng = _make_rng(episode_seed)

            sequence, is_target = cls._generate_sequence(
                ep_rng, n, seq_len, target_rate, lure_rate, token_pool,
            )

            turns: list[Dict[str, Any]] = []
            for pos, (token, tgt) in enumerate(zip(sequence, is_target)):
                expected = "MATCH" if tgt else "NO MATCH"
                turns.append({
                    "position": pos,
                    "stimulus": token,
                    "expected": expected,
                })

            # Build a human-readable stimulus description (overview)
            stimulus_text = (
                f"N-Back Task (n={n}).\n"
                f"You will see a sequence of {seq_len} tokens, presented one "
                f"at a time.\n"
                f"For each token, respond MATCH if it is the same as the token "
                f"presented {n} position(s) earlier, otherwise respond NO MATCH.\n"
                f"First token: {sequence[0]}"
            )

            task_id = (
                f"wm_nback_{'probe' if contamination_probe else 'gen'}"
                f"_{difficulty}_n{n}_s{episode_seed}"
            )

            metadata = TaskMetadata(
                dimension=cls.DIMENSION,
                paradigm=cls.PARADIGM,
                mode=EvalMode.AGENT_INTERACTIVE,
                parameters={
                    "n": n,
                    "sequence_length": seq_len,
                    "target_rate": target_rate,
                    "lure_rate": lure_rate,
                    "episode_seed": episode_seed,
                    "contamination_probe": contamination_probe,
                    "multi_turn": True,
                    "turns": turns,
                },
                scoring=ScoringConfig(
                    method="custom",
                    params={
                        "fn": "cogarena.dimensions.working_memory.score_nback",
                        "n": n,
                    },
                ),
                difficulty=_difficulty_enum(difficulty),
                adaptation_distance=AdaptationDistance.LOW,
                description=(
                    f"{n}-back task with {'classic' if contamination_probe else 'synthetic'} tokens"
                ),
            )

            items.append(TaskInstance(
                task_id=task_id,
                metadata=metadata,
                stimulus=stimulus_text,
                expected_response=[t["expected"] for t in turns],
            ))

        return items

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score(task: TaskInstance, responses: list[str]) -> Dict[str, float]:
        """Score a completed N-Back episode.

        Parameters
        ----------
        task : TaskInstance
            The generated task (must have turns in metadata.parameters).
        responses : list[str]
            Model responses, one per turn.  Each should be "MATCH" or
            "NO MATCH" (case-insensitive).

        Returns
        -------
        dict with keys: accuracy, hit_rate, false_alarm_rate, d_prime,
        n_targets, n_non_targets.
        """
        turns = task.metadata.parameters["turns"]
        assert len(responses) == len(turns), (
            f"Expected {len(turns)} responses, got {len(responses)}"
        )

        hits = 0
        misses = 0
        false_alarms = 0
        correct_rejections = 0

        for turn, resp in zip(turns, responses):
            expected = turn["expected"]
            answer = resp.strip().upper()
            is_match_response = "MATCH" in answer and "NO" not in answer

            if expected == "MATCH":
                if is_match_response:
                    hits += 1
                else:
                    misses += 1
            else:  # NO MATCH expected
                if is_match_response:
                    false_alarms += 1
                else:
                    correct_rejections += 1

        n_targets = hits + misses
        n_non_targets = false_alarms + correct_rejections
        total = len(turns)

        hit_rate = hits / max(n_targets, 1)
        fa_rate = false_alarms / max(n_non_targets, 1)
        accuracy = (hits + correct_rejections) / max(total, 1)
        d_prime = _dprime(hit_rate, fa_rate)

        return {
            "accuracy": round(accuracy, 4),
            "hit_rate": round(hit_rate, 4),
            "false_alarm_rate": round(fa_rate, 4),
            "d_prime": round(d_prime, 4),
            "hits": float(hits),
            "misses": float(misses),
            "false_alarms": float(false_alarms),
            "correct_rejections": float(correct_rejections),
            "n_targets": float(n_targets),
            "n_non_targets": float(n_non_targets),
        }


# Module-level scoring function referenced by ScoringConfig custom fn path
def score_nback(
    response: Any,
    expected: Any,
    metadata: TaskMetadata,
) -> Dict[str, float]:
    """Scoring function for N-Back (used by TaskInstance.score via custom fn).

    ``response`` should be a list[str] of per-turn answers.
    ``expected`` should be a list[str] of expected per-turn answers.
    """
    turns = metadata.parameters["turns"]
    if isinstance(response, str):
        # Single response -- shouldn't happen for multi-turn but handle gracefully
        responses = [response]
    else:
        responses = list(response)

    # Pad or truncate to match turns length
    while len(responses) < len(turns):
        responses.append("")
    responses = responses[: len(turns)]

    hits = 0
    misses = 0
    false_alarms = 0
    correct_rejections = 0

    for turn, resp in zip(turns, responses):
        exp = turn["expected"]
        answer = resp.strip().upper()
        is_match_response = "MATCH" in answer and "NO" not in answer

        if exp == "MATCH":
            if is_match_response:
                hits += 1
            else:
                misses += 1
        else:
            if is_match_response:
                false_alarms += 1
            else:
                correct_rejections += 1

    n_targets = hits + misses
    n_non_targets = false_alarms + correct_rejections
    total = len(turns)

    hit_rate = hits / max(n_targets, 1)
    fa_rate = false_alarms / max(n_non_targets, 1)
    accuracy = (hits + correct_rejections) / max(total, 1)
    d_prime = _dprime(hit_rate, fa_rate)

    return {
        "accuracy": round(accuracy, 4),
        "hit_rate": round(hit_rate, 4),
        "false_alarm_rate": round(fa_rate, 4),
        "d_prime": round(d_prime, 4),
    }


# ===================================================================
# PARADIGM 2 -- DIGIT SPAN
# ===================================================================

class DigitSpanGenerator:
    """Procedural generator for the Digit Span task.

    Three sub-tasks (modes):
      - **forward**: repeat the digit sequence in presentation order.
      - **backward**: repeat the sequence in reverse order.
      - **sequencing**: reorder digits from smallest to largest.

    Sequences start short and grow; two trials per length.
    """

    PARADIGM = "digit_span"
    DIMENSION = "working_memory"

    DIFFICULTY_MAP: Dict[str, Dict[str, Any]] = {
        "easy":   {"start_length": 3, "max_length": 7,  "trials_per_length": 2},
        "medium": {"start_length": 3, "max_length": 11, "trials_per_length": 2},
        "hard":   {"start_length": 3, "max_length": 15, "trials_per_length": 2},
    }

    SUB_MODES = ("forward", "backward", "sequencing")

    @classmethod
    def _generate_digit_sequence(
        cls, rng: random.Random, length: int
    ) -> list[int]:
        """Return a random digit sequence of *length*, digits 0-9, no
        consecutive repeats."""
        seq: list[int] = []
        for _ in range(length):
            available = [d for d in range(10) if not seq or d != seq[-1]]
            seq.append(rng.choice(available))
        return seq

    @classmethod
    def _expected_response(cls, digits: list[int], sub_mode: str) -> str:
        """Return the correct response string for a given sub-mode."""
        if sub_mode == "forward":
            return " ".join(str(d) for d in digits)
        elif sub_mode == "backward":
            return " ".join(str(d) for d in reversed(digits))
        elif sub_mode == "sequencing":
            return " ".join(str(d) for d in sorted(digits))
        else:
            raise ValueError(f"Unknown sub_mode: {sub_mode}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def generate(
        cls,
        seed: int,
        n_items: int = 10,
        difficulty: str = "medium",
        contamination_probe: bool = False,
    ) -> list[TaskInstance]:
        """Generate Digit Span items.

        Each item is an independent trial at a specific length and sub-mode.
        The generator cycles through sub-modes and increasing lengths to
        produce *n_items* total items (distributing across sub-modes).

        Parameters
        ----------
        seed : int
            Base random seed.
        n_items : int
            Number of items to generate (spread across sub-modes).
        difficulty : str
            Controls max_length range.
        contamination_probe : bool
            No effect for this paradigm (digits are always random), but
            the flag is stored in metadata for consistency.
        """
        params = dict(cls.DIFFICULTY_MAP.get(difficulty, cls.DIFFICULTY_MAP["medium"]))
        start_len: int = params["start_length"]
        max_len: int = params["max_length"]
        trials_per_len: int = params["trials_per_length"]

        rng = _make_rng(seed)
        items: list[TaskInstance] = []

        # Distribute items roughly evenly across sub-modes
        items_per_mode = max(1, n_items // len(cls.SUB_MODES))
        remainder = n_items - items_per_mode * len(cls.SUB_MODES)

        for mode_idx, sub_mode in enumerate(cls.SUB_MODES):
            count = items_per_mode + (1 if mode_idx < remainder else 0)
            generated = 0
            current_length = start_len

            while generated < count and current_length <= max_len:
                for trial in range(trials_per_len):
                    if generated >= count:
                        break
                    digits = cls._generate_digit_sequence(rng, current_length)
                    expected = cls._expected_response(digits, sub_mode)

                    digit_str = " ".join(str(d) for d in digits)
                    if sub_mode == "forward":
                        instruction = (
                            f"Digit Span (Forward).\n"
                            f"Listen to the following sequence of digits, then "
                            f"repeat them back in the SAME order.\n"
                            f"Digits: {digit_str}\n"
                            f"Your response (digits separated by spaces):"
                        )
                    elif sub_mode == "backward":
                        instruction = (
                            f"Digit Span (Backward).\n"
                            f"Listen to the following sequence of digits, then "
                            f"repeat them back in REVERSE order.\n"
                            f"Digits: {digit_str}\n"
                            f"Your response (digits separated by spaces):"
                        )
                    else:
                        instruction = (
                            f"Digit Span (Sequencing).\n"
                            f"Listen to the following sequence of digits, then "
                            f"reorder them from SMALLEST to LARGEST.\n"
                            f"Digits: {digit_str}\n"
                            f"Your response (digits separated by spaces):"
                        )

                    trial_seed = rng.randint(0, 2**31)
                    task_id = (
                        f"wm_dspan_{sub_mode}"
                        f"_{'probe' if contamination_probe else 'gen'}"
                        f"_{difficulty}_len{current_length}_s{trial_seed}"
                    )

                    metadata = TaskMetadata(
                        dimension=cls.DIMENSION,
                        paradigm=cls.PARADIGM,
                        mode=EvalMode.LLM_STATIC,
                        parameters={
                            "sub_mode": sub_mode,
                            "digit_length": current_length,
                            "digits": digits,
                            "start_length": start_len,
                            "max_length": max_len,
                            "trials_per_length": trials_per_len,
                            "contamination_probe": contamination_probe,
                            "multi_turn": False,
                        },
                        scoring=ScoringConfig(
                            method="custom",
                            params={
                                "fn": "cogarena.dimensions.working_memory.score_digit_span",
                                "sub_mode": sub_mode,
                            },
                        ),
                        difficulty=_difficulty_enum(difficulty),
                        adaptation_distance=AdaptationDistance.LOW,
                        description=(
                            f"Digit span {sub_mode}, length {current_length}"
                        ),
                    )

                    items.append(TaskInstance(
                        task_id=task_id,
                        metadata=metadata,
                        stimulus=instruction,
                        expected_response=expected,
                    ))
                    generated += 1
                current_length += 1

        return items

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score(task: TaskInstance, response: str) -> Dict[str, float]:
        """Score a single Digit Span trial.

        Returns accuracy (1.0 or 0.0 for exact match), partial credit
        (proportion of digits in correct position), and the achieved span
        flag (1 if correct, 0 if not -- the caller aggregates across
        lengths to compute max span).
        """
        expected_tokens = str(task.expected_response).split()
        response_tokens = response.strip().split()

        # Exact match
        exact = 1.0 if response_tokens == expected_tokens else 0.0

        # Partial credit: proportion of positions matching
        max_len = max(len(expected_tokens), len(response_tokens), 1)
        matches = sum(
            1 for a, b in zip(expected_tokens, response_tokens) if a == b
        )
        partial = matches / max_len

        return {
            "accuracy": exact,
            "partial_credit": round(partial, 4),
            "digit_length": float(task.metadata.parameters["digit_length"]),
            "span_correct": exact,
        }

    @staticmethod
    def compute_max_span(
        scored_trials: list[Dict[str, float]],
    ) -> Dict[str, int]:
        """Given a list of scored trial dicts (each containing sub_mode
        from the task's metadata), compute the maximum span achieved per
        sub-mode.

        The max span for a sub-mode is the longest digit_length at which
        at least one trial was correct (exact match).
        """
        spans: Dict[str, int] = {}
        for trial in scored_trials:
            # Caller should attach sub_mode from task.metadata.parameters
            sm = trial.get("sub_mode", "forward")
            if trial.get("span_correct", 0.0) >= 1.0:
                dl = int(trial.get("digit_length", 0))
                spans[sm] = max(spans.get(sm, 0), dl)
        for sm in DigitSpanGenerator.SUB_MODES:
            spans.setdefault(sm, 0)
        return spans


# Module-level scoring function referenced by ScoringConfig custom fn path
def score_digit_span(
    response: Any,
    expected: Any,
    metadata: TaskMetadata,
) -> Dict[str, float]:
    """Scoring function for Digit Span (used by TaskInstance.score via custom fn)."""
    expected_tokens = str(expected).split()
    response_tokens = str(response).strip().split()

    exact = 1.0 if response_tokens == expected_tokens else 0.0

    max_len = max(len(expected_tokens), len(response_tokens), 1)
    matches = sum(
        1 for a, b in zip(expected_tokens, response_tokens) if a == b
    )
    partial = matches / max_len

    return {
        "accuracy": exact,
        "partial_credit": round(partial, 4),
    }


# ===================================================================
# PARADIGM 3 -- OPERATION SPAN
# ===================================================================

class OperationSpanGenerator:
    """Procedural generator for the Operation Span (OSpan) task.

    Each *set* consists of 3-7 operation-letter pairs:
      1. An arithmetic verification problem (e.g. "Is (3 x 4) + 2 = 14?")
         -- the model answers YES or NO.
      2. A letter to remember.

    After all pairs in a set, the model must recall all letters in
    presentation order.

    Scoring uses *partial-credit unit scoring*: proportion of letters
    recalled in the correct serial position.
    """

    PARADIGM = "operation_span"
    DIMENSION = "working_memory"

    # Letters to use for recall (avoid I/O/Q to reduce ambiguity)
    _RECALL_LETTERS = list("BCDEFGHJKLMNPRSTVWXYZ")

    DIFFICULTY_MAP: Dict[str, Dict[str, Any]] = {
        "easy":   {"set_sizes": [3, 4],       "num_sets_per_size": 3},
        "medium": {"set_sizes": [3, 4, 5, 6], "num_sets_per_size": 3},
        "hard":   {"set_sizes": [4, 5, 6, 7], "num_sets_per_size": 3},
    }

    # ------------------------------------------------------------------
    # Arithmetic problem generation
    # ------------------------------------------------------------------

    @classmethod
    def _generate_arithmetic(
        cls, rng: random.Random
    ) -> tuple[str, bool]:
        """Generate an arithmetic verification problem.

        Returns (problem_string, correct_answer_is_yes).
        """
        # Pick an operation pattern
        pattern = rng.choice(["mul_add", "mul_sub", "add_mul", "div_add"])

        if pattern == "mul_add":
            a = rng.randint(1, 9)
            b = rng.randint(1, 9)
            c = rng.randint(1, 9)
            correct = a * b + c
            expr = f"({a} x {b}) + {c}"
        elif pattern == "mul_sub":
            a = rng.randint(2, 9)
            b = rng.randint(1, 9)
            c = rng.randint(1, min(a * b - 1, 9))
            correct = a * b - c
            expr = f"({a} x {b}) - {c}"
        elif pattern == "add_mul":
            a = rng.randint(1, 9)
            b = rng.randint(1, 9)
            c = rng.randint(1, 5)
            correct = (a + b) * c
            expr = f"({a} + {b}) x {c}"
        else:  # div_add
            # ensure clean division
            b = rng.randint(1, 9)
            divisor = rng.randint(2, 5)
            a = b * divisor
            c = rng.randint(1, 9)
            correct = a // divisor + c
            expr = f"({a} / {divisor}) + {c}"

        # Decide whether to present the correct or incorrect answer
        show_correct = rng.random() < 0.5
        if show_correct:
            presented = correct
        else:
            # Offset by +/-1 or +/-2, avoiding the correct answer
            offset = rng.choice([-2, -1, 1, 2])
            presented = correct + offset
            if presented == correct:
                presented = correct + 1

        problem = f"Is {expr} = {presented}?"
        answer_is_yes = presented == correct
        return problem, answer_is_yes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def generate(
        cls,
        seed: int,
        n_items: int = 15,
        difficulty: str = "medium",
        contamination_probe: bool = False,
    ) -> list[TaskInstance]:
        """Generate Operation Span sets.

        Each ``TaskInstance`` represents one set (multi-turn episode).
        Trial-level data is stored in ``metadata.parameters["turns"]``.
        The final turn is the recall prompt.

        Parameters
        ----------
        seed : int
            Base random seed.
        n_items : int
            Total number of sets to generate (distributed across set sizes).
        difficulty : str
            Controls set sizes.
        contamination_probe : bool
            If True, uses classic letters A-E in fixed order as recall
            targets (unrealistic but tests memorisation from training data).
        """
        params = dict(cls.DIFFICULTY_MAP.get(difficulty, cls.DIFFICULTY_MAP["medium"]))
        set_sizes: list[int] = params["set_sizes"]
        num_sets_per_size: int = params["num_sets_per_size"]

        rng = _make_rng(seed)
        items: list[TaskInstance] = []

        # Distribute n_items across set_sizes
        items_per_size = max(1, n_items // len(set_sizes))
        remainder = n_items - items_per_size * len(set_sizes)

        for size_idx, set_size in enumerate(set_sizes):
            count = items_per_size + (1 if size_idx < remainder else 0)

            for _ in range(count):
                ep_seed = rng.randint(0, 2**31)
                ep_rng = _make_rng(ep_seed)

                # Pick recall letters
                if contamination_probe:
                    letters = list(_CLASSIC_LETTERS[:set_size])
                else:
                    letters = ep_rng.sample(cls._RECALL_LETTERS, set_size)

                turns: list[Dict[str, Any]] = []
                math_problems: list[Dict[str, Any]] = []

                for i in range(set_size):
                    problem, answer_is_yes = cls._generate_arithmetic(ep_rng)
                    letter = letters[i]

                    turn_stimulus = (
                        f"{problem} YES or NO.\n"
                        f"Remember the letter: {letter}"
                    )
                    turns.append({
                        "position": i,
                        "type": "operation_letter",
                        "stimulus": turn_stimulus,
                        "math_problem": problem,
                        "math_expected": "YES" if answer_is_yes else "NO",
                        "expected": "YES" if answer_is_yes else "NO",
                        "recall_letter": letter,
                    })
                    math_problems.append({
                        "problem": problem,
                        "correct": "YES" if answer_is_yes else "NO",
                    })

                # Final recall turn
                recall_expected = " ".join(letters)
                turns.append({
                    "position": set_size,
                    "type": "recall",
                    "stimulus": (
                        "Now recall ALL the letters you were asked to remember, "
                        "in the order they were presented.\n"
                        "Your response (letters separated by spaces):"
                    ),
                    "expected": recall_expected,
                })

                # Overview stimulus
                stimulus_text = (
                    f"Operation Span Task (set size = {set_size}).\n"
                    f"You will see {set_size} items. For each item:\n"
                    f"  1. Verify an arithmetic equation (answer YES or NO).\n"
                    f"  2. Remember the letter shown.\n"
                    f"After all {set_size} items, recall all letters in order.\n\n"
                    f"Item 1: {turns[0]['stimulus']}"
                )

                task_id = (
                    f"wm_ospan_{'probe' if contamination_probe else 'gen'}"
                    f"_{difficulty}_sz{set_size}_s{ep_seed}"
                )

                metadata = TaskMetadata(
                    dimension=cls.DIMENSION,
                    paradigm=cls.PARADIGM,
                    mode=EvalMode.AGENT_INTERACTIVE,
                    parameters={
                        "set_size": set_size,
                        "set_sizes": set_sizes,
                        "num_sets_per_size": num_sets_per_size,
                        "letters": letters,
                        "math_problems": math_problems,
                        "episode_seed": ep_seed,
                        "contamination_probe": contamination_probe,
                        "multi_turn": True,
                        "turns": turns,
                    },
                    scoring=ScoringConfig(
                        method="custom",
                        params={
                            "fn": "cogarena.dimensions.working_memory.score_ospan",
                            "set_size": set_size,
                        },
                    ),
                    difficulty=_difficulty_enum(difficulty),
                    adaptation_distance=AdaptationDistance.LOW,
                    description=(
                        f"Operation span, set size {set_size}, "
                        f"{'classic' if contamination_probe else 'procedural'} letters"
                    ),
                )

                items.append(TaskInstance(
                    task_id=task_id,
                    metadata=metadata,
                    stimulus=stimulus_text,
                    expected_response=recall_expected,
                ))

        return items

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score(
        task: TaskInstance, responses: list[str]
    ) -> Dict[str, float]:
        """Score a completed Operation Span set.

        Parameters
        ----------
        task : TaskInstance
            The generated OSpan set.
        responses : list[str]
            Model responses -- one per turn.  For operation-letter turns
            the response should contain YES/NO (and possibly the letter,
            which is ignored).  The final response is the recall string.

        Returns
        -------
        dict with: recall_partial_credit (=accuracy), recall_exact_match,
        math_accuracy, set_size.
        """
        turns = task.metadata.parameters["turns"]
        set_size: int = task.metadata.parameters["set_size"]

        # --- Math verification accuracy ---
        math_correct = 0
        math_total = 0
        for turn, resp in zip(turns, responses):
            if turn["type"] == "operation_letter":
                math_total += 1
                resp_upper = resp.strip().upper()
                expected_math = turn["math_expected"]
                if expected_math == "YES" and "YES" in resp_upper:
                    math_correct += 1
                elif expected_math == "NO" and "NO" in resp_upper:
                    math_correct += 1

        math_accuracy = math_correct / max(math_total, 1)

        # --- Letter recall (partial-credit unit scoring) ---
        # The last response should be the recall
        recall_response = (
            responses[-1].strip().upper().split() if responses else []
        )
        expected_letters = task.metadata.parameters["letters"]

        # Partial credit: proportion of letters in correct serial position
        correct_positions = 0
        for i, expected_letter in enumerate(expected_letters):
            if (
                i < len(recall_response)
                and recall_response[i].upper() == expected_letter.upper()
            ):
                correct_positions += 1

        partial_credit = correct_positions / max(len(expected_letters), 1)
        exact_match = (
            1.0
            if recall_response == [lt.upper() for lt in expected_letters]
            else 0.0
        )

        return {
            "accuracy": round(partial_credit, 4),  # primary metric
            "recall_partial_credit": round(partial_credit, 4),
            "recall_exact_match": exact_match,
            "recall_correct_positions": float(correct_positions),
            "math_accuracy": round(math_accuracy, 4),
            "math_correct": float(math_correct),
            "math_total": float(math_total),
            "set_size": float(set_size),
        }


# Module-level scoring function referenced by ScoringConfig custom fn path
def score_ospan(
    response: Any,
    expected: Any,
    metadata: TaskMetadata,
) -> Dict[str, float]:
    """Scoring function for Operation Span (used by TaskInstance.score).

    For the custom scoring path, ``response`` is expected to be the recall
    string (the last turn's response).  Full multi-turn scoring should use
    ``OperationSpanGenerator.score()`` directly.
    """
    expected_letters = metadata.parameters["letters"]
    recall_tokens = str(response).strip().upper().split()

    correct_positions = 0
    for i, exp_letter in enumerate(expected_letters):
        if i < len(recall_tokens) and recall_tokens[i] == exp_letter.upper():
            correct_positions += 1

    partial_credit = correct_positions / max(len(expected_letters), 1)
    exact_match = (
        1.0
        if recall_tokens == [lt.upper() for lt in expected_letters]
        else 0.0
    )

    return {
        "accuracy": round(partial_credit, 4),
        "recall_partial_credit": round(partial_credit, 4),
        "recall_exact_match": exact_match,
    }


# ===================================================================
# Convenience dispatch
# ===================================================================

_GENERATORS: Dict[str, type] = {
    "n_back": NBackGenerator,
    "digit_span": DigitSpanGenerator,
    "operation_span": OperationSpanGenerator,
}


def generate(
    paradigm: str,
    seed: int,
    n_items: int = 10,
    difficulty: str = "medium",
    contamination_probe: bool = False,
) -> list[TaskInstance]:
    """Unified entry-point for generating Working Memory items.

    Parameters
    ----------
    paradigm : str
        One of "n_back", "digit_span", "operation_span".
    seed, n_items, difficulty, contamination_probe
        Forwarded to the paradigm generator.
    """
    gen_cls = _GENERATORS.get(paradigm)
    if gen_cls is None:
        raise ValueError(
            f"Unknown paradigm '{paradigm}'. Choose from {list(_GENERATORS)}"
        )
    return gen_cls.generate(
        seed=seed,
        n_items=n_items,
        difficulty=difficulty,
        contamination_probe=contamination_probe,
    )


def score(task: TaskInstance, response: Any) -> Dict[str, float]:
    """Unified scoring dispatcher.

    *response* should be:
      - ``list[str]`` for multi-turn paradigms (n_back, operation_span)
      - ``str`` for single-turn paradigms (digit_span)
    """
    gen_cls = _GENERATORS.get(task.metadata.paradigm)
    if gen_cls is None:
        raise ValueError(f"Unknown paradigm '{task.metadata.paradigm}'")
    return gen_cls.score(task, response)
