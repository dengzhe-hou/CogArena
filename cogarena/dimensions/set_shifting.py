"""Set Shifting / Cognitive Flexibility dimension for CogArena.

Implements two paradigms with procedural generation:
  1. Wisconsin Card Sorting Test (WCST) -- multi-turn rule discovery & shifting
  2. Reversal Learning                  -- multi-turn contingency tracking

Both paradigms use NON-STANDARD dimension names and values to minimise
contamination from training corpora (the classic colour/shape/number WCST
is only generated for contamination probes).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

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


def _make_rng(seed: int) -> random.Random:
    """Return a seeded Random instance (reproducible, thread-safe)."""
    return random.Random(seed)


def _difficulty_enum(s: str) -> DifficultyLevel:
    return DifficultyLevel(s.lower())


# ---------------------------------------------------------------------------
# Non-standard dimension pools (contamination avoidance)
# ---------------------------------------------------------------------------

# Each entry is (dimension_name, [value1, value2, value3, value4])
_NONSTANDARD_DIM_POOLS: List[Tuple[str, List[str]]] = [
    ("texture",  ["smooth", "rough", "striped", "dotted"]),
    ("size",     ["tiny", "small", "medium", "large"]),
    ("border",   ["none", "thin", "thick", "double"]),
    ("weight",   ["feather", "light", "heavy", "anchor"]),
    ("opacity",  ["transparent", "translucent", "frosted", "opaque"]),
    ("pattern",  ["solid", "checkered", "zigzag", "spiral"]),
    ("material", ["glass", "wood", "metal", "stone"]),
    ("finish",   ["matte", "satin", "glossy", "mirror"]),
    ("edge",     ["rounded", "beveled", "sharp", "wavy"]),
    ("density",  ["sparse", "scattered", "packed", "dense"]),
]

# Classic WCST dimensions -- ONLY for contamination probes
_CLASSIC_DIMS: List[Tuple[str, List[str]]] = [
    ("color",  ["red", "blue", "green", "yellow"]),
    ("shape",  ["circle", "triangle", "square", "star"]),
    ("number", ["1", "2", "3", "4"]),
]

# Non-standard option label pools for Reversal Learning
_OPTION_LABEL_POOL: List[Tuple[str, str]] = [
    ("Circle Path", "Square Path"),
    ("River Route", "Mountain Route"),
    ("Dawn Gate", "Dusk Gate"),
    ("Iron Door", "Copper Door"),
    ("North Trail", "South Trail"),
    ("Amber Lever", "Jade Lever"),
    ("Falcon Nest", "Raven Nest"),
    ("Pine Bridge", "Oak Bridge"),
    ("Frost Lane", "Ember Lane"),
    ("Silver Key", "Bronze Key"),
]


# ===================================================================
# PARADIGM 1 -- Wisconsin Card Sorting Test (WCST)
# ===================================================================

class WCSTGenerator:
    """Procedural generator for a WCST-like card sorting task (multi-turn).

    On each turn the model sees one target card and four reference cards.
    It must choose which reference card the target matches by the *current*
    hidden rule (one of the card dimensions).  Feedback is given after each
    choice ("CORRECT" or "WRONG").  The rule switches (without announcement)
    after N consecutive correct responses.

    Non-standard dimension names and values are used by default to reduce
    contamination from LLM training data.
    """

    PARADIGM = "wcst"
    DIMENSION = "set_shifting"

    DIFFICULTY_MAP: Dict[str, Dict[str, Any]] = {
        "easy": {
            "n_rules": 3,
            "consecutive_correct_to_switch": 10,
            "max_trials": 48,
        },
        "medium": {
            "n_rules": 4,
            "consecutive_correct_to_switch": 10,
            "max_trials": 64,
        },
        "hard": {
            "n_rules": 6,
            "consecutive_correct_to_switch": 10,
            "max_trials": 80,
        },
    }

    # ------------------------------------------------------------------
    # Card helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_dimensions(
        rng: random.Random,
        n_dims: int,
        contamination_probe: bool,
    ) -> List[Tuple[str, List[str]]]:
        """Select *n_dims* dimension definitions to use for this episode.

        For contamination probes, use the classic colour/shape/number set.
        Otherwise, sample from the non-standard pool.
        """
        if contamination_probe:
            return list(_CLASSIC_DIMS[:n_dims])
        selected = rng.sample(_NONSTANDARD_DIM_POOLS, min(n_dims, len(_NONSTANDARD_DIM_POOLS)))
        return selected

    @staticmethod
    def _make_card(
        rng: random.Random,
        dims: List[Tuple[str, List[str]]],
    ) -> Dict[str, str]:
        """Generate a random card with one value per dimension."""
        return {dim_name: rng.choice(values) for dim_name, values in dims}

    @staticmethod
    def _card_str(card: Dict[str, str]) -> str:
        """Human-readable description of a card."""
        parts = [f"{dim}={val}" for dim, val in card.items()]
        return "[" + ", ".join(parts) + "]"

    @classmethod
    def _make_reference_cards(
        cls,
        rng: random.Random,
        dims: List[Tuple[str, List[str]]],
    ) -> List[Dict[str, str]]:
        """Generate 4 reference cards.

        Each reference card has a *unique* value for every dimension so that
        matching on any single dimension yields exactly one reference card.
        We achieve this by assigning one of the 4 values per dimension in a
        permuted order.
        """
        n_vals = 4  # always 4 reference cards
        ref_cards: List[Dict[str, str]] = [{} for _ in range(n_vals)]
        for dim_name, values in dims:
            # Ensure we have exactly 4 values; pad or trim
            pool = list(values)
            while len(pool) < n_vals:
                pool.append(pool[-1] + "'")
            pool = pool[:n_vals]
            perm = list(pool)
            rng.shuffle(perm)
            for i in range(n_vals):
                ref_cards[i][dim_name] = perm[i]
        return ref_cards

    @classmethod
    def _make_target_card(
        cls,
        rng: random.Random,
        ref_cards: List[Dict[str, str]],
        dims: List[Tuple[str, List[str]]],
    ) -> Dict[str, str]:
        """Generate a target card that matches exactly one reference card per
        dimension (and ideally different reference cards for different dimensions).

        For each dimension, pick a value that equals one of the reference
        cards' values in that dimension (chosen at random), but try to ensure
        that the matching reference differs across dimensions.
        """
        n_refs = len(ref_cards)
        dim_names = [d[0] for d in dims]

        # For each dimension, build a map: value -> which ref index has it
        dim_val_to_ref: Dict[str, Dict[str, int]] = {}
        for dim_name in dim_names:
            mapping: Dict[str, int] = {}
            for ri, rc in enumerate(ref_cards):
                mapping[rc[dim_name]] = ri
            dim_val_to_ref[dim_name] = mapping

        # Assign targets: try to pick distinct reference matches per dimension
        ref_indices = list(range(n_refs))
        rng.shuffle(ref_indices)

        card: Dict[str, str] = {}
        for di, dim_name in enumerate(dim_names):
            target_ref_idx = ref_indices[di % n_refs]
            card[dim_name] = ref_cards[target_ref_idx][dim_name]

        return card

    @classmethod
    def _find_match(
        cls,
        target: Dict[str, str],
        ref_cards: List[Dict[str, str]],
        rule_dim: str,
    ) -> int:
        """Return the index of the reference card that matches the target
        on the given rule dimension."""
        for i, rc in enumerate(ref_cards):
            if rc[rule_dim] == target[rule_dim]:
                return i
        # Fallback -- should not happen with well-constructed cards
        return 0

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
    ) -> List[TaskInstance]:
        """Generate *n_items* WCST episodes.

        Each episode is a multi-turn sequence.  Turn-level data is stored
        in ``metadata.parameters["turns"]``.

        Parameters
        ----------
        seed : int
            Base random seed.
        n_items : int
            Number of independent WCST episodes to generate.
        difficulty : "easy" | "medium" | "hard"
            Controls n_rules and max_trials.
        contamination_probe : bool
            If True, use classic color/shape/number dimensions.
        """
        params = dict(cls.DIFFICULTY_MAP.get(difficulty, cls.DIFFICULTY_MAP["medium"]))
        n_rules: int = params["n_rules"]
        consec_to_switch: int = params["consecutive_correct_to_switch"]
        max_trials: int = params["max_trials"]

        rng = _make_rng(seed)
        items: List[TaskInstance] = []

        for idx in range(n_items):
            ep_seed = rng.randint(0, 2**31)
            ep_rng = _make_rng(ep_seed)

            # Pick dimensions for this episode (at least 3)
            n_dims = max(3, min(n_rules, len(_NONSTANDARD_DIM_POOLS)))
            dims = cls._pick_dimensions(ep_rng, n_dims, contamination_probe)
            dim_names = [d[0] for d in dims]

            # Build the rule sequence (cycle through dimensions)
            rule_sequence: List[str] = []
            available_dims = list(dim_names)
            ep_rng.shuffle(available_dims)
            for ri in range(n_rules):
                rule_sequence.append(available_dims[ri % len(available_dims)])

            # Generate trials
            turns: List[Dict[str, Any]] = []
            current_rule_idx = 0
            current_rule = rule_sequence[0]
            consecutive_correct = 0
            categories_completed = 0

            for trial_num in range(max_trials):
                if current_rule_idx >= len(rule_sequence):
                    # All rules exhausted -- pad remaining trials with last rule
                    current_rule = rule_sequence[-1]

                # Generate reference cards (new set each trial for variety)
                ref_cards = cls._make_reference_cards(ep_rng, dims)

                # Generate target card
                target = cls._make_target_card(ep_rng, ref_cards, dims)

                # Find the correct answer under the current rule
                correct_ref_idx = cls._find_match(target, ref_cards, current_rule)

                # Also compute what the *previous* rule would have said
                # (for perseverative error detection)
                prev_rule_idx = min(current_rule_idx - 1, len(rule_sequence) - 1)
                prev_rule = rule_sequence[prev_rule_idx] if current_rule_idx > 0 else None
                prev_rule_ref_idx = None
                if prev_rule is not None:
                    prev_rule_ref_idx = cls._find_match(target, ref_cards, prev_rule)

                turns.append({
                    "trial": trial_num,
                    "target_card": target,
                    "reference_cards": ref_cards,
                    "current_rule": current_rule,
                    "current_rule_idx": current_rule_idx,
                    "correct_ref_idx": correct_ref_idx,
                    "prev_rule": prev_rule,
                    "prev_rule_ref_idx": prev_rule_ref_idx,
                    "expected": str(correct_ref_idx + 1),  # 1-indexed for human
                    "feedback": None,  # filled during scoring
                    "consecutive_correct_before": consecutive_correct,
                })

                # Simulate what happens after the model answers correctly
                # (for seeding the rule-switch schedule in advance)
                # We store the rule-switch points so scoring can track them
                consecutive_correct += 1
                if consecutive_correct >= consec_to_switch:
                    categories_completed += 1
                    current_rule_idx += 1
                    if current_rule_idx < len(rule_sequence):
                        current_rule = rule_sequence[current_rule_idx]
                    consecutive_correct = 0

            # Stimulus text (overview + first trial)
            dim_desc = ", ".join(
                f"{dn} ({'/'.join(dv)})" for dn, dv in dims
            )
            first_target_str = cls._card_str(turns[0]["target_card"])
            ref_strs = [
                f"  {i+1}. {cls._card_str(rc)}"
                for i, rc in enumerate(turns[0]["reference_cards"])
            ]
            ref_block = "\n".join(ref_strs)

            stimulus_text = (
                f"Card Sorting Task.\n"
                f"Cards have the following dimensions: {dim_desc}.\n\n"
                f"On each turn you will see a target card and 4 reference cards.\n"
                f"You must figure out the CURRENT sorting rule (which single "
                f"dimension to match on) by trial and error.\n"
                f"After each choice you will receive feedback: CORRECT or WRONG.\n"
                f"The sorting rule may CHANGE without warning after several "
                f"consecutive correct answers.\n\n"
                f"Respond with ONLY the reference card number (1, 2, 3, or 4).\n\n"
                f"--- Trial 1 ---\n"
                f"Target card: {first_target_str}\n"
                f"Reference cards:\n{ref_block}\n"
                f"Your choice:"
            )

            task_id = (
                f"ss_wcst_{'probe' if contamination_probe else 'gen'}"
                f"_{difficulty}_r{n_rules}_s{ep_seed}"
            )

            metadata = TaskMetadata(
                dimension=cls.DIMENSION,
                paradigm=cls.PARADIGM,
                mode=EvalMode.AGENT_INTERACTIVE,
                parameters={
                    "n_rules": n_rules,
                    "consecutive_correct_to_switch": consec_to_switch,
                    "max_trials": max_trials,
                    "n_dims": n_dims,
                    "dim_names": dim_names,
                    "dims": [(dn, dv) for dn, dv in dims],
                    "rule_sequence": rule_sequence,
                    "episode_seed": ep_seed,
                    "contamination_probe": contamination_probe,
                    "multi_turn": True,
                    "turns": turns,
                },
                scoring=ScoringConfig(
                    method="custom",
                    params={
                        "fn": "cogarena.dimensions.set_shifting.score_wcst",
                        "n_rules": n_rules,
                        "consec_to_switch": consec_to_switch,
                    },
                ),
                difficulty=_difficulty_enum(difficulty),
                adaptation_distance=(
                    AdaptationDistance.LOW if contamination_probe
                    else AdaptationDistance.MEDIUM
                ),
                description=(
                    f"WCST with {n_rules} rules, "
                    f"{'classic' if contamination_probe else 'procedural'} dimensions"
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
    def score(task: TaskInstance, responses: List[str]) -> Dict[str, float]:
        """Score a completed WCST episode.

        Parameters
        ----------
        task : TaskInstance
            The generated WCST episode.
        responses : list[str]
            Model responses, one per turn.  Each should be "1", "2", "3",
            or "4".

        Returns
        -------
        dict with keys: accuracy, categories_completed, perseverative_errors,
        total_errors, trials_to_first_category, perseverative_error_rate,
        n_trials.
        """
        turns = task.metadata.parameters["turns"]
        consec_to_switch: int = task.metadata.parameters["consecutive_correct_to_switch"]
        rule_sequence: List[str] = task.metadata.parameters["rule_sequence"]

        # Pad or truncate responses to match turns
        resps = list(responses)
        while len(resps) < len(turns):
            resps.append("")
        resps = resps[:len(turns)]

        # Replay the episode with actual model responses
        current_rule_idx = 0
        consecutive_correct = 0
        categories_completed = 0
        perseverative_errors = 0
        total_errors = 0
        total_correct = 0
        trials_to_first_category: Optional[int] = None

        for i, (turn, resp) in enumerate(zip(turns, resps)):
            # Parse response
            resp_clean = resp.strip()
            # Extract first digit 1-4
            chosen = None
            for ch in resp_clean:
                if ch in "1234":
                    chosen = int(ch)
                    break

            if chosen is None:
                # Invalid response counts as error
                total_errors += 1
                consecutive_correct = 0
                continue

            # Determine the correct answer under current rule
            current_rule = rule_sequence[min(current_rule_idx, len(rule_sequence) - 1)]

            # Recompute correct_ref_idx using actual rule tracking
            target = turn["target_card"]
            ref_cards = turn["reference_cards"]
            correct_idx = None
            for ri, rc in enumerate(ref_cards):
                if rc[current_rule] == target[current_rule]:
                    correct_idx = ri + 1  # 1-indexed
                    break
            if correct_idx is None:
                correct_idx = int(turn["expected"])

            is_correct = (chosen == correct_idx)

            if is_correct:
                total_correct += 1
                consecutive_correct += 1

                if consecutive_correct >= consec_to_switch:
                    categories_completed += 1
                    if trials_to_first_category is None:
                        trials_to_first_category = i + 1
                    current_rule_idx += 1
                    consecutive_correct = 0
            else:
                total_errors += 1
                consecutive_correct = 0

                # Check if this is a perseverative error
                # (response matches the *previous* rule's correct answer)
                if current_rule_idx > 0:
                    prev_rule = rule_sequence[current_rule_idx - 1]
                    prev_correct = None
                    for ri, rc in enumerate(ref_cards):
                        if rc[prev_rule] == target[prev_rule]:
                            prev_correct = ri + 1
                            break
                    if prev_correct is not None and chosen == prev_correct:
                        perseverative_errors += 1

        n_trials = len(turns)
        accuracy = total_correct / max(n_trials, 1)
        perseverative_error_rate = perseverative_errors / max(total_errors, 1)

        if trials_to_first_category is None:
            trials_to_first_category = n_trials  # never completed first cat

        return {
            "accuracy": round(accuracy, 4),
            "categories_completed": float(categories_completed),
            "perseverative_errors": float(perseverative_errors),
            "total_errors": float(total_errors),
            "total_correct": float(total_correct),
            "trials_to_first_category": float(trials_to_first_category),
            "perseverative_error_rate": round(perseverative_error_rate, 4),
            "n_trials": float(n_trials),
        }


# Module-level scoring function referenced by ScoringConfig custom fn path
def score_wcst(
    response: Any,
    expected: Any,
    metadata: TaskMetadata,
) -> Dict[str, float]:
    """Scoring function for WCST (used by TaskInstance.score via custom fn).

    ``response`` should be a list[str] of per-turn answers.
    ``expected`` should be a list[str] of expected per-turn answers.
    """
    turns = metadata.parameters["turns"]
    consec_to_switch: int = metadata.parameters["consecutive_correct_to_switch"]
    rule_sequence: List[str] = metadata.parameters["rule_sequence"]

    if isinstance(response, str):
        responses = [response]
    else:
        responses = list(response)

    # Pad or truncate
    while len(responses) < len(turns):
        responses.append("")
    responses = responses[:len(turns)]

    current_rule_idx = 0
    consecutive_correct = 0
    categories_completed = 0
    perseverative_errors = 0
    total_errors = 0
    total_correct = 0
    trials_to_first_category: Optional[int] = None

    for i, (turn, resp) in enumerate(zip(turns, responses)):
        resp_clean = resp.strip()
        chosen = None
        for ch in resp_clean:
            if ch in "1234":
                chosen = int(ch)
                break

        if chosen is None:
            total_errors += 1
            consecutive_correct = 0
            continue

        current_rule = rule_sequence[min(current_rule_idx, len(rule_sequence) - 1)]
        target = turn["target_card"]
        ref_cards = turn["reference_cards"]

        correct_idx = None
        for ri, rc in enumerate(ref_cards):
            if rc[current_rule] == target[current_rule]:
                correct_idx = ri + 1
                break
        if correct_idx is None:
            correct_idx = int(turn["expected"])

        is_correct = (chosen == correct_idx)

        if is_correct:
            total_correct += 1
            consecutive_correct += 1
            if consecutive_correct >= consec_to_switch:
                categories_completed += 1
                if trials_to_first_category is None:
                    trials_to_first_category = i + 1
                current_rule_idx += 1
                consecutive_correct = 0
        else:
            total_errors += 1
            consecutive_correct = 0

            if current_rule_idx > 0:
                prev_rule = rule_sequence[current_rule_idx - 1]
                prev_correct = None
                for ri, rc in enumerate(ref_cards):
                    if rc[prev_rule] == target[prev_rule]:
                        prev_correct = ri + 1
                        break
                if prev_correct is not None and chosen == prev_correct:
                    perseverative_errors += 1

    n_trials = len(turns)
    accuracy = total_correct / max(n_trials, 1)
    perseverative_error_rate = perseverative_errors / max(total_errors, 1)

    if trials_to_first_category is None:
        trials_to_first_category = n_trials

    return {
        "accuracy": round(accuracy, 4),
        "categories_completed": float(categories_completed),
        "perseverative_errors": float(perseverative_errors),
        "total_errors": float(total_errors),
        "perseverative_error_rate": round(perseverative_error_rate, 4),
    }


# ===================================================================
# PARADIGM 2 -- Reversal Learning
# ===================================================================

class ReversalLearningGenerator:
    """Procedural generator for a probabilistic reversal learning task.

    Two options (with procedurally generated labels).  One option is
    rewarded with probability *reward_prob* (e.g. 0.8), the other with
    *1 - reward_prob* (e.g. 0.2).  After a fixed number of trials per
    phase, the contingencies reverse.

    The task is multi-turn: each turn presents the two options, the model
    chooses one, and receives feedback ("REWARDED" or "NOT REWARDED").
    """

    PARADIGM = "reversal_learning"
    DIMENSION = "set_shifting"

    DIFFICULTY_MAP: Dict[str, Dict[str, Any]] = {
        "easy": {
            "n_trials_per_phase": 20,
            "n_reversals": 2,
            "reward_prob": 0.9,
        },
        "medium": {
            "n_trials_per_phase": 25,
            "n_reversals": 3,
            "reward_prob": 0.8,
        },
        "hard": {
            "n_trials_per_phase": 30,
            "n_reversals": 4,
            "reward_prob": 0.7,
        },
    }

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
    ) -> List[TaskInstance]:
        """Generate *n_items* Reversal Learning episodes.

        Each episode is a multi-turn sequence.  Turn-level data is stored
        in ``metadata.parameters["turns"]``.

        Parameters
        ----------
        seed : int
            Base random seed.
        n_items : int
            Number of independent episodes to generate.
        difficulty : "easy" | "medium" | "hard"
        contamination_probe : bool
            If True, use generic "A"/"B" labels instead of procedural names.
        """
        params = dict(cls.DIFFICULTY_MAP.get(difficulty, cls.DIFFICULTY_MAP["medium"]))
        n_trials_per_phase: int = params["n_trials_per_phase"]
        n_reversals: int = params["n_reversals"]
        reward_prob: float = params["reward_prob"]

        rng = _make_rng(seed)
        items: List[TaskInstance] = []

        for idx in range(n_items):
            ep_seed = rng.randint(0, 2**31)
            ep_rng = _make_rng(ep_seed)

            # Pick option labels
            if contamination_probe:
                option_a, option_b = "A", "B"
            else:
                label_pair = ep_rng.choice(_OPTION_LABEL_POOL)
                option_a, option_b = label_pair

            # Total trials = initial phase + n_reversals * n_trials_per_phase
            total_trials = n_trials_per_phase * (n_reversals + 1)

            # Build reward schedule
            # Phase 0: A is high, B is low
            # Phase 1 (first reversal): B is high, A is low
            # Phase 2: A is high again, etc.
            turns: List[Dict[str, Any]] = []
            phase_boundaries: List[int] = []  # trial index of each reversal

            for trial_num in range(total_trials):
                phase = trial_num // n_trials_per_phase
                trial_in_phase = trial_num % n_trials_per_phase

                if trial_in_phase == 0 and phase > 0:
                    phase_boundaries.append(trial_num)

                # Determine which option is "good" this phase
                if phase % 2 == 0:
                    good_option = option_a
                    bad_option = option_b
                else:
                    good_option = option_b
                    bad_option = option_a

                # Pre-generate reward outcomes for both choices
                roll_a = ep_rng.random()
                roll_b = ep_rng.random()

                if good_option == option_a:
                    reward_if_a = roll_a < reward_prob
                    reward_if_b = roll_b < (1.0 - reward_prob)
                else:
                    reward_if_a = roll_a < (1.0 - reward_prob)
                    reward_if_b = roll_b < reward_prob

                turns.append({
                    "trial": trial_num,
                    "phase": phase,
                    "trial_in_phase": trial_in_phase,
                    "good_option": good_option,
                    "bad_option": bad_option,
                    "option_a": option_a,
                    "option_b": option_b,
                    "reward_if_a": reward_if_a,
                    "reward_if_b": reward_if_b,
                    "expected": good_option,  # optimal choice
                })

            # Stimulus text
            stimulus_text = (
                f"Reversal Learning Task.\n"
                f"On each trial you must choose between two options: "
                f'"{option_a}" or "{option_b}".\n'
                f"One option is more likely to be rewarded than the other, "
                f"but the reward contingencies may REVERSE at some point.\n"
                f"After each choice you will be told: REWARDED or NOT REWARDED.\n"
                f"Your goal is to maximise total reward.\n\n"
                f'Respond with ONLY the option name (either "{option_a}" or '
                f'"{option_b}").\n\n'
                f"--- Trial 1 ---\n"
                f'Choose: "{option_a}" or "{option_b}"'
            )

            task_id = (
                f"ss_reversal_{'probe' if contamination_probe else 'gen'}"
                f"_{difficulty}_rev{n_reversals}_s{ep_seed}"
            )

            metadata = TaskMetadata(
                dimension=cls.DIMENSION,
                paradigm=cls.PARADIGM,
                mode=EvalMode.AGENT_INTERACTIVE,
                parameters={
                    "n_trials_per_phase": n_trials_per_phase,
                    "n_reversals": n_reversals,
                    "reward_prob": reward_prob,
                    "option_a": option_a,
                    "option_b": option_b,
                    "total_trials": total_trials,
                    "phase_boundaries": phase_boundaries,
                    "episode_seed": ep_seed,
                    "contamination_probe": contamination_probe,
                    "multi_turn": True,
                    "turns": turns,
                },
                scoring=ScoringConfig(
                    method="custom",
                    params={
                        "fn": "cogarena.dimensions.set_shifting.score_reversal",
                        "n_reversals": n_reversals,
                        "n_trials_per_phase": n_trials_per_phase,
                    },
                ),
                difficulty=_difficulty_enum(difficulty),
                adaptation_distance=(
                    AdaptationDistance.LOW if contamination_probe
                    else AdaptationDistance.MEDIUM
                ),
                description=(
                    f"Reversal learning with {n_reversals} reversals, "
                    f"p={reward_prob}, "
                    f"{'classic' if contamination_probe else 'procedural'} labels"
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
    def score(task: TaskInstance, responses: List[str]) -> Dict[str, float]:
        """Score a completed Reversal Learning episode.

        Parameters
        ----------
        task : TaskInstance
            The generated Reversal Learning episode.
        responses : list[str]
            Model responses, one per turn.

        Returns
        -------
        dict with keys: accuracy, accuracy_per_phase (list),
        switch_cost, trials_to_criterion_after_reversal,
        win_stay_rate, lose_shift_rate, total_reward.
        """
        turns = task.metadata.parameters["turns"]
        option_a: str = task.metadata.parameters["option_a"]
        option_b: str = task.metadata.parameters["option_b"]
        n_trials_per_phase: int = task.metadata.parameters["n_trials_per_phase"]
        n_reversals: int = task.metadata.parameters["n_reversals"]

        resps = list(responses)
        while len(resps) < len(turns):
            resps.append("")
        resps = resps[:len(turns)]

        # Parse responses and track outcomes
        choices: List[Optional[str]] = []
        rewards: List[bool] = []
        optimal_choices: List[bool] = []

        for turn, resp in zip(turns, resps):
            resp_clean = resp.strip()

            # Determine which option the model chose
            chosen = None
            resp_lower = resp_clean.lower()
            opt_a_lower = option_a.lower()
            opt_b_lower = option_b.lower()

            if opt_a_lower in resp_lower and opt_b_lower not in resp_lower:
                chosen = option_a
            elif opt_b_lower in resp_lower and opt_a_lower not in resp_lower:
                chosen = option_b
            elif opt_a_lower in resp_lower and opt_b_lower in resp_lower:
                # Both present -- pick whichever appears first
                idx_a = resp_lower.index(opt_a_lower)
                idx_b = resp_lower.index(opt_b_lower)
                chosen = option_a if idx_a < idx_b else option_b
            else:
                # Try simple A/B matching as fallback
                if resp_clean in (option_a, option_b):
                    chosen = resp_clean

            choices.append(chosen)

            # Determine reward
            if chosen == option_a:
                got_reward = turn["reward_if_a"]
            elif chosen == option_b:
                got_reward = turn["reward_if_b"]
            else:
                got_reward = False  # invalid choice gets no reward

            rewards.append(got_reward)
            optimal_choices.append(chosen == turn["good_option"])

        # --- Compute metrics ---

        # Overall accuracy (proportion of optimal choices)
        valid_choices = [o for c, o in zip(choices, optimal_choices) if c is not None]
        accuracy = sum(valid_choices) / max(len(valid_choices), 1)

        # Per-phase accuracy
        n_phases = n_reversals + 1
        phase_accuracies: List[float] = []
        for phase in range(n_phases):
            start = phase * n_trials_per_phase
            end = start + n_trials_per_phase
            phase_optimal = optimal_choices[start:end]
            phase_choices = choices[start:end]
            valid_in_phase = [
                o for c, o in zip(phase_choices, phase_optimal) if c is not None
            ]
            if valid_in_phase:
                phase_accuracies.append(sum(valid_in_phase) / len(valid_in_phase))
            else:
                phase_accuracies.append(0.0)

        # Switch cost: average accuracy drop in the first half of each
        # post-reversal phase compared to the second half of the preceding phase
        switch_costs: List[float] = []
        half = n_trials_per_phase // 2
        for rev_idx in range(n_reversals):
            # Second half of preceding phase
            prev_phase = rev_idx
            prev_start = prev_phase * n_trials_per_phase + half
            prev_end = prev_phase * n_trials_per_phase + n_trials_per_phase
            prev_opt = [
                o for c, o in zip(
                    choices[prev_start:prev_end],
                    optimal_choices[prev_start:prev_end],
                ) if c is not None
            ]
            prev_acc = sum(prev_opt) / max(len(prev_opt), 1)

            # First half of new phase
            new_phase = rev_idx + 1
            new_start = new_phase * n_trials_per_phase
            new_end = new_start + half
            new_opt = [
                o for c, o in zip(
                    choices[new_start:new_end],
                    optimal_choices[new_start:new_end],
                ) if c is not None
            ]
            new_acc = sum(new_opt) / max(len(new_opt), 1)

            switch_costs.append(prev_acc - new_acc)

        switch_cost = sum(switch_costs) / max(len(switch_costs), 1)

        # Trials to criterion after each reversal (5 consecutive optimal)
        criterion = 5
        trials_to_criterion: List[int] = []
        for rev_idx in range(n_reversals):
            phase = rev_idx + 1
            start = phase * n_trials_per_phase
            end = start + n_trials_per_phase
            consec = 0
            found = False
            for t in range(start, min(end, len(optimal_choices))):
                if choices[t] is not None and optimal_choices[t]:
                    consec += 1
                    if consec >= criterion:
                        trials_to_criterion.append(t - start + 1)
                        found = True
                        break
                else:
                    consec = 0
            if not found:
                trials_to_criterion.append(n_trials_per_phase)

        avg_trials_to_criterion = (
            sum(trials_to_criterion) / max(len(trials_to_criterion), 1)
        )

        # Win-Stay rate: P(same choice | previous trial rewarded)
        win_stay = 0
        win_total = 0
        for t in range(1, len(choices)):
            if choices[t] is None or choices[t - 1] is None:
                continue
            if rewards[t - 1]:
                win_total += 1
                if choices[t] == choices[t - 1]:
                    win_stay += 1
        win_stay_rate = win_stay / max(win_total, 1)

        # Lose-Shift rate: P(different choice | previous trial not rewarded)
        lose_shift = 0
        lose_total = 0
        for t in range(1, len(choices)):
            if choices[t] is None or choices[t - 1] is None:
                continue
            if not rewards[t - 1]:
                lose_total += 1
                if choices[t] != choices[t - 1]:
                    lose_shift += 1
        lose_shift_rate = lose_shift / max(lose_total, 1)

        total_reward = sum(1 for r in rewards if r)

        return {
            "accuracy": round(accuracy, 4),
            "accuracy_phase_0": round(phase_accuracies[0], 4) if phase_accuracies else 0.0,
            "accuracy_post_reversal_mean": round(
                sum(phase_accuracies[1:]) / max(len(phase_accuracies) - 1, 1), 4
            ) if len(phase_accuracies) > 1 else 0.0,
            "switch_cost": round(switch_cost, 4),
            "trials_to_criterion_after_reversal": round(avg_trials_to_criterion, 2),
            "win_stay_rate": round(win_stay_rate, 4),
            "lose_shift_rate": round(lose_shift_rate, 4),
            "total_reward": float(total_reward),
            "total_trials": float(len(turns)),
            "n_reversals": float(n_reversals),
        }


# Module-level scoring function referenced by ScoringConfig custom fn path
def score_reversal(
    response: Any,
    expected: Any,
    metadata: TaskMetadata,
) -> Dict[str, float]:
    """Scoring function for Reversal Learning (used by TaskInstance.score).

    ``response`` should be a list[str] of per-turn choices.
    """
    turns = metadata.parameters["turns"]
    option_a: str = metadata.parameters["option_a"]
    option_b: str = metadata.parameters["option_b"]
    n_trials_per_phase: int = metadata.parameters["n_trials_per_phase"]
    n_reversals: int = metadata.parameters["n_reversals"]

    if isinstance(response, str):
        responses = [response]
    else:
        responses = list(response)

    while len(responses) < len(turns):
        responses.append("")
    responses = responses[:len(turns)]

    # Parse choices
    choices: List[Optional[str]] = []
    rewards: List[bool] = []
    optimal_choices: List[bool] = []

    for turn, resp in zip(turns, responses):
        resp_lower = resp.strip().lower()
        opt_a_lower = option_a.lower()
        opt_b_lower = option_b.lower()

        chosen = None
        if opt_a_lower in resp_lower and opt_b_lower not in resp_lower:
            chosen = option_a
        elif opt_b_lower in resp_lower and opt_a_lower not in resp_lower:
            chosen = option_b
        elif opt_a_lower in resp_lower and opt_b_lower in resp_lower:
            idx_a = resp_lower.index(opt_a_lower)
            idx_b = resp_lower.index(opt_b_lower)
            chosen = option_a if idx_a < idx_b else option_b

        choices.append(chosen)

        if chosen == option_a:
            got_reward = turn["reward_if_a"]
        elif chosen == option_b:
            got_reward = turn["reward_if_b"]
        else:
            got_reward = False

        rewards.append(got_reward)
        optimal_choices.append(chosen == turn["good_option"])

    valid_choices = [o for c, o in zip(choices, optimal_choices) if c is not None]
    accuracy = sum(valid_choices) / max(len(valid_choices), 1)

    # Win-Stay
    win_stay = 0
    win_total = 0
    for t in range(1, len(choices)):
        if choices[t] is None or choices[t - 1] is None:
            continue
        if rewards[t - 1]:
            win_total += 1
            if choices[t] == choices[t - 1]:
                win_stay += 1
    win_stay_rate = win_stay / max(win_total, 1)

    # Lose-Shift
    lose_shift = 0
    lose_total = 0
    for t in range(1, len(choices)):
        if choices[t] is None or choices[t - 1] is None:
            continue
        if not rewards[t - 1]:
            lose_total += 1
            if choices[t] != choices[t - 1]:
                lose_shift += 1
    lose_shift_rate = lose_shift / max(lose_total, 1)

    return {
        "accuracy": round(accuracy, 4),
        "win_stay_rate": round(win_stay_rate, 4),
        "lose_shift_rate": round(lose_shift_rate, 4),
    }


# ===================================================================
# Convenience dispatch
# ===================================================================

_GENERATORS: Dict[str, type] = {
    "wcst": WCSTGenerator,
    "reversal_learning": ReversalLearningGenerator,
}


def generate(
    paradigm: str,
    seed: int,
    n_items: int = 10,
    difficulty: str = "medium",
    contamination_probe: bool = False,
) -> List[TaskInstance]:
    """Unified entry-point for generating Set Shifting items.

    Parameters
    ----------
    paradigm : str
        One of "wcst", "reversal_learning".
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

    *response* should be a ``list[str]`` (both paradigms are multi-turn).
    """
    gen_cls = _GENERATORS.get(task.metadata.paradigm)
    if gen_cls is None:
        raise ValueError(f"Unknown paradigm '{task.metadata.paradigm}'")
    return gen_cls.score(task, response)
