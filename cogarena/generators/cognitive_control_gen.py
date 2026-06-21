"""
Procedural generator for Cognitive Control / Inhibition dimension.

Produces a full battery of TaskInstance items across three paradigms
(Stroop, Flanker, Go/No-Go) with optional contamination probes.

Usage::

    from cogarena.generators.cognitive_control_gen import generate_cc_items
    items = generate_cc_items(seed=42, n_per_paradigm=100)
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from cogarena.dimensions.cognitive_control import (
    FlankerParadigm,
    GoNoGoParadigm,
    StroopParadigm,
    TaskInstance,
    _CATEGORY_PAIRS,
)


def generate_cc_items(
    seed: int = 42,
    n_per_paradigm: int = 100,
    include_contamination_probes: bool = True,
) -> List[TaskInstance]:
    """Generate a complete Cognitive Control item battery.

    Produces items across all three paradigms with balanced conditions and
    multiple difficulty levels.

    Parameters
    ----------
    seed : int
        Master random seed.
    n_per_paradigm : int
        Approximate number of items per paradigm.  The actual count may vary
        slightly due to rounding (e.g., splitting across difficulty levels).
    include_contamination_probes : bool
        If True, each paradigm will include a small set of classic / high-
        contamination-risk items to enable Gate 2 contamination detection.

    Returns
    -------
    list of TaskInstance
        All generated items, shuffled.
    """
    rng = random.Random(seed)
    all_items: List[TaskInstance] = []

    # We generate across three difficulty levels to support IRT analysis
    difficulties = ["easy", "medium", "hard"]
    n_per_diff = max(1, n_per_paradigm // len(difficulties))
    remainder = n_per_paradigm - n_per_diff * len(difficulties)

    # ------------------------------------------------------------------
    # 1. Stroop Task
    # ------------------------------------------------------------------
    # Use all non-color conflict types; color-word only as contamination probe
    stroop_conflict_types = ["size_word", "direction_word", "number_quantity"]

    for di, diff in enumerate(difficulties):
        n_this = n_per_diff + (1 if di < remainder else 0)
        n_cong = n_this // 2
        n_incong = n_this - n_cong

        # Rotate through conflict types
        ct = stroop_conflict_types[di % len(stroop_conflict_types)]
        sub_seed = seed + di * 100

        items = StroopParadigm.generate(
            seed=sub_seed,
            n_congruent=n_cong,
            n_incongruent=n_incong,
            conflict_type=ct,
            difficulty=diff,
            contamination_probe=include_contamination_probes,
        )
        all_items.extend(items)

    # Also generate a mixed-conflict-type batch at medium difficulty
    # (provides cross-type comparisons within the same batch)
    mixed_seed = seed + 999
    n_mixed_cong = n_per_diff // 2
    n_mixed_incong = n_per_diff - n_mixed_cong
    all_items.extend(
        StroopParadigm.generate(
            seed=mixed_seed,
            n_congruent=n_mixed_cong,
            n_incongruent=n_mixed_incong,
            conflict_type="mixed",
            difficulty="medium",
            contamination_probe=include_contamination_probes,
        )
    )

    # ------------------------------------------------------------------
    # 2. Flanker Task
    # ------------------------------------------------------------------
    flanker_symbol_sets = ["arrows", "letters", "numbers_parity", "numbers_magnitude"]

    for di, diff in enumerate(difficulties):
        n_this = n_per_diff + (1 if di < remainder else 0)
        n_cong = n_this // 2
        n_incong = n_this - n_cong

        ss = flanker_symbol_sets[di % len(flanker_symbol_sets)]
        sub_seed = seed + 1000 + di * 100

        items = FlankerParadigm.generate(
            seed=sub_seed,
            n_congruent=n_cong,
            n_incongruent=n_incong,
            symbol_set=ss,
            difficulty=diff,
            contamination_probe=include_contamination_probes,
        )
        all_items.extend(items)

    # Mixed symbol-set batch
    all_items.extend(
        FlankerParadigm.generate(
            seed=seed + 1999,
            n_congruent=n_mixed_cong,
            n_incongruent=n_mixed_incong,
            symbol_set="mixed",
            difficulty="medium",
            contamination_probe=include_contamination_probes,
        )
    )

    # ------------------------------------------------------------------
    # 3. Go/No-Go Task
    # ------------------------------------------------------------------
    # Generate multiple episodes with different category pairs
    n_episodes = max(1, n_per_paradigm // 60)  # ~60 trials per episode
    trials_per_episode = max(50, n_per_paradigm // n_episodes)

    # Select category pairs
    available_pairs = list(_CATEGORY_PAIRS)
    rng.shuffle(available_pairs)

    for ep_idx in range(n_episodes):
        pair = available_pairs[ep_idx % len(available_pairs)]
        diff = difficulties[ep_idx % len(difficulties)]
        sub_seed = seed + 2000 + ep_idx * 100

        items = GoNoGoParadigm.generate(
            seed=sub_seed,
            n_trials=trials_per_episode,
            category_pair=pair,
            difficulty=diff,
            contamination_probe=False,
        )
        all_items.extend(items)

    # Contamination probe episode (classic animals vs plants)
    if include_contamination_probes:
        all_items.extend(
            GoNoGoParadigm.generate(
                seed=seed + 2999,
                n_trials=min(30, trials_per_episode),
                contamination_probe=True,
                difficulty="medium",
            )
        )

    # ------------------------------------------------------------------
    # Summary stats (for logging / debugging)
    # ------------------------------------------------------------------
    paradigm_counts = {}
    for item in all_items:
        p = item.metadata.paradigm if hasattr(item, "metadata") and item.metadata else getattr(item, "paradigm", "?")
        paradigm_counts[p] = paradigm_counts.get(p, 0) + 1

    # Tag every item with a globally-unique task_id prefix
    for i, item in enumerate(all_items):
        item.task_id = f"cc_{i:05d}_{item.task_id}"

    # Final shuffle (preserving go/nogo episode order within episodes is
    # handled separately by the evaluator; here we interleave paradigms)
    # NOTE: We do NOT shuffle go_nogo items across episodes because trial
    # order matters. Instead, shuffle only Stroop and Flanker items together,
    # then append go_nogo episodes in sequence.
    def _get_mode(it):
        if hasattr(it, "metadata") and it.metadata:
            m = it.metadata.mode
            return m.value if hasattr(m, "value") else str(m)
        return getattr(it, "mode", "llm_static")
    static_items = [it for it in all_items if _get_mode(it) in ("static", "llm_static")]
    multiturn_items = [it for it in all_items if _get_mode(it) not in ("static", "llm_static")]
    rng.shuffle(static_items)
    # Group multi-turn items by episode (by their go_cat + nogo_cat + seed)
    # They are already in trial order within each episode.

    final = static_items + multiturn_items
    return final


def generate_cc_dev_set(seed: int = 0) -> List[TaskInstance]:
    """Generate a small development set (5 items per paradigm per condition).

    Useful for few-shot prompting and sanity checks.
    """
    items: List[TaskInstance] = []

    # Stroop: 5 congruent + 5 incongruent, mixed conflict types
    items.extend(
        StroopParadigm.generate(
            seed=seed,
            n_congruent=5,
            n_incongruent=5,
            conflict_type="mixed",
            difficulty="medium",
            contamination_probe=False,
        )
    )

    # Flanker: 5 + 5
    items.extend(
        FlankerParadigm.generate(
            seed=seed + 100,
            n_congruent=5,
            n_incongruent=5,
            symbol_set="arrows",
            difficulty="medium",
            contamination_probe=False,
        )
    )

    # Go/No-Go: 10 trials
    items.extend(
        GoNoGoParadigm.generate(
            seed=seed + 200,
            n_trials=10,
            difficulty="medium",
            contamination_probe=False,
        )
    )

    return items


def generate_cc_contamination_set(seed: int = 7777) -> List[TaskInstance]:
    """Generate a dedicated contamination probe set for Gate 2 testing.

    Returns classic / high-contamination-risk items alongside matched
    procedurally-generated items for paired comparison.
    """
    items: List[TaskInstance] = []

    # Classic color-word Stroop (contamination probe)
    items.extend(
        StroopParadigm.generate(
            seed=seed,
            n_congruent=25,
            n_incongruent=25,
            conflict_type="color_word",
            difficulty="medium",
            contamination_probe=True,
        )
    )

    # Matched non-color Stroop (control)
    items.extend(
        StroopParadigm.generate(
            seed=seed + 1,
            n_congruent=25,
            n_incongruent=25,
            conflict_type="mixed",
            difficulty="medium",
            contamination_probe=False,
        )
    )

    # Classic arrow Flanker (contamination probe)
    items.extend(
        FlankerParadigm.generate(
            seed=seed + 100,
            n_congruent=25,
            n_incongruent=25,
            symbol_set="arrows",
            difficulty="medium",
            contamination_probe=True,
        )
    )

    # Matched non-arrow Flanker (control)
    items.extend(
        FlankerParadigm.generate(
            seed=seed + 101,
            n_congruent=25,
            n_incongruent=25,
            symbol_set="numbers_parity",
            difficulty="medium",
            contamination_probe=False,
        )
    )

    # Classic animals-vs-plants Go/No-Go (contamination probe)
    items.extend(
        GoNoGoParadigm.generate(
            seed=seed + 200,
            n_trials=50,
            contamination_probe=True,
            difficulty="medium",
        )
    )

    # Non-classic category pair (control)
    items.extend(
        GoNoGoParadigm.generate(
            seed=seed + 201,
            n_trials=50,
            category_pair=("tools", "instruments"),
            contamination_probe=False,
            difficulty="medium",
        )
    )

    return items
