"""Convenience wrapper that generates the full Theory of Mind item pool.

Usage::

    from cogarena.generators.theory_of_mind_gen import generate_tom_items

    items = generate_tom_items(seed=42, n_per_paradigm=100,
                               include_contamination_probes=True)

Each call returns a deterministic list of ``TaskInstance`` objects
spanning both paradigms (False Belief, EPITOME-style Multi-aspect ToM)
at a balanced distribution of difficulty levels and sub-conditions.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from cogarena.core import TaskInstance
from cogarena.dimensions.theory_of_mind import (
    FalseBeliefGenerator,
    EpitomeToMGenerator,
)

# Difficulty tiers and their relative proportions
_DIFFICULTY_DISTRIBUTION: Dict[str, float] = {
    "easy": 0.25,
    "medium": 0.50,
    "hard": 0.25,
}

# Number of contamination-probe items per paradigm
_PROBE_COUNT_PER_PARADIGM = 10


def _split_by_difficulty(
    n_total: int,
    distribution: Dict[str, float] | None = None,
) -> Dict[str, int]:
    """Split *n_total* into per-difficulty counts.

    Guarantees the counts sum to exactly *n_total* (rounding residuals
    are added to "medium").
    """
    dist = distribution or _DIFFICULTY_DISTRIBUTION
    counts: Dict[str, int] = {}
    allocated = 0
    for diff, frac in dist.items():
        c = int(round(n_total * frac))
        counts[diff] = c
        allocated += c
    # Fix rounding
    counts["medium"] += n_total - allocated
    return counts


def generate_tom_items(
    seed: int = 42,
    n_per_paradigm: int = 100,
    include_contamination_probes: bool = True,
) -> list[TaskInstance]:
    """Generate the full Theory of Mind item pool.

    Parameters
    ----------
    seed : int
        Master random seed.  Each paradigm x difficulty x condition
        combination receives a deterministic sub-seed.
    n_per_paradigm : int
        Number of *main* (non-probe) items to generate per paradigm.
        The total pool is roughly ``2 * n_per_paradigm`` plus probes.
    include_contamination_probes : bool
        If True, an additional small set of classic Sally-Anne items
        is generated per false-belief order for contamination analysis.

    Returns
    -------
    list[TaskInstance]
        Flat list of all generated items, tagged with paradigm,
        difficulty, and contamination-probe status.
    """
    rng = random.Random(seed)
    all_items: list[TaskInstance] = []

    # =================================================================
    # 1. False Belief paradigm
    # =================================================================
    # Split items between order-1 and order-2
    n_fb_per_order = n_per_paradigm // 2
    n_fb_remainder = n_per_paradigm - 2 * n_fb_per_order

    for order in (1, 2):
        n_order = n_fb_per_order + (n_fb_remainder if order == 1 else 0)
        diff_counts = _split_by_difficulty(n_order)

        for difficulty, count in diff_counts.items():
            sub_seed = rng.randint(0, 2**31)
            items = FalseBeliefGenerator.generate(
                seed=sub_seed,
                n_items=count,
                order=order,
                difficulty=difficulty,
                contamination_probe=False,
            )
            all_items.extend(items)

        # Contamination probes: classic Sally-Anne, medium difficulty
        if include_contamination_probes:
            probe_seed = rng.randint(0, 2**31)
            probes = FalseBeliefGenerator.generate(
                seed=probe_seed,
                n_items=_PROBE_COUNT_PER_PARADIGM // 2,
                order=order,
                difficulty="medium",
                contamination_probe=True,
            )
            all_items.extend(probes)

    # =================================================================
    # 2. EPITOME-style Multi-aspect ToM
    # =================================================================
    diff_counts = _split_by_difficulty(n_per_paradigm)

    for difficulty, count in diff_counts.items():
        sub_seed = rng.randint(0, 2**31)
        items = EpitomeToMGenerator.generate(
            seed=sub_seed,
            n_items=count,
            sub_capacity="all",
            difficulty=difficulty,
            contamination_probe=False,
        )
        all_items.extend(items)

    # Small probe set for EPITOME (flag-only, no classic scenario)
    if include_contamination_probes:
        probe_seed = rng.randint(0, 2**31)
        probes = EpitomeToMGenerator.generate(
            seed=probe_seed,
            n_items=_PROBE_COUNT_PER_PARADIGM,
            sub_capacity="all",
            difficulty="medium",
            contamination_probe=True,
        )
        all_items.extend(probes)

    return all_items


# ---------------------------------------------------------------------------
# Quick summary helpers
# ---------------------------------------------------------------------------

def summarise_pool(items: list[TaskInstance]) -> Dict[str, Any]:
    """Return a summary dict describing the generated item pool."""
    from collections import Counter

    by_paradigm: Counter[str] = Counter()
    by_difficulty: Counter[str] = Counter()
    by_sub_capacity: Counter[str] = Counter()
    probes = 0
    by_order: Counter[int] = Counter()

    for item in items:
        by_paradigm[item.metadata.paradigm] += 1
        by_difficulty[item.metadata.difficulty.value] += 1
        if item.metadata.parameters.get("contamination_probe", False):
            probes += 1
        sc = item.metadata.parameters.get("sub_capacity")
        if sc:
            by_sub_capacity[sc] += 1
        order = item.metadata.parameters.get("order")
        if order:
            by_order[order] += 1

    return {
        "total_items": len(items),
        "by_paradigm": dict(by_paradigm),
        "by_difficulty": dict(by_difficulty),
        "by_sub_capacity": dict(by_sub_capacity),
        "by_order": dict(by_order),
        "contamination_probes": probes,
    }


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    items = generate_tom_items(
        seed=42, n_per_paradigm=20, include_contamination_probes=True,
    )
    summary = summarise_pool(items)
    print("Theory of Mind item pool generated successfully.")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Show example items
    print("\n--- Example 1st-order False Belief ---")
    fb1 = [
        i for i in items
        if i.metadata.paradigm == "false_belief"
        and i.metadata.parameters.get("order") == 1
        and not i.metadata.parameters.get("contamination_probe")
    ]
    if fb1:
        print(f"  task_id: {fb1[0].task_id}")
        print(f"  stimulus:\n{fb1[0].stimulus[:400]}...")
        print(f"  expected: {fb1[0].expected_response}")

    print("\n--- Example 2nd-order False Belief ---")
    fb2 = [
        i for i in items
        if i.metadata.paradigm == "false_belief"
        and i.metadata.parameters.get("order") == 2
        and not i.metadata.parameters.get("contamination_probe")
    ]
    if fb2:
        print(f"  task_id: {fb2[0].task_id}")
        print(f"  stimulus:\n{fb2[0].stimulus[:400]}...")
        print(f"  expected: {fb2[0].expected_response}")

    print("\n--- Example EPITOME ToM (emotion) ---")
    emo = [
        i for i in items
        if i.metadata.paradigm == "epitome_tom"
        and i.metadata.parameters.get("sub_capacity") == "emotion"
    ]
    if emo:
        print(f"  task_id: {emo[0].task_id}")
        print(f"  stimulus:\n{emo[0].stimulus[:400]}")
        print(f"  expected: {emo[0].expected_response}")

    print("\n--- Example contamination probe (Sally-Anne) ---")
    probes = [
        i for i in items
        if i.metadata.parameters.get("contamination_probe")
        and i.metadata.paradigm == "false_belief"
    ]
    if probes:
        print(f"  task_id: {probes[0].task_id}")
        print(f"  stimulus:\n{probes[0].stimulus[:400]}")
        print(f"  expected: {probes[0].expected_response}")
