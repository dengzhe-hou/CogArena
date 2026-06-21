"""Convenience wrapper that generates the full Set Shifting item pool.

Usage::

    from cogarena.generators.set_shifting_gen import generate_ss_items

    items = generate_ss_items(seed=42, n_per_paradigm=50,
                              include_contamination_probes=True)

Each call returns a deterministic list of ``TaskInstance`` objects
spanning both paradigms (WCST, Reversal Learning) at a balanced
distribution of difficulty levels.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List

from cogarena.core import TaskInstance
from cogarena.dimensions.set_shifting import (
    WCSTGenerator,
    ReversalLearningGenerator,
)

# The three difficulty tiers and their relative proportions (25/50/25)
_DIFFICULTY_DISTRIBUTION: Dict[str, float] = {
    "easy": 0.25,
    "medium": 0.50,
    "hard": 0.25,
}

# Number of contamination-probe items per paradigm (small fixed set)
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


def generate_ss_items(
    seed: int = 42,
    n_per_paradigm: int = 50,
    include_contamination_probes: bool = True,
) -> List[TaskInstance]:
    """Generate the full Set Shifting / Cognitive Flexibility item pool.

    Parameters
    ----------
    seed : int
        Master random seed.  Each paradigm x difficulty combination
        receives a deterministic sub-seed derived from this.
    n_per_paradigm : int
        Number of *main* (non-probe) items to generate per paradigm.
        The total pool size is approximately ``2 * n_per_paradigm``
        plus contamination probes.
    include_contamination_probes : bool
        If True, an additional small set of items per paradigm is
        generated using classic / well-known stimuli (for contamination
        detection analysis).  For WCST this means classic color/shape/
        number dimensions; for Reversal Learning this means generic
        A/B labels.

    Returns
    -------
    list[TaskInstance]
        Flat list of all generated items, tagged with paradigm, difficulty,
        and contamination-probe status.
    """
    rng = random.Random(seed)
    all_items: List[TaskInstance] = []

    paradigm_generators = [
        ("wcst", WCSTGenerator),
        ("reversal_learning", ReversalLearningGenerator),
    ]

    for paradigm_name, gen_cls in paradigm_generators:
        diff_counts = _split_by_difficulty(n_per_paradigm)

        for difficulty, count in diff_counts.items():
            sub_seed = rng.randint(0, 2**31)
            items = gen_cls.generate(
                seed=sub_seed,
                n_items=count,
                difficulty=difficulty,
                contamination_probe=False,
            )
            all_items.extend(items)

        # Contamination probes (small fixed set, medium difficulty)
        if include_contamination_probes:
            probe_seed = rng.randint(0, 2**31)
            probes = gen_cls.generate(
                seed=probe_seed,
                n_items=_PROBE_COUNT_PER_PARADIGM,
                difficulty="medium",
                contamination_probe=True,
            )
            all_items.extend(probes)

    return all_items


# ---------------------------------------------------------------------------
# Quick summary helpers
# ---------------------------------------------------------------------------

def summarise_pool(items: List[TaskInstance]) -> Dict[str, Any]:
    """Return a summary dict describing the generated item pool.

    Useful for sanity-checking during development.
    """
    by_paradigm: Counter[str] = Counter()
    by_difficulty: Counter[str] = Counter()
    probes = 0
    multi_turn = 0

    for item in items:
        by_paradigm[item.metadata.paradigm] += 1
        by_difficulty[item.metadata.difficulty.value] += 1
        if item.metadata.parameters.get("contamination_probe", False):
            probes += 1
        if item.metadata.parameters.get("multi_turn", False):
            multi_turn += 1

    return {
        "total_items": len(items),
        "by_paradigm": dict(by_paradigm),
        "by_difficulty": dict(by_difficulty),
        "contamination_probes": probes,
        "multi_turn_episodes": multi_turn,
        "single_turn_items": len(items) - multi_turn,
    }


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    items = generate_ss_items(
        seed=42, n_per_paradigm=10, include_contamination_probes=True,
    )
    summary = summarise_pool(items)
    print("Set Shifting item pool generated successfully.")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Show a few example items
    print("\n--- Example WCST episode ---")
    wcst_items = [
        i for i in items
        if i.metadata.paradigm == "wcst"
        and not i.metadata.parameters.get("contamination_probe")
    ]
    if wcst_items:
        ex = wcst_items[0]
        print(f"  task_id: {ex.task_id}")
        print(f"  difficulty: {ex.metadata.difficulty.value}")
        print(f"  dim_names: {ex.metadata.parameters['dim_names']}")
        print(f"  rule_sequence: {ex.metadata.parameters['rule_sequence']}")
        print(f"  max_trials: {ex.metadata.parameters['max_trials']}")
        print(f"  stimulus (first 300 chars):\n{ex.stimulus[:300]}...")
        turns = ex.metadata.parameters["turns"]
        print(f"  total turns: {len(turns)}")
        print(f"  first turn expected: {turns[0]['expected']}")

    print("\n--- Example WCST contamination probe ---")
    wcst_probes = [
        i for i in items
        if i.metadata.paradigm == "wcst"
        and i.metadata.parameters.get("contamination_probe")
    ]
    if wcst_probes:
        ex = wcst_probes[0]
        print(f"  task_id: {ex.task_id}")
        print(f"  dim_names: {ex.metadata.parameters['dim_names']}")
        print(f"  (uses classic color/shape/number)")

    print("\n--- Example Reversal Learning episode ---")
    rev_items = [
        i for i in items
        if i.metadata.paradigm == "reversal_learning"
        and not i.metadata.parameters.get("contamination_probe")
    ]
    if rev_items:
        ex = rev_items[0]
        print(f"  task_id: {ex.task_id}")
        print(f"  difficulty: {ex.metadata.difficulty.value}")
        print(f"  options: {ex.metadata.parameters['option_a']} / {ex.metadata.parameters['option_b']}")
        print(f"  n_reversals: {ex.metadata.parameters['n_reversals']}")
        print(f"  total_trials: {ex.metadata.parameters['total_trials']}")
        print(f"  phase_boundaries: {ex.metadata.parameters['phase_boundaries']}")
        print(f"  stimulus (first 300 chars):\n{ex.stimulus[:300]}...")
