"""Convenience wrapper that generates the full Working Memory item pool.

Usage::

    from cogarena.generators.working_memory_gen import generate_wm_items

    items = generate_wm_items(seed=42, n_per_paradigm=100,
                              include_contamination_probes=True)

Each call returns a deterministic list of ``TaskInstance`` objects
spanning all three paradigms (N-Back, Digit Span, Operation Span) at
a balanced distribution of difficulty levels.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from cogarena.core import TaskInstance
from cogarena.dimensions.working_memory import (
    NBackGenerator,
    DigitSpanGenerator,
    OperationSpanGenerator,
)

# The three difficulty tiers and their relative proportions
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


def generate_wm_items(
    seed: int = 42,
    n_per_paradigm: int = 100,
    include_contamination_probes: bool = True,
) -> list[TaskInstance]:
    """Generate the full Working Memory item pool.

    Parameters
    ----------
    seed : int
        Master random seed.  Each paradigm x difficulty combination
        receives a deterministic sub-seed derived from this.
    n_per_paradigm : int
        Number of *main* (non-probe) items to generate per paradigm.
        The total pool size is approximately ``3 * n_per_paradigm``
        plus contamination probes.
    include_contamination_probes : bool
        If True, an additional small set of items per paradigm is
        generated using classic / well-known stimuli (for contamination
        detection analysis as described in the CogArena plan).

    Returns
    -------
    list[TaskInstance]
        Flat list of all generated items, tagged with paradigm, difficulty,
        and contamination-probe status.
    """
    rng = random.Random(seed)
    all_items: list[TaskInstance] = []

    paradigm_generators = [
        ("n_back", NBackGenerator),
        ("digit_span", DigitSpanGenerator),
        ("operation_span", OperationSpanGenerator),
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

def summarise_pool(items: list[TaskInstance]) -> Dict[str, Any]:
    """Return a summary dict describing the generated item pool.

    Useful for sanity-checking during development.
    """
    from collections import Counter

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
    items = generate_wm_items(
        seed=42, n_per_paradigm=20, include_contamination_probes=True
    )
    summary = summarise_pool(items)
    print("Working Memory item pool generated successfully.")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Show a few example items
    print("\n--- Example N-Back episode ---")
    nb = [
        i for i in items
        if i.metadata.paradigm == "n_back"
        and not i.metadata.parameters.get("contamination_probe")
    ][0]
    print(f"  task_id: {nb.task_id}")
    print(f"  difficulty: {nb.metadata.difficulty.value}")
    print(f"  stimulus: {nb.stimulus[:200]}...")
    turns = nb.metadata.parameters["turns"]
    print(f"  turns (first 5): {turns[:5]}")

    print("\n--- Example Digit Span item ---")
    ds = [i for i in items if i.metadata.paradigm == "digit_span"][0]
    print(f"  task_id: {ds.task_id}")
    print(f"  stimulus: {ds.stimulus}")
    print(f"  expected: {ds.expected_response}")

    print("\n--- Example Operation Span set ---")
    os_ = [i for i in items if i.metadata.paradigm == "operation_span"][0]
    print(f"  task_id: {os_.task_id}")
    print(f"  set_size: {os_.metadata.parameters['set_size']}")
    print(f"  stimulus: {os_.stimulus[:200]}...")
    print(f"  expected recall: {os_.expected_response}")
