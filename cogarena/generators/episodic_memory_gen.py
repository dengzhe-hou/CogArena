"""Convenience wrapper that generates the full Episodic Memory item pool.

Usage::

    from cogarena.generators.episodic_memory_gen import generate_em_items

    items = generate_em_items(seed=42, n_per_paradigm=50,
                              include_contamination_probes=True)

Each call returns a deterministic list of ``TaskInstance`` objects
spanning all three paradigms (CVLT Word List Learning, DRM False Memory,
Source Monitoring) at a balanced distribution of difficulty levels.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from cogarena.core import TaskInstance
from cogarena.dimensions.episodic_memory import (
    CVLTGenerator,
    DRMGenerator,
    SourceMonitoringGenerator,
)

# The three difficulty tiers and their relative proportions
_DIFFICULTY_DISTRIBUTION: Dict[str, float] = {
    "easy": 0.25,
    "medium": 0.50,
    "hard": 0.25,
}

# Number of contamination-probe items per paradigm (small fixed set)
_PROBE_COUNT_PER_PARADIGM = 5


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


def generate_em_items(
    seed: int = 42,
    n_per_paradigm: int = 50,
    include_contamination_probes: bool = True,
) -> List[TaskInstance]:
    """Generate the full Episodic Memory item pool.

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
    all_items: List[TaskInstance] = []

    paradigm_generators = [
        ("cvlt_word_list", CVLTGenerator),
        ("drm_false_memory", DRMGenerator),
        ("source_monitoring", SourceMonitoringGenerator),
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
    items = generate_em_items(
        seed=42, n_per_paradigm=10, include_contamination_probes=True
    )
    summary = summarise_pool(items)
    print("Episodic Memory item pool generated successfully.")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Show example items
    print("\n--- Example CVLT episode ---")
    cvlt_items = [
        i for i in items
        if i.metadata.paradigm == "cvlt_word_list"
        and not i.metadata.parameters.get("contamination_probe")
    ]
    if cvlt_items:
        cvlt = cvlt_items[0]
        print(f"  task_id: {cvlt.task_id}")
        print(f"  difficulty: {cvlt.metadata.difficulty.value}")
        print(f"  category: {cvlt.metadata.parameters['primary_category']}")
        print(f"  list_length: {cvlt.metadata.parameters['list_length']}")
        n_turns = len(cvlt.metadata.parameters['turns'])
        print(f"  n_turns: {n_turns}")
        print(f"  stimulus (first 300 chars): {cvlt.stimulus[:300]}...")
        turns = cvlt.metadata.parameters["turns"]
        for t in turns[:3]:
            print(f"    turn {t['position']}: type={t['type']}")

    print("\n--- Example DRM episode ---")
    drm_items = [
        i for i in items
        if i.metadata.paradigm == "drm_false_memory"
        and not i.metadata.parameters.get("contamination_probe")
    ]
    if drm_items:
        drm = drm_items[0]
        print(f"  task_id: {drm.task_id}")
        print(f"  difficulty: {drm.metadata.difficulty.value}")
        print(f"  n_lists: {drm.metadata.parameters['n_lists']}")
        n_probes = len(drm.metadata.parameters['recognition_probes'])
        print(f"  n_recognition_probes: {n_probes}")
        print(f"  critical_lures: {drm.metadata.parameters['critical_lures']}")
        print(f"  stimulus (first 400 chars): {drm.stimulus[:400]}...")

    print("\n--- Example Source Monitoring episode ---")
    sm_items = [
        i for i in items
        if i.metadata.paradigm == "source_monitoring"
        and not i.metadata.parameters.get("contamination_probe")
    ]
    if sm_items:
        sm = sm_items[0]
        print(f"  task_id: {sm.task_id}")
        print(f"  difficulty: {sm.metadata.difficulty.value}")
        print(f"  n_sources: {sm.metadata.parameters['n_sources']}")
        print(f"  sources: {sm.metadata.parameters['sources']}")
        n_test = len(sm.metadata.parameters['test_items'])
        print(f"  n_test_items: {n_test}")
        print(f"  stimulus (first 400 chars): {sm.stimulus[:400]}...")

    # Quick scoring test with mock responses
    print("\n--- Scoring smoke test (DRM) ---")
    if drm_items:
        drm = drm_items[0]
        # Build a perfect response
        probes = drm.metadata.parameters['recognition_probes']
        perfect_resp_lines = []
        for p in probes:
            perfect_resp_lines.append(f"{p['word']}: {p['correct']}")
        perfect_resp = "\n".join(perfect_resp_lines)
        from cogarena.dimensions.episodic_memory import DRMGenerator
        scores = DRMGenerator.score(drm, perfect_resp)
        print(f"  Perfect response scores: {scores}")

    print("\n--- Scoring smoke test (Source Monitoring) ---")
    if sm_items:
        sm = sm_items[0]
        # Build a perfect response
        test_data = sm.metadata.parameters['test_items']
        perfect_lines = []
        for td in test_data:
            perfect_lines.append(f"{td['test_position']}. {td['correct_source']}")
        perfect_resp = "\n".join(perfect_lines)
        from cogarena.dimensions.episodic_memory import SourceMonitoringGenerator
        scores = SourceMonitoringGenerator.score(sm, perfect_resp)
        print(f"  Perfect response scores: {scores}")
