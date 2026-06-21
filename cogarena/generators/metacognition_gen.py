"""Convenience wrapper that generates the full Metacognition item pool.

Usage::

    from cogarena.generators.metacognition_gen import generate_mc_items

    items = generate_mc_items(seed=42, n_per_paradigm=100,
                              include_contamination_probes=True)

Each call returns a deterministic list of ``TaskInstance`` objects
spanning both paradigms (Confidence Calibration, Post-Decision Wagering)
at a balanced distribution of difficulty levels.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from cogarena.core import TaskInstance
from cogarena.dimensions.metacognition import (
    ConfidenceCalibrationGenerator,
    PostDecisionWageringGenerator,
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
    counts["medium"] += n_total - allocated
    return counts


def generate_mc_items(
    seed: int = 42,
    n_per_paradigm: int = 100,
    include_contamination_probes: bool = True,
) -> list[TaskInstance]:
    """Generate the full Metacognition item pool.

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
        If True, an additional small set of well-known (easy) questions
        is generated per paradigm for contamination analysis.

    Returns
    -------
    list[TaskInstance]
        Flat list of all generated items, tagged with paradigm, difficulty,
        and contamination-probe status.
    """
    rng = random.Random(seed)
    all_items: list[TaskInstance] = []

    paradigm_generators = [
        ("confidence_calibration", ConfidenceCalibrationGenerator),
        ("post_decision_wagering", PostDecisionWageringGenerator),
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

        # Contamination probes (well-known questions, medium difficulty)
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
    """Return a summary dict describing the generated item pool."""
    from collections import Counter

    by_paradigm: Counter[str] = Counter()
    by_difficulty: Counter[str] = Counter()
    by_domain: Counter[str] = Counter()
    probes = 0

    for item in items:
        by_paradigm[item.metadata.paradigm] += 1
        by_difficulty[item.metadata.difficulty.value] += 1
        domain = item.metadata.parameters.get("domain", "unknown")
        by_domain[domain] += 1
        if item.metadata.parameters.get("contamination_probe", False):
            probes += 1

    return {
        "total_items": len(items),
        "by_paradigm": dict(by_paradigm),
        "by_difficulty": dict(by_difficulty),
        "by_domain": dict(by_domain),
        "contamination_probes": probes,
    }


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    items = generate_mc_items(
        seed=42, n_per_paradigm=20, include_contamination_probes=True,
    )
    summary = summarise_pool(items)
    print("Metacognition item pool generated successfully.")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Show example items
    print("\n--- Example Confidence Calibration item ---")
    cc_items = [
        i for i in items
        if i.metadata.paradigm == "confidence_calibration"
        and not i.metadata.parameters.get("contamination_probe")
    ]
    if cc_items:
        print(f"  task_id: {cc_items[0].task_id}")
        print(f"  difficulty: {cc_items[0].metadata.difficulty.value}")
        print(f"  domain: {cc_items[0].metadata.parameters['domain']}")
        print(f"  stimulus:\n{cc_items[0].stimulus[:300]}...")
        print(f"  expected: {cc_items[0].expected_response}")

    print("\n--- Example Post-Decision Wagering item ---")
    pdw_items = [
        i for i in items
        if i.metadata.paradigm == "post_decision_wagering"
        and not i.metadata.parameters.get("contamination_probe")
    ]
    if pdw_items:
        print(f"  task_id: {pdw_items[0].task_id}")
        print(f"  difficulty: {pdw_items[0].metadata.difficulty.value}")
        print(f"  domain: {pdw_items[0].metadata.parameters['domain']}")
        print(f"  stimulus:\n{pdw_items[0].stimulus[:400]}...")
        print(f"  expected: {pdw_items[0].expected_response}")

    print("\n--- Example contamination probe ---")
    probes_list = [
        i for i in items
        if i.metadata.parameters.get("contamination_probe")
    ]
    if probes_list:
        print(f"  task_id: {probes_list[0].task_id}")
        print(f"  stimulus:\n{probes_list[0].stimulus[:300]}...")
        print(f"  expected: {probes_list[0].expected_response}")

    # Test scoring with a mock response
    print("\n--- Scoring test ---")
    from cogarena.dimensions.metacognition import (
        ConfidenceCalibrationGenerator,
        PostDecisionWageringGenerator,
    )

    if cc_items:
        mock_resp = f"Answer: {cc_items[0].expected_response}\nConfidence: 85%"
        score_result = ConfidenceCalibrationGenerator.score(cc_items[0], mock_resp)
        print(f"  Confidence calibration score: {score_result}")

    if pdw_items:
        mock_resp = f"Answer: {pdw_items[0].expected_response}\nBet: YES"
        score_result = PostDecisionWageringGenerator.score(pdw_items[0], mock_resp)
        print(f"  Post-decision wagering score: {score_result}")
