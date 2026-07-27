import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/experiments/profile_validity_20260720/family_structure.py"
SPEC = importlib.util.spec_from_file_location("family_structure", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_theory_partition_count_and_uniqueness():
    seen = set(MOD.theory_partitions())
    assert len(seen) == 600_600
    assert all(sorted(map(len, partition)) == [2, 2, 3, 3, 3] for partition in seen)


def test_delta_detects_block_structure():
    labels = ["a", "a", "b", "b"]
    rng = np.random.default_rng(7)
    latent_a = rng.normal(size=300)
    latent_b = rng.normal(size=300)
    matrix = np.column_stack([
        latent_a + rng.normal(scale=.1, size=300),
        latent_a + rng.normal(scale=.1, size=300),
        latent_b + rng.normal(scale=.1, size=300),
        latent_b + rng.normal(scale=.1, size=300),
    ])
    result = MOD.delta_from_matrix(matrix, labels)
    assert result["delta"] > 0.8


def test_family_centering_removes_between_family_profile():
    labels = ["a", "a", "b", "b"]
    family_profiles = {
        "f1": np.array([1.0, 1.0, 0.0, 0.0]),
        "f2": np.array([0.0, 0.0, 1.0, 1.0]),
        "f3": np.array([0.8, 0.8, 0.2, 0.2]),
    }
    rows, families = [], []
    rng = np.random.default_rng(11)
    for family, profile in family_profiles.items():
        for _ in range(6):
            rows.append(profile + rng.normal(scale=.02, size=4))
            families.append(family)
    matrix = np.asarray(rows)
    before = MOD.delta_from_matrix(matrix, labels)["delta"]
    centered = matrix.copy()
    for family in set(families):
        idx = [i for i, f in enumerate(families) if f == family]
        centered[idx] -= centered[idx].mean(axis=0)
    after = MOD.delta_from_matrix(centered, labels)["delta"]
    assert before > 0.5
    assert abs(after) < 0.3


def test_paired_decomposition_bootstrap_detects_between_family_signal():
    labels = ["a", "a", "b", "b"]
    profiles = {
        f"f{i}": np.array([1.0, 1.0, 0.0, 0.0])
        if i < 6
        else np.array([0.0, 0.0, 1.0, 1.0])
        for i in range(12)
    }
    rng = np.random.default_rng(19)
    rows, index, mapping = [], [], {}
    for family, profile in profiles.items():
        for checkpoint in range(3):
            model = f"{family}_m{checkpoint}"
            index.append(model)
            mapping[model] = family
            rows.append(profile + rng.normal(scale=0.03, size=4))
    import pandas as pd

    frame = pd.DataFrame(rows, index=index, columns=["p0", "p1", "p2", "p3"])
    result = MOD.paired_decomposition_bootstrap(
        frame, mapping, labels, n_boot=200, seed=7
    )
    assert result["n_effective"] >= 190
    assert result["between_minus_within_delta"]["mean"] > 0.5
