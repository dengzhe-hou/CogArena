import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/experiments/profile_validity_20260720/family_heldout_utility.py"
SPEC = importlib.util.spec_from_file_location("family_heldout_utility", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_build_features_excludes_target_and_is_train_only():
    rng = np.random.default_rng(4)
    columns = [p for ps in MOD.GROUPS.values() for p in sorted(ps)]
    models = [f"m{i}" for i in range(12)]
    frame = pd.DataFrame(rng.normal(size=(12, 13)), index=models, columns=columns)
    labels = MOD.group_map(columns)
    target = columns[0]
    got = MOD.build_features(frame, models[:9], models[9:], target, labels)
    assert got[0].shape == (9, 1)
    assert got[1].shape == (3, 1)
    assert got[2].shape == (9, 2)
    assert got[3].shape == (3, 2)
    # Mutating held-out target values cannot change any feature.
    changed = frame.copy()
    changed.loc[models[9:], target] += 1_000
    got_changed = MOD.build_features(changed, models[:9], models[9:], target, labels)
    for a, b in zip(got, got_changed):
        np.testing.assert_allclose(a, b)


def test_metrics_equal_models_have_zero_gain():
    pred = pd.DataFrame({
        "model": ["m1", "m2"],
        "family": ["f1", "f2"],
        "target": ["p", "p"],
        "truth_z": [0.0, 1.0],
        "pred_g_z": [0.2, 0.8],
        "pred_g_group_z": [0.2, 0.8],
    })
    result = MOD.metrics(pred)
    assert abs(result["relative_rmse_gain"]) < 1e-12


def test_feature_builder_accepts_family_centroids_and_preserves_test_rows():
    rng = np.random.default_rng(9)
    columns = [p for ps in MOD.GROUPS.values() for p in sorted(ps)]
    train = pd.DataFrame(
        rng.normal(size=(23, 13)), index=[f"f{i}" for i in range(23)], columns=columns
    )
    test = pd.DataFrame(
        rng.normal(size=(3, 13)), index=[f"m{i}" for i in range(3)], columns=columns
    )
    got = MOD.build_features_from_frames(train, test, columns[0], MOD.group_map(columns))
    assert got[0].shape == (23, 1)
    assert got[1].shape == (3, 1)
    assert got[2].shape == (23, 2)
    assert got[3].shape == (3, 2)
