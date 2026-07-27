from __future__ import annotations

import numpy as np
import pytest

from ..analyze import compute_estimate
from ..analyze_target_baseline import (
    BOOTSTRAP_SEED,
    build_contrasts,
    joint_crossed_family_item_bootstrap,
    validate_sensitivity_spec,
)
from .test_analyzer import _synthetic_data


def test_spec_is_explicitly_post_hoc_and_nonconfirmatory():
    spec = validate_sensitivity_spec()
    assert spec["status"].startswith("post_hoc_sensitivity_")
    assert spec["confirmatory_role"] == "none"
    assert spec["bootstrap"]["seed"] == BOOTSTRAP_SEED
    assert "post-hoc" in spec["interpretation_constraints"][0]


def test_target_placebo_baseline_point_identity_and_weighting():
    data = _synthetic_data()
    contrasts, _, _, _, groups, masks = build_contrasts(data)
    estimates = {
        name: compute_estimate(values, groups, masks)
        for name, values in contrasts.items()
    }
    tp = estimates["target_minus_placebo"]
    tb = estimates["target_minus_baseline"]
    pb = estimates["placebo_minus_baseline_differential"]
    assert tp["gamma"] == pytest.approx(0.15)
    assert tb["gamma"] == pytest.approx(0.15)
    assert pb["gamma"] == pytest.approx(0.0, abs=1e-15)
    assert tp["gamma"] == pytest.approx(tb["gamma"] - pb["gamma"])
    assert tp["s"] == pytest.approx(tb["s"] - pb["s"])


def test_joint_bootstrap_is_seeded_and_preserves_drawwise_identity():
    data = _synthetic_data()
    contrasts, _, _, _, groups, masks = build_contrasts(data)
    first = joint_crossed_family_item_bootstrap(
        contrasts,
        data["families"],
        groups,
        masks,
        n_boot=40,
        seed=BOOTSTRAP_SEED,
    )
    second = joint_crossed_family_item_bootstrap(
        contrasts,
        data["families"],
        groups,
        masks,
        n_boot=40,
        seed=BOOTSTRAP_SEED,
    )
    assert first == second
    assert first["maximum_gamma_identity_error"] < 1e-12
    assert first["maximum_s_identity_error"] < 1e-12
    assert first["contrasts"]["target_minus_placebo"]["gamma_ci95"] == pytest.approx(
        [0.15, 0.15]
    )


def test_group_differential_placebo_effect_is_separated_from_global_harm():
    data = _synthetic_data()
    contrasts, _, placebo_index, baseline_index, groups, masks = build_contrasts(data)
    working_memory = masks["working_memory"]

    # Make placebo worse than baseline only on working-memory paradigms.
    # Rebuild the three paired contrasts after the score change.
    data["accuracy"][:, placebo_index, working_memory, :] -= 0.06
    contrasts, _, _, _, groups, masks = build_contrasts(data)
    estimates = {
        name: compute_estimate(values, groups, masks)
        for name, values in contrasts.items()
    }
    tp = estimates["target_minus_placebo"]
    tb = estimates["target_minus_baseline"]
    pb = estimates["placebo_minus_baseline_differential"]
    assert float(
        (
            data["accuracy"][:, placebo_index]
            - data["accuracy"][:, baseline_index]
        ).mean()
    ) < 0
    assert pb["gamma"] != pytest.approx(0.0)
    assert tp["gamma"] == pytest.approx(tb["gamma"] - pb["gamma"])
    assert np.max(np.abs(tp["s"] - tb["s"] + pb["s"])) < 1e-12
