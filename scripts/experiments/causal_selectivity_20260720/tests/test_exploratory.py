from __future__ import annotations

import numpy as np
import pytest

from ..analyze_exploratory import (
    _primary_inputs,
    family_paradigm_item_bootstrap,
    leave_one_paradigm_out,
    neutral_placebo_audit,
    observable_evaluability,
    response_validity_accuracy_decomposition,
    small_g_family_inference,
    validate_exploratory_spec,
)
from ..analyze import compute_estimate
from .test_analyzer import _synthetic_data


def test_exploratory_spec_is_explicitly_post_hoc_and_nonconfirmatory():
    spec = validate_exploratory_spec()
    assert spec["status"] == "post_hoc_exploratory_authored_after_primary_outcome_inspection"
    assert spec["confirmatory_role"] == "none"
    assert len(spec["analyses"]) == 5


def test_lopo_and_paradigm_bootstrap_preserve_equal_weighting():
    data = _synthetic_data()
    gains, groups, masks, _, _ = _primary_inputs(data)
    lopo = leave_one_paradigm_out(gains, data["paradigms"], groups, masks)
    assert len(lopo["rows"]) == 13
    assert lopo["gamma_min"] == pytest.approx(0.15)
    assert lopo["gamma_max"] == pytest.approx(0.15)

    first = family_paradigm_item_bootstrap(
        gains, data["families"], groups, masks, n_boot=40, seed=5201
    )
    second = family_paradigm_item_bootstrap(
        gains, data["families"], groups, masks, n_boot=40, seed=5201
    )
    assert first == second
    assert first["gamma_ci95"] == pytest.approx([0.15, 0.15])


def test_observable_evaluability_accuracy_partition_reconstructs_primary():
    data = _synthetic_data()
    gains, groups, masks, target_indices, placebo_index = _primary_inputs(data)
    ospan = data["paradigms"].index("operation_span")

    # Populate all four paired evaluability strata without altering the score
    # array; the contribution identity must reconstruct any observed gains.
    data["empty"][0, target_indices[0], 0, 0] = True
    data["empty"][1, placebo_index, 1, 1] = True
    data["empty"][2, target_indices[1], 2, 2] = True
    data["empty"][2, placebo_index, 2, 2] = True
    data["ospan_parse_none"][3, target_indices[2], ospan, 3] = True

    evaluable = observable_evaluability(data)
    assert evaluable.dtype == bool
    assert not evaluable[0, target_indices[0], 0, 0]
    assert not evaluable[3, target_indices[2], ospan, 3]

    result = response_validity_accuracy_decomposition(
        data,
        gains,
        groups,
        masks,
        target_indices,
        placebo_index,
        n_boot=30,
    )
    identity = result["paired_accuracy_contribution_by_observed_evaluability_stratum"]
    assert identity["primary_gamma"] == pytest.approx(
        compute_estimate(gains, groups, masks)["gamma"]
    )
    assert identity["absolute_reconstruction_error"] < 1e-12
    assert identity["maximum_s_reconstruction_error"] < 1e-12
    assert sum(row["pair_count"] for row in identity["rows"]) == gains.size


def test_neutral_placebo_audit_and_small_g_exact_inference():
    data = _synthetic_data()
    audit = neutral_placebo_audit(data, n_boot=30)
    assert audit["primary_accuracy"]["paired_difference"] == pytest.approx(0.01)
    assert len(audit["primary_accuracy"]["by_paradigm"]) == 13
    assert len(audit["primary_accuracy"]["by_family"]) == 6

    # Five positive family means and one negative mean give the exact
    # one-sided sign-test probability 7/64.
    family_effect = np.asarray([0.02, 0.01, 0.03, 0.04, 0.05, -0.01])
    model_s = np.repeat(np.repeat(family_effect, 2)[:, None], 5, axis=1)
    result = small_g_family_inference(model_s, data["families"])
    sign = result["exact_binomial_sign_test"]
    assert sign["positive"] == 5
    assert sign["negative"] == 1
    assert sign["p_one_sided_positive"] == pytest.approx(7 / 64)
    assert result["exact_family_sign_flip_test"]["n_assignments"] == 64
    assert len(result["leave_one_family_out"]["rows"]) == 6
