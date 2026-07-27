from __future__ import annotations

import copy
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from ..analyze import (
    CONFIRMATORY_GATE_IDS,
    _contrast_from_paradigm_means,
    _primary_response_texts,
    _validate_formal_run_manifest,
    analyze_arrays,
    benjamini_hochberg,
    compute_estimate,
    confirmatory_gate_thresholds,
    crossed_family_item_bootstrap,
    exact_intervention_mapping_permutation,
    validate_aggregate_output,
    validate_canonical_analysis_paths,
)
from ..analyze_amended import analyze_arrays as analyze_arrays_amended
from ..common import RESULTS_ROOT, load_spec, request_stop_policy


def _synthetic_data() -> dict:
    spec = copy.deepcopy(load_spec())
    models_meta = spec["formal_model_panel"]
    models = [entry["model"] for entry in models_meta]
    families = [entry["family"] for entry in models_meta]
    conditions = list(spec["conditions"])
    condition_ids = [entry["id"] for entry in conditions]
    paradigms = [name for group in spec["grouping"].values() for name in group]
    difficulties = np.asarray(
        [["easy"] * 6 + ["medium"] * 6 + ["hard"] * 6 for _ in paradigms],
        dtype=object,
    )

    m_count, c_count, p_count, i_count = 12, 7, 13, 18
    accuracy = np.zeros((m_count, c_count, p_count, i_count), dtype=float)
    response_chars = np.zeros_like(accuracy)
    for m in range(m_count):
        for p in range(p_count):
            for i in range(i_count):
                placebo = 0.24 + 0.008 * m + 0.006 * (p % 4) + 0.002 * (i % 6)
                accuracy[m, 0, p, i] = placebo - 0.01
                accuracy[m, 1, p, i] = placebo
                base_chars = 12 + m + p + i
                response_chars[m, 0, p, i] = base_chars - 1
                response_chars[m, 1, p, i] = base_chars
                for j, condition in enumerate(conditions[2:]):
                    matched = paradigms[p] in spec["grouping"][condition["target_group"]]
                    accuracy[m, j + 2, p, i] = placebo + (0.18 if matched else 0.03)
                    response_chars[m, j + 2, p, i] = (
                        base_chars + 1 + j + (p % 2) + ((m + i) % 3)
                    )
    ospan_math = np.full_like(accuracy, np.nan)
    ospan_index = paradigms.index("operation_span")
    ospan_math[:, :, ospan_index, :] = accuracy[:, :, ospan_index, :]
    condition_record_counts = {condition_id: 12 * 13 * 18 for condition_id in condition_ids}
    condition_truncated_record_counts = {condition_id: 0 for condition_id in condition_ids}
    condition_truncation_rates = {condition_id: 0.0 for condition_id in condition_ids}
    condition_invalid_record_counts = {condition_id: 0 for condition_id in condition_ids}
    condition_invalid_record_rates = {condition_id: 0.0 for condition_id in condition_ids}
    return {
        "spec": spec,
        "models": models,
        "families": families,
        "conditions": conditions,
        "condition_ids": condition_ids,
        "paradigms": paradigms,
        "difficulties": difficulties,
        "accuracy": accuracy,
        "canonical_accuracy": accuracy.copy(),
        "ospan_math_accuracy": ospan_math,
        "empty": np.zeros_like(accuracy, dtype=bool),
        "response_chars": response_chars,
        "ospan_parse_none": np.zeros_like(accuracy, dtype=bool),
        "protocol_invalid": np.zeros_like(accuracy, dtype=bool),
        "recovered_terminal_metadata_fault_exposed": np.zeros_like(
            accuracy, dtype=bool
        ),
        "run": {
            "truncated_completion_count": 0,
            "truncated_api_call_count": 0,
            "truncated_record_count": 0,
            "transport_incomplete_logical_call_count": 0,
            "recovered_terminal_metadata_fault_logical_call_count": 0,
            "recovered_terminal_metadata_fault_record_count": 0,
            "transport_protocol_invalid_record_count": 0,
            "invalid_record_count": 0,
            "condition_record_counts": condition_record_counts,
            "condition_truncated_record_counts": condition_truncated_record_counts,
            "condition_truncation_rates": condition_truncation_rates,
            "condition_invalid_record_counts": condition_invalid_record_counts,
            "condition_invalid_record_rates": condition_invalid_record_rates,
            "condition_recovered_terminal_metadata_fault_record_counts": {
                condition_id: 0 for condition_id in condition_ids
            },
            "condition_recovered_terminal_metadata_fault_record_rates": {
                condition_id: 0.0 for condition_id in condition_ids
            },
        },
    }


def _primary_inputs(data: dict):
    targeted = [entry for entry in data["conditions"] if entry["kind"] == "targeted"]
    target_indices = [data["condition_ids"].index(entry["id"]) for entry in targeted]
    placebo_index = data["condition_ids"].index("neutral_placebo")
    gains = (
        data["accuracy"][:, target_indices]
        - data["accuracy"][:, placebo_index, None]
    )
    groups = [entry["target_group"] for entry in targeted]
    masks = {
        group: np.asarray([p in members for p in data["paradigms"]], dtype=bool)
        for group, members in data["spec"]["grouping"].items()
    }
    return gains, groups, masks


def test_primary_estimator_is_equal_weighted_at_each_declared_level():
    data = _synthetic_data()
    gains, groups, masks = _primary_inputs(data)
    result = compute_estimate(gains, groups, masks)
    assert result["gamma"] == pytest.approx(0.15)
    assert result["s"] == pytest.approx(np.repeat(0.15, 5))
    assert result["model_s"].shape == (12, 5)
    assert np.all(result["cell_item_counts"] == 18)


def test_crossed_family_item_bootstrap_is_seeded_and_family_aware():
    data = _synthetic_data()
    gains, groups, masks = _primary_inputs(data)
    first = crossed_family_item_bootstrap(
        gains, data["families"], groups, masks, n_boot=80, seed=42
    )
    second = crossed_family_item_bootstrap(
        gains, data["families"], groups, masks, n_boot=80, seed=42
    )
    assert first == second
    assert first["n_bootstrap"] == 80
    assert first["gamma_ci95"] == pytest.approx([0.15, 0.15])
    assert len(first["s_ci95"]) == 5


def test_exact_mapping_permutation_enumerates_all_120_and_bh_is_valid():
    data = _synthetic_data()
    gains, groups, masks = _primary_inputs(data)
    estimate = compute_estimate(gains, groups, masks)
    result = exact_intervention_mapping_permutation(
        estimate["intervention_paradigm_mean_gain"], groups, masks
    )
    assert result["n_exact_mappings"] == 120
    assert 1 / 120 <= result["gamma_p_one"] <= 1
    assert len(result["s_p_two_bh"]) == 5
    assert all(0 <= value <= 1 for value in result["s_p_two_bh"])
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.20, 0.50]) == pytest.approx(
        [0.05, 1 / 15, 1 / 15, 0.25, 0.50]
    )


def test_formal_run_manifest_gate_accepts_only_closed_formal_raw_data():
    spec = copy.deepcopy(load_spec())
    spec["status"] = "formal_frozen_after_pilot"
    revision = "a" * 40
    run = {
        "schema_version": "cogarena.causal_selectivity.run_manifest.v3",
        "profile": "formal",
        "status": "formal_raw_complete",
        "all_model_replays_passed": True,
        "execution_guard_count": 12,
        "execution_guard_tree_sha256": "b" * 64,
        "all_execution_guards_verified_complete": True,
        "record_reuse_allowed": False,
        "profile_array_job_id": "12345",
        "reported_usage_context_budget_verified": True,
        "max_reported_prompt_tokens": 100,
        "max_reported_completion_tokens": 20,
        "max_reported_total_tokens": 120,
        "minimum_reported_prompt_reservation_margin_tokens": 3484,
        "source_revision": revision,
        "reasoning_effort": spec["scope"]["reasoning_effort"],
        "reasoning_request_verified": True,
        "stop_policy": request_stop_policy(spec),
        "stop_sequence_request_verified": True,
        "model_count": 12,
        "record_count": 12 * 234 * 7,
        "api_call_count": 30000,
        "transport_attempt_count": 30000,
        "transport_retry_count": 0,
        "terminal_metadata_fault_attempt_count": 0,
        "request_error_attempt_count": 0,
        "transport_incomplete_logical_call_count": 0,
        "recovered_terminal_metadata_fault_logical_call_count": 0,
        "usage_metadata_valid_logical_call_count": 30000,
        "static_prompt_budget_verified_for_all_logical_calls": True,
        "truncated_completion_count": 0,
        "truncated_api_call_count": 0,
        "truncated_record_count": 0,
        "transport_protocol_invalid_record_count": 0,
        "invalid_record_count": 0,
        "condition_record_counts": {
            condition["id"]: 12 * 234 for condition in spec["conditions"]
        },
        "condition_truncated_record_counts": {
            condition["id"]: 0 for condition in spec["conditions"]
        },
        "condition_truncation_rates": {
            condition["id"]: 0.0 for condition in spec["conditions"]
        },
        "condition_transport_protocol_invalid_record_counts": {
            condition["id"]: 0 for condition in spec["conditions"]
        },
        "condition_transport_protocol_invalid_rates": {
            condition["id"]: 0.0 for condition in spec["conditions"]
        },
        "recovered_terminal_metadata_fault_record_count": 0,
        "condition_recovered_terminal_metadata_fault_record_counts": {
            condition["id"]: 0 for condition in spec["conditions"]
        },
        "condition_recovered_terminal_metadata_fault_record_rates": {
            condition["id"]: 0.0 for condition in spec["conditions"]
        },
        "condition_invalid_record_counts": {
            condition["id"]: 0 for condition in spec["conditions"]
        },
        "condition_invalid_record_rates": {
            condition["id"]: 0.0 for condition in spec["conditions"]
        },
        "fully_gpu_served_model_count": 12,
        "all_models_fully_gpu_served": True,
        "processor_requirement": "100% GPU",
    }
    _validate_formal_run_manifest(spec, run, revision)
    for key, value in (
        ("status", "provisional"),
        ("all_model_replays_passed", False),
    ):
        bad = dict(run)
        bad[key] = value
        with pytest.raises(RuntimeError):
            _validate_formal_run_manifest(spec, bad, revision)

    counted = copy.deepcopy(run)
    first = spec["conditions"][0]["id"]
    counted["truncated_completion_count"] = 2
    counted["truncated_api_call_count"] = 2
    counted["truncated_record_count"] = 1
    counted["condition_truncated_record_counts"][first] = 1
    counted["condition_truncation_rates"][first] = 1 / (12 * 234)
    counted["invalid_record_count"] = 1
    counted["condition_invalid_record_counts"][first] = 1
    counted["condition_invalid_record_rates"][first] = 1 / (12 * 234)
    _validate_formal_run_manifest(spec, counted, revision)


def test_full_synthetic_analysis_is_aggregate_only_and_detects_selectivity():
    result = analyze_arrays(_synthetic_data(), n_boot=40)
    assert result["primary"]["gamma"] == pytest.approx(0.15)
    assert result["primary"]["families_positive"] == 6
    assert result["primary"]["family_consistency_gate_at_least_4_of_6"] is True
    assert len(result["primary"]["intervention_by_group_gain_matrix"]) == 5
    assert all(
        len(row["group_mean_gain"]) == 5
        for row in result["primary"]["intervention_by_group_gain_matrix"]
    )
    assert result["predictive_family_lofo"]["delta_log_likelihood"] > 0
    assert result["predictive_family_lofo"]["families_improved"] == 6
    assert result["sensitivities"]["exclude_pair_if_either_response_empty"][
        "excluded_pairs"
    ] == 0
    assert set(result["sensitivities"]["difficulty_stratified"]) == {
        "easy",
        "medium",
        "hard",
    }
    frozen_gate_ids = {
        rule["id"]
        for rule in _synthetic_data()["spec"]["estimands"]["confirmatory_success_gate"]["rules"]
    }
    assert set(result["confirmatory_gate"]["components"]) == frozen_gate_ids
    assert list(result["confirmatory_gate"]["components"]) == list(CONFIRMATORY_GATE_IDS)
    assert result["confirmatory_gate"]["numeric_thresholds"] == (
        confirmatory_gate_thresholds(_synthetic_data()["spec"])
    )
    assert result["confirmatory_gate"]["pass"] is True
    rendered = repr(result).lower()
    assert "response_text" not in rendered
    assert "stimulus" not in rendered
    assert "expected_response" not in rendered


def test_amended_analyzer_reports_unestimable_empty_sensitivity_fail_closed():
    data = _synthetic_data()
    target_index = next(
        index
        for index, condition in enumerate(data["conditions"])
        if condition.get("target_group") == "theory_of_mind"
    )
    matched = [
        data["paradigms"].index(paradigm)
        for paradigm in data["spec"]["grouping"]["theory_of_mind"]
    ]
    data["empty"][0, target_index, matched, :] = True

    result = analyze_arrays_amended(data, n_boot=12)
    sensitivity = result["sensitivities"]["exclude_pair_if_either_response_empty"]
    assert result["primary"]["gamma"] == pytest.approx(0.15)
    assert sensitivity["estimable"] is False
    assert sensitivity["minimum_items_per_model_intervention_paradigm"] == 0
    assert sensitivity["gamma"] is None
    assert sensitivity["preservation_ratio_vs_primary"] is None
    assert sensitivity["s"] is None
    assert sensitivity["bootstrap"] is None
    assert "cell empty" in sensitivity["failure_reason"]
    assert result["confirmatory_gate"]["components"][
        "empty_exclusion_preserves_at_least_half_gamma_and_three_items_per_cell"
    ] is False
    assert result["confirmatory_gate"]["pass"] is False


def test_confirmatory_gate_ids_and_numeric_thresholds_fail_closed():
    spec = copy.deepcopy(load_spec())
    thresholds = confirmatory_gate_thresholds(spec)
    assert thresholds["minimum_sensitivity_gamma_preservation_ratio"] == 0.5
    assert thresholds["minimum_sensitivity_items_per_cell"] == 3

    drifted = copy.deepcopy(spec)
    drifted["estimands"]["confirmatory_success_gate"]["rules"] = list(reversed(
        drifted["estimands"]["confirmatory_success_gate"]["rules"]
    ))
    with pytest.raises(RuntimeError, match="IDs/order"):
        confirmatory_gate_thresholds(drifted)

    missing = copy.deepcopy(spec)
    del missing["estimands"]["confirmatory_success_gate"]["numeric_thresholds"][
        "minimum_sensitivity_items_per_cell"
    ]
    with pytest.raises(RuntimeError, match="threshold schema"):
        confirmatory_gate_thresholds(missing)


def test_protocol_invalid_is_primary_zero_but_has_frozen_paired_exclusion_sensitivity():
    data = _synthetic_data()
    target_index = next(
        index for index, condition in enumerate(data["conditions"])
        if condition["kind"] == "targeted"
    )
    condition_id = data["condition_ids"][target_index]
    original = float(data["accuracy"][0, target_index, 0, 0])
    data["accuracy"][0, target_index, 0, 0] = 0.0
    data["canonical_accuracy"][0, target_index, 0, 0] = 0.0
    data["protocol_invalid"][0, target_index, 0, 0] = True
    data["run"]["truncated_completion_count"] = 1
    data["run"]["truncated_api_call_count"] = 1
    data["run"]["truncated_record_count"] = 1
    data["run"]["invalid_record_count"] = 1
    data["run"]["condition_truncated_record_counts"][condition_id] = 1
    data["run"]["condition_truncation_rates"][condition_id] = 1 / (12 * 13 * 18)
    data["run"]["condition_invalid_record_counts"][condition_id] = 1
    data["run"]["condition_invalid_record_rates"][condition_id] = 1 / (12 * 13 * 18)
    data["response_chars"][0, target_index, 0, 0] = 10**9
    result = analyze_arrays(data, n_boot=12)
    sensitivity = result["sensitivities"][
        "exclude_pair_if_either_task_record_protocol_invalid"
    ]
    assert original > 0
    assert sensitivity["excluded_pairs"] == 1
    assert sensitivity["minimum_items_per_model_intervention_paradigm"] == 17
    assert sensitivity["estimable"] is True
    assert result["confirmatory_gate"]["components"][
        "each_condition_task_record_invalid_rate_le_0_01"
    ] is True
    length_sensitivity = result["sensitivities"]["response_character_length_adjustment"]
    assert length_sensitivity["excluded_protocol_invalid_pairs"] == 1
    assert length_sensitivity["minimum_items_per_model_intervention_paradigm"] == 17
    length_gamma = length_sensitivity["gamma"]
    data["response_chars"][0, target_index, 0, 0] = 0
    assert analyze_arrays(data, n_boot=12)["sensitivities"][
        "response_character_length_adjustment"
    ]["gamma"] == pytest.approx(length_gamma)

    data["spec"]["estimands"]["confirmatory_success_gate"]["numeric_thresholds"][
        "maximum_formal_condition_task_record_invalid_rate"
    ] = 0.0
    result = analyze_arrays(data, n_boot=12)
    assert result["confirmatory_gate"]["components"][
        "each_condition_task_record_invalid_rate_le_0_01"
    ] is False


def test_transport_incomplete_uses_the_same_generic_invalid_pair_mask():
    data = _synthetic_data()
    target_index = next(
        index for index, condition in enumerate(data["conditions"])
        if condition["kind"] == "targeted"
    )
    condition_id = data["condition_ids"][target_index]
    data["accuracy"][0, target_index, 0, 0] = 0.0
    data["canonical_accuracy"][0, target_index, 0, 0] = 0.0
    data["protocol_invalid"][0, target_index, 0, 0] = True
    data["run"]["transport_incomplete_logical_call_count"] = 1
    data["run"]["transport_protocol_invalid_record_count"] = 1
    data["run"]["invalid_record_count"] = 1
    data["run"]["condition_invalid_record_counts"][condition_id] = 1
    data["run"]["condition_invalid_record_rates"][condition_id] = 1 / (12 * 13 * 18)
    result = analyze_arrays(data, n_boot=12)
    sensitivity = result["sensitivities"][
        "exclude_pair_if_either_task_record_protocol_invalid"
    ]
    assert sensitivity["excluded_pairs"] == 1
    assert result["sensitivities"]["protocol_invalid_completion_policy"][
        "observed_transport_incomplete_call_count"
    ] == 1


def test_recovered_terminal_metadata_fault_has_separate_paired_exclusion_sensitivity():
    data = _synthetic_data()
    target_index = next(
        index for index, condition in enumerate(data["conditions"])
        if condition["kind"] == "targeted"
    )
    condition_id = data["condition_ids"][target_index]
    data["recovered_terminal_metadata_fault_exposed"][
        0, target_index, 0, 0
    ] = True
    data["run"]["recovered_terminal_metadata_fault_logical_call_count"] = 1
    data["run"]["recovered_terminal_metadata_fault_record_count"] = 1
    data["run"][
        "condition_recovered_terminal_metadata_fault_record_counts"
    ][condition_id] = 1
    data["run"][
        "condition_recovered_terminal_metadata_fault_record_rates"
    ][condition_id] = 1 / (12 * 13 * 18)
    result = analyze_arrays(data, n_boot=12)
    sensitivity = result["sensitivities"][
        "exclude_pair_if_either_task_record_recovered_terminal_metadata_fault_exposed"
    ]
    assert sensitivity["status"] == (
        "prespecified_descriptive_sensitivity_not_a_confirmatory_gate"
    )
    assert sensitivity["excluded_pairs"] == 1
    assert sensitivity["minimum_items_per_model_intervention_paradigm"] == 17
    assert sensitivity["estimable"] is True
    assert sensitivity["gamma"] == pytest.approx(result["primary"]["gamma"])
    assert sensitivity["preservation_ratio_vs_primary"] == pytest.approx(1.0)
    assert not any("recovered_terminal" in key for key in result["confirmatory_gate"]["components"])


def test_canonical_ospan_and_hard_minus_easy_are_separate_frozen_secondaries():
    data = _synthetic_data()
    strict_gamma = analyze_arrays(data, n_boot=12)["primary"]["gamma"]
    ospan = data["paradigms"].index("operation_span")
    targeted = [
        data["condition_ids"].index(entry["id"])
        for entry in data["conditions"]
        if entry["kind"] == "targeted"
    ]
    data["canonical_accuracy"][:, targeted, ospan, :] += 0.04
    result = analyze_arrays(data, n_boot=12)
    assert result["primary"]["gamma"] == pytest.approx(strict_gamma)
    assert result["secondary"]["canonical_whitespace_ospan_scoring"]["gamma"] != pytest.approx(
        strict_gamma
    )

    difficulty = _synthetic_data()
    for ci, condition in enumerate(difficulty["conditions"]):
        if condition["kind"] != "targeted":
            continue
        matched = set(difficulty["spec"]["grouping"][condition["target_group"]])
        for p, paradigm in enumerate(difficulty["paradigms"]):
            if paradigm in matched:
                difficulty["accuracy"][:, ci, p, 12:18] += 0.04
                difficulty["canonical_accuracy"][:, ci, p, 12:18] += 0.04
    interaction = analyze_arrays(difficulty, n_boot=12)["secondary"]["difficulty_interaction"]
    assert interaction["delta_gamma"] == pytest.approx(0.04)
    assert interaction["delta_s"] == pytest.approx([0.04] * 5)


def test_exact_two_sided_mapping_uses_permutation_mean_for_unequal_groups():
    data = _synthetic_data()
    _, groups, masks = _primary_inputs(data)
    means = np.asarray(
        [[0.02 * (j + 1) + 0.01 * (p + 1) ** 2 for p in range(13)] for j in range(5)]
    )
    result = exact_intervention_mapping_permutation(means, groups, masks)
    null = []
    for mapping in itertools.permutations(masks):
        _, gamma = _contrast_from_paradigm_means(means, list(mapping), masks)
        null.append(gamma)
    null = np.asarray(null)
    _, observed = _contrast_from_paradigm_means(means, groups, masks)
    expected = np.mean(
        np.abs(null - null.mean()) >= abs(observed - null.mean()) - 1e-15
    )
    assert result["gamma_null_mean"] == pytest.approx(null.mean())
    assert result["gamma_p_two"] == pytest.approx(expected)


def test_primary_response_units_exclude_ospan_math_and_cvlt_filler():
    ospan_record = {"responses": [{"response": ""}, {"response": "A B C"}]}
    ospan_entry = {"is_multiturn": True, "paradigm": "operation_span"}
    assert _primary_response_texts(ospan_record, ospan_entry, object()) == ["A B C"]

    cvlt_record = {"responses": [{"response": ""}, {"response": "apple pear"}]}
    cvlt_entry = {"is_multiturn": True, "paradigm": "cvlt_word_list"}
    cvlt_item = SimpleNamespace(
        metadata=SimpleNamespace(
            parameters={"turns": [{"type": "filler_task"}, {"expected_words": ["apple"]}]}
        )
    )
    assert _primary_response_texts(cvlt_record, cvlt_entry, cvlt_item) == ["apple pear"]


def test_aggregate_output_denylist_rejects_item_or_raw_payloads():
    validate_aggregate_output({"primary": {"gamma": 0.1}})
    for forbidden in ("response", "stimulus", "gold", "task_id", "strict_tokens"):
        with pytest.raises(RuntimeError, match="leaked"):
            validate_aggregate_output({"nested": {forbidden: "secret"}})


def test_analyzer_refuses_noncanonical_raw_or_output_roots(tmp_path):
    validate_canonical_analysis_paths(RESULTS_ROOT / "raw", RESULTS_ROOT / "analysis")
    with pytest.raises(RuntimeError, match="noncanonical formal raw root"):
        validate_canonical_analysis_paths(tmp_path / "raw-copy", RESULTS_ROOT / "analysis")
    with pytest.raises(RuntimeError, match="noncanonical analysis output"):
        validate_canonical_analysis_paths(RESULTS_ROOT / "raw", tmp_path / "analysis-copy")
