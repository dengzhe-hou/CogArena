#!/usr/bin/env python3
"""Post-hoc target-versus-baseline sensitivity for causal selectivity.

The frozen study compared five targeted scaffolds with a length-matched neutral
placebo.  A later control audit found that the placebo reduced aggregate
accuracy relative to the no-scaffold baseline.  This isolated sensitivity
replays the same closed records and the same strict primary scores, replaces
the placebo reference with baseline, and uses the frozen diagonal weighting.

The analysis is explicitly post-hoc and cannot alter the primary estimand,
thresholds, artifacts, or nine-gate FAIL decision.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .analyze import (
    _contrast_from_paradigm_means,
    _target_structure,
    compute_estimate,
    exact_intervention_mapping_permutation,
    load_formal_arrays,
    validate_aggregate_output,
)
from .common import (
    ROOT,
    RESULTS_ROOT,
    SPEC_PATH,
    atomic_write_json,
    jsonable,
    load_json,
    manifest_path,
    require,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
SENSITIVITY_SPEC_PATH = HERE / "TARGET_BASELINE_SENSITIVITY_SPEC.json"
LAUNCHER_PATH = HERE / "analyze_target_baseline.sbatch"
FROZEN_ANALYZER_PATH = HERE / "analyze.py"
PRIMARY_OUTPUT_DIR = RESULTS_ROOT / "analysis"
EXPLORATORY_OUTPUT_DIR = PRIMARY_OUTPUT_DIR / "exploratory"
OUTPUT_DIR = PRIMARY_OUTPUT_DIR / "target_baseline_sensitivity"
OUTPUT_PATH = OUTPUT_DIR / "target_baseline_results.json"
MANIFEST_PATH = OUTPUT_DIR / "TARGET_BASELINE_MANIFEST.json"
SCHEMA = "cogarena.causal_selectivity.target_baseline_sensitivity.v1"
MANIFEST_SCHEMA = "cogarena.causal_selectivity.target_baseline_manifest.v1"
SPEC_SCHEMA = "cogarena.causal_selectivity.target_baseline_sensitivity_spec.v1"
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 42
RUNTIME_MANIFEST_NAME = "TARGET_BASELINE_RUNTIME_MANIFEST.json"
CONTRAST_ORDER = (
    "target_minus_placebo",
    "target_minus_baseline",
    "placebo_minus_baseline_differential",
)


def validate_sensitivity_spec() -> dict[str, Any]:
    require(SENSITIVITY_SPEC_PATH.is_file(), "target-baseline sensitivity spec missing")
    spec = load_json(SENSITIVITY_SPEC_PATH)
    require(
        spec.get("schema_version") == SPEC_SCHEMA
        and spec.get("study_id") == "causal_selectivity_20260720"
        and spec.get("status")
        == "post_hoc_sensitivity_authored_after_primary_and_control_audit_inspection"
        and spec.get("confirmatory_role") == "none",
        "target-baseline sensitivity spec identity/status mismatch",
    )
    estimands = spec.get("estimands")
    require(
        isinstance(estimands, dict)
        and list(estimands)
        == [
            "target_minus_baseline_diagonal",
            "placebo_reference_identity",
            "family_item_uncertainty",
            "exact_mapping",
            "small_g_family",
        ],
        "target-baseline estimand list/order mismatch",
    )
    bootstrap = spec.get("bootstrap")
    require(
        isinstance(bootstrap, dict)
        and bootstrap.get("replicates") == N_BOOTSTRAP
        and bootstrap.get("seed") == BOOTSTRAP_SEED
        and bootstrap.get("quantiles") == [0.025, 0.975],
        "target-baseline bootstrap contract mismatch",
    )
    constraints = spec.get("interpretation_constraints")
    require(
        isinstance(constraints, list)
        and len(constraints) >= 8
        and all(isinstance(value, str) and value for value in constraints),
        "target-baseline interpretation constraints missing",
    )
    return spec


def _analysis_revision() -> str:
    revision = os.environ.get("COGARENA_TARGET_BASELINE_GIT_HEAD", "").strip()
    require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "COGARENA_TARGET_BASELINE_GIT_HEAD must be an injected 40-hex revision",
    )
    return revision


def validate_runtime_binding(
    sensitivity_spec: dict[str, Any], analysis_revision: str
) -> dict[str, Any]:
    """Require execution from a frozen-source archive plus three-file overlay."""
    authority_text = os.environ.get("COGARENA_ANALYSIS_AUTHORITY_ROOT", "").strip()
    runtime_manifest_sha = os.environ.get(
        "COGARENA_TARGET_BASELINE_RUNTIME_MANIFEST_SHA256", ""
    ).strip()
    require(authority_text, "COGARENA_ANALYSIS_AUTHORITY_ROOT is required")
    require(
        len(runtime_manifest_sha) == 64
        and all(character in "0123456789abcdef" for character in runtime_manifest_sha),
        "runtime-manifest SHA-256 must be injected",
    )
    authority_root = Path(authority_text).resolve()
    formal_revision = sensitivity_spec["formal_raw_source_revision"]
    expected_runtime_root = (
        authority_root
        / "results"
        / "causal_selectivity_20260720"
        / "runtime"
        / f"formal_{formal_revision}__analysis_{analysis_revision}"
    )
    require(
        ROOT.resolve() == expected_runtime_root.resolve(),
        "COGARENA_ROOT is not the canonical frozen target-baseline runtime",
    )
    expected_module = (
        ROOT
        / "scripts"
        / "experiments"
        / "causal_selectivity_20260720"
        / "analyze_target_baseline.py"
    )
    require(
        Path(__file__).resolve() == expected_module.resolve(),
        "target-baseline module was not imported from the frozen runtime overlay",
    )
    results_link = ROOT / "results"
    require(results_link.is_symlink(), "frozen runtime results path is not a symlink")
    require(
        results_link.resolve() == (authority_root / "results").resolve(),
        "frozen runtime results symlink does not target the authority tree",
    )
    runtime_manifest_path = ROOT / RUNTIME_MANIFEST_NAME
    require(runtime_manifest_path.is_file(), "frozen runtime manifest missing")
    require(
        sha256_file(runtime_manifest_path) == runtime_manifest_sha,
        "frozen runtime manifest hash differs from submission binding",
    )
    runtime_manifest = load_json(runtime_manifest_path)
    overlay_paths = [
        "scripts/experiments/causal_selectivity_20260720/"
        "TARGET_BASELINE_SENSITIVITY_SPEC.json",
        "scripts/experiments/causal_selectivity_20260720/"
        "analyze_target_baseline.py",
        "scripts/experiments/causal_selectivity_20260720/"
        "analyze_target_baseline.sbatch",
    ]
    require(
        runtime_manifest.get("schema_version")
        == "cogarena.causal_selectivity.target_baseline_runtime.v1"
        and runtime_manifest.get("formal_source_revision") == formal_revision
        and runtime_manifest.get("analysis_overlay_revision") == analysis_revision
        and runtime_manifest.get("overlay_paths") == overlay_paths,
        "frozen runtime manifest identity/revision/overlay mismatch",
    )
    overlay_sha = runtime_manifest.get("overlay_sha256")
    require(
        isinstance(overlay_sha, dict) and list(overlay_sha) == overlay_paths,
        "frozen runtime overlay hash map mismatch",
    )
    for relative, expected_sha in overlay_sha.items():
        require(
            sha256_file(ROOT / relative) == expected_sha,
            f"frozen runtime analysis overlay drift: {relative}",
        )
    require(
        sha256_file(SENSITIVITY_SPEC_PATH)
        == overlay_sha[overlay_paths[0]]
        and sha256_file(Path(__file__)) == overlay_sha[overlay_paths[1]]
        and sha256_file(LAUNCHER_PATH) == overlay_sha[overlay_paths[2]],
        "active target-baseline files differ from the verified overlay",
    )
    return {
        "authority_root": authority_root,
        "manifest_path": runtime_manifest_path,
        "manifest_sha256": runtime_manifest_sha,
        "manifest": runtime_manifest,
    }


def validate_primary_bindings(
    data: dict[str, Any], sensitivity_spec: dict[str, Any]
) -> dict[str, Any]:
    """Bind authoritative results while replaying raw data with frozen sources."""
    result_path = PRIMARY_OUTPUT_DIR / "analysis_results.json"
    primary_manifest_path = PRIMARY_OUTPUT_DIR / "ANALYSIS_MANIFEST.json"
    require(result_path.is_file(), "primary analysis result missing")
    require(primary_manifest_path.is_file(), "primary analysis manifest missing")
    result = load_json(result_path)
    manifest = load_json(primary_manifest_path)
    require(
        manifest.get("schema_version")
        == "cogarena.causal_selectivity.analysis_manifest.v3"
        and manifest.get("status") == "complete"
        and manifest.get("confirmatory_gate_pass") is False,
        "primary analysis manifest status mismatch",
    )
    require(
        sensitivity_spec.get("authoritative_primary_results_sha256")
        == sha256_file(result_path)
        == manifest.get("outputs_sha256", {}).get("analysis_results.json"),
        "primary result hash differs from sensitivity/manifest binding",
    )
    require(
        sensitivity_spec.get("authoritative_primary_manifest_sha256")
        == sha256_file(primary_manifest_path),
        "primary manifest hash differs from sensitivity binding",
    )
    require(
        manifest.get("analyzer_sha256")
        == sensitivity_spec.get("authoritative_primary_amended_analyzer_sha256"),
        "primary manifest amended-analyzer binding mismatch",
    )
    require(
        manifest.get("frozen_analyzer_sha256")
        == sensitivity_spec.get("authoritative_frozen_analyzer_sha256")
        == sha256_file(FROZEN_ANALYZER_PATH),
        "executed frozen analyzer differs from primary manifest binding",
    )
    require(
        manifest.get("spec_sha256") == sha256_file(SPEC_PATH)
        and manifest.get("formal_item_manifest_sha256")
        == sha256_file(manifest_path("formal"))
        and manifest.get("formal_run_manifest_sha256")
        == sha256_file(RESULTS_ROOT / "RUN_MANIFEST_formal.json"),
        "primary manifest input binding mismatch",
    )
    require(
        data["source_revision"] == manifest.get("source_revision"),
        "replayed raw source revision differs from primary analysis",
    )
    require(
        result.get("confirmatory_gate", {}).get("pass") is False,
        "primary nine-gate decision is not FAIL",
    )
    return {"result": result, "manifest": manifest}


def _binomial_upper_tail(n: int, observed: int) -> float:
    return float(
        sum(math.comb(n, value) for value in range(observed, n + 1)) / 2**n
    )


def small_g_family_inference(
    model_s: np.ndarray, families: list[str]
) -> dict[str, Any]:
    """Exact six-family summaries copied into this isolated overlay."""
    require(model_s.ndim == 2 and model_s.shape[1] == 5, "bad model-S matrix")
    model_gamma = model_s.mean(axis=1)
    family_order = list(dict.fromkeys(families))
    family_gamma = np.asarray(
        [
            model_gamma[np.asarray(families) == family].mean()
            for family in family_order
        ],
        dtype=float,
    )
    require(len(family_gamma) == 6, "small-G analysis requires six families")
    nonzero = family_gamma[np.abs(family_gamma) > 1e-15]
    positive = int((nonzero > 0).sum())
    negative = int((nonzero < 0).sum())
    n = len(nonzero)
    p_upper = _binomial_upper_tail(n, positive)
    p_lower = float(
        sum(math.comb(n, value) for value in range(0, positive + 1)) / 2**n
    )
    sign_vectors = np.asarray(list(itertools.product((-1.0, 1.0), repeat=6)))
    sign_flip_null = (sign_vectors * family_gamma).mean(axis=1)
    observed = float(family_gamma.mean())
    sign_flip_one = float(np.mean(sign_flip_null >= observed - 1e-15))
    sign_flip_two = float(
        np.mean(np.abs(sign_flip_null) >= abs(observed) - 1e-15)
    )
    lofo = []
    for index, family in enumerate(family_order):
        keep = np.ones(6, dtype=bool)
        keep[index] = False
        lofo.append(
            {
                "omitted_family": family,
                "gamma": float(family_gamma[keep].mean()),
            }
        )
    return {
        "status": "post_hoc_sensitivity_small_G",
        "family_unit": "mean of the two frozen checkpoints within each model family",
        "family_gamma": {
            family: float(value)
            for family, value in zip(family_order, family_gamma)
        },
        "exact_binomial_sign_test": {
            "nonzero_families": n,
            "positive": positive,
            "negative": negative,
            "p_one_sided_positive": p_upper,
            "p_two_sided_doubled_minimum_tail": min(
                1.0, 2 * min(p_upper, p_lower)
            ),
            "assumption": "under the null, positive and negative family effects are equiprobable; magnitudes are ignored",
        },
        "exact_family_sign_flip_test": {
            "n_assignments": 64,
            "minimum_attainable_one_sided_p": 1 / 64,
            "observed_mean_family_gamma": observed,
            "p_one_sided_positive": sign_flip_one,
            "p_two_sided_absolute": sign_flip_two,
            "assumption": "the six family-level effects are independent and sign-symmetric under the null",
        },
        "leave_one_family_out": {
            "rows": lofo,
            "gamma_min": min(row["gamma"] for row in lofo),
            "gamma_max": max(row["gamma"] for row in lofo),
        },
        "interpretation": "six clusters give coarse exact-test resolution; report every family estimate and do not substitute asymptotic cluster-robust inference",
    }


def validate_control_audit_binding(
    data: dict[str, Any], sensitivity_spec: dict[str, Any]
) -> dict[str, Any]:
    result_path = EXPLORATORY_OUTPUT_DIR / "exploratory_results.json"
    manifest_file = EXPLORATORY_OUTPUT_DIR / "EXPLORATORY_MANIFEST.json"
    require(result_path.is_file(), "exploratory control-audit result missing")
    require(manifest_file.is_file(), "exploratory control-audit manifest missing")
    result = load_json(result_path)
    manifest = load_json(manifest_file)
    require(
        result.get("schema_version")
        == "cogarena.causal_selectivity.exploratory_analysis.v1"
        and result.get("status") == "post_hoc_exploratory_only"
        and result.get("confirmatory_role") == "none",
        "exploratory control-audit result identity mismatch",
    )
    require(
        manifest.get("schema_version")
        == "cogarena.causal_selectivity.exploratory_manifest.v1"
        and manifest.get("status") == "complete_post_hoc_exploratory"
        and manifest.get("confirmatory_role") == "none",
        "exploratory control-audit manifest identity mismatch",
    )
    require(
        manifest.get("outputs_sha256", {}).get("exploratory_results.json")
        == sha256_file(result_path),
        "exploratory control-audit result hash mismatch",
    )
    require(
        sensitivity_spec.get("authoritative_exploratory_results_sha256")
        == sha256_file(result_path),
        "sensitivity spec binds a different exploratory result",
    )
    require(
        sensitivity_spec.get("formal_raw_source_revision") == data["source_revision"],
        "sensitivity spec binds a different formal raw revision",
    )
    return {"result": result, "manifest": manifest}


def build_contrasts(
    data: dict[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    list[int],
    int,
    int,
    list[str],
    dict[str, np.ndarray],
]:
    target_indices, placebo_index, target_groups, group_masks = _target_structure(data)
    baseline_index = data["condition_ids"].index("baseline")
    accuracy = np.asarray(data["accuracy"], dtype=float)
    target = accuracy[:, target_indices, :, :]
    placebo = accuracy[:, placebo_index, :, :]
    baseline = accuracy[:, baseline_index, :, :]
    target_minus_placebo = target - placebo[:, None, :, :]
    target_minus_baseline = target - baseline[:, None, :, :]
    placebo_minus_baseline = np.broadcast_to(
        (placebo - baseline)[:, None, :, :], target.shape
    ).copy()
    require(
        np.allclose(
            target_minus_placebo,
            target_minus_baseline - placebo_minus_baseline,
            rtol=0,
            atol=1e-15,
        ),
        "paired target/placebo/baseline score identity failed",
    )
    contrasts = {
        "target_minus_placebo": target_minus_placebo,
        "target_minus_baseline": target_minus_baseline,
        "placebo_minus_baseline_differential": placebo_minus_baseline,
    }
    return (
        contrasts,
        target_indices,
        placebo_index,
        baseline_index,
        target_groups,
        group_masks,
    )


def _sign_tail(samples: np.ndarray) -> float:
    return float(
        min(1.0, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0)))
    )


def joint_crossed_family_item_bootstrap(
    contrasts: dict[str, np.ndarray],
    families: list[str],
    target_groups: list[str],
    group_masks: dict[str, np.ndarray],
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Frozen family-by-item draws shared by all three linear contrasts."""
    require(
        list(contrasts) == list(CONTRAST_ORDER),
        "joint-bootstrap contrast list/order mismatch",
    )
    shapes = {values.shape for values in contrasts.values()}
    require(len(shapes) == 1, "joint-bootstrap contrasts have unequal shapes")
    shape = next(iter(shapes))
    require(
        len(shape) == 4 and shape[1:] == (5, 13, 18),
        f"joint-bootstrap contrast shape mismatch: {shape}",
    )
    family_order = list(dict.fromkeys(families))
    family_models = [
        np.flatnonzero(np.asarray(families) == family) for family in family_order
    ]
    require(
        len(family_models) == 6 and all(len(indices) == 2 for indices in family_models),
        "joint bootstrap requires six families with two checkpoints each",
    )
    gamma_samples = {
        name: np.empty(n_boot, dtype=float) for name in CONTRAST_ORDER
    }
    s_samples = {
        name: np.empty((n_boot, len(target_groups)), dtype=float)
        for name in CONTRAST_ORDER
    }
    rng = np.random.default_rng(seed)
    for draw in range(n_boot):
        sampled_families = rng.integers(0, 6, size=6)
        model_indices = np.concatenate(
            [family_models[index] for index in sampled_families]
        )
        means = {
            name: np.empty((shape[1], shape[2]), dtype=float)
            for name in CONTRAST_ORDER
        }
        for paradigm_index in range(shape[2]):
            item_indices = rng.integers(0, shape[3], size=shape[3])
            for name, values in contrasts.items():
                sampled = values[model_indices, :, paradigm_index, :][
                    :, :, item_indices
                ]
                means[name][:, paradigm_index] = sampled.mean(axis=(0, 2))
        for name in CONTRAST_ORDER:
            s, gamma = _contrast_from_paradigm_means(
                means[name], target_groups, group_masks
            )
            s_samples[name][draw] = s
            gamma_samples[name][draw] = gamma

    gamma_identity_error = (
        gamma_samples["target_minus_placebo"]
        - gamma_samples["target_minus_baseline"]
        + gamma_samples["placebo_minus_baseline_differential"]
    )
    s_identity_error = (
        s_samples["target_minus_placebo"]
        - s_samples["target_minus_baseline"]
        + s_samples["placebo_minus_baseline_differential"]
    )
    require(
        np.max(np.abs(gamma_identity_error)) < 1e-12
        and np.max(np.abs(s_identity_error)) < 1e-12,
        "joint-bootstrap linear identity failed",
    )
    summaries = {}
    for name in CONTRAST_ORDER:
        summaries[name] = {
            "gamma_ci95": np.quantile(
                gamma_samples[name], [0.025, 0.975]
            ).tolist(),
            "descriptive_bootstrap_sign_tail_fraction_two_sided": _sign_tail(
                gamma_samples[name]
            ),
            "s_ci95": np.quantile(
                s_samples[name], [0.025, 0.975], axis=0
            ).T.tolist(),
        }

    placebo_contribution = (
        gamma_samples["target_minus_placebo"]
        - gamma_samples["target_minus_baseline"]
    )
    placebo_contribution_s = (
        s_samples["target_minus_placebo"]
        - s_samples["target_minus_baseline"]
    )
    denominator = gamma_samples["target_minus_placebo"]
    ratio_mask = np.abs(denominator) > 1e-12
    ratios = placebo_contribution[ratio_mask] / denominator[ratio_mask]
    require(ratio_mask.any(), "all bootstrap share denominators are zero")
    return {
        "method": "frozen crossed family-by-item bootstrap: resample six families retaining both checkpoints, then 18 items independently within each of 13 fixed paradigms; identical draws for all contrasts",
        "n_bootstrap": n_boot,
        "seed": seed,
        "quantile_method": "numpy.quantile default linear",
        "contrasts": summaries,
        "placebo_reference_contribution_to_target_minus_placebo_selectivity": {
            "definition": "Gamma(target-minus-placebo) minus Gamma(target-minus-baseline), equivalently Gamma(baseline-minus-placebo)",
            "gamma_ci95": np.quantile(
                placebo_contribution, [0.025, 0.975]
            ).tolist(),
            "descriptive_bootstrap_sign_tail_fraction_two_sided": _sign_tail(
                placebo_contribution
            ),
            "s_ci95": np.quantile(
                placebo_contribution_s, [0.025, 0.975], axis=0
            ).T.tolist(),
            "share_ratio_defined_draws": int(ratio_mask.sum()),
            "share_ratio_total_draws": n_boot,
            "share_ratio_median": float(np.median(ratios)),
            "share_ratio_ci95": np.quantile(ratios, [0.025, 0.975]).tolist(),
            "ratio_warning": "descriptive ratio with an estimated denominator; absolute Gamma contribution and interval take priority",
        },
        "maximum_gamma_identity_error": float(
            np.max(np.abs(gamma_identity_error))
        ),
        "maximum_s_identity_error": float(np.max(np.abs(s_identity_error))),
    }


def _contrast_rows(
    data: dict[str, Any],
    estimates: dict[str, dict[str, Any]],
    target_indices: list[int],
    target_groups: list[str],
    group_masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for intervention_index, (condition_index, group) in enumerate(
        zip(target_indices, target_groups)
    ):
        matched = group_masks[group]
        values = {}
        for contrast_name, estimate in estimates.items():
            paradigm_means = estimate["intervention_paradigm_mean_gain"][
                intervention_index
            ]
            values[contrast_name] = {
                "all_paradigm_mean": float(paradigm_means.mean()),
                "matched_mean": float(paradigm_means[matched].mean()),
                "unmatched_mean": float(paradigm_means[~matched].mean()),
                "matched_minus_unmatched_s": float(
                    estimate["s"][intervention_index]
                ),
            }
        identity_error = (
            values["target_minus_placebo"]["matched_minus_unmatched_s"]
            - values["target_minus_baseline"]["matched_minus_unmatched_s"]
            + values["placebo_minus_baseline_differential"][
                "matched_minus_unmatched_s"
            ]
        )
        rows.append(
            {
                "intervention": data["condition_ids"][condition_index],
                "target_group": group,
                "contrasts": values,
                "s_identity_error": identity_error,
            }
        )
    return rows


def analyze_target_baseline(
    data: dict[str, Any],
    primary_binding: dict[str, Any],
    control_binding: dict[str, Any],
    sensitivity_spec: dict[str, Any],
    *,
    n_boot: int = N_BOOTSTRAP,
) -> dict[str, Any]:
    (
        contrasts,
        target_indices,
        placebo_index,
        baseline_index,
        target_groups,
        group_masks,
    ) = build_contrasts(data)
    estimates = {
        name: compute_estimate(values, target_groups, group_masks)
        for name, values in contrasts.items()
    }
    target_placebo = estimates["target_minus_placebo"]
    target_baseline = estimates["target_minus_baseline"]
    placebo_baseline = estimates["placebo_minus_baseline_differential"]
    primary = primary_binding["result"]["primary"]
    require(
        abs(target_placebo["gamma"] - float(primary["gamma"])) < 1e-12,
        "target-minus-placebo Gamma does not reproduce authoritative primary",
    )
    require(
        np.allclose(
            target_placebo["s"],
            np.asarray([row["s"] for row in primary["interventions"]], dtype=float),
            rtol=0,
            atol=1e-12,
        ),
        "target-minus-placebo S does not reproduce authoritative primary",
    )
    point_gamma_identity_error = (
        target_placebo["gamma"]
        - target_baseline["gamma"]
        + placebo_baseline["gamma"]
    )
    point_s_identity_error = (
        target_placebo["s"] - target_baseline["s"] + placebo_baseline["s"]
    )
    require(
        abs(point_gamma_identity_error) < 1e-12
        and np.max(np.abs(point_s_identity_error)) < 1e-12,
        "point-estimate target/placebo/baseline identity failed",
    )

    bootstrap = joint_crossed_family_item_bootstrap(
        contrasts,
        data["families"],
        target_groups,
        group_masks,
        n_boot=n_boot,
        seed=BOOTSTRAP_SEED,
    )
    authoritative_bootstrap = primary["bootstrap"]
    require(
        np.allclose(
            bootstrap["contrasts"]["target_minus_placebo"]["gamma_ci95"],
            authoritative_bootstrap["gamma_ci95"],
            rtol=0,
            atol=1e-15,
        )
        and np.allclose(
            bootstrap["contrasts"]["target_minus_placebo"]["s_ci95"],
            authoritative_bootstrap["s_ci95"],
            rtol=0,
            atol=1e-15,
        ),
        "joint bootstrap does not reproduce authoritative primary interval",
    )

    target_baseline_permutation = exact_intervention_mapping_permutation(
        target_baseline["intervention_paradigm_mean_gain"],
        target_groups,
        group_masks,
    )
    target_placebo_permutation = exact_intervention_mapping_permutation(
        target_placebo["intervention_paradigm_mean_gain"],
        target_groups,
        group_masks,
    )
    require(
        target_placebo_permutation == primary["exact_mapping_permutation"],
        "target-minus-placebo exact mapping test differs from authoritative primary",
    )

    accuracy = np.asarray(data["accuracy"], dtype=float)
    target = accuracy[:, target_indices, :, :]
    placebo = accuracy[:, placebo_index, :, :]
    baseline = accuracy[:, baseline_index, :, :]
    global_target_placebo = float((target - placebo[:, None]).mean())
    global_target_baseline = float((target - baseline[:, None]).mean())
    global_placebo_baseline = float((placebo - baseline).mean())
    global_placebo_reference_contribution = -global_placebo_baseline
    require(
        abs(
            global_target_placebo
            - global_target_baseline
            - global_placebo_reference_contribution
        )
        < 1e-12,
        "global target/placebo/baseline identity failed",
    )
    audited_placebo_difference = control_binding["result"][
        "neutral_placebo_vs_baseline"
    ]["primary_accuracy"]["paired_difference"]
    require(
        abs(global_placebo_baseline - float(audited_placebo_difference)) < 1e-12,
        "raw replay differs from existing placebo-minus-baseline control audit",
    )

    gamma_placebo_contribution = (
        target_placebo["gamma"] - target_baseline["gamma"]
    )
    gamma_share = (
        gamma_placebo_contribution / target_placebo["gamma"]
        if abs(target_placebo["gamma"]) > 1e-12
        else None
    )
    global_share = (
        global_placebo_reference_contribution / global_target_placebo
        if abs(global_target_placebo) > 1e-12
        else None
    )
    family_target_placebo = small_g_family_inference(
        target_placebo["model_s"], data["families"]
    )
    family_target_baseline = small_g_family_inference(
        target_baseline["model_s"], data["families"]
    )
    family_placebo_baseline = small_g_family_inference(
        placebo_baseline["model_s"], data["families"]
    )
    family_rows = []
    for family in family_target_baseline["family_gamma"]:
        tp = family_target_placebo["family_gamma"][family]
        tb = family_target_baseline["family_gamma"][family]
        pb = family_placebo_baseline["family_gamma"][family]
        require(abs(tp - tb + pb) < 1e-12, f"family identity failed: {family}")
        family_rows.append(
            {
                "family": family,
                "gamma_target_minus_placebo": tp,
                "gamma_target_minus_baseline": tb,
                "gamma_placebo_minus_baseline_differential": pb,
                "gamma_placebo_reference_contribution": tp - tb,
            }
        )

    output = {
        "schema_version": SCHEMA,
        "status": "complete_post_hoc_sensitivity",
        "confirmatory_role": "none",
        "primary_confirmatory_gate_remains_fail": True,
        "estimand": sensitivity_spec["estimands"]["target_minus_baseline_diagonal"],
        "contrast_identity": (
            "target-minus-placebo = target-minus-baseline + baseline-minus-placebo; "
            "therefore Gamma(T-P) = Gamma(T-B) - Gamma(P-B)"
        ),
        "point_estimates": {
            "gamma_target_minus_placebo": target_placebo["gamma"],
            "gamma_target_minus_baseline": target_baseline["gamma"],
            "gamma_placebo_minus_baseline_differential": placebo_baseline["gamma"],
            "gamma_placebo_reference_contribution_to_target_minus_placebo": (
                gamma_placebo_contribution
            ),
            "gamma_placebo_reference_contribution_share": gamma_share,
            "s_target_minus_placebo": target_placebo["s"].tolist(),
            "s_target_minus_baseline": target_baseline["s"].tolist(),
            "s_placebo_minus_baseline_differential": placebo_baseline["s"].tolist(),
            "maximum_point_s_identity_error": float(
                np.max(np.abs(point_s_identity_error))
            ),
            "point_gamma_identity_error": point_gamma_identity_error,
        },
        "global_accuracy_levels_and_differences": {
            "baseline_mean": float(baseline.mean()),
            "neutral_placebo_mean": float(placebo.mean()),
            "targeted_arms_mean": float(target.mean()),
            "target_minus_placebo_mean": global_target_placebo,
            "target_minus_baseline_mean": global_target_baseline,
            "neutral_placebo_minus_baseline_mean": global_placebo_baseline,
            "baseline_minus_placebo_contribution_to_target_minus_placebo_mean": (
                global_placebo_reference_contribution
            ),
            "placebo_reference_contribution_share_of_target_minus_placebo_mean": (
                global_share
            ),
            "identity_error": (
                global_target_placebo
                - global_target_baseline
                - global_placebo_reference_contribution
            ),
            "weighting": "equal items, paradigms, models, and five targeted arms in this balanced design",
        },
        "per_scaffold": _contrast_rows(
            data,
            estimates,
            target_indices,
            target_groups,
            group_masks,
        ),
        "joint_crossed_family_item_bootstrap": bootstrap,
        "target_minus_baseline_exact_mapping_permutation": (
            target_baseline_permutation
        ),
        "small_g_family_inference_target_minus_baseline": (
            family_target_baseline
        ),
        "family_contrast_decomposition": family_rows,
        "authoritative_target_minus_placebo_replay": {
            "gamma": target_placebo["gamma"],
            "bootstrap_reproduced_exactly": True,
            "exact_mapping_permutation_reproduced_exactly": True,
        },
        "existing_control_audit_replay": {
            "neutral_placebo_minus_baseline_accuracy": global_placebo_baseline,
            "reproduced_exactly": True,
        },
        "interpretation": {
            "global_mean_question": "A positive target-minus-placebo global mean can be generated by placebo harm even when targeted arms do not exceed baseline.",
            "selectivity_question": "Only the grouping-differential placebo-minus-baseline contrast affects diagonal Gamma; the overall placebo decrement cancels from matched-minus-unmatched selectivity.",
            "ratio_warning": "Contribution shares are descriptive ratios with estimated denominators; use absolute effects and intervals for inference.",
        },
        "interpretation_constraints": sensitivity_spec[
            "interpretation_constraints"
        ],
    }
    validate_aggregate_output(output)
    return output


def validate_paths(output_dir: Path) -> None:
    require(
        output_dir.resolve() == OUTPUT_DIR.resolve(),
        "noncanonical target-baseline output directory refused",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    validate_paths(args.output_dir)
    sensitivity_spec = validate_sensitivity_spec()
    analysis_revision = _analysis_revision()
    runtime_binding = validate_runtime_binding(
        sensitivity_spec, analysis_revision
    )
    data = load_formal_arrays()
    primary_binding = validate_primary_bindings(
        data, sensitivity_spec
    )
    control_binding = validate_control_audit_binding(data, sensitivity_spec)
    results = analyze_target_baseline(
        data,
        primary_binding,
        control_binding,
        sensitivity_spec,
        n_boot=N_BOOTSTRAP,
    )
    validate_aggregate_output(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / OUTPUT_PATH.name, jsonable(results))
    require(LAUNCHER_PATH.is_file(), "target-baseline Slurm launcher missing")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "study_id": "causal_selectivity_20260720",
        "status": "complete_post_hoc_sensitivity",
        "confirmatory_role": "none",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "formal_raw_source_revision": data["source_revision"],
        "target_baseline_analysis_revision": analysis_revision,
        "frozen_runtime": {
            "runtime_manifest_sha256": runtime_binding["manifest_sha256"],
            "formal_source_revision": runtime_binding["manifest"][
                "formal_source_revision"
            ],
            "formal_source_git_tree": runtime_binding["manifest"][
                "formal_source_git_tree"
            ],
            "analysis_overlay_revision": runtime_binding["manifest"][
                "analysis_overlay_revision"
            ],
            "analysis_overlay_sha256": runtime_binding["manifest"][
                "overlay_sha256"
            ],
            "frozen_dependency_sha256": load_json(
                manifest_path("formal")
            )["source_sha256"],
            "all_frozen_sources_revalidated_by_loader": True,
            "active_module_loaded_from_frozen_runtime_overlay": True,
            "authority_results_symlink_verified": True,
        },
        "inputs_sha256": {
            "PREPILOT_SPEC.json": sha256_file(SPEC_PATH),
            "formal_item_manifest.json": sha256_file(manifest_path("formal")),
            "RUN_MANIFEST_formal.json": sha256_file(
                RESULTS_ROOT / "RUN_MANIFEST_formal.json"
            ),
            "analysis_results.json": sha256_file(
                PRIMARY_OUTPUT_DIR / "analysis_results.json"
            ),
            "ANALYSIS_MANIFEST.json": sha256_file(
                PRIMARY_OUTPUT_DIR / "ANALYSIS_MANIFEST.json"
            ),
            "exploratory_results.json": sha256_file(
                EXPLORATORY_OUTPUT_DIR / "exploratory_results.json"
            ),
            "EXPLORATORY_MANIFEST.json": sha256_file(
                EXPLORATORY_OUTPUT_DIR / "EXPLORATORY_MANIFEST.json"
            ),
            "TARGET_BASELINE_SENSITIVITY_SPEC.json": sha256_file(
                SENSITIVITY_SPEC_PATH
            ),
            RUNTIME_MANIFEST_NAME: runtime_binding["manifest_sha256"],
        },
        "code_sha256": {
            "analyze_target_baseline.py": sha256_file(Path(__file__)),
            "analyze_target_baseline.sbatch": sha256_file(LAUNCHER_PATH),
            "frozen_loader_analyze.py": sha256_file(FROZEN_ANALYZER_PATH),
        },
        "model_manifest_sha256": data["model_manifest_hashes"],
        "execution_guard_sha256": data["execution_guard_hashes"],
        "outputs_sha256": {
            OUTPUT_PATH.name: sha256_file(args.output_dir / OUTPUT_PATH.name)
        },
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "numeric_runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "quantile_method": "numpy.quantile default linear",
        },
        "raw_content_emitted": False,
        "new_model_inference_performed": False,
        "rescoring_performed": False,
        "primary_artifacts_modified": False,
        "primary_confirmatory_gate_remains_fail": True,
    }
    atomic_write_json(args.output_dir / MANIFEST_PATH.name, manifest)
    print(
        "Target-vs-baseline post-hoc sensitivity complete: "
        f"Gamma(T-P)={results['point_estimates']['gamma_target_minus_placebo']:.6f}; "
        f"Gamma(T-B)={results['point_estimates']['gamma_target_minus_baseline']:.6f}; "
        "placebo-reference contribution="
        f"{results['point_estimates']['gamma_placebo_reference_contribution_to_target_minus_placebo']:.6f}"
    )


if __name__ == "__main__":
    main()
