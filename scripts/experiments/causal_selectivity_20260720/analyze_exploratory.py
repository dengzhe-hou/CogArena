#!/usr/bin/env python3
"""Post-hoc, aggregate-only robustness analyses for causal selectivity.

This module is intentionally separate from both the frozen analyzer and its
reporting-only amendment.  It replays the same closed raw records through the
already frozen loader, verifies the published primary aggregate, and writes a
new exploratory artifact.  Nothing here can alter the primary analysis,
confirmatory thresholds, or the nine-gate FAIL decision.
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

from .analyze_amended import (
    _target_structure,
    compute_estimate,
    crossed_family_item_bootstrap,
    load_formal_arrays,
    validate_aggregate_output,
)
from .common import (
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
EXPLORATORY_SPEC_PATH = HERE / "EXPLORATORY_ANALYSIS_SPEC.json"
EXPLORATORY_LAUNCHER_PATH = HERE / "analyze_exploratory.sbatch"
PRIMARY_ANALYZER_PATH = HERE / "analyze_amended.py"
PRIMARY_OUTPUT_DIR = RESULTS_ROOT / "analysis"
OUTPUT_DIR = PRIMARY_OUTPUT_DIR / "exploratory"
OUTPUT_PATH = OUTPUT_DIR / "exploratory_results.json"
MANIFEST_PATH = OUTPUT_DIR / "EXPLORATORY_MANIFEST.json"
SCHEMA = "cogarena.causal_selectivity.exploratory_analysis.v1"
MANIFEST_SCHEMA = "cogarena.causal_selectivity.exploratory_manifest.v1"
N_BOOTSTRAP = 20_000
SEEDS = {
    "family_paradigm_item_primary": 5201,
    "observable_evaluability": 5202,
    "neutral_placebo_accuracy": 5203,
    "neutral_placebo_evaluability": 5204,
    "neutral_placebo_log_response_length": 5205,
}


def validate_exploratory_spec() -> dict[str, Any]:
    require(EXPLORATORY_SPEC_PATH.is_file(), "exploratory analysis spec missing")
    spec = load_json(EXPLORATORY_SPEC_PATH)
    require(
        spec.get("schema_version")
        == "cogarena.causal_selectivity.exploratory_spec.v1"
        and spec.get("study_id") == "causal_selectivity_20260720"
        and spec.get("status")
        == "post_hoc_exploratory_authored_after_primary_outcome_inspection"
        and spec.get("confirmatory_role") == "none",
        "exploratory spec identity/status mismatch",
    )
    expected_ids = [
        "leave_one_paradigm_out",
        "family_paradigm_item_bootstrap",
        "observable_response_validity_and_accuracy_contributions",
        "neutral_placebo_vs_baseline",
        "small_g_family_inference",
    ]
    analyses = spec.get("analyses")
    require(
        isinstance(analyses, list)
        and [entry.get("id") for entry in analyses if isinstance(entry, dict)]
        == expected_ids,
        "exploratory analysis list/order mismatch",
    )
    bootstrap = spec.get("bootstrap")
    require(
        isinstance(bootstrap, dict)
        and bootstrap.get("replicates") == N_BOOTSTRAP
        and bootstrap.get("seeds") == SEEDS,
        "exploratory bootstrap contract mismatch",
    )
    constraints = spec.get("interpretation_constraints")
    require(
        isinstance(constraints, list)
        and len(constraints) >= 6
        and all(isinstance(value, str) and value for value in constraints),
        "exploratory interpretation constraints missing",
    )
    return spec


def _exploratory_revision() -> str:
    revision = os.environ.get("COGARENA_EXPLORATORY_GIT_HEAD", "").strip()
    require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "COGARENA_EXPLORATORY_GIT_HEAD must be an injected 40-hex revision",
    )
    return revision


def validate_primary_bindings(data: dict[str, Any]) -> dict[str, Any]:
    """Verify the existing primary result without changing or regenerating it."""
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
        manifest.get("outputs_sha256", {}).get("analysis_results.json")
        == sha256_file(result_path),
        "primary analysis result hash mismatch",
    )
    require(
        manifest.get("analyzer_sha256") == sha256_file(PRIMARY_ANALYZER_PATH),
        "current replay loader differs from the analyzer bound to the primary result",
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


def _primary_inputs(
    data: dict[str, Any], accuracy_key: str = "accuracy"
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray], list[int], int]:
    target_indices, placebo_index, target_groups, group_masks = _target_structure(data)
    accuracy = np.asarray(data[accuracy_key], dtype=float)
    gains = (
        accuracy[:, target_indices, :, :]
        - accuracy[:, placebo_index, None, :, :]
    )
    return gains, target_groups, group_masks, target_indices, placebo_index


def leave_one_paradigm_out(
    gains: np.ndarray,
    paradigms: list[str],
    target_groups: list[str],
    group_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    require(gains.shape[2] == len(paradigms), "LOPO paradigm dimension mismatch")
    rows = []
    for omitted_index, omitted in enumerate(paradigms):
        keep = np.ones(len(paradigms), dtype=bool)
        keep[omitted_index] = False
        reduced_masks = {group: mask[keep] for group, mask in group_masks.items()}
        require(
            all(mask.any() and (~mask).any() for mask in reduced_masks.values()),
            f"LOPO omission makes a target contrast undefined: {omitted}",
        )
        estimate = compute_estimate(
            gains[:, :, keep, :], target_groups, reduced_masks
        )
        rows.append(
            {
                "omitted_paradigm": omitted,
                "gamma": estimate["gamma"],
                "s": estimate["s"].tolist(),
            }
        )
    gamma_values = np.asarray([row["gamma"] for row in rows])
    return {
        "status": "post_hoc_exploratory",
        "method": "recompute frozen estimator after each of all 13 single-paradigm omissions; no selected exclusions",
        "rows": rows,
        "gamma_min": float(gamma_values.min()),
        "gamma_max": float(gamma_values.max()),
        "gamma_range": float(gamma_values.max() - gamma_values.min()),
        "all_same_sign": bool(np.all(gamma_values > 0) or np.all(gamma_values < 0)),
    }


def family_paradigm_item_bootstrap(
    gains: np.ndarray,
    families: list[str],
    target_groups: list[str],
    group_masks: dict[str, np.ndarray],
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEEDS["family_paradigm_item_primary"],
) -> dict[str, Any]:
    """Three-level family x paradigm-within-group x item bootstrap."""
    family_order = list(dict.fromkeys(families))
    family_models = [
        np.flatnonzero(np.asarray(families) == family) for family in family_order
    ]
    require(
        len(family_models) == 6 and all(len(indices) == 2 for indices in family_models),
        "paradigm bootstrap requires six two-checkpoint families",
    )
    group_order = list(group_masks)
    paradigm_strata = [np.flatnonzero(group_masks[group]) for group in group_order]
    require(
        sorted(len(indices) for indices in paradigm_strata) == [2, 2, 3, 3, 3],
        "paradigm strata do not match frozen 3/3/3/2/2 design",
    )
    rng = np.random.default_rng(seed)
    gamma_samples = np.empty(n_boot, dtype=float)
    s_samples = np.empty((n_boot, len(target_groups)), dtype=float)
    for draw in range(n_boot):
        sampled_family_ids = rng.integers(0, 6, size=6)
        model_indices = np.concatenate(
            [family_models[index] for index in sampled_family_ids]
        )
        sampled_paradigms: list[int] = []
        sampled_labels: list[str] = []
        means: list[np.ndarray] = []
        for group, eligible in zip(group_order, paradigm_strata):
            paradigm_draw = eligible[
                rng.integers(0, len(eligible), size=len(eligible))
            ]
            for paradigm_index in paradigm_draw:
                item_draw = rng.integers(
                    0, gains.shape[3], size=gains.shape[3]
                )
                sampled = gains[model_indices, :, paradigm_index, :][:, :, item_draw]
                means.append(sampled.mean(axis=(0, 2)))
                sampled_paradigms.append(int(paradigm_index))
                sampled_labels.append(group)
        mean_matrix = np.stack(means, axis=1)
        sampled_masks = {
            group: np.asarray([label == group for label in sampled_labels], dtype=bool)
            for group in group_order
        }
        s = np.empty(len(target_groups), dtype=float)
        for intervention_index, group in enumerate(target_groups):
            matched = sampled_masks[group]
            s[intervention_index] = (
                mean_matrix[intervention_index, matched].mean()
                - mean_matrix[intervention_index, ~matched].mean()
            )
        s_samples[draw] = s
        gamma_samples[draw] = s.mean()
    return {
        "status": "post_hoc_exploratory",
        "method": "resample six families retaining both checkpoints; resample paradigms within each frozen theoretical group retaining 3/3/3/2/2 stratum sizes; resample 18 items within each selected paradigm",
        "inferential_scope_warning": "only two or three observed paradigms occur in each theoretical stratum; this interval is a weak design-sensitivity diagnostic, not strong population-of-paradigms inference",
        "n_bootstrap": n_boot,
        "seed": seed,
        "gamma_ci95": np.quantile(gamma_samples, [0.025, 0.975]).tolist(),
        "s_ci95": np.quantile(s_samples, [0.025, 0.975], axis=0).T.tolist(),
        "descriptive_sign_tail_fraction_two_sided": float(
            min(
                1.0,
                2
                * min(
                    np.mean(gamma_samples <= 0),
                    np.mean(gamma_samples >= 0),
                ),
            )
        ),
    }


def observable_evaluability(data: dict[str, Any]) -> np.ndarray:
    """Observable response availability/evaluability, not semantic correctness."""
    evaluable = ~(np.asarray(data["protocol_invalid"]) | np.asarray(data["empty"]))
    ospan_index = data["paradigms"].index("operation_span")
    evaluable[:, :, ospan_index, :] &= ~np.asarray(data["ospan_parse_none"])[
        :, :, ospan_index, :
    ]
    require(evaluable.dtype == bool, "observable evaluability is not boolean")
    return evaluable


def response_validity_accuracy_decomposition(
    data: dict[str, Any],
    gains: np.ndarray,
    target_groups: list[str],
    group_masks: dict[str, np.ndarray],
    target_indices: list[int],
    placebo_index: int,
    *,
    n_boot: int = N_BOOTSTRAP,
) -> dict[str, Any]:
    evaluable = observable_evaluability(data)
    target_valid = evaluable[:, target_indices, :, :]
    placebo_valid = evaluable[:, placebo_index, :, :]
    validity_gain = target_valid.astype(float) - placebo_valid[:, None].astype(float)
    validity_estimate = compute_estimate(validity_gain, target_groups, group_masks)
    validity_bootstrap = crossed_family_item_bootstrap(
        validity_gain,
        data["families"],
        target_groups,
        group_masks,
        n_boot=n_boot,
        seed=SEEDS["observable_evaluability"],
    )
    strata = {
        "both_evaluable": target_valid & placebo_valid[:, None],
        "target_only_evaluable": target_valid & ~placebo_valid[:, None],
        "placebo_only_evaluable": ~target_valid & placebo_valid[:, None],
        "neither_evaluable": ~target_valid & ~placebo_valid[:, None],
    }
    require(
        np.all(sum(mask.astype(np.int8) for mask in strata.values()) == 1),
        "evaluability strata are not a complete disjoint partition",
    )
    contribution_rows = []
    summed_s = np.zeros(len(target_groups), dtype=float)
    summed_gamma = 0.0
    for name, mask in strata.items():
        contribution = gains * mask
        estimate = compute_estimate(contribution, target_groups, group_masks)
        summed_s += estimate["s"]
        summed_gamma += estimate["gamma"]
        contribution_rows.append(
            {
                "stratum": name,
                "pair_count": int(mask.sum()),
                "pair_fraction": float(mask.mean()),
                "accuracy_contribution_gamma": estimate["gamma"],
                "accuracy_contribution_s": estimate["s"].tolist(),
            }
        )
    primary = compute_estimate(gains, target_groups, group_masks)
    require(
        abs(summed_gamma - primary["gamma"]) < 1e-12
        and np.allclose(summed_s, primary["s"], rtol=0, atol=1e-12),
        "evaluability-stratum contributions do not reconstruct primary accuracy",
    )
    return {
        "status": "post_hoc_exploratory",
        "observable_evaluability_rule": "complete task record is protocol-valid; every response unit consumed by the primary scorer is nonempty; operation-span strict parser status is not none",
        "scope_limitation": "does not identify semantic parse failures outside operation span and is not a mediator or counterfactual decomposition",
        "evaluable_rate": {
            "baseline": float(evaluable[:, data["condition_ids"].index("baseline")].mean()),
            "neutral_placebo": float(evaluable[:, placebo_index].mean()),
            "targeted": {
                data["condition_ids"][condition_index]: float(
                    evaluable[:, condition_index].mean()
                )
                for condition_index in target_indices
            },
        },
        "target_minus_placebo_evaluability_selectivity": {
            "gamma": validity_estimate["gamma"],
            "s": validity_estimate["s"].tolist(),
            "bootstrap": validity_bootstrap,
        },
        "paired_accuracy_contribution_by_observed_evaluability_stratum": {
            "identity": "for each pair, primary gain equals the sum of gain times the four mutually exclusive evaluability indicators; the frozen estimator is linear",
            "rows": contribution_rows,
            "sum_gamma": summed_gamma,
            "primary_gamma": primary["gamma"],
            "absolute_reconstruction_error": abs(summed_gamma - primary["gamma"]),
            "maximum_s_reconstruction_error": float(
                np.max(np.abs(summed_s - primary["s"]))
            ),
        },
    }


def crossed_family_item_mean_bootstrap(
    values: np.ndarray,
    families: list[str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    require(values.ndim == 3, "mean bootstrap expects model x paradigm x item")
    family_order = list(dict.fromkeys(families))
    family_models = [
        np.flatnonzero(np.asarray(families) == family) for family in family_order
    ]
    require(
        len(family_models) == 6 and all(len(indices) == 2 for indices in family_models),
        "mean bootstrap requires six two-checkpoint families",
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        selected_families = rng.integers(0, 6, size=6)
        model_indices = np.concatenate(
            [family_models[index] for index in selected_families]
        )
        paradigm_means = np.empty(values.shape[1], dtype=float)
        for paradigm_index in range(values.shape[1]):
            item_indices = rng.integers(0, values.shape[2], size=values.shape[2])
            sampled = values[model_indices, paradigm_index, :][:, item_indices]
            paradigm_means[paradigm_index] = sampled.mean()
        samples[draw] = paradigm_means.mean()
    return {
        "n_bootstrap": n_boot,
        "seed": seed,
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "descriptive_sign_tail_fraction_two_sided": float(
            min(1.0, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0)))
        ),
    }


def neutral_placebo_audit(
    data: dict[str, Any], *, n_boot: int = N_BOOTSTRAP
) -> dict[str, Any]:
    baseline_index = data["condition_ids"].index("baseline")
    placebo_index = data["condition_ids"].index("neutral_placebo")
    accuracy = np.asarray(data["accuracy"], dtype=float)
    accuracy_gain = accuracy[:, placebo_index] - accuracy[:, baseline_index]
    evaluable = observable_evaluability(data).astype(float)
    evaluability_gain = evaluable[:, placebo_index] - evaluable[:, baseline_index]
    chars = np.asarray(data["response_chars"], dtype=float)
    length_gain = (
        np.log1p(chars[:, placebo_index]) - np.log1p(chars[:, baseline_index])
    )
    grouping = data["spec"]["grouping"]
    paradigm_index = {name: index for index, name in enumerate(data["paradigms"])}
    model_means = accuracy_gain.mean(axis=(1, 2))
    family_order = list(dict.fromkeys(data["families"]))
    family_means = {
        family: float(
            model_means[np.asarray(data["families"]) == family].mean()
        )
        for family in family_order
    }
    return {
        "status": "post_hoc_exploratory_control_condition_audit",
        "contrast": "neutral_placebo_minus_baseline_on_identical_items",
        "weighting": "equal item within paradigm, equal paradigm within model, equal model",
        "primary_accuracy": {
            "baseline_mean": float(accuracy[:, baseline_index].mean()),
            "neutral_placebo_mean": float(accuracy[:, placebo_index].mean()),
            "paired_difference": float(accuracy_gain.mean()),
            "crossed_family_item_bootstrap": crossed_family_item_mean_bootstrap(
                accuracy_gain,
                data["families"],
                n_boot=n_boot,
                seed=SEEDS["neutral_placebo_accuracy"],
            ),
            "by_paradigm": {
                paradigm: float(accuracy_gain[:, index].mean())
                for index, paradigm in enumerate(data["paradigms"])
            },
            "by_grouping": {
                group: float(
                    np.mean(
                        [
                            accuracy_gain[:, paradigm_index[paradigm]].mean()
                            for paradigm in paradigms
                        ]
                    )
                )
                for group, paradigms in grouping.items()
            },
            "by_family": family_means,
        },
        "observable_evaluability": {
            "baseline_mean": float(evaluable[:, baseline_index].mean()),
            "neutral_placebo_mean": float(evaluable[:, placebo_index].mean()),
            "paired_difference": float(evaluability_gain.mean()),
            "crossed_family_item_bootstrap": crossed_family_item_mean_bootstrap(
                evaluability_gain,
                data["families"],
                n_boot=n_boot,
                seed=SEEDS["neutral_placebo_evaluability"],
            ),
        },
        "log1p_primary_response_characters": {
            "baseline_mean": float(np.log1p(chars[:, baseline_index]).mean()),
            "neutral_placebo_mean": float(np.log1p(chars[:, placebo_index]).mean()),
            "paired_difference": float(length_gain.mean()),
            "crossed_family_item_bootstrap": crossed_family_item_mean_bootstrap(
                length_gain,
                data["families"],
                n_boot=n_boot,
                seed=SEEDS["neutral_placebo_log_response_length"],
            ),
        },
    }


def _binomial_upper_tail(n: int, observed: int) -> float:
    return float(sum(math.comb(n, value) for value in range(observed, n + 1)) / 2**n)


def small_g_family_inference(
    model_s: np.ndarray, families: list[str]
) -> dict[str, Any]:
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
    p_lower = float(sum(math.comb(n, value) for value in range(0, positive + 1)) / 2**n)
    sign_vectors = np.asarray(list(itertools.product((-1.0, 1.0), repeat=6)))
    sign_flip_null = (sign_vectors * family_gamma).mean(axis=1)
    observed = float(family_gamma.mean())
    sign_flip_one = float(np.mean(sign_flip_null >= observed - 1e-15))
    sign_flip_two = float(np.mean(np.abs(sign_flip_null) >= abs(observed) - 1e-15))
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
        "status": "post_hoc_exploratory_small_G",
        "family_unit": "mean of the two frozen checkpoints within each model family",
        "family_gamma": {
            family: float(value) for family, value in zip(family_order, family_gamma)
        },
        "exact_binomial_sign_test": {
            "nonzero_families": n,
            "positive": positive,
            "negative": negative,
            "p_one_sided_positive": p_upper,
            "p_two_sided_doubled_minimum_tail": min(1.0, 2 * min(p_upper, p_lower)),
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


def analyze_exploratory(
    data: dict[str, Any], primary_binding: dict[str, Any], *, n_boot: int = N_BOOTSTRAP
) -> dict[str, Any]:
    gains, target_groups, group_masks, target_indices, placebo_index = _primary_inputs(data)
    primary = compute_estimate(gains, target_groups, group_masks)
    upstream_primary = primary_binding["result"]["primary"]
    require(
        abs(primary["gamma"] - float(upstream_primary["gamma"])) < 1e-12,
        "replayed primary Gamma differs from authoritative analysis",
    )
    upstream_s = np.asarray(
        [entry["s"] for entry in upstream_primary["interventions"]], dtype=float
    )
    require(
        np.allclose(primary["s"], upstream_s, rtol=0, atol=1e-12),
        "replayed primary intervention effects differ from authoritative analysis",
    )
    output = {
        "schema_version": SCHEMA,
        "status": "post_hoc_exploratory_only",
        "confirmatory_role": "none",
        "authoritative_primary_gamma_reproduced": primary["gamma"],
        "authoritative_confirmatory_gate_remains": False,
        "leave_one_paradigm_out": leave_one_paradigm_out(
            gains, data["paradigms"], target_groups, group_masks
        ),
        "family_paradigm_item_bootstrap": family_paradigm_item_bootstrap(
            gains,
            data["families"],
            target_groups,
            group_masks,
            n_boot=n_boot,
            seed=SEEDS["family_paradigm_item_primary"],
        ),
        "observable_response_validity_and_accuracy_contributions": (
            response_validity_accuracy_decomposition(
                data,
                gains,
                target_groups,
                group_masks,
                target_indices,
                placebo_index,
                n_boot=n_boot,
            )
        ),
        "neutral_placebo_vs_baseline": neutral_placebo_audit(data, n_boot=n_boot),
        "small_g_family_inference": small_g_family_inference(
            primary["model_s"], data["families"]
        ),
        "interpretation_constraints": validate_exploratory_spec()[
            "interpretation_constraints"
        ],
    }
    validate_aggregate_output(output)
    return output


def validate_paths(output_dir: Path) -> None:
    require(
        output_dir.resolve() == OUTPUT_DIR.resolve(),
        "noncanonical exploratory output directory refused",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    validate_paths(args.output_dir)
    validate_exploratory_spec()
    exploratory_revision = _exploratory_revision()
    data = load_formal_arrays()
    primary_binding = validate_primary_bindings(data)
    results = analyze_exploratory(data, primary_binding, n_boot=N_BOOTSTRAP)
    validate_aggregate_output(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / OUTPUT_PATH.name, jsonable(results))
    require(EXPLORATORY_LAUNCHER_PATH.is_file(), "exploratory Slurm launcher missing")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "study_id": "causal_selectivity_20260720",
        "status": "complete_post_hoc_exploratory",
        "confirmatory_role": "none",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "formal_raw_source_revision": data["source_revision"],
        "exploratory_analysis_revision": exploratory_revision,
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
            "EXPLORATORY_ANALYSIS_SPEC.json": sha256_file(EXPLORATORY_SPEC_PATH),
        },
        "code_sha256": {
            "analyze_exploratory.py": sha256_file(Path(__file__)),
            "analyze_exploratory.sbatch": sha256_file(EXPLORATORY_LAUNCHER_PATH),
            "primary_loader_analyze_amended.py": sha256_file(PRIMARY_ANALYZER_PATH),
        },
        "model_manifest_sha256": data["model_manifest_hashes"],
        "execution_guard_sha256": data["execution_guard_hashes"],
        "outputs_sha256": {OUTPUT_PATH.name: sha256_file(args.output_dir / OUTPUT_PATH.name)},
        "n_bootstrap": N_BOOTSTRAP,
        "seeds": SEEDS,
        "numeric_runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "quantile_method": "numpy.quantile default linear",
        },
        "raw_content_emitted": False,
        "primary_artifacts_modified": False,
        "primary_confirmatory_gate_remains_fail": True,
    }
    atomic_write_json(args.output_dir / MANIFEST_PATH.name, manifest)
    print(
        "Exploratory robustness complete: "
        f"Gamma={results['authoritative_primary_gamma_reproduced']:.6f}; "
        f"LOPO=[{results['leave_one_paradigm_out']['gamma_min']:.6f}, "
        f"{results['leave_one_paradigm_out']['gamma_max']:.6f}]; "
        f"paradigm-bootstrap CI="
        f"{results['family_paradigm_item_bootstrap']['gamma_ci95']}"
    )


if __name__ == "__main__":
    main()
