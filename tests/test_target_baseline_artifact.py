"""Release-level checks for the target-versus-baseline sensitivity artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    ROOT
    / "results"
    / "causal_selectivity_20260720"
    / "analysis"
    / "target_baseline_sensitivity"
)
RESULT_PATH = RESULT_DIR / "target_baseline_results.json"
MANIFEST_PATH = RESULT_DIR / "TARGET_BASELINE_MANIFEST.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_target_baseline_release_is_hash_bound_and_nonconfirmatory():
    result = json.loads(RESULT_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert result["status"] == "complete_post_hoc_sensitivity"
    assert result["confirmatory_role"] == "none"
    assert result["primary_confirmatory_gate_remains_fail"] is True
    assert manifest["status"] == "complete_post_hoc_sensitivity"
    assert manifest["confirmatory_role"] == "none"
    assert manifest["outputs_sha256"] == {
        RESULT_PATH.name: _sha256(RESULT_PATH)
    }
    assert manifest["raw_content_emitted"] is False
    assert manifest["new_model_inference_performed"] is False
    assert manifest["rescoring_performed"] is False
    assert manifest["primary_artifacts_modified"] is False


def test_target_placebo_baseline_identity_and_headline_values():
    result = json.loads(RESULT_PATH.read_text())
    point = result["point_estimates"]
    assert point["gamma_target_minus_placebo"] == pytest.approx(
        point["gamma_target_minus_baseline"]
        - point["gamma_placebo_minus_baseline_differential"],
        abs=1e-15,
    )
    assert point["gamma_target_minus_baseline"] == pytest.approx(
        0.02066685697584309
    )
    assert point[
        "gamma_placebo_reference_contribution_to_target_minus_placebo"
    ] == pytest.approx(-0.0007466965955854887)
    bootstrap = result["joint_crossed_family_item_bootstrap"]
    assert bootstrap["contrasts"]["target_minus_baseline"]["gamma_ci95"] == (
        pytest.approx([0.0034244460347522838, 0.03814595780256263])
    )
    assert bootstrap[
        "placebo_reference_contribution_to_target_minus_placebo_selectivity"
    ]["gamma_ci95"] == pytest.approx(
        [-0.0037495329512051818, 0.0024061978622200698]
    )
    exact = result["target_minus_baseline_exact_mapping_permutation"]
    assert exact["n_exact_mappings"] == 120
    assert exact["gamma_p_two"] == pytest.approx(1 / 60)


def test_global_gain_and_selectivity_are_not_conflated():
    result = json.loads(RESULT_PATH.read_text())
    levels = result["global_accuracy_levels_and_differences"]
    assert levels["target_minus_placebo_mean"] > 0
    assert levels["target_minus_baseline_mean"] < 0
    assert levels[
        "baseline_minus_placebo_contribution_to_target_minus_placebo_mean"
    ] > 0
    assert levels["identity_error"] == pytest.approx(0, abs=1e-15)
