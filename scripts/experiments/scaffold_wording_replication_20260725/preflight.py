#!/usr/bin/env python3
"""Fail-closed preflight for the exploratory wording replication."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.experiments.causal_selectivity_20260720.common import (
    RESULTS_ROOT,
    SPEC_PATH,
    condition_map,
    load_json,
    load_spec,
    manifest_path,
    profile_models,
    require,
    sha256_file,
)
from scripts.experiments.causal_selectivity_20260720.preflight import (
    validate_manifest_sources,
)


STUDY_ID = "scaffold_wording_replication_20260725"
EXPECTED_PARENT_BINDINGS = {
    "spec_sha256": "8367e0db2e9bd03a4fa3d3dc283b0be955203f938b889550e3a61a7b45b0ff40",
    "formal_item_manifest_sha256": "0b8a54402104bcb266d06b7ee3c281d225876107e4b41e9c50c51bf8917202a0",
    "pilot_manifest_sha256": "eadfc1c52eab95a40f40d8fcba965f6d6b83d2982e2d27061d0ed60c42cf55c1",
    "released_capacity_manifest_sha256": "7d63e46146e4d86bbb8fd3b6daed69e1530043503c00efab5912cc41a3ffdb25",
}


def validate_static() -> dict:
    spec = load_spec()
    require(spec.get("study_id") == STUDY_ID, "wrong replication study")
    require(
        spec.get("analysis_role")
        == "post_hoc_exploratory_wording_replication_not_part_of_the_parent_confirmation_decision",
        "replication analysis role is missing or changed",
    )
    require(
        spec.get("result_interpretation", {}).get("confirmatory_status") == "none",
        "exploratory replication cannot be labeled confirmatory",
    )
    require(
        spec.get("parent_study", {})
        == {
            **EXPECTED_PARENT_BINDINGS,
            "study_id": "causal_selectivity_20260720",
            "operational_gate_inheritance": spec["parent_study"][
                "operational_gate_inheritance"
            ],
        },
        "parent-study bindings drifted",
    )
    conditions = condition_map(spec)
    require(
        len(conditions) == 7
        and conditions["baseline"]["scaffold"] == ""
        and conditions["neutral_placebo"]["kind"] == "placebo"
        and sum(row["kind"] == "targeted" for row in conditions.values()) == 5,
        "replication condition matrix is malformed",
    )
    manifest = validate_manifest_sources(spec, "formal")
    require(
        manifest["condition_count"] == 7
        and manifest["item_count"] == 234
        and manifest["task_record_count_per_model"] == 1638,
        "replication item manifest has the wrong dimensions",
    )
    require(
        sha256_file(SPEC_PATH) == manifest["spec_sha256"],
        "replication manifest does not bind the active spec",
    )
    root = Path(os.environ["COGARENA_ROOT"])
    parent_results = root / "results" / "causal_selectivity_20260720"
    require(
        sha256_file(
            root
            / "scripts"
            / "experiments"
            / "causal_selectivity_20260720"
            / "PREPILOT_SPEC.json"
        )
        == EXPECTED_PARENT_BINDINGS["spec_sha256"],
        "parent spec changed",
    )
    require(
        sha256_file(parent_results / "item_manifest_formal.json")
        == EXPECTED_PARENT_BINDINGS["formal_item_manifest_sha256"],
        "parent formal item manifest changed",
    )
    require(
        sha256_file(parent_results / "RUN_MANIFEST_pilot.json")
        == EXPECTED_PARENT_BINDINGS["pilot_manifest_sha256"],
        "parent pilot manifest changed",
    )
    require(
        sha256_file(parent_results / "CAPACITY_GATE_MANIFEST.json")
        == EXPECTED_PARENT_BINDINGS["released_capacity_manifest_sha256"],
        "parent capacity manifest changed",
    )
    return spec


def validate(model: str, revision: str) -> None:
    require(os.environ.get("SLURM_JOB_ID"), "replication inference requires Slurm")
    require(os.environ.get("CUDA_VISIBLE_DEVICES"), "replication requires a GPU allocation")
    require(
        len(revision) == 40 and all(char in "0123456789abcdef" for char in revision),
        "COGARENA_GIT_HEAD must be injected as 40 lowercase hex characters",
    )
    spec = validate_static()
    require(
        model in {row["model"] for row in profile_models(spec, "formal")},
        f"model is outside the replication panel: {model}",
    )
    model_root = RESULTS_ROOT / "raw" / "formal" / model.replace(":", "__")
    require(not model_root.exists(), f"fresh-only model root already exists: {model_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        require(args.model is None, "--manifest-only does not accept --model")
        validate_static()
        print("replication static preflight PASS")
        return
    require(bool(args.model), "--model is required for runtime preflight")
    validate(args.model, os.environ.get("COGARENA_GIT_HEAD", "").strip())
    print(f"replication preflight PASS model={args.model}")


if __name__ == "__main__":
    main()
