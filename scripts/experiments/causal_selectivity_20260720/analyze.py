#!/usr/bin/env python3
"""Frozen formal analysis for the causal-selectivity matrix.

The analyzer is deliberately downstream of raw closure. It accepts only a
``formal_raw_complete`` RUN_MANIFEST, independently enumerates and replays all
12 x 234 x 7 records, and emits aggregate statistics without copying any raw
response, stimulus, or gold value into an output artifact.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    EXECUTION_GUARD_FILENAME,
    RESULTS_ROOT,
    SPEC_PATH,
    atomic_write_json,
    condition_map,
    ensure_finite_accuracy,
    execution_guard_identity_sha256,
    execution_guard_records_complete_sha256,
    jsonable,
    load_json,
    load_spec,
    manifest_path,
    model_safe,
    profile_models,
    recovered_terminal_metadata_fault_exposure,
    request_reasoning_effort,
    request_stop_policy,
    require,
    is_protocol_invalid_finish_reason,
    TRANSPORT_INCOMPLETE_FINISH_REASON,
    validate_reported_usage_summary,
    sha256_file,
    tree_hash,
    validate_execution_guard,
)
from .run_model import reconstruct_items, result_path, validate_record
from .preflight import validate_manifest_sources, validate_source_revision


ANALYSIS_SCHEMA = "cogarena.causal_selectivity.analysis.v3"
ANALYSIS_SEED = 42
N_BOOTSTRAP = 20_000
RIDGE_LAMBDA = 1.0
DIFFICULTY_SEEDS = {"easy": 4201, "medium": 4202, "hard": 4203}
DIFFICULTY_INTERACTION_SEED = 4204

CONFIRMATORY_GATE_IDS = (
    "gamma_family_item_ci_lower_gt_zero",
    "at_least_four_of_six_family_gamma_positive",
    "family_lofo_selective_delta_log_likelihood_gt_zero",
    "exact_mapping_one_sided_p_le_0_05",
    "each_condition_task_record_invalid_rate_le_0_01",
    "protocol_invalid_exclusion_preserves_at_least_half_gamma_and_three_items_per_cell",
    "empty_exclusion_preserves_at_least_half_gamma_and_three_items_per_cell",
    "ospan_parse_exclusion_preserves_at_least_half_gamma_and_three_items_per_cell",
    "response_length_adjustment_preserves_at_least_half_gamma",
)
CONFIRMATORY_THRESHOLD_KEYS = {
    "gamma_ci_lower_strictly_above",
    "minimum_positive_families",
    "predictive_delta_log_likelihood_strictly_above",
    "maximum_exact_mapping_one_sided_p",
    "maximum_formal_condition_task_record_invalid_rate",
    "minimum_sensitivity_gamma_preservation_ratio",
    "minimum_sensitivity_items_per_cell",
}


def confirmatory_gate_thresholds(spec: dict[str, Any]) -> dict[str, float | int]:
    """Validate and return the single frozen source for all nine gate cutoffs."""
    gate = spec.get("estimands", {}).get("confirmatory_success_gate")
    require(isinstance(gate, dict), "confirmatory success gate is missing")
    rules = gate.get("rules")
    require(isinstance(rules, list), "confirmatory gate rules are missing")
    rule_ids = [rule.get("id") for rule in rules if isinstance(rule, dict)]
    require(
        rule_ids == list(CONFIRMATORY_GATE_IDS),
        "confirmatory gate IDs/order differ from the frozen analyzer contract",
    )
    thresholds = gate.get("numeric_thresholds")
    require(
        isinstance(thresholds, dict) and set(thresholds) == CONFIRMATORY_THRESHOLD_KEYS,
        "confirmatory numeric-threshold schema mismatch",
    )
    numeric = {
        key: value
        for key, value in thresholds.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    require(len(numeric) == len(thresholds), "confirmatory threshold is not numeric")
    require(all(math.isfinite(float(value)) for value in numeric.values()),
            "confirmatory threshold is non-finite")
    require(
        isinstance(numeric["minimum_positive_families"], int)
        and 1 <= numeric["minimum_positive_families"] <= 6,
        "minimum-positive-family threshold is invalid",
    )
    require(
        isinstance(numeric["minimum_sensitivity_items_per_cell"], int)
        and 1 <= numeric["minimum_sensitivity_items_per_cell"] <= 18,
        "minimum-item threshold is invalid",
    )
    require(0 <= numeric["maximum_exact_mapping_one_sided_p"] <= 1,
            "mapping-p threshold is invalid")
    require(0 <= numeric["maximum_formal_condition_task_record_invalid_rate"] <= 1,
            "formal invalid-rate threshold is invalid")
    require(numeric["minimum_sensitivity_gamma_preservation_ratio"] >= 0,
            "preservation-ratio threshold is invalid")
    return numeric


def _source_revision() -> str:
    revision = os.environ.get("COGARENA_GIT_HEAD", "").strip()
    require(
        len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
        "COGARENA_GIT_HEAD must be injected at sbatch submission",
    )
    return revision


def _validate_formal_run_manifest(spec: dict[str, Any], run: dict[str, Any], revision: str) -> None:
    reasoning_effort = request_reasoning_effort(spec)
    stop_policy = request_stop_policy(spec)
    require(spec.get("status") == "formal_frozen_after_pilot", "formal spec is not frozen")
    require(run.get("schema_version") == "cogarena.causal_selectivity.run_manifest.v3",
            "bad formal RUN_MANIFEST schema")
    require(run.get("profile") == "formal", "analyzer accepts formal profile only")
    require(run.get("status") == "formal_raw_complete", "formal raw run is not closed")
    require(run.get("all_model_replays_passed") is True, "formal model replay gate did not pass")
    require(
        "condition_paradigm_mean_accuracy" not in run,
        "formal RUN_MANIFEST leaks pre-analysis arm outcomes",
    )
    validate_reported_usage_summary(spec, run)
    require(run.get("source_revision") == revision, "RUN_MANIFEST source revision mismatch")
    require(run.get("reasoning_effort") == reasoning_effort,
            "RUN_MANIFEST reasoning-effort contract mismatch")
    require(run.get("reasoning_request_verified") is True,
            "RUN_MANIFEST lacks replay verification of the reasoning request")
    require(run.get("stop_policy") == stop_policy
            and run.get("stop_sequence_request_verified") is True,
            "RUN_MANIFEST lacks replay verification of the response-format policy")
    require(run.get("model_count") == 12, "formal RUN_MANIFEST must contain 12 models")
    require(run.get("record_count") == 12 * 234 * 7, "formal RUN_MANIFEST record count mismatch")
    guard_tree = run.get("execution_guard_tree_sha256")
    profile_array_job_id = run.get("profile_array_job_id")
    require(
        run.get("execution_guard_count") == 12
        and run.get("all_execution_guards_verified_complete") is True
        and run.get("record_reuse_allowed") is False
        and isinstance(guard_tree, str)
        and len(guard_tree) == 64
        and all(character in "0123456789abcdef" for character in guard_tree),
        "formal RUN_MANIFEST execution-guard closure is invalid",
    )
    require(
        isinstance(profile_array_job_id, str) and bool(profile_array_job_id),
        "formal RUN_MANIFEST lacks a single profile Slurm-array identity",
    )
    condition_ids = [condition["id"] for condition in spec["conditions"]]
    condition_counts = run.get("condition_record_counts")
    truncated_counts = run.get("condition_truncated_record_counts")
    truncation_rates = run.get("condition_truncation_rates")
    invalid_counts = run.get("condition_invalid_record_counts")
    invalid_rates = run.get("condition_invalid_record_rates")
    transport_invalid_counts = run.get(
        "condition_transport_protocol_invalid_record_counts"
    )
    transport_invalid_rates = run.get(
        "condition_transport_protocol_invalid_rates"
    )
    recovered_fault_counts = run.get(
        "condition_recovered_terminal_metadata_fault_record_counts"
    )
    recovered_fault_rates = run.get(
        "condition_recovered_terminal_metadata_fault_record_rates"
    )
    require(
        isinstance(condition_counts, dict)
        and isinstance(truncated_counts, dict)
        and isinstance(truncation_rates, dict)
        and isinstance(invalid_counts, dict)
        and isinstance(invalid_rates, dict)
        and isinstance(transport_invalid_counts, dict)
        and isinstance(transport_invalid_rates, dict)
        and isinstance(recovered_fault_counts, dict)
        and isinstance(recovered_fault_rates, dict)
        and set(condition_counts) == set(condition_ids)
        and set(truncated_counts) == set(condition_ids)
        and set(truncation_rates) == set(condition_ids)
        and set(invalid_counts) == set(condition_ids)
        and set(invalid_rates) == set(condition_ids)
        and set(transport_invalid_counts) == set(condition_ids)
        and set(transport_invalid_rates) == set(condition_ids)
        and set(recovered_fault_counts) == set(condition_ids)
        and set(recovered_fault_rates) == set(condition_ids)
        and all(condition_counts[cid] == 12 * 234 for cid in condition_ids),
        "formal RUN_MANIFEST condition invalid coverage mismatch",
    )
    require(
        all(
            isinstance(truncated_counts[cid], int)
            and not isinstance(truncated_counts[cid], bool)
            and 0 <= truncated_counts[cid] <= condition_counts[cid]
            and abs(
                float(truncation_rates[cid])
                - truncated_counts[cid] / condition_counts[cid]
            ) < 1e-15
            for cid in condition_ids
        )
        and run.get("truncated_record_count") == sum(truncated_counts.values())
        and run.get("truncated_api_call_count") == run.get("truncated_completion_count"),
        "formal RUN_MANIFEST truncation totals mismatch",
    )
    require(
        all(
            isinstance(invalid_counts[cid], int)
            and not isinstance(invalid_counts[cid], bool)
            and isinstance(transport_invalid_counts[cid], int)
            and not isinstance(transport_invalid_counts[cid], bool)
            and 0 <= truncated_counts[cid] <= invalid_counts[cid]
            and 0 <= transport_invalid_counts[cid] <= invalid_counts[cid]
            and invalid_counts[cid]
            <= truncated_counts[cid] + transport_invalid_counts[cid]
            and abs(float(invalid_rates[cid])
                    - invalid_counts[cid] / condition_counts[cid]) < 1e-15
            and abs(float(transport_invalid_rates[cid])
                    - transport_invalid_counts[cid] / condition_counts[cid]) < 1e-15
            and isinstance(recovered_fault_counts[cid], int)
            and not isinstance(recovered_fault_counts[cid], bool)
            and 0 <= recovered_fault_counts[cid] <= condition_counts[cid]
            and abs(float(recovered_fault_rates[cid])
                    - recovered_fault_counts[cid] / condition_counts[cid]) < 1e-15
            for cid in condition_ids
        )
        and run.get("invalid_record_count") == sum(invalid_counts.values())
        and run.get("transport_protocol_invalid_record_count")
        == sum(transport_invalid_counts.values())
        and run.get("recovered_terminal_metadata_fault_record_count")
        == sum(recovered_fault_counts.values())
        and isinstance(
            run.get("recovered_terminal_metadata_fault_logical_call_count"), int
        )
        and not isinstance(
            run.get("recovered_terminal_metadata_fault_logical_call_count"), bool
        )
        and 0 <= run["recovered_terminal_metadata_fault_logical_call_count"]
        <= run["usage_metadata_valid_logical_call_count"],
        "formal RUN_MANIFEST protocol-invalid totals mismatch",
    )
    require(run.get("all_models_fully_gpu_served") is True
            and run.get("processor_requirement") == "100% GPU"
            and run.get("fully_gpu_served_model_count") == 12,
            "formal RUN_MANIFEST lacks full-GPU closure")


def _response_texts(record: dict[str, Any], multiturn: bool) -> list[str]:
    if multiturn:
        payload = record["responses"]
        require(isinstance(payload, list), "bad multi-turn response payload")
        texts = [entry.get("response") for entry in payload]
    else:
        texts = [record.get("response")]
    require(all(isinstance(text, str) for text in texts), "non-string raw response")
    return texts


def _primary_response_texts(
    record: dict[str, Any], entry: dict[str, Any], item: Any
) -> list[str]:
    """Response units actually consumed by the frozen primary score."""
    texts = _response_texts(record, entry["is_multiturn"])
    if not entry["is_multiturn"]:
        return texts
    paradigm = entry["paradigm"]
    if paradigm == "operation_span":
        return [texts[-1]]
    if paradigm == "n_back":
        return texts
    if paradigm == "cvlt_word_list":
        turns = item.metadata.parameters.get("turns", [])
        require(len(turns) == len(texts), "CVLT turn/response mismatch")
        selected = [text for text, turn in zip(texts, turns) if turn.get("expected_words")]
        require(selected, "CVLT has no designated primary-scoring recall turn")
        return selected
    raise RuntimeError(f"unfrozen multi-turn primary-unit rule: {paradigm}")


def load_formal_arrays(raw_root: Path = RESULTS_ROOT / "raw") -> dict[str, Any]:
    """Read, enumerate, identity-check, and replay every formal record."""
    revision = _source_revision()
    spec = load_spec()
    reasoning_effort = request_reasoning_effort(spec)
    stop_policy = request_stop_policy(spec)
    validate_source_revision(spec, "formal", revision)
    validate_manifest_sources(spec, "formal")
    run_path = RESULTS_ROOT / "RUN_MANIFEST_formal.json"
    require(run_path.is_file(), "formal RUN_MANIFEST missing")
    run = load_json(run_path)
    _validate_formal_run_manifest(spec, run, revision)

    item_path = manifest_path("formal")
    item_manifest = load_json(item_path)
    require(item_manifest.get("profile") == "formal", "bad formal item manifest")
    spec_sha = sha256_file(SPEC_PATH)
    item_sha = sha256_file(item_path)
    require(run.get("study_id") == spec["study_id"], "RUN_MANIFEST study mismatch")
    require(run.get("spec_sha256") == spec_sha, "RUN_MANIFEST spec hash mismatch")
    require(run.get("item_manifest_sha256") == item_sha,
            "RUN_MANIFEST item-manifest hash mismatch")
    require(item_manifest.get("schema_version") == "cogarena.causal_selectivity.item_manifest.v1",
            "bad formal item-manifest schema")
    require(item_manifest.get("study_id") == spec["study_id"], "item-manifest study mismatch")
    require(item_manifest.get("spec_sha256") == spec_sha,
            "item manifest binds stale spec")
    require(item_manifest.get("item_count") == 234, "formal item manifest must contain 234 items")
    items = reconstruct_items(spec, item_manifest)

    models_meta = profile_models(spec, "formal")
    models = [entry["model"] for entry in models_meta]
    families = [entry["family"] for entry in models_meta]
    family_counts = Counter(families)
    require(len(family_counts) == 6 and set(family_counts.values()) == {2},
            f"formal family panel is not 6 x 2: {family_counts}")

    conditions = list(condition_map(spec).values())
    condition_ids = [entry["id"] for entry in conditions]
    require(len(condition_ids) == 7 and len(set(condition_ids)) == 7, "condition set mismatch")
    paradigms = [p for group in spec["grouping"].values() for p in group]
    require(len(paradigms) == 13 and len(set(paradigms)) == 13, "paradigm set mismatch")
    entries_by_paradigm = {
        paradigm: [x for x in item_manifest["items"] if x["paradigm"] == paradigm]
        for paradigm in paradigms
    }
    require(all(len(entries) == 18 for entries in entries_by_paradigm.values()),
            "each formal paradigm must have 18 items")
    item_index = {
        entry["task_id"]: (pi, ii)
        for pi, paradigm in enumerate(paradigms)
        for ii, entry in enumerate(entries_by_paradigm[paradigm])
    }
    entry_by_task = {entry["task_id"]: entry for entry in item_manifest["items"]}
    difficulties = np.empty((len(paradigms), 18), dtype=object)
    for task_id, (pi, ii) in item_index.items():
        difficulties[pi, ii] = entry_by_task[task_id]["difficulty"]

    shape = (len(models), len(condition_ids), len(paradigms), 18)
    accuracy = np.full(shape, np.nan, dtype=float)
    canonical_accuracy = np.full(shape, np.nan, dtype=float)
    ospan_math_accuracy = np.full(shape, np.nan, dtype=float)
    empty = np.zeros(shape, dtype=bool)
    response_chars = np.zeros(shape, dtype=float)
    ospan_parse_none = np.zeros(shape, dtype=bool)
    protocol_invalid = np.zeros(shape, dtype=bool)
    truncated_record = np.zeros(shape, dtype=bool)
    transport_incomplete_record = np.zeros(shape, dtype=bool)
    recovered_terminal_metadata_fault_exposed = np.zeros(shape, dtype=bool)
    profile_root = raw_root.resolve() / "formal"
    require(profile_root.is_dir(), "formal raw profile directory missing")
    require(
        {x.name for x in profile_root.iterdir() if x.is_dir()} == {model_safe(x) for x in models},
        "formal raw model directory set mismatch",
    )

    model_manifest_paths: list[Path] = []
    model_manifest_hashes: dict[str, str] = {}
    execution_guard_paths: list[Path] = []
    execution_guard_hashes: dict[str, str] = {}
    seen_records = 0
    replayed_run_transport_counts: Counter[str] = Counter()
    replayed_run_max_prompt_tokens = 0
    replayed_run_max_completion_tokens = 0
    replayed_run_max_total_tokens = 0
    replayed_run_response_unit_count = 0
    replayed_run_empty_response_count = 0
    for mi, model in enumerate(models):
        model_root = profile_root / model_safe(model)
        mm_path = model_root / "MODEL_MANIFEST.json"
        guard_path = model_root / EXECUTION_GUARD_FILENAME
        require(mm_path.is_file(), f"MODEL_MANIFEST missing for {model}")
        require(guard_path.is_file(), f"execution guard missing for {model}")
        mm = load_json(mm_path)
        guard = load_json(guard_path)
        validate_execution_guard(guard, expected_state="verified_complete")
        require(mm.get("schema_version") == "cogarena.causal_selectivity.model_manifest.v3",
                f"bad MODEL_MANIFEST schema: {model}")
        require(mm.get("study_id") == spec["study_id"] and mm.get("profile") == "formal",
                f"MODEL_MANIFEST study/profile mismatch: {model}")
        require(mm.get("status") == "complete" and mm.get("all_records_replayed") is True,
                f"model closure incomplete: {model}")
        validate_reported_usage_summary(spec, mm)
        require(
            "condition_paradigm_mean_accuracy" not in mm,
            f"MODEL_MANIFEST leaks pre-analysis arm outcomes: {model}",
        )
        require(mm.get("model") == model and mm.get("family") == families[mi],
                f"model identity mismatch: {model}")
        require(mm.get("source_revision") == revision, f"model source revision mismatch: {model}")
        require(mm.get("reasoning_effort") == reasoning_effort,
                f"model reasoning-effort contract mismatch: {model}")
        require(mm.get("reasoning_request_verified") is True,
                f"model reasoning-effort request was not replay-verified: {model}")
        require(mm.get("stop_policy") == stop_policy
                and mm.get("stop_sequence_request_verified") is True,
                f"model response-format policy was not replay-verified: {model}")
        require(mm.get("spec_sha256") == spec_sha, f"model spec drift: {model}")
        require(mm.get("item_manifest_sha256") == item_sha,
                f"model item-manifest drift: {model}")
        require(mm.get("record_count") == 234 * 7, f"model record count mismatch: {model}")
        require(
            guard.get("study_id") == spec["study_id"]
            and guard.get("profile") == "formal"
            and guard.get("slurm_array_job_id") == run.get("profile_array_job_id")
            and guard.get("model") == model
            and guard.get("source_revision") == revision
            and guard.get("spec_sha256") == spec_sha
            and guard.get("item_manifest_sha256") == item_sha
            and guard.get("expected_record_count") == mm.get("record_count")
            and guard.get("record_tree_sha256") == mm.get("record_tree_sha256")
            and guard.get("serving_provenance_sha256")
            == mm.get("serving_provenance_sha256")
            and guard.get("run_summary_sha256") == mm.get("run_summary_sha256")
            and guard.get("model_manifest_sha256") == sha256_file(mm_path)
            and mm.get("execution_guard_identity_sha256")
            == execution_guard_identity_sha256(guard)
            and mm.get("execution_guard_records_complete_sha256")
            == execution_guard_records_complete_sha256(guard)
            and mm.get("execution_guard_same_job_verified") is True
            and mm.get("record_reuse_allowed") is False,
            f"model execution-guard closure mismatch: {model}",
        )
        mm_condition_counts = mm.get("condition_record_counts")
        mm_truncated_counts = mm.get("condition_truncated_record_counts")
        mm_truncation_rates = mm.get("condition_truncation_rates")
        mm_invalid_counts = mm.get("condition_invalid_record_counts")
        mm_invalid_rates = mm.get("condition_invalid_record_rates")
        mm_transport_invalid_counts = mm.get(
            "condition_transport_protocol_invalid_record_counts"
        )
        mm_transport_invalid_rates = mm.get(
            "condition_transport_protocol_invalid_rates"
        )
        mm_recovered_fault_counts = mm.get(
            "condition_recovered_terminal_metadata_fault_record_counts"
        )
        mm_recovered_fault_rates = mm.get(
            "condition_recovered_terminal_metadata_fault_record_rates"
        )
        require(
            isinstance(mm_condition_counts, dict)
            and isinstance(mm_truncated_counts, dict)
            and isinstance(mm_truncation_rates, dict)
            and isinstance(mm_invalid_counts, dict)
            and isinstance(mm_invalid_rates, dict)
            and isinstance(mm_transport_invalid_counts, dict)
            and isinstance(mm_transport_invalid_rates, dict)
            and isinstance(mm_recovered_fault_counts, dict)
            and isinstance(mm_recovered_fault_rates, dict)
            and set(mm_condition_counts) == set(condition_ids)
            and set(mm_truncated_counts) == set(condition_ids)
            and set(mm_truncation_rates) == set(condition_ids)
            and set(mm_invalid_counts) == set(condition_ids)
            and set(mm_invalid_rates) == set(condition_ids)
            and set(mm_transport_invalid_counts) == set(condition_ids)
            and set(mm_transport_invalid_rates) == set(condition_ids)
            and set(mm_recovered_fault_counts) == set(condition_ids)
            and set(mm_recovered_fault_rates) == set(condition_ids)
            and all(mm_condition_counts[cid] == 234 for cid in condition_ids),
            f"model condition invalid coverage mismatch: {model}",
        )
        require(
            all(
                isinstance(mm_truncated_counts[cid], int)
                and not isinstance(mm_truncated_counts[cid], bool)
                and 0 <= mm_truncated_counts[cid] <= 234
                and abs(mm_truncation_rates[cid] - mm_truncated_counts[cid] / 234) < 1e-15
                for cid in condition_ids
            )
            and mm.get("truncated_record_count") == sum(mm_truncated_counts.values())
            and mm.get("truncated_api_call_count") == mm.get("truncated_completion_count"),
            f"model truncation totals mismatch: {model}",
        )
        require(
            all(
                isinstance(mm_invalid_counts[cid], int)
                and not isinstance(mm_invalid_counts[cid], bool)
                and isinstance(mm_transport_invalid_counts[cid], int)
                and not isinstance(mm_transport_invalid_counts[cid], bool)
                and 0 <= mm_truncated_counts[cid] <= mm_invalid_counts[cid]
                and 0 <= mm_transport_invalid_counts[cid] <= mm_invalid_counts[cid]
                and mm_invalid_counts[cid]
                <= mm_truncated_counts[cid] + mm_transport_invalid_counts[cid]
                and abs(mm_invalid_rates[cid] - mm_invalid_counts[cid] / 234) < 1e-15
                and abs(
                    mm_transport_invalid_rates[cid]
                    - mm_transport_invalid_counts[cid] / 234
                ) < 1e-15
                and isinstance(mm_recovered_fault_counts[cid], int)
                and not isinstance(mm_recovered_fault_counts[cid], bool)
                and 0 <= mm_recovered_fault_counts[cid] <= 234
                and abs(
                    mm_recovered_fault_rates[cid]
                    - mm_recovered_fault_counts[cid] / 234
                ) < 1e-15
                for cid in condition_ids
            )
            and mm.get("invalid_record_count") == sum(mm_invalid_counts.values())
            and mm.get("transport_protocol_invalid_record_count")
            == sum(mm_transport_invalid_counts.values())
            and mm.get("recovered_terminal_metadata_fault_record_count")
            == sum(mm_recovered_fault_counts.values())
            and isinstance(
                mm.get("recovered_terminal_metadata_fault_logical_call_count"), int
            )
            and not isinstance(
                mm.get("recovered_terminal_metadata_fault_logical_call_count"), bool
            )
            and 0 <= mm["recovered_terminal_metadata_fault_logical_call_count"]
            <= mm["usage_metadata_valid_logical_call_count"],
            f"model protocol-invalid totals mismatch: {model}",
        )
        require(mm.get("fully_gpu_served") is True and mm.get("processor") == "100% GPU",
                f"model was not fully GPU-served: {model}")
        digest = mm.get("served_model_digest")
        require(isinstance(digest, str) and len(digest) >= 32, f"bad digest: {model}")
        model_manifest_paths.append(mm_path)
        model_manifest_hashes[model] = sha256_file(mm_path)
        execution_guard_paths.append(guard_path)
        execution_guard_hashes[model] = sha256_file(guard_path)

        expected_paths = {
            result_path(model_root, condition["id"], entry): (ci, condition, entry)
            for ci, condition in enumerate(conditions)
            for entry in item_manifest["items"]
        }
        metadata_paths = {
            mm_path,
            guard_path,
            model_root / "serving_provenance.json",
            model_root / "run_summary.json",
        }
        require(all(path.is_file() for path in metadata_paths),
                f"serving/run/model metadata closure incomplete: {model}")
        all_json_paths = set(model_root.rglob("*.json"))
        require(all_json_paths == set(expected_paths) | metadata_paths,
                f"raw JSON missing/extra paths for {model}: "
                f"missing={len((set(expected_paths) | metadata_paths)-all_json_paths)} "
                f"extra={len(all_json_paths-(set(expected_paths) | metadata_paths))}")
        actual_paths = all_json_paths - metadata_paths
        temporary = list(model_root.rglob("*.tmp")) + list(model_root.rglob(".*.tmp"))
        require(not temporary, f"temporary raw records remain for {model}")
        require(actual_paths == set(expected_paths),
                f"raw missing/extra records for {model}: missing={len(set(expected_paths)-actual_paths)} extra={len(actual_paths-set(expected_paths))}")

        replayed_model_transport_counts: Counter[str] = Counter()
        replayed_model_finish_reasons: Counter[str] = Counter()
        replayed_model_max_prompt_tokens = 0
        replayed_model_max_completion_tokens = 0
        replayed_model_max_total_tokens = 0
        replayed_model_response_unit_count = 0
        replayed_model_empty_response_count = 0
        for path in sorted(actual_paths):
            ci, condition, entry = expected_paths[path]
            record = load_json(path)
            validate_record(
                record,
                model=model,
                profile="formal",
                condition=condition,
                entry=entry,
                item=items[entry["task_id"]],
                spec_sha=spec_sha,
                manifest_sha=item_sha,
                served_digest=digest,
                source_revision=revision,
                spec=spec,
            )
            expected_identity = {
                "family": families[mi],
                "condition_kind": condition["kind"],
                "target_group": condition["target_group"],
                "dimension": entry["dimension"],
                "group": entry["group"],
                "presentation_sha256": entry["presentation_sha256"],
            }
            for key, expected in expected_identity.items():
                require(record.get(key) == expected,
                        f"record {key} mismatch for {model}/{entry['task_id']}")
            pi, ii = item_index[entry["task_id"]]
            require(not np.isfinite(accuracy[mi, ci, pi, ii]), "duplicate raw record assignment")
            value = ensure_finite_accuracy(record["score"]["primary_accuracy"])
            accuracy[mi, ci, pi, ii] = value
            record_invalid = any(
                is_protocol_invalid_finish_reason(call.get("finish_reason"))
                for call in record["api_calls"]
            )
            protocol_invalid[mi, ci, pi, ii] = record_invalid
            record_truncated = any(
                call.get("finish_reason") == "length"
                for call in record["api_calls"]
            )
            record_transport_incomplete = any(
                call.get("finish_reason") == TRANSPORT_INCOMPLETE_FINISH_REASON
                for call in record["api_calls"]
            )
            record_recovered_fault_exposed = any(
                recovered_terminal_metadata_fault_exposure(call)
                for call in record["api_calls"]
            )
            truncated_record[mi, ci, pi, ii] = record_truncated
            transport_incomplete_record[mi, ci, pi, ii] = (
                record_transport_incomplete
            )
            recovered_terminal_metadata_fault_exposed[mi, ci, pi, ii] = (
                record_recovered_fault_exposed
            )
            for call in record["api_calls"]:
                attempts = call["attempts"]
                replayed_model_transport_counts["api_call_count"] += 1
                replayed_model_transport_counts["transport_attempt_count"] += len(
                    attempts
                )
                replayed_model_transport_counts["transport_retry_count"] += (
                    len(attempts) - 1
                )
                replayed_model_transport_counts[
                    "terminal_metadata_fault_attempt_count"
                ] += sum(
                    attempt["status"] == "protocol_fault" for attempt in attempts
                )
                replayed_model_transport_counts["request_error_attempt_count"] += sum(
                    attempt["status"] == "request_error" for attempt in attempts
                )
                replayed_model_transport_counts[
                    "usage_metadata_valid_logical_call_count"
                ] += int(call["usage_metadata_valid"] is True)
                if call["usage_metadata_valid"] is True:
                    usage = call["usage"]
                    replayed_model_max_prompt_tokens = max(
                        replayed_model_max_prompt_tokens, int(usage["prompt_tokens"])
                    )
                    replayed_model_max_completion_tokens = max(
                        replayed_model_max_completion_tokens,
                        int(usage["completion_tokens"]),
                    )
                    replayed_model_max_total_tokens = max(
                        replayed_model_max_total_tokens, int(usage["total_tokens"])
                    )
                replayed_model_transport_counts[
                    "transport_incomplete_logical_call_count"
                ] += int(
                    call.get("finish_reason")
                    == TRANSPORT_INCOMPLETE_FINISH_REASON
                )
                replayed_model_transport_counts[
                    "recovered_terminal_metadata_fault_logical_call_count"
                ] += int(recovered_terminal_metadata_fault_exposure(call))
                replayed_model_transport_counts["truncated_api_call_count"] += int(
                    call.get("finish_reason") == "length"
                )
                replayed_model_finish_reasons[str(call.get("finish_reason"))] += 1
            all_response_texts = _response_texts(record, entry["is_multiturn"])
            replayed_model_response_unit_count += len(all_response_texts)
            replayed_model_empty_response_count += sum(
                text == "" for text in all_response_texts
            )
            completion_contract = record["score"]["metrics"].get("completion_contract")
            invalid_call_count = sum(
                is_protocol_invalid_finish_reason(call.get("finish_reason"))
                for call in record["api_calls"]
            )
            transport_invalid_call_count = sum(
                call.get("finish_reason") == TRANSPORT_INCOMPLETE_FINISH_REASON
                for call in record["api_calls"]
            )
            require(
                isinstance(completion_contract, dict)
                and completion_contract.get("task_invalidated") is record_invalid
                and completion_contract.get("native_scorer_evaluated") is (not record_invalid)
                and completion_contract.get("invalid_call_count") == invalid_call_count
                and completion_contract.get("transport_protocol_invalid_call_count")
                == transport_invalid_call_count
                and completion_contract.get("truncated_call_count")
                == sum(call.get("finish_reason") == "length" for call in record["api_calls"]),
                f"completion-contract metadata mismatch: {model}/{entry['task_id']}",
            )
            require(not record_invalid or value == 0.0,
                    f"protocol-invalid task did not receive zero primary accuracy: {model}/{entry['task_id']}")
            texts = _primary_response_texts(
                record, entry, items[entry["task_id"]]
            )
            empty[mi, ci, pi, ii] = any(text == "" for text in texts)
            response_chars[mi, ci, pi, ii] = sum(len(text) for text in texts)
            if entry["paradigm"] == "operation_span":
                parse_status = record["score"]["metrics"]["strict_parse_status"]
                native_scorer_evaluated = completion_contract[
                    "native_scorer_evaluated"
                ]
                native_statuses = {
                    "final", "inline", "compact", "multiline", "single", "none"
                }
                require(
                    (
                        native_scorer_evaluated is True
                        and parse_status in native_statuses
                    )
                    or (
                        native_scorer_evaluated is False
                        and record_invalid
                        and parse_status == "not_evaluated_protocol_invalid"
                    ),
                    f"OSpan parse status/native-scorer contract mismatch: {parse_status}",
                )
                ospan_parse_none[mi, ci, pi, ii] = parse_status == "none"
                native_canonical = ensure_finite_accuracy(
                    record["score"]["metrics"]["canonical_accuracy"],
                    "operation-span canonical accuracy",
                )
                canonical_accuracy[mi, ci, pi, ii] = (
                    0.0 if record_invalid else native_canonical
                )
                ospan_math_accuracy[mi, ci, pi, ii] = ensure_finite_accuracy(
                    record["score"]["metrics"]["math_accuracy"],
                    "operation-span math accuracy",
                )
            else:
                canonical_accuracy[mi, ci, pi, ii] = value
            seen_records += 1
        replayed_model_truncated_counts = {
            condition_id: int(truncated_record[mi, ci, :, :].sum())
            for ci, condition_id in enumerate(condition_ids)
        }
        replayed_model_transport_incomplete_counts = {
            condition_id: int(transport_incomplete_record[mi, ci, :, :].sum())
            for ci, condition_id in enumerate(condition_ids)
        }
        replayed_model_invalid_counts = {
            condition_id: int(protocol_invalid[mi, ci, :, :].sum())
            for ci, condition_id in enumerate(condition_ids)
        }
        replayed_model_recovered_fault_counts = {
            condition_id: int(
                recovered_terminal_metadata_fault_exposed[mi, ci, :, :].sum()
            )
            for ci, condition_id in enumerate(condition_ids)
        }
        require(
            mm_truncated_counts == replayed_model_truncated_counts
            and mm_transport_invalid_counts
            == replayed_model_transport_incomplete_counts
            and mm_invalid_counts == replayed_model_invalid_counts,
            f"model protocol-invalid record counts do not replay: {model}",
        )
        require(
            mm_recovered_fault_counts == replayed_model_recovered_fault_counts,
            f"model recovered terminal-metadata-fault exposure counts do not replay: {model}",
        )
        ospan_index_for_replay = paradigms.index("operation_span")
        require(
            mm.get("operation_span_protocol_invalid_not_evaluated_count")
            == int(protocol_invalid[mi, :, ospan_index_for_replay, :].sum()),
            f"model operation-span protocol-invalid not-evaluated count does not replay: {model}",
        )
        require(
            mm.get("operation_span_nonparseable_count")
            == int(ospan_parse_none[mi, :, ospan_index_for_replay, :].sum()),
            f"model operation-span native nonparseable count does not replay: {model}",
        )
        replayed_model_expected_totals = {
            "api_call_count": replayed_model_transport_counts["api_call_count"],
            "transport_attempt_count": replayed_model_transport_counts[
                "transport_attempt_count"
            ],
            "transport_retry_count": replayed_model_transport_counts[
                "transport_retry_count"
            ],
            "terminal_metadata_fault_attempt_count": replayed_model_transport_counts[
                "terminal_metadata_fault_attempt_count"
            ],
            "request_error_attempt_count": replayed_model_transport_counts[
                "request_error_attempt_count"
            ],
            "usage_metadata_valid_logical_call_count": replayed_model_transport_counts[
                "usage_metadata_valid_logical_call_count"
            ],
            "transport_incomplete_logical_call_count": replayed_model_transport_counts[
                "transport_incomplete_logical_call_count"
            ],
            "recovered_terminal_metadata_fault_logical_call_count": (
                replayed_model_transport_counts[
                    "recovered_terminal_metadata_fault_logical_call_count"
                ]
            ),
            "truncated_completion_count": replayed_model_transport_counts[
                "truncated_api_call_count"
            ],
            "truncated_api_call_count": replayed_model_transport_counts[
                "truncated_api_call_count"
            ],
            "truncated_record_count": sum(replayed_model_truncated_counts.values()),
            "transport_protocol_invalid_record_count": sum(
                replayed_model_transport_incomplete_counts.values()
            ),
            "invalid_record_count": sum(replayed_model_invalid_counts.values()),
            "recovered_terminal_metadata_fault_record_count": sum(
                replayed_model_recovered_fault_counts.values()
            ),
            "response_unit_count": replayed_model_response_unit_count,
            "empty_response_count": replayed_model_empty_response_count,
            "max_reported_prompt_tokens": replayed_model_max_prompt_tokens,
            "max_reported_completion_tokens": replayed_model_max_completion_tokens,
            "max_reported_total_tokens": replayed_model_max_total_tokens,
            "minimum_reported_prompt_reservation_margin_tokens": (
                int(spec["scope"]["served_context_tokens"])
                - int(spec["scope"]["max_completion_tokens"])
                - replayed_model_max_prompt_tokens
            ),
        }
        require(
            all(mm.get(key) == value for key, value in replayed_model_expected_totals.items()),
            f"model call/attempt transport totals do not replay: {model}",
        )
        require(
            mm.get("finish_reason_counts")
            == dict(sorted(replayed_model_finish_reasons.items())),
            f"model finish-reason totals do not replay: {model}",
        )
        replayed_run_transport_counts.update(replayed_model_expected_totals)
        replayed_run_max_prompt_tokens = max(
            replayed_run_max_prompt_tokens, replayed_model_max_prompt_tokens
        )
        replayed_run_max_completion_tokens = max(
            replayed_run_max_completion_tokens,
            replayed_model_max_completion_tokens,
        )
        replayed_run_max_total_tokens = max(
            replayed_run_max_total_tokens, replayed_model_max_total_tokens
        )
        replayed_run_response_unit_count += replayed_model_response_unit_count
        replayed_run_empty_response_count += replayed_model_empty_response_count
        require(mm.get("record_tree_sha256") == tree_hash(actual_paths, model_root),
                f"MODEL_MANIFEST raw record tree mismatch: {model}")
        require(mm.get("serving_provenance_sha256") == sha256_file(model_root / "serving_provenance.json"),
                f"MODEL_MANIFEST serving provenance mismatch: {model}")
        require(mm.get("run_summary_sha256") == sha256_file(model_root / "run_summary.json"),
                f"MODEL_MANIFEST run summary mismatch: {model}")
        serving = load_json(model_root / "serving_provenance.json")
        expected_context = spec["scope"]["served_context_tokens"]
        require(serving.get("actual_context_tokens") == expected_context,
                f"served context mismatch: {model}")
        require(mm.get("actual_context_tokens") == expected_context,
                f"MODEL_MANIFEST context mismatch: {model}")
        sessions = serving.get("sessions")
        require(isinstance(sessions, list) and sessions,
                f"serving sessions missing: {model}")
        require(all(session.get("actual_context_tokens") == expected_context
                    for session in sessions),
                f"mixed serving context: {model}")

    require(seen_records == 12 * 234 * 7, "analyzer enumeration count mismatch")
    require(np.isfinite(accuracy).all(), "non-finite or unfilled formal accuracy cell")
    require(np.isfinite(canonical_accuracy).all(),
            "non-finite or unfilled canonical-sensitivity accuracy cell")
    require(((0.0 <= accuracy) & (accuracy <= 1.0)).all(), "accuracy outside [0,1]")
    recomputed_condition_counts = {
        condition_id: int(protocol_invalid[:, ci, :, :].size)
        for ci, condition_id in enumerate(condition_ids)
    }
    recomputed_invalid_counts = {
        condition_id: int(protocol_invalid[:, ci, :, :].sum())
        for ci, condition_id in enumerate(condition_ids)
    }
    recomputed_truncated_counts = {
        condition_id: int(truncated_record[:, ci, :, :].sum())
        for ci, condition_id in enumerate(condition_ids)
    }
    recomputed_transport_incomplete_counts = {
        condition_id: int(transport_incomplete_record[:, ci, :, :].sum())
        for ci, condition_id in enumerate(condition_ids)
    }
    recomputed_recovered_fault_counts = {
        condition_id: int(
            recovered_terminal_metadata_fault_exposed[:, ci, :, :].sum()
        )
        for ci, condition_id in enumerate(condition_ids)
    }
    require(run.get("condition_record_counts") == recomputed_condition_counts,
            "formal RUN_MANIFEST condition record counts do not replay")
    require(run.get("condition_invalid_record_counts") == recomputed_invalid_counts,
            "formal RUN_MANIFEST condition invalid counts do not replay")
    require(
        run.get("condition_truncated_record_counts") == recomputed_truncated_counts
        and run.get("condition_transport_protocol_invalid_record_counts")
        == recomputed_transport_incomplete_counts,
        "formal RUN_MANIFEST condition invalid-subtype counts do not replay",
    )
    require(
        run.get("condition_recovered_terminal_metadata_fault_record_counts")
        == recomputed_recovered_fault_counts,
        "formal RUN_MANIFEST recovered terminal-metadata-fault exposure counts do not replay",
    )
    run_replayed_fields = (
        "api_call_count",
        "transport_attempt_count",
        "transport_retry_count",
        "terminal_metadata_fault_attempt_count",
        "request_error_attempt_count",
        "usage_metadata_valid_logical_call_count",
        "transport_incomplete_logical_call_count",
        "recovered_terminal_metadata_fault_logical_call_count",
        "truncated_completion_count",
        "truncated_api_call_count",
        "truncated_record_count",
        "transport_protocol_invalid_record_count",
        "invalid_record_count",
        "recovered_terminal_metadata_fault_record_count",
        "response_unit_count",
        "empty_response_count",
    )
    require(
        all(
            run.get(key) == replayed_run_transport_counts[key]
            for key in run_replayed_fields
        ),
        "formal RUN_MANIFEST call/attempt/record transport totals do not replay",
    )
    require(
        run.get("max_reported_prompt_tokens") == replayed_run_max_prompt_tokens
        and run.get("max_reported_completion_tokens")
        == replayed_run_max_completion_tokens
        and run.get("max_reported_total_tokens") == replayed_run_max_total_tokens
        and run.get("minimum_reported_prompt_reservation_margin_tokens")
        == int(spec["scope"]["served_context_tokens"])
        - int(spec["scope"]["max_completion_tokens"])
        - replayed_run_max_prompt_tokens,
        "formal RUN_MANIFEST reported-usage extrema do not replay",
    )
    ospan_index_for_replay = paradigms.index("operation_span")
    require(
        run.get("operation_span_nonparseable_count")
        == int(ospan_parse_none[:, :, ospan_index_for_replay, :].sum())
        and run.get("operation_span_protocol_invalid_not_evaluated_count")
        == int(protocol_invalid[:, :, ospan_index_for_replay, :].sum()),
        "formal RUN_MANIFEST operation-span parser/not-evaluated totals do not replay",
    )
    require(
        tree_hash(model_manifest_paths, profile_root) == run.get("model_manifest_tree_sha256"),
        "formal RUN_MANIFEST model-manifest tree hash mismatch",
    )
    require(
        tree_hash(execution_guard_paths, profile_root)
        == run.get("execution_guard_tree_sha256"),
        "formal RUN_MANIFEST execution-guard tree hash mismatch",
    )

    return {
        "spec": spec,
        "run": run,
        "run_path": run_path,
        "run_sha256": sha256_file(run_path),
        "item_manifest": item_manifest,
        "item_path": item_path,
        "item_sha256": item_sha,
        "spec_sha256": spec_sha,
        "models": models,
        "families": families,
        "conditions": conditions,
        "condition_ids": condition_ids,
        "paradigms": paradigms,
        "difficulties": difficulties,
        "accuracy": accuracy,
        "canonical_accuracy": canonical_accuracy,
        "ospan_math_accuracy": ospan_math_accuracy,
        "empty": empty,
        "response_chars": response_chars,
        "ospan_parse_none": ospan_parse_none,
        "protocol_invalid": protocol_invalid,
        "recovered_terminal_metadata_fault_exposed": (
            recovered_terminal_metadata_fault_exposed
        ),
        "model_manifest_hashes": model_manifest_hashes,
        "execution_guard_hashes": execution_guard_hashes,
        "source_revision": revision,
    }


def _target_structure(data: dict[str, Any]) -> tuple[list[int], int, list[str], dict[str, np.ndarray]]:
    targeted = [condition for condition in data["conditions"] if condition["kind"] == "targeted"]
    target_indices = [data["condition_ids"].index(condition["id"]) for condition in targeted]
    placebo_index = data["condition_ids"].index("neutral_placebo")
    target_groups = [condition["target_group"] for condition in targeted]
    masks = {
        group: np.array([p in paradigms for p in data["paradigms"]], dtype=bool)
        for group, paradigms in data["spec"]["grouping"].items()
    }
    require(len(targeted) == 5 and set(target_groups) == set(masks), "target/group map mismatch")
    return target_indices, placebo_index, target_groups, masks


def compute_estimate(
    gains: np.ndarray,
    target_groups: list[str],
    group_masks: dict[str, np.ndarray],
    valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Equal-item, equal-paradigm, equal-model estimator; NaNs only for sensitivities."""
    values = gains.copy()
    if valid is not None:
        require(valid.shape == values.shape, "validity mask shape mismatch")
        values[~valid] = np.nan
    finite = np.isfinite(values)
    counts = finite.sum(axis=3)
    sums = np.nansum(values, axis=3)
    cell = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    model_s = np.full((values.shape[0], values.shape[1]), np.nan)
    for j, group in enumerate(target_groups):
        matched = group_masks[group]
        matched_mean = np.nanmean(cell[:, j, matched], axis=1)
        nonmatched_mean = np.nanmean(cell[:, j, ~matched], axis=1)
        model_s[:, j] = matched_mean - nonmatched_mean
    require(np.isfinite(model_s).all(), "sensitivity left an unestimable model/intervention contrast")
    s = model_s.mean(axis=0)
    gamma = float(s.mean())
    return {
        "gamma": gamma,
        "s": s,
        "model_s": model_s,
        "cell_item_counts": counts,
        "intervention_paradigm_mean_gain": np.nanmean(values, axis=(0, 3)),
    }


def _contrast_from_paradigm_means(
    means: np.ndarray, target_groups: list[str], group_masks: dict[str, np.ndarray]
) -> tuple[np.ndarray, float]:
    s = np.empty(len(target_groups), dtype=float)
    for j, group in enumerate(target_groups):
        matched = group_masks[group]
        s[j] = means[j, matched].mean() - means[j, ~matched].mean()
    return s, float(s.mean())


def crossed_family_item_bootstrap(
    gains: np.ndarray,
    families: list[str],
    target_groups: list[str],
    group_masks: dict[str, np.ndarray],
    *,
    item_indices: list[np.ndarray] | None = None,
    valid: np.ndarray | None = None,
    n_boot: int = N_BOOTSTRAP,
    seed: int = ANALYSIS_SEED,
) -> dict[str, Any]:
    values = gains.copy()
    if valid is not None:
        require(valid.shape == values.shape, "bootstrap validity mask shape mismatch")
        values[~valid] = np.nan
    family_order = list(dict.fromkeys(families))
    family_models = [np.flatnonzero(np.array(families) == family) for family in family_order]
    require(len(family_models) == 6 and all(len(x) == 2 for x in family_models),
            "bootstrap requires six families with two checkpoints each")
    if item_indices is None:
        item_indices = [np.arange(gains.shape[3]) for _ in range(gains.shape[2])]
    require(len(item_indices) == gains.shape[2] and all(len(x) > 0 for x in item_indices),
            "empty bootstrap item stratum")
    rng = np.random.default_rng(seed)
    gamma_samples = np.empty(n_boot, dtype=float)
    s_samples = np.empty((n_boot, gains.shape[1]), dtype=float)
    for b in range(n_boot):
        sampled_families = rng.integers(0, len(family_models), size=len(family_models))
        model_indices = np.concatenate([family_models[index] for index in sampled_families])
        means = np.empty((gains.shape[1], gains.shape[2]), dtype=float)
        for p, eligible in enumerate(item_indices):
            sampled_items = eligible[rng.integers(0, len(eligible), size=len(eligible))]
            sampled = values[model_indices, :, p, :][:, :, sampled_items]
            counts = np.isfinite(sampled).sum(axis=(0, 2))
            sums = np.nansum(sampled, axis=(0, 2))
            means[:, p] = np.divide(
                sums, counts, out=np.full(gains.shape[1], np.nan), where=counts > 0
            )
        require(np.isfinite(means).all(),
                "bootstrap draw contains an unestimable intervention/paradigm cell")
        s, gamma = _contrast_from_paradigm_means(means, target_groups, group_masks)
        s_samples[b] = s
        gamma_samples[b] = gamma
    gamma_ci = np.quantile(gamma_samples, [0.025, 0.975])
    s_ci = np.quantile(s_samples, [0.025, 0.975], axis=0).T
    return {
        "n_bootstrap": n_boot,
        "seed": seed,
        "gamma_ci95": gamma_ci.tolist(),
        "descriptive_bootstrap_sign_tail_fraction_two_sided": float(
            min(1.0, 2 * min(np.mean(gamma_samples <= 0), np.mean(gamma_samples >= 0)))
        ),
        "s_ci95": s_ci.tolist(),
    }


def crossed_difficulty_interaction_bootstrap(
    gains: np.ndarray,
    families: list[str],
    target_groups: list[str],
    group_masks: dict[str, np.ndarray],
    difficulties: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = DIFFICULTY_INTERACTION_SEED,
) -> dict[str, Any]:
    """Crossed family/item CI for the frozen hard-minus-easy selectivity contrast."""
    family_order = list(dict.fromkeys(families))
    family_models = [np.flatnonzero(np.asarray(families) == family) for family in family_order]
    require(len(family_models) == 6 and all(len(x) == 2 for x in family_models),
            "difficulty bootstrap requires six 2-checkpoint families")
    strata = {
        difficulty: [np.flatnonzero(difficulties[p] == difficulty)
                     for p in range(gains.shape[2])]
        for difficulty in ("easy", "hard")
    }
    require(all(len(items) == 6 for group in strata.values() for items in group),
            "difficulty interaction requires six easy and six hard items per paradigm")
    rng = np.random.default_rng(seed)
    delta_gamma = np.empty(n_boot, dtype=float)
    delta_s = np.empty((n_boot, gains.shape[1]), dtype=float)
    for b in range(n_boot):
        sampled_families = rng.integers(0, 6, size=6)
        model_indices = np.concatenate([family_models[index] for index in sampled_families])
        estimates: dict[str, tuple[np.ndarray, float]] = {}
        for difficulty in ("easy", "hard"):
            means = np.empty((gains.shape[1], gains.shape[2]), dtype=float)
            for p, eligible in enumerate(strata[difficulty]):
                sampled_items = eligible[rng.integers(0, len(eligible), size=len(eligible))]
                means[:, p] = gains[model_indices, :, p, :][:, :, sampled_items].mean(
                    axis=(0, 2)
                )
            estimates[difficulty] = _contrast_from_paradigm_means(
                means, target_groups, group_masks
            )
        easy_s, easy_gamma = estimates["easy"]
        hard_s, hard_gamma = estimates["hard"]
        delta_s[b] = hard_s - easy_s
        delta_gamma[b] = hard_gamma - easy_gamma
    return {
        "contrast": "hard_minus_easy",
        "n_bootstrap": n_boot,
        "seed": seed,
        "delta_gamma_ci95": np.quantile(delta_gamma, [0.025, 0.975]).tolist(),
        "delta_s_ci95": np.quantile(delta_s, [0.025, 0.975], axis=0).T.tolist(),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    require(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), "invalid p values")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    for rank_index in range(len(p) - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p[original_index] * len(p) / rank)
        adjusted[original_index] = running
    return adjusted.tolist()


def exact_intervention_mapping_permutation(
    means: np.ndarray, target_groups: list[str], group_masks: dict[str, np.ndarray]
) -> dict[str, Any]:
    observed_s, observed_gamma = _contrast_from_paradigm_means(means, target_groups, group_masks)
    gamma_null = []
    s_null = [[] for _ in target_groups]
    mappings = list(itertools.permutations(group_masks.keys()))
    require(len(mappings) == math.factorial(5), "mapping permutation is not exact 5!")
    for mapping in mappings:
        s, gamma = _contrast_from_paradigm_means(means, list(mapping), group_masks)
        gamma_null.append(gamma)
        for j, value in enumerate(s):
            s_null[j].append(value)
    gamma_null_array = np.asarray(gamma_null)
    gamma_center = float(gamma_null_array.mean())
    p_one = float(np.mean(gamma_null_array >= observed_gamma - 1e-15))
    p_two = float(np.mean(
        np.abs(gamma_null_array - gamma_center)
        >= abs(observed_gamma - gamma_center) - 1e-15
    ))
    s_p_two = [
        float(np.mean(
            np.abs(values - values.mean())
            >= abs(observed_s[j] - values.mean()) - 1e-15
        ))
        for j, values in enumerate(map(np.asarray, s_null))
    ]
    return {
        "n_exact_mappings": len(mappings),
        "tie_rule": "inclusive with absolute tolerance 1e-15",
        "two_sided_rule": "distance from the exact permutation-null mean, not distance from zero",
        "gamma_null_mean": gamma_center,
        "gamma_p_one": p_one,
        "gamma_p_two": p_two,
        "gamma_null_quantiles": np.quantile(gamma_null_array, [0, 0.025, 0.5, 0.975, 1]).tolist(),
        "s_distinct_group_assignments_each": 5,
        "s_exact_p_resolution": 0.2,
        "s_p_two": s_p_two,
        "s_p_two_bh": benjamini_hochberg(s_p_two),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def _fit_ridge_logistic(x: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> tuple[np.ndarray, int]:
    require(x.ndim == 2 and y.ndim == 1 and len(x) == len(y), "bad logistic design")
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1], dtype=float)
    mean_y = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    beta[0] = math.log(mean_y / (1 - mean_y))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    for iteration in range(1, 101):
        probability = _sigmoid(design @ beta)
        weights = np.clip(probability * (1 - probability), 1e-6, None)
        gradient = design.T @ (probability - y) + lam * (penalty @ beta)
        hessian = design.T @ (weights[:, None] * design) + lam * penalty
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-9:
            return beta, iteration
    raise RuntimeError("fixed ridge-logistic solver did not converge")


def _soft_bernoulli_log_likelihood(y: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability, 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def family_lofo_predictive(
    target_accuracy: np.ndarray,
    placebo_accuracy: np.ndarray,
    families: list[str],
    target_groups: list[str],
    group_masks: dict[str, np.ndarray],
    difficulties: np.ndarray,
) -> dict[str, Any]:
    """Fixed-lambda family-LOFO soft-Bernoulli global vs selective comparison."""
    m_count, j_count, p_count, i_count = target_accuracy.shape
    rows = []
    y = []
    family_row = []
    for m in range(m_count):
        for j in range(j_count):
            for p in range(p_count):
                for i in range(i_count):
                    # Global model: placebo accuracy + additive intervention,
                    # paradigm, and difficulty. Selective adds the five frozen
                    # intervention-by-matched-group diagonal indicators.
                    global_features = [float(placebo_accuracy[m, p, i])]
                    global_features += [float(j == k) for k in range(1, j_count)]
                    global_features += [float(p == k) for k in range(1, p_count)]
                    global_features += [float(difficulties[p, i] == d) for d in ("medium", "hard")]
                    selective = [float(j == k and group_masks[target_groups[k]][p]) for k in range(j_count)]
                    rows.append((global_features, selective))
                    y.append(float(target_accuracy[m, j, p, i]))
                    family_row.append(families[m])
    y_array = np.asarray(y)
    family_row_array = np.asarray(family_row)
    global_x = np.asarray([row[0] for row in rows], dtype=float)
    selective_x = np.asarray([row[0] + row[1] for row in rows], dtype=float)
    family_order = list(dict.fromkeys(families))
    per_family = []
    total_global = total_selective = 0.0
    iterations_global = iterations_selective = 0
    for family in family_order:
        test = family_row_array == family
        train = ~test
        # Only the continuous placebo feature is transformed; its mean/std
        # are learned on the training families and applied to held-out rows.
        mean = global_x[train, 0].mean()
        std = global_x[train, 0].std()
        if std < 1e-12:
            std = 1.0
        gx_train = global_x[train].copy()
        gx_test = global_x[test].copy()
        sx_train = selective_x[train].copy()
        sx_test = selective_x[test].copy()
        for matrix in (gx_train, gx_test, sx_train, sx_test):
            matrix[:, 0] = (matrix[:, 0] - mean) / std
        global_beta, gi = _fit_ridge_logistic(gx_train, y_array[train])
        selective_beta, si = _fit_ridge_logistic(sx_train, y_array[train])
        gp = _sigmoid(np.column_stack([np.ones(test.sum()), gx_test]) @ global_beta)
        sp = _sigmoid(np.column_stack([np.ones(test.sum()), sx_test]) @ selective_beta)
        global_ll = _soft_bernoulli_log_likelihood(y_array[test], gp)
        selective_ll = _soft_bernoulli_log_likelihood(y_array[test], sp)
        total_global += global_ll
        total_selective += selective_ll
        iterations_global += gi
        iterations_selective += si
        per_family.append(
            {
                "family": family,
                "n_test_records": int(test.sum()),
                "global_log_likelihood": global_ll,
                "selective_log_likelihood": selective_ll,
                "delta_log_likelihood": selective_ll - global_ll,
            }
        )
    return {
        "method": "six-fold leave-one-family-out soft-Bernoulli ridge logistic; lambda=1 fixed; train-only placebo-accuracy standardization; global=placebo+additive intervention+paradigm+difficulty; selective adds five frozen diagonal interactions",
        "ridge_lambda": RIDGE_LAMBDA,
        "global_log_likelihood": total_global,
        "selective_log_likelihood": total_selective,
        "delta_log_likelihood": total_selective - total_global,
        "delta_mean_log_likelihood_per_record": (total_selective - total_global) / len(y_array),
        "families_improved": sum(row["delta_log_likelihood"] > 0 for row in per_family),
        "per_family": per_family,
        "solver_iterations_total": {
            "global": iterations_global,
            "selective": iterations_selective,
        },
    }


def response_length_adjustment(
    gains: np.ndarray,
    target_chars: np.ndarray,
    placebo_chars: np.ndarray,
    target_groups: list[str],
    group_masks: dict[str, np.ndarray],
    difficulties: np.ndarray,
    valid: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    delta_length = np.log1p(target_chars) - np.log1p(placebo_chars[:, None, :, :])
    require(valid.shape == gains.shape, "response-length validity mask shape mismatch")
    require(valid.dtype == bool, "response-length validity mask is not boolean")
    m_count, j_count, p_count, i_count = gains.shape
    outcome = gains.reshape(-1)
    length = delta_length.reshape(-1)
    selected = valid.reshape(-1)
    require(selected.any(), "response-length adjustment has no protocol-valid paired observations")
    design_rows = []
    for m in range(m_count):
        for j in range(j_count):
            for p in range(p_count):
                for i in range(i_count):
                    row = [1.0]
                    row += [float(m == k) for k in range(1, m_count)]
                    row += [float(j == k) for k in range(1, j_count)]
                    row += [float(p == k) for k in range(1, p_count)]
                    row += [float(difficulties[p, i] == d) for d in ("medium", "hard")]
                    design_rows.append(row)
    nuisance = np.asarray(design_rows, dtype=float)[selected]
    outcome = outcome[selected]
    length = length[selected]
    require(np.linalg.matrix_rank(nuisance) == nuisance.shape[1],
            "response-length nuisance design is rank deficient")
    length_residual = length - nuisance @ np.linalg.lstsq(nuisance, length, rcond=None)[0]
    outcome_residual = outcome - nuisance @ np.linalg.lstsq(nuisance, outcome, rcond=None)[0]
    length_information = float(length_residual @ length_residual)
    identifiable = length_information > 1e-12
    beta = float(length_residual @ outcome_residual / length_information) if identifiable else 0.0
    adjusted = gains - beta * delta_length
    estimate = compute_estimate(adjusted, target_groups, group_masks, valid)
    return (
        {
            "method": "descriptive OLS nuisance adjustment of paired gain by log1p target-minus-placebo primary-scoring response characters, with additive model/intervention/paradigm/difficulty fixed effects; pairs with either task record protocol-invalid are excluded before fitting and estimation; no tuning",
            "protocol_invalid_pair_rule": "exclude paired target-placebo observation if either complete task record contains a length completion or an exhausted terminal-metadata fault",
            "excluded_protocol_invalid_pairs": int((~valid).sum()),
            "minimum_items_per_model_intervention_paradigm": int(
                estimate["cell_item_counts"].min()
            ),
            "length_slope": beta,
            "length_slope_identifiable": identifiable,
            "mean_log_length_difference": float(length.mean()),
            "gamma": estimate["gamma"],
            "s": estimate["s"].tolist(),
        },
        adjusted,
    )


def condition_pc1_shares(
    accuracy: np.ndarray, condition_ids: list[str], paradigms: list[str]
) -> dict[str, Any]:
    """PC1 share after column-wise z scoring of each condition's 12 x 13 matrix."""
    cell = accuracy.mean(axis=3)
    placebo_index = condition_ids.index("neutral_placebo")
    rows = []
    shares = []
    for ci, condition_id in enumerate(condition_ids):
        matrix = cell[:, ci, :]
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=0)
        invariant = std < 1e-12
        safe_std = std.copy()
        safe_std[invariant] = 1.0
        z = (matrix - mean) / safe_std
        z[:, invariant] = 0.0
        singular = np.linalg.svd(z, full_matrices=False, compute_uv=False)
        variance = singular ** 2
        require(variance.sum() > 0, f"PC1 undefined for condition: {condition_id}")
        share = float(variance[0] / variance.sum())
        shares.append(share)
        rows.append(
            {
                "condition": condition_id,
                "pc1_share": share,
                "delta_vs_neutral_placebo": None,
                "invariant_paradigms": [
                    paradigms[index] for index in np.flatnonzero(invariant)
                ],
            }
        )
    for row, share in zip(rows, shares):
        row["delta_vs_neutral_placebo"] = share - shares[placebo_index]
    return {
        "method": "item-mean 12-model x 13-paradigm matrix per condition; paradigm-wise population-z scores; invariant columns set to zero; SVD squared-singular-value share",
        "conditions": rows,
    }


FORBIDDEN_AGGREGATE_KEYS = {
    "response",
    "responses",
    "response_text",
    "stimulus",
    "expected_response",
    "gold",
    "task_id",
    "item_fingerprint_sha256",
    "presentation_sha256",
    "scoring_gold_sha256",
    "strict_tokens",
    "item_scores",
    "item_accuracy",
    "model_item_scores",
}


def validate_aggregate_output(value: Any, path: str = "analysis_results") -> None:
    """Recursively prevent item/raw payloads from drifting into public analysis output."""
    if isinstance(value, dict):
        forbidden = set(value) & FORBIDDEN_AGGREGATE_KEYS
        forbidden.update(
            key for key in value
            if "stimulus" in key.lower()
            or "scoring_gold" in key.lower()
            or "item_fingerprint" in key.lower()
            or key.lower().endswith("task_id")
        )
        require(not forbidden, f"raw/item field leaked at {path}: {sorted(forbidden)}")
        for key, child in value.items():
            validate_aggregate_output(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_aggregate_output(child, f"{path}[{index}]")


def validate_canonical_analysis_paths(raw_root: Path, output_dir: Path) -> None:
    require(raw_root.resolve() == (RESULTS_ROOT / "raw").resolve(),
            "noncanonical formal raw root refused")
    require(output_dir.resolve() == (RESULTS_ROOT / "analysis").resolve(),
            "noncanonical analysis output directory refused")


def analyze_arrays(data: dict[str, Any], *, n_boot: int = N_BOOTSTRAP) -> dict[str, Any]:
    thresholds = confirmatory_gate_thresholds(data["spec"])
    target_indices, placebo_index, target_groups, group_masks = _target_structure(data)
    target_accuracy = data["accuracy"][:, target_indices, :, :]
    placebo_accuracy = data["accuracy"][:, placebo_index, :, :]
    gains = target_accuracy - placebo_accuracy[:, None, :, :]
    primary = compute_estimate(gains, target_groups, group_masks)

    canonical_target = data["canonical_accuracy"][:, target_indices, :, :]
    canonical_placebo = data["canonical_accuracy"][:, placebo_index, :, :]
    canonical_gains = canonical_target - canonical_placebo[:, None, :, :]
    canonical = compute_estimate(canonical_gains, target_groups, group_masks)

    bootstrap = crossed_family_item_bootstrap(
        gains,
        data["families"],
        target_groups,
        group_masks,
        n_boot=n_boot,
        seed=ANALYSIS_SEED,
    )
    permutation = exact_intervention_mapping_permutation(
        primary["intervention_paradigm_mean_gain"], target_groups, group_masks
    )
    canonical_bootstrap = crossed_family_item_bootstrap(
        canonical_gains,
        data["families"],
        target_groups,
        group_masks,
        n_boot=n_boot,
        seed=ANALYSIS_SEED,
    )
    canonical_permutation = exact_intervention_mapping_permutation(
        canonical["intervention_paradigm_mean_gain"], target_groups, group_masks
    )
    family_order = list(dict.fromkeys(data["families"]))
    model_gamma = primary["model_s"].mean(axis=1)
    family_gamma = {
        family: float(model_gamma[np.array(data["families"]) == family].mean())
        for family in family_order
    }

    target_empty = data["empty"][:, target_indices, :, :]
    placebo_empty = data["empty"][:, placebo_index, :, :]
    empty_valid = ~(target_empty | placebo_empty[:, None, :, :])
    empty_estimate = compute_estimate(gains, target_groups, group_masks, empty_valid)
    empty_bootstrap = crossed_family_item_bootstrap(
        gains,
        data["families"],
        target_groups,
        group_masks,
        valid=empty_valid,
        n_boot=n_boot,
        seed=ANALYSIS_SEED,
    )

    target_invalid = data["protocol_invalid"][:, target_indices, :, :]
    placebo_invalid = data["protocol_invalid"][:, placebo_index, :, :]
    protocol_valid = ~(
        target_invalid | placebo_invalid[:, None, :, :]
    )
    protocol_valid_cell_counts = protocol_valid.sum(axis=3)
    protocol_valid_estimable = bool((protocol_valid_cell_counts > 0).all())
    protocol_valid_estimate = (
        compute_estimate(gains, target_groups, group_masks, protocol_valid)
        if protocol_valid_estimable
        else None
    )
    protocol_valid_bootstrap = (
        crossed_family_item_bootstrap(
            gains,
            data["families"],
            target_groups,
            group_masks,
            valid=protocol_valid,
            n_boot=n_boot,
            seed=ANALYSIS_SEED,
        )
        if protocol_valid_estimable
        else None
    )

    target_retry_exposed = data[
        "recovered_terminal_metadata_fault_exposed"
    ][:, target_indices, :, :]
    placebo_retry_exposed = data[
        "recovered_terminal_metadata_fault_exposed"
    ][:, placebo_index, :, :]
    retry_unexposed = ~(
        target_retry_exposed | placebo_retry_exposed[:, None, :, :]
    )
    retry_unexposed_cell_counts = retry_unexposed.sum(axis=3)
    retry_unexposed_estimable = bool((retry_unexposed_cell_counts > 0).all())
    retry_unexposed_estimate = (
        compute_estimate(gains, target_groups, group_masks, retry_unexposed)
        if retry_unexposed_estimable
        else None
    )

    parse_valid = np.ones_like(gains, dtype=bool)
    ospan_index = data["paradigms"].index("operation_span")
    target_parse_none = data["ospan_parse_none"][:, target_indices, ospan_index, :]
    placebo_parse_none = data["ospan_parse_none"][:, placebo_index, ospan_index, :]
    parse_valid[:, :, ospan_index, :] = ~(
        target_parse_none | placebo_parse_none[:, None, :]
    )
    parse_estimate = compute_estimate(gains, target_groups, group_masks, parse_valid)
    parse_bootstrap = crossed_family_item_bootstrap(
        gains,
        data["families"],
        target_groups,
        group_masks,
        valid=parse_valid,
        n_boot=n_boot,
        seed=ANALYSIS_SEED,
    )

    difficulty_results = {}
    for difficulty, seed in DIFFICULTY_SEEDS.items():
        indices = [np.flatnonzero(data["difficulties"][p] == difficulty)
                   for p in range(len(data["paradigms"]))]
        require(all(len(x) == 6 for x in indices), f"difficulty stratum is not 6/item: {difficulty}")
        valid = np.zeros_like(gains, dtype=bool)
        for p, selected in enumerate(indices):
            valid[:, :, p, selected] = True
        estimate = compute_estimate(gains, target_groups, group_masks, valid)
        boot = crossed_family_item_bootstrap(
            gains,
            data["families"],
            target_groups,
            group_masks,
            item_indices=indices,
            n_boot=n_boot,
            seed=seed,
        )
        difficulty_results[difficulty] = {
            "gamma": estimate["gamma"],
            "s": estimate["s"].tolist(),
            "bootstrap": boot,
        }

    difficulty_interaction_bootstrap = crossed_difficulty_interaction_bootstrap(
        gains,
        data["families"],
        target_groups,
        group_masks,
        data["difficulties"],
        n_boot=n_boot,
        seed=DIFFICULTY_INTERACTION_SEED,
    )
    difficulty_interaction = {
        "contrast": "hard_minus_easy",
        "delta_gamma": (
            difficulty_results["hard"]["gamma"] - difficulty_results["easy"]["gamma"]
        ),
        "delta_s": (
            np.asarray(difficulty_results["hard"]["s"])
            - np.asarray(difficulty_results["easy"]["s"])
        ).tolist(),
        "bootstrap": difficulty_interaction_bootstrap,
    }

    length_adjusted, adjusted_gains = response_length_adjustment(
        gains,
        data["response_chars"][:, target_indices, :, :],
        data["response_chars"][:, placebo_index, :, :],
        target_groups,
        group_masks,
        data["difficulties"],
        protocol_valid,
    )
    length_adjusted["bootstrap"] = crossed_family_item_bootstrap(
        adjusted_gains,
        data["families"],
        target_groups,
        group_masks,
        valid=protocol_valid,
        n_boot=n_boot,
        seed=ANALYSIS_SEED,
    )
    predictive = family_lofo_predictive(
        target_accuracy,
        placebo_accuracy,
        data["families"],
        target_groups,
        group_masks,
        data["difficulties"],
    )
    pc1 = condition_pc1_shares(data["accuracy"], data["condition_ids"], data["paradigms"])

    math = data["ospan_math_accuracy"]
    math_placebo = math[:, placebo_index, ospan_index, :]
    require(np.isfinite(math_placebo).all(), "OSpan placebo math metric is incomplete")
    math_rows = []
    for j, index in enumerate(target_indices):
        target_math = math[:, index, ospan_index, :]
        require(np.isfinite(target_math).all(), "OSpan targeted math metric is incomplete")
        math_rows.append(
            {
                "intervention": data["condition_ids"][index],
                "target_group": target_groups[j],
                "target_mean_math_accuracy": float(target_math.mean()),
                "neutral_placebo_mean_math_accuracy": float(math_placebo.mean()),
                "paired_gain": float((target_math - math_placebo).mean()),
            }
        )

    intervention_ids = [data["conditions"][index]["id"] for index in target_indices]
    paradigm_means = primary["intervention_paradigm_mean_gain"]
    group_order = list(data["spec"]["grouping"])
    intervention_group_matrix = []
    s_rows = []
    for j, intervention in enumerate(intervention_ids):
        matched = group_masks[target_groups[j]]
        s_rows.append(
            {
                "intervention": intervention,
                "target_group": target_groups[j],
                "matched_mean_gain": float(paradigm_means[j, matched].mean()),
                "nonmatched_mean_gain": float(paradigm_means[j, ~matched].mean()),
                "all_paradigm_mean_gain": float(paradigm_means[j].mean()),
                "s": float(primary["s"][j]),
                "bootstrap_ci95": bootstrap["s_ci95"][j],
                "mapping_p_two": permutation["s_p_two"][j],
                "mapping_p_two_bh": permutation["s_p_two_bh"][j],
            }
        )
        intervention_group_matrix.append(
            {
                "intervention": intervention,
                "target_group": target_groups[j],
                "group_mean_gain": {
                    group: float(paradigm_means[j, group_masks[group]].mean())
                    for group in group_order
                },
            }
        )

    primary_gamma = float(primary["gamma"])
    positive_family_count = sum(
        value > thresholds["gamma_ci_lower_strictly_above"]
        for value in family_gamma.values()
    )

    def preservation_ratio(value: float) -> float | None:
        return None if abs(primary_gamma) < 1e-12 else float(value / primary_gamma)

    empty_ratio = preservation_ratio(empty_estimate["gamma"])
    protocol_valid_ratio = (
        preservation_ratio(protocol_valid_estimate["gamma"])
        if protocol_valid_estimate is not None
        else None
    )
    parse_ratio = preservation_ratio(parse_estimate["gamma"])
    length_ratio = preservation_ratio(length_adjusted["gamma"])
    retry_unexposed_ratio = (
        preservation_ratio(retry_unexposed_estimate["gamma"])
        if retry_unexposed_estimate is not None
        else None
    )
    minimum_empty_items = int(empty_estimate["cell_item_counts"].min())
    minimum_protocol_valid_items = int(protocol_valid_cell_counts.min())
    minimum_parse_items = int(parse_estimate["cell_item_counts"][:, :, ospan_index].min())
    minimum_retry_unexposed_items = int(retry_unexposed_cell_counts.min())
    invalid_call_count = int(
        data["run"]["truncated_api_call_count"]
        + data["run"]["transport_incomplete_logical_call_count"]
    )
    condition_invalid_rates = {
        condition_id: float(rate)
        for condition_id, rate in data["run"]["condition_invalid_record_rates"].items()
    }
    max_condition_invalid_rate = max(condition_invalid_rates.values())
    gate_vector = {
        "gamma_family_item_ci_lower_gt_zero": (
            bootstrap["gamma_ci95"][0]
            > thresholds["gamma_ci_lower_strictly_above"]
        ),
        "at_least_four_of_six_family_gamma_positive": sum(
            value > thresholds["gamma_ci_lower_strictly_above"]
            for value in family_gamma.values()
        ) >= thresholds["minimum_positive_families"],
        "family_lofo_selective_delta_log_likelihood_gt_zero": (
            predictive["delta_log_likelihood"]
            > thresholds["predictive_delta_log_likelihood_strictly_above"]
        ),
        "exact_mapping_one_sided_p_le_0_05": (
            permutation["gamma_p_one"]
            <= thresholds["maximum_exact_mapping_one_sided_p"]
        ),
        "each_condition_task_record_invalid_rate_le_0_01": (
            max_condition_invalid_rate
            <= thresholds["maximum_formal_condition_task_record_invalid_rate"]
        ),
        "protocol_invalid_exclusion_preserves_at_least_half_gamma_and_three_items_per_cell": (
            protocol_valid_ratio is not None
            and protocol_valid_ratio
            >= thresholds["minimum_sensitivity_gamma_preservation_ratio"]
            and minimum_protocol_valid_items
            >= thresholds["minimum_sensitivity_items_per_cell"]
        ),
        "empty_exclusion_preserves_at_least_half_gamma_and_three_items_per_cell": (
            empty_ratio is not None
            and empty_ratio >= thresholds["minimum_sensitivity_gamma_preservation_ratio"]
            and minimum_empty_items >= thresholds["minimum_sensitivity_items_per_cell"]
        ),
        "ospan_parse_exclusion_preserves_at_least_half_gamma_and_three_items_per_cell": (
            parse_ratio is not None
            and parse_ratio >= thresholds["minimum_sensitivity_gamma_preservation_ratio"]
            and minimum_parse_items >= thresholds["minimum_sensitivity_items_per_cell"]
        ),
        "response_length_adjustment_preserves_at_least_half_gamma": (
            length_ratio is not None
            and length_ratio >= thresholds["minimum_sensitivity_gamma_preservation_ratio"]
        ),
    }
    require(list(gate_vector) == list(CONFIRMATORY_GATE_IDS),
            "analyzer gate vector drifted from the frozen gate IDs")
    output = {
        "schema_version": ANALYSIS_SCHEMA,
        "estimand": "per-item targeted-minus-placebo paired gain; item mean within model/intervention/paradigm; equal paradigm means for matched minus nonmatched S_j; equal five-intervention mean Gamma",
        "dimensions": {
            "models": len(data["models"]),
            "families": len(set(data["families"])),
            "checkpoints_per_family": 2,
            "paradigms": len(data["paradigms"]),
            "items_per_paradigm": 18,
            "conditions": len(data["conditions"]),
            "records": int(np.prod(data["accuracy"].shape)),
        },
        "primary": {
            "gamma": primary["gamma"],
            "bootstrap": bootstrap,
            "exact_mapping_permutation": permutation,
            "family_gamma": family_gamma,
            "families_positive": positive_family_count,
            "family_consistency_gate_at_least_4_of_6": (
                positive_family_count >= thresholds["minimum_positive_families"]
            ),
            "interventions": s_rows,
            "intervention_by_group_gain_matrix": intervention_group_matrix,
        },
        "predictive_family_lofo": predictive,
        "confirmatory_gate": {
            "rule": (
                f"all {len(CONFIRMATORY_GATE_IDS)} frozen booleans must be true; "
                "no analyst override"
            ),
            "numeric_thresholds": thresholds,
            "components": gate_vector,
            "pass": all(gate_vector.values()),
        },
        "secondary": {
            "canonical_whitespace_ospan_scoring": {
                "gamma": canonical["gamma"],
                "s": canonical["s"].tolist(),
                "bootstrap": canonical_bootstrap,
                "exact_mapping_permutation": canonical_permutation,
            },
            "difficulty_interaction": difficulty_interaction,
            "condition_pc1_share": pc1,
            "operation_span_math_process_metric": {
                "status": "descriptive_only_not_part_of_gamma_or_success_gate",
                "interventions": math_rows,
            },
        },
        "sensitivities": {
            "exclude_pair_if_either_response_empty": {
                "response_unit": "only responses consumed by the primary scorer; OSpan recall, CVLT designated recall turns, all n-back turns",
                "excluded_pairs": int((~empty_valid).sum()),
                "minimum_items_per_model_intervention_paradigm": minimum_empty_items,
                "gamma": empty_estimate["gamma"],
                "preservation_ratio_vs_primary": empty_ratio,
                "s": empty_estimate["s"].tolist(),
                "bootstrap": empty_bootstrap,
            },
            "exclude_pair_if_either_task_record_protocol_invalid": {
                "response_unit": "complete task record; a length completion or exhausted terminal-metadata fault in any logical call invalidates that record",
                "excluded_pairs": int((~protocol_valid).sum()),
                "minimum_items_per_model_intervention_paradigm": minimum_protocol_valid_items,
                "estimable": protocol_valid_estimable,
                "gamma": (
                    protocol_valid_estimate["gamma"]
                    if protocol_valid_estimate is not None
                    else None
                ),
                "preservation_ratio_vs_primary": protocol_valid_ratio,
                "s": (
                    protocol_valid_estimate["s"].tolist()
                    if protocol_valid_estimate is not None
                    else None
                ),
                "bootstrap": protocol_valid_bootstrap,
            },
            "exclude_pair_if_either_task_record_recovered_terminal_metadata_fault_exposed": {
                "status": "prespecified_descriptive_sensitivity_not_a_confirmatory_gate",
                "exposure_unit": "complete task record; exposed when any ultimately valid logical call had at least one preceding HTTP-200 terminal-metadata-fault attempt; request-error-only retries are not exposed",
                "excluded_pairs": int((~retry_unexposed).sum()),
                "minimum_items_per_model_intervention_paradigm": (
                    minimum_retry_unexposed_items
                ),
                "estimable": retry_unexposed_estimable,
                "gamma": (
                    retry_unexposed_estimate["gamma"]
                    if retry_unexposed_estimate is not None
                    else None
                ),
                "preservation_ratio_vs_primary": retry_unexposed_ratio,
                "s": (
                    retry_unexposed_estimate["s"].tolist()
                    if retry_unexposed_estimate is not None
                    else None
                ),
            },
            "exclude_ospan_pair_if_target_or_placebo_parse_none": {
                "excluded_pairs": int((~parse_valid[:, :, ospan_index, :]).sum()),
                "minimum_ospan_items_per_model_intervention": minimum_parse_items,
                "gamma": parse_estimate["gamma"],
                "preservation_ratio_vs_primary": parse_ratio,
                "s": parse_estimate["s"].tolist(),
                "bootstrap": parse_bootstrap,
            },
            "response_character_length_adjustment": length_adjusted,
            "difficulty_stratified": difficulty_results,
            "protocol_invalid_completion_policy": {
                "rule": "length completions are persisted once; listed terminal/usage-metadata faults retry the same canonical request-payload SHA-256 three total times; either qualifying exhausted failure invalidates the complete task record and receives primary accuracy zero",
                "observed_invalid_logical_call_count": invalid_call_count,
                "observed_length_call_count": int(data["run"]["truncated_api_call_count"]),
                "observed_transport_incomplete_call_count": int(
                    data["run"]["transport_incomplete_logical_call_count"]
                ),
                "observed_recovered_terminal_metadata_fault_logical_call_count": int(
                    data["run"][
                        "recovered_terminal_metadata_fault_logical_call_count"
                    ]
                ),
                "observed_recovered_terminal_metadata_fault_task_record_count": int(
                    data["run"][
                        "recovered_terminal_metadata_fault_record_count"
                    ]
                ),
                "condition_recovered_terminal_metadata_fault_task_record_rates": {
                    condition_id: float(rate)
                    for condition_id, rate in data["run"][
                        "condition_recovered_terminal_metadata_fault_record_rates"
                    ].items()
                },
                "observed_invalid_task_record_count": int(data["run"]["invalid_record_count"]),
                "condition_task_record_rates": condition_invalid_rates,
                "maximum_condition_task_record_rate": max_condition_invalid_rate,
                "confirmatory_rate_threshold": thresholds[
                    "maximum_formal_condition_task_record_invalid_rate"
                ],
            },
        },
    }
    output["sensitivities"]["response_character_length_adjustment"][
        "preservation_ratio_vs_primary"
    ] = length_ratio
    validate_aggregate_output(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RESULTS_ROOT / "raw")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "analysis")
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID"), "formal analysis must run in Slurm on c01")
    node = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    require(node == "c01" or node.startswith("c01."), f"formal analyzer must run on c01, got {node}")
    validate_canonical_analysis_paths(args.raw_root, args.output_dir)
    allowed_outputs = {"analysis_results.json", "ANALYSIS_MANIFEST.json"}
    if args.output_dir.exists():
        unexpected = {path.name for path in args.output_dir.iterdir()} - allowed_outputs
        require(not unexpected, f"stale/unknown analysis outputs refused: {sorted(unexpected)}")

    data = load_formal_arrays(args.raw_root)
    results = analyze_arrays(data)
    validate_aggregate_output(results)
    results_path = args.output_dir / "analysis_results.json"

    model_manifest_paths = {
        model: RESULTS_ROOT / "raw" / "formal" / model_safe(model) / "MODEL_MANIFEST.json"
        for model in data["models"]
    }
    current_model_manifest_hashes = {
        model: sha256_file(path) for model, path in model_manifest_paths.items()
    }
    execution_guard_paths = {
        model: RESULTS_ROOT / "raw" / "formal" / model_safe(model)
        / EXECUTION_GUARD_FILENAME
        for model in data["models"]
    }
    current_execution_guard_hashes = {
        model: sha256_file(path) for model, path in execution_guard_paths.items()
    }
    require(current_model_manifest_hashes == data["model_manifest_hashes"],
            "MODEL_MANIFEST changed while analysis was running")
    require(current_execution_guard_hashes == data["execution_guard_hashes"],
            "execution guard changed while analysis was running")
    require(sha256_file(SPEC_PATH) == data["spec_sha256"],
            "spec changed while analysis was running")
    require(sha256_file(data["run_path"]) == data["run_sha256"],
            "RUN_MANIFEST changed while analysis was running")
    require(sha256_file(data["item_path"]) == data["item_sha256"],
            "item manifest changed while analysis was running")
    atomic_write_json(results_path, jsonable(results))
    manifest = {
        "schema_version": "cogarena.causal_selectivity.analysis_manifest.v2",
        "study_id": data["spec"]["study_id"],
        "status": "complete",
        "confirmatory_gate_pass": results["confirmatory_gate"]["pass"],
        "source_revision": data["source_revision"],
        "spec_sha256": sha256_file(SPEC_PATH),
        "formal_run_manifest_sha256": sha256_file(data["run_path"]),
        "formal_item_manifest_sha256": sha256_file(data["item_path"]),
        "model_manifest_sha256": current_model_manifest_hashes,
        "execution_guard_sha256": current_execution_guard_hashes,
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "launcher_sha256": sha256_file(Path(__file__).with_name("analyze.sbatch")),
        "seeds": {
            "main_canonical_and_format_sensitivity_crossed_bootstraps": ANALYSIS_SEED,
            "difficulty_crossed_bootstrap": DIFFICULTY_SEEDS,
            "difficulty_interaction_crossed_bootstrap": DIFFICULTY_INTERACTION_SEED,
        },
        "n_bootstrap": N_BOOTSTRAP,
        "exact_mapping_permutations": math.factorial(5),
        "ridge_lambda_fixed": RIDGE_LAMBDA,
        "numeric_runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "quantile_method": "numpy.quantile default linear",
        },
        "outputs_sha256": {"analysis_results.json": sha256_file(results_path)},
        "raw_content_emitted": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(args.output_dir / "ANALYSIS_MANIFEST.json", manifest)
    require({path.name for path in args.output_dir.iterdir()} == allowed_outputs,
            "analysis output directory failed exact allowlist postflight")
    print(
        f"ANALYSIS COMPLETE confirmatory_gate={'PASS' if results['confirmatory_gate']['pass'] else 'FAIL'} "
        f"gamma={results['primary']['gamma']:.6f} "
        f"families+={results['primary']['families_positive']}/6 "
        f"LOFO dLL={results['predictive_family_lofo']['delta_log_likelihood']:.3f}"
    )


if __name__ == "__main__":
    main()
