#!/usr/bin/env python3
"""Independently enumerate, identity-check, and replay one model's raw records."""

from __future__ import annotations

import argparse
import os
import socket
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    EXECUTION_GUARD_FILENAME,
    RESULTS_ROOT,
    SPEC_PATH,
    atomic_write_json,
    condition_map,
    execution_guard_identity_sha256,
    execution_guard_records_complete_sha256,
    load_json,
    load_spec,
    manifest_path,
    model_safe,
    profile_models,
    request_reasoning_effort,
    request_stop_policy,
    recovered_terminal_metadata_fault_exposure,
    require,
    sha256_file,
    tree_hash,
    TRANSPORT_INCOMPLETE_FINISH_REASON,
    is_protocol_invalid_finish_reason,
    validate_execution_guard,
)
from .run_model import reconstruct_items, result_path, validate_record


def outcome_aggregate_fields(
    profile: str, by_condition_paradigm: dict[tuple[str, str], list[float]]
) -> dict[str, Any]:
    """Per-model closure is operational-only and never exposes arm outcomes."""
    require(profile in {"pilot", "formal"}, f"unknown profile: {profile}")
    return {}


def verify(profile: str, model: str, raw_root: Path) -> dict[str, Any]:
    spec = load_spec()
    reasoning_effort = request_reasoning_effort(spec)
    stop_policy = request_stop_policy(spec)
    panel = {x["model"]: x for x in profile_models(spec, profile)}
    require(model in panel, f"{model} not in {profile} panel")
    manifest_file = manifest_path(profile)
    manifest = load_json(manifest_file)
    spec_sha = sha256_file(SPEC_PATH)
    manifest_sha = sha256_file(manifest_file)
    require(manifest["spec_sha256"] == spec_sha, "manifest/spec hash mismatch")
    items = reconstruct_items(spec, manifest)
    entries = {x["task_id"]: x for x in manifest["items"]}
    conditions = condition_map(spec)
    model_root = raw_root.resolve() / profile / model_safe(model)
    require(model_root.is_dir(), f"missing model output directory: {model_root}")
    guard_path = model_root / EXECUTION_GUARD_FILENAME
    serving_path = model_root / "serving_provenance.json"
    summary_path = model_root / "run_summary.json"
    require(
        guard_path.is_file() and serving_path.is_file() and summary_path.is_file(),
        "missing execution guard/serving provenance/run summary",
    )
    guard = load_json(guard_path)
    validate_execution_guard(guard, expected_state="records_complete")
    guard_file_sha256 = sha256_file(guard_path)
    current_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    current_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "").strip()
    current_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0").strip() or "0"
    current_node = socket.gethostname().split(".", 1)[0]
    require(
        current_job_id
        and guard.get("slurm_job_id") == current_job_id
        and current_array_job_id
        and guard.get("slurm_array_job_id") == current_array_job_id
        and guard.get("slurm_array_task_id") == current_task_id
        and guard.get("execution_node") == current_node,
        "model verification must run in the same Slurm job/task/node as acquisition",
    )
    serving = load_json(serving_path)
    summary = load_json(summary_path)
    require(
        summary.get("schema_version") == "cogarena.causal_selectivity.run_summary.v3",
        "run summary schema mismatch",
    )
    digest = serving.get("tag", {}).get("digest")
    source_revision = summary.get("source_revision")
    require(isinstance(digest, str) and len(digest) >= 32, "invalid served digest")
    require(isinstance(source_revision, str) and len(source_revision) == 40, "invalid source revision")
    require(
        os.environ.get("COGARENA_GIT_HEAD", "").strip() == source_revision
        and guard.get("study_id") == spec["study_id"]
        and guard.get("profile") == profile
        and guard.get("model") == model
        and guard.get("source_revision") == source_revision
        and guard.get("spec_sha256") == spec_sha
        and guard.get("item_manifest_sha256") == manifest_sha
        and guard.get("expected_record_count")
        == manifest["task_record_count_per_model"]
        and summary.get("execution_guard_identity_sha256")
        == guard["guard_identity_sha256"],
        "execution guard identity/binding mismatch",
    )
    require(serving.get("actual_context_tokens") == spec["scope"]["served_context_tokens"],
            "served context differs from specification")
    require(serving.get("reasoning_effort") == reasoning_effort,
            "serving provenance reasoning-effort contract mismatch")
    require(serving.get("stop_policy") == stop_policy,
            "serving provenance response-format contract mismatch")
    require(serving.get("fully_gpu_served") is True
            and serving.get("processor") == "100% GPU",
            "selected model was not proven fully GPU-served")
    require(
        bool(serving.get("sessions"))
        and all(s.get("actual_context_tokens") == spec["scope"]["served_context_tokens"]
                and s.get("fully_gpu_served") is True
                and s.get("processor") == "100% GPU"
                and s.get("reasoning_effort") == reasoning_effort
                and s.get("stop_policy") == stop_policy
            for s in serving.get("sessions", [])),
        "a serving session used a different context/reasoning request or was not fully GPU-served",
    )
    allowed_nodes = spec["execution_contract"]["eligible_inference_nodes"]
    required_gpu = spec["execution_contract"]["required_gpu_name_fragment"]
    session_nodes = sorted({
        str(session.get("hostname", "")).split(".", 1)[0]
        for session in serving["sessions"]
    })
    require(
        session_nodes and all(node in allowed_nodes for node in session_nodes)
        and all(required_gpu.lower() in str(session.get("nvidia_smi", "")).lower()
                for session in serving["sessions"]),
        "a serving session used a node or GPU outside the frozen hardware scope",
    )

    expected_paths: dict[Path, tuple[dict, dict]] = {}
    for condition in conditions.values():
        for entry in manifest["items"]:
            expected_paths[result_path(model_root, condition["id"], entry)] = (condition, entry)
    actual_paths = {
        path for condition_id in conditions for path in (model_root / condition_id).rglob("*.json")
    }
    expected_metadata_paths = {guard_path, serving_path, summary_path}
    all_json_paths = set(model_root.rglob("*.json"))
    require(
        all_json_paths == set(expected_paths) | expected_metadata_paths,
        "missing/extra raw or metadata JSON before model verification",
    )
    temporary = list(model_root.rglob("*.tmp")) + list(model_root.rglob(".*.tmp"))
    require(not temporary, f"temporary files remain: {temporary[:3]}")
    missing = set(expected_paths) - actual_paths
    extra = actual_paths - set(expected_paths)
    require(not missing, f"missing {len(missing)} result records; first={sorted(missing)[:1]}")
    require(not extra, f"unexpected {len(extra)} result records; first={sorted(extra)[:1]}")
    require(len(actual_paths) == manifest["task_record_count_per_model"], "record total mismatch")
    actual_record_tree_sha256 = tree_hash(actual_paths, model_root)
    require(
        guard.get("record_tree_sha256") == actual_record_tree_sha256
        and guard.get("serving_provenance_sha256") == sha256_file(serving_path)
        and guard.get("run_summary_sha256") == sha256_file(summary_path)
        and summary.get("record_tree_sha256") == actual_record_tree_sha256
        and summary.get("serving_provenance_sha256") == sha256_file(serving_path)
        and summary.get("expected_records") == len(actual_paths)
        and summary.get("new_records") == len(actual_paths)
        and summary.get("record_reuse_allowed") is False
        and "reused_records" not in summary,
        "execution guard/run-summary completed-tree binding mismatch",
    )

    finish_reasons: dict[str, int] = defaultdict(int)
    total_api_calls = 0
    transport_attempt_count = 0
    transport_retry_count = 0
    terminal_metadata_fault_attempt_count = 0
    request_error_attempt_count = 0
    transport_incomplete_logical_call_count = 0
    recovered_terminal_metadata_fault_logical_call_count = 0
    empty_response_count = 0
    response_unit_count = 0
    ospan_parse_status_counts: dict[str, int] = defaultdict(int)
    max_reported_prompt_tokens = 0
    max_reported_completion_tokens = 0
    max_reported_total_tokens = 0
    condition_record_counts: dict[str, int] = defaultdict(int)
    condition_truncated_record_counts: dict[str, int] = defaultdict(int)
    condition_transport_invalid_record_counts: dict[str, int] = defaultdict(int)
    condition_invalid_record_counts: dict[str, int] = defaultdict(int)
    condition_recovered_terminal_metadata_fault_record_counts: dict[str, int] = (
        defaultdict(int)
    )
    for path in sorted(actual_paths):
        condition, entry = expected_paths[path]
        record = load_json(path)
        validate_record(
            record,
            model=model,
            profile=profile,
            condition=condition,
            entry=entry,
            item=items[entry["task_id"]],
            spec_sha=spec_sha,
            manifest_sha=manifest_sha,
            served_digest=digest,
            source_revision=source_revision,
            spec=spec,
        )
        total_api_calls += len(record["api_calls"])
        record_truncated = False
        record_transport_invalid = False
        record_invalid = False
        record_recovered_terminal_metadata_fault_exposed = False
        for call in record["api_calls"]:
            finish = str(call.get("finish_reason", ""))
            finish_reasons[finish] += 1
            record_truncated = record_truncated or finish == "length"
            record_transport_invalid = (
                record_transport_invalid
                or finish == TRANSPORT_INCOMPLETE_FINISH_REASON
            )
            record_invalid = record_invalid or is_protocol_invalid_finish_reason(finish)
            attempts = call["attempts"]
            transport_attempt_count += len(attempts)
            transport_retry_count += len(attempts) - 1
            terminal_metadata_fault_attempt_count += sum(
                attempt["status"] == "protocol_fault" for attempt in attempts
            )
            request_error_attempt_count += sum(
                attempt["status"] == "request_error" for attempt in attempts
            )
            transport_incomplete_logical_call_count += int(
                finish == TRANSPORT_INCOMPLETE_FINISH_REASON
            )
            call_recovered_terminal_fault = (
                recovered_terminal_metadata_fault_exposure(call)
            )
            recovered_terminal_metadata_fault_logical_call_count += int(
                call_recovered_terminal_fault
            )
            record_recovered_terminal_metadata_fault_exposed = (
                record_recovered_terminal_metadata_fault_exposed
                or call_recovered_terminal_fault
            )
            if call["usage_metadata_valid"] is True:
                usage = call["usage"]
                max_reported_prompt_tokens = max(
                    max_reported_prompt_tokens, int(usage["prompt_tokens"])
                )
                max_reported_completion_tokens = max(
                    max_reported_completion_tokens, int(usage["completion_tokens"])
                )
                max_reported_total_tokens = max(
                    max_reported_total_tokens, int(usage["total_tokens"])
                )
        condition_id = condition["id"]
        condition_record_counts[condition_id] += 1
        condition_truncated_record_counts[condition_id] += int(record_truncated)
        condition_transport_invalid_record_counts[condition_id] += int(
            record_transport_invalid
        )
        condition_invalid_record_counts[condition_id] += int(record_invalid)
        condition_recovered_terminal_metadata_fault_record_counts[condition_id] += int(
            record_recovered_terminal_metadata_fault_exposed
        )
        payload = record.get("responses") if entry["is_multiturn"] else record.get("response")
        texts = [x["response"] for x in payload] if isinstance(payload, list) else [payload]
        response_unit_count += len(texts)
        empty_response_count += sum(1 for text in texts if text == "")
        if entry["paradigm"] == "operation_span":
            status = record["score"]["metrics"]["strict_parse_status"]
            ospan_parse_status_counts[str(status)] += 1

    require(summary.get("expected_records") == len(actual_paths), "run summary expected count mismatch")
    require(summary.get("served_model_digest") == digest, "run summary digest mismatch")
    require(summary.get("source_revision") == source_revision, "run summary revision mismatch")
    require(summary.get("reasoning_effort") == reasoning_effort,
            "run summary reasoning-effort contract mismatch")
    require(summary.get("stop_policy") == stop_policy,
            "run summary response-format contract mismatch")
    condition_ids = sorted(conditions)
    require(set(condition_record_counts) == set(condition_ids), "condition count coverage mismatch")
    condition_truncation_rates = {
        condition_id: (
            condition_truncated_record_counts[condition_id]
            / condition_record_counts[condition_id]
        )
        for condition_id in condition_ids
    }
    condition_transport_invalid_rates = {
        condition_id: (
            condition_transport_invalid_record_counts[condition_id]
            / condition_record_counts[condition_id]
        )
        for condition_id in condition_ids
    }
    condition_invalid_rates = {
        condition_id: (
            condition_invalid_record_counts[condition_id]
            / condition_record_counts[condition_id]
        )
        for condition_id in condition_ids
    }
    condition_recovered_terminal_metadata_fault_rates = {
        condition_id: (
            condition_recovered_terminal_metadata_fault_record_counts[condition_id]
            / condition_record_counts[condition_id]
        )
        for condition_id in condition_ids
    }
    truncated_api_calls = finish_reasons.get("length", 0)
    truncated_records = sum(condition_truncated_record_counts.values())
    transport_invalid_records = sum(condition_transport_invalid_record_counts.values())
    invalid_records = sum(condition_invalid_record_counts.values())
    recovered_terminal_metadata_fault_records = sum(
        condition_recovered_terminal_metadata_fault_record_counts.values()
    )
    usage_metadata_valid_logical_call_count = (
        total_api_calls - transport_incomplete_logical_call_count
    )
    require(
        transport_attempt_count
        == usage_metadata_valid_logical_call_count
        + terminal_metadata_fault_attempt_count
        + request_error_attempt_count,
        "physical transport attempt accounting does not partition into "
        "accepted, terminal-metadata-fault, and request-error attempts",
    )
    result = {
        "schema_version": "cogarena.causal_selectivity.model_manifest.v3",
        "study_id": spec["study_id"],
        "profile": profile,
        "model": model,
        "family": panel[model]["family"],
        "source_revision": source_revision,
        "reasoning_effort": reasoning_effort,
        "stop_policy": stop_policy,
        "reasoning_request_verified": True,
        "stop_sequence_request_verified": True,
        "served_model_digest": digest,
        "actual_context_tokens": serving["actual_context_tokens"],
        "processor": serving["processor"],
        "fully_gpu_served": True,
        "execution_nodes": session_nodes,
        "required_gpu_name_fragment": required_gpu,
        "spec_sha256": spec_sha,
        "item_manifest_sha256": manifest_sha,
        "record_count": len(actual_paths),
        "api_call_count": total_api_calls,
        "transport_attempt_count": transport_attempt_count,
        "transport_retry_count": transport_retry_count,
        "terminal_metadata_fault_attempt_count": (
            terminal_metadata_fault_attempt_count
        ),
        "request_error_attempt_count": request_error_attempt_count,
        "transport_incomplete_logical_call_count": (
            transport_incomplete_logical_call_count
        ),
        "recovered_terminal_metadata_fault_logical_call_count": (
            recovered_terminal_metadata_fault_logical_call_count
        ),
        "usage_metadata_valid_logical_call_count": (
            usage_metadata_valid_logical_call_count
        ),
        "static_prompt_budget_verified_for_all_logical_calls": True,
        "reported_usage_context_budget_verified": True,
        "max_reported_prompt_tokens": max_reported_prompt_tokens,
        "max_reported_completion_tokens": max_reported_completion_tokens,
        "max_reported_total_tokens": max_reported_total_tokens,
        "minimum_reported_prompt_reservation_margin_tokens": (
            int(spec["scope"]["served_context_tokens"])
            - int(spec["scope"]["max_completion_tokens"])
            - max_reported_prompt_tokens
        ),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "truncated_completion_count": truncated_api_calls,
        "truncated_api_call_count": truncated_api_calls,
        "truncated_record_count": truncated_records,
        "transport_protocol_invalid_record_count": transport_invalid_records,
        "invalid_record_count": invalid_records,
        "recovered_terminal_metadata_fault_record_count": (
            recovered_terminal_metadata_fault_records
        ),
        "condition_record_counts": {
            condition_id: condition_record_counts[condition_id]
            for condition_id in condition_ids
        },
        "condition_truncated_record_counts": {
            condition_id: condition_truncated_record_counts[condition_id]
            for condition_id in condition_ids
        },
        "condition_truncation_rates": condition_truncation_rates,
        "condition_transport_protocol_invalid_record_counts": {
            condition_id: condition_transport_invalid_record_counts[condition_id]
            for condition_id in condition_ids
        },
        "condition_transport_protocol_invalid_rates": condition_transport_invalid_rates,
        "condition_invalid_record_counts": {
            condition_id: condition_invalid_record_counts[condition_id]
            for condition_id in condition_ids
        },
        "condition_invalid_record_rates": condition_invalid_rates,
        "condition_recovered_terminal_metadata_fault_record_counts": {
            condition_id: condition_recovered_terminal_metadata_fault_record_counts[
                condition_id
            ]
            for condition_id in condition_ids
        },
        "condition_recovered_terminal_metadata_fault_record_rates": (
            condition_recovered_terminal_metadata_fault_rates
        ),
        "response_unit_count": response_unit_count,
        "empty_response_count": empty_response_count,
        "operation_span_parse_status_counts": dict(sorted(ospan_parse_status_counts.items())),
        "operation_span_nonparseable_count": ospan_parse_status_counts.get("none", 0),
        "operation_span_protocol_invalid_not_evaluated_count": (
            ospan_parse_status_counts.get("not_evaluated_protocol_invalid", 0)
        ),
        "all_records_replayed": True,
        "unknown_records_rejected": True,
        "record_tree_sha256": actual_record_tree_sha256,
        "serving_provenance_sha256": sha256_file(serving_path),
        "run_summary_sha256": sha256_file(summary_path),
        "execution_guard_identity_sha256": execution_guard_identity_sha256(guard),
        "execution_guard_records_complete_sha256": (
            execution_guard_records_complete_sha256(guard)
        ),
        "execution_guard_same_job_verified": True,
        "record_reuse_allowed": False,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
    }
    result.update(outcome_aggregate_fields(profile, {}))
    manifest_output_path = model_root / "MODEL_MANIFEST.json"
    require(not manifest_output_path.exists(), "MODEL_MANIFEST already exists; verification resume refused")
    require(
        sha256_file(guard_path) == guard_file_sha256,
        "execution guard changed during independent model replay",
    )
    atomic_write_json(manifest_output_path, result)
    verified_guard = dict(guard)
    verified_guard.update(
        {
            "state": "verified_complete",
            "verified_at": result["verified_at"],
            "verification_slurm_job_id": current_job_id,
            "verification_slurm_array_task_id": current_task_id,
            "verification_execution_node": current_node,
            "model_manifest_sha256": sha256_file(manifest_output_path),
        }
    )
    validate_execution_guard(verified_guard, expected_state="verified_complete")
    atomic_write_json(guard_path, verified_guard)
    validate_execution_guard(load_json(guard_path), expected_state="verified_complete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("formal", "pilot"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--raw-root", type=Path, default=RESULTS_ROOT / "raw")
    args = parser.parse_args()
    result = verify(args.profile, args.model, args.raw_root)
    print(
        f"{args.profile}/{args.model}: records={result['record_count']} "
        f"calls={result['api_call_count']} replay=PASS"
    )


if __name__ == "__main__":
    main()
