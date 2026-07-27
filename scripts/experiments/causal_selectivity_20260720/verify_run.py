#!/usr/bin/env python3
"""Close a complete pilot or formal array after every per-model replay gate passes."""

from __future__ import annotations

import argparse
import os
import socket
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .common import (
    EXECUTION_GUARD_FILENAME,
    RESULTS_ROOT,
    SPEC_PATH,
    atomic_write_json,
    execution_guard_identity_sha256,
    execution_guard_records_complete_sha256,
    load_json,
    load_spec,
    manifest_path,
    model_safe,
    profile_models,
    request_reasoning_effort,
    request_stop_policy,
    require,
    sha256_file,
    tree_hash,
    validate_reported_usage_summary,
    validate_execution_guard,
)


def enforce_pilot_invalid_rate_gate(
    spec: dict,
    profile: str,
    condition_invalid_rates: dict[str, float],
) -> None:
    """Fail closure when the pilot exceeds its frozen generic protocol-invalid gate."""
    if profile != "pilot":
        return
    limit = spec["execution_contract"][
        "pilot_maximum_condition_task_record_invalid_rate"
    ]
    require(
        isinstance(limit, (int, float))
        and not isinstance(limit, bool)
        and 0.0 <= float(limit) <= 1.0,
        "pilot invalid-rate threshold is missing or invalid",
    )
    worst_condition, worst_rate = max(
        condition_invalid_rates.items(), key=lambda pair: pair[1]
    )
    require(
        float(worst_rate) <= float(limit),
        "pilot closure refused: condition task-record invalid rate "
        f"{worst_condition}={worst_rate:.6f} exceeds frozen limit {float(limit):.6f}",
    )


def enforce_pilot_truncation_gate(
    spec: dict,
    profile: str,
    condition_truncation_rates: dict[str, float],
) -> None:
    """Compatibility alias for tests/scripts predating the generic invalid name."""
    enforce_pilot_invalid_rate_gate(spec, profile, condition_truncation_rates)


def require_single_profile_array_job_id(values: set[str]) -> str:
    """Reject arrays assembled by filling failed models from a later submission."""
    require(
        len(values) == 1 and all(isinstance(value, str) and value for value in values),
        "model guards came from multiple Slurm array attempts; quarantine the entire profile",
    )
    return next(iter(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("formal", "pilot"), required=True)
    parser.add_argument("--raw-root", type=Path, default=RESULTS_ROOT / "raw")
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID"), "run closure must execute in Slurm, not on a login node")
    require(socket.gethostname().split(".", 1)[0].startswith("c01"),
            "run closure must execute on c01")
    injected_revision = os.environ.get("COGARENA_GIT_HEAD", "")
    require(len(injected_revision) == 40
            and all(c in "0123456789abcdef" for c in injected_revision),
            "COGARENA_GIT_HEAD must be a full frozen revision")

    spec = load_spec()
    reasoning_effort = request_reasoning_effort(spec)
    stop_policy = request_stop_policy(spec)
    models = profile_models(spec, args.profile)
    expected_names = {model_safe(x["model"]) for x in models}
    profile_root = args.raw_root.resolve() / args.profile
    require(profile_root.is_dir(), f"missing profile root: {profile_root}")
    actual_names = {x.name for x in profile_root.iterdir() if x.is_dir()}
    require(actual_names == expected_names, f"model directory mismatch: {actual_names ^ expected_names}")

    manifests = []
    revisions = set()
    manifest_paths = []
    guard_paths = []
    profile_array_job_ids = set()
    condition_ids = [condition["id"] for condition in spec["conditions"]]
    expected_per_condition_per_model = load_json(manifest_path(args.profile))["item_count"]
    for model in models:
        model_root = profile_root / model_safe(model["model"])
        path = model_root / "MODEL_MANIFEST.json"
        guard_path = model_root / EXECUTION_GUARD_FILENAME
        require(path.is_file(), f"missing MODEL_MANIFEST for {model['model']}")
        require(guard_path.is_file(), f"missing execution guard for {model['model']}")
        data = load_json(path)
        guard = load_json(guard_path)
        validate_execution_guard(guard, expected_state="verified_complete")
        profile_array_job_ids.add(guard["slurm_array_job_id"])
        require(data.get("schema_version") == "cogarena.causal_selectivity.model_manifest.v3"
                and data.get("study_id") == spec["study_id"]
                and data.get("profile") == args.profile,
                f"MODEL_MANIFEST schema/study/profile mismatch: {model['model']}")
        require(data.get("status") == "complete" and data.get("all_records_replayed") is True,
                f"incomplete model gate: {model['model']}")
        require(data.get("model") == model["model"] and data.get("family") == model["family"],
                f"model manifest identity mismatch: {model['model']}")
        require(data.get("spec_sha256") == sha256_file(SPEC_PATH), "model spec hash mismatch")
        require(data.get("item_manifest_sha256") == sha256_file(manifest_path(args.profile)),
                "model item-manifest hash mismatch")
        require(
            guard.get("study_id") == spec["study_id"]
            and guard.get("profile") == args.profile
            and guard.get("model") == model["model"]
            and guard.get("source_revision") == data.get("source_revision")
            and guard.get("spec_sha256") == data.get("spec_sha256")
            and guard.get("item_manifest_sha256") == data.get("item_manifest_sha256")
            and guard.get("expected_record_count") == data.get("record_count")
            and guard.get("record_tree_sha256") == data.get("record_tree_sha256")
            and guard.get("serving_provenance_sha256")
            == data.get("serving_provenance_sha256")
            and guard.get("run_summary_sha256") == data.get("run_summary_sha256")
            and guard.get("model_manifest_sha256") == sha256_file(path)
            and data.get("execution_guard_identity_sha256")
            == execution_guard_identity_sha256(guard)
            and data.get("execution_guard_records_complete_sha256")
            == execution_guard_records_complete_sha256(guard)
            and data.get("execution_guard_same_job_verified") is True
            and data.get("record_reuse_allowed") is False,
            f"execution guard closure mismatch: {model['model']}",
        )
        require(data.get("reasoning_effort") == reasoning_effort,
                f"model reasoning-effort contract mismatch: {model['model']}")
        require(data.get("reasoning_request_verified") is True,
                f"model reasoning-effort request was not replay-verified: {model['model']}")
        require(data.get("stop_policy") == stop_policy
                and data.get("stop_sequence_request_verified") is True,
                f"model response-format policy was not replay-verified: {model['model']}")
        require(data.get("fully_gpu_served") is True and data.get("processor") == "100% GPU",
                f"model was not proven fully GPU-served: {model['model']}")
        validate_reported_usage_summary(spec, data)
        require(
            data.get("execution_nodes") == spec["execution_contract"]["eligible_inference_nodes"]
            and data.get("required_gpu_name_fragment")
            == spec["execution_contract"]["required_gpu_name_fragment"],
            f"model used hardware outside the frozen scope: {model['model']}",
        )
        condition_counts = data.get("condition_record_counts")
        truncated_counts = data.get("condition_truncated_record_counts")
        truncation_rates = data.get("condition_truncation_rates")
        invalid_counts = data.get("condition_invalid_record_counts")
        invalid_rates = data.get("condition_invalid_record_rates")
        transport_invalid_counts = data.get(
            "condition_transport_protocol_invalid_record_counts"
        )
        transport_invalid_rates = data.get(
            "condition_transport_protocol_invalid_rates"
        )
        recovered_fault_counts = data.get(
            "condition_recovered_terminal_metadata_fault_record_counts"
        )
        recovered_fault_rates = data.get(
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
            and set(truncation_rates) == set(condition_ids),
            f"condition accounting coverage mismatch: {model['model']}",
        )
        require(
            set(invalid_counts) == set(condition_ids)
            and set(invalid_rates) == set(condition_ids)
            and set(transport_invalid_counts) == set(condition_ids)
            and set(transport_invalid_rates) == set(condition_ids)
            and set(recovered_fault_counts) == set(condition_ids)
            and set(recovered_fault_rates) == set(condition_ids),
            f"condition invalid coverage mismatch: {model['model']}",
        )
        require(
            all(condition_counts[cid] == expected_per_condition_per_model for cid in condition_ids),
            f"condition record count mismatch: {model['model']}",
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
            ),
            f"invalid condition truncation accounting: {model['model']}",
        )
        require(
            all(
                isinstance(invalid_counts[cid], int)
                and not isinstance(invalid_counts[cid], bool)
                and isinstance(transport_invalid_counts[cid], int)
                and not isinstance(transport_invalid_counts[cid], bool)
                and 0 <= transport_invalid_counts[cid] <= invalid_counts[cid]
                <= condition_counts[cid]
                and truncated_counts[cid] <= invalid_counts[cid]
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
            ),
            f"invalid generic protocol accounting: {model['model']}",
        )
        require(
            data.get("truncated_record_count") == sum(truncated_counts.values())
            and data.get("truncated_api_call_count") == data.get("truncated_completion_count"),
            f"truncation totals mismatch: {model['model']}",
        )
        require(
            data.get("invalid_record_count") == sum(invalid_counts.values())
            and data.get("transport_protocol_invalid_record_count")
            == sum(transport_invalid_counts.values())
            and data.get("transport_incomplete_logical_call_count")
            == data.get("finish_reason_counts", {}).get("transport_incomplete", 0)
            and data.get("recovered_terminal_metadata_fault_record_count")
            == sum(recovered_fault_counts.values())
            and isinstance(
                data.get("recovered_terminal_metadata_fault_logical_call_count"), int
            )
            and not isinstance(
                data.get("recovered_terminal_metadata_fault_logical_call_count"), bool
            )
            and 0 <= data["recovered_terminal_metadata_fault_logical_call_count"]
            <= data["usage_metadata_valid_logical_call_count"],
            f"transport totals mismatch: {model['model']}",
        )
        ospan_status_counts = data.get("operation_span_parse_status_counts")
        require(
            isinstance(ospan_status_counts, dict)
            and data.get("operation_span_nonparseable_count")
            == ospan_status_counts.get("none", 0)
            and data.get("operation_span_protocol_invalid_not_evaluated_count")
            == ospan_status_counts.get("not_evaluated_protocol_invalid", 0),
            f"operation-span native-parse/protocol-invalid accounting mismatch: {model['model']}",
        )
        require(
            "condition_paradigm_mean_accuracy" not in data,
            f"MODEL_MANIFEST leaks pre-analysis arm outcomes: {model['model']}",
        )
        revisions.add(data["source_revision"])
        manifests.append(data)
        manifest_paths.append(path)
        guard_paths.append(guard_path)
    require(len(revisions) == 1, f"mixed source revisions: {revisions}")
    profile_array_job_id = require_single_profile_array_job_id(profile_array_job_ids)
    require(revisions == {injected_revision},
            "model manifests do not match the closure source revision")

    other = "pilot" if args.profile == "formal" else "formal"
    require(
        not ({x["model"] for x in profile_models(spec, args.profile)}
             & {x["model"] for x in profile_models(spec, other)}),
        "pilot/formal model overlap",
    )
    condition_record_counts: dict[str, int] = defaultdict(int)
    condition_truncated_record_counts: dict[str, int] = defaultdict(int)
    condition_transport_invalid_record_counts: dict[str, int] = defaultdict(int)
    condition_invalid_record_counts: dict[str, int] = defaultdict(int)
    condition_recovered_terminal_metadata_fault_record_counts: dict[str, int] = (
        defaultdict(int)
    )
    for data in manifests:
        for condition_id in condition_ids:
            condition_record_counts[condition_id] += data["condition_record_counts"][condition_id]
            condition_truncated_record_counts[condition_id] += data[
                "condition_truncated_record_counts"
            ][condition_id]
            condition_transport_invalid_record_counts[condition_id] += data[
                "condition_transport_protocol_invalid_record_counts"
            ][condition_id]
            condition_invalid_record_counts[condition_id] += data[
                "condition_invalid_record_counts"
            ][condition_id]
            condition_recovered_terminal_metadata_fault_record_counts[
                condition_id
            ] += data[
                "condition_recovered_terminal_metadata_fault_record_counts"
            ][condition_id]
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
    enforce_pilot_invalid_rate_gate(spec, args.profile, condition_invalid_rates)
    closure = {
        "schema_version": "cogarena.causal_selectivity.run_manifest.v3",
        "study_id": spec["study_id"],
        "profile": args.profile,
        "source_revision": next(iter(revisions)),
        "reasoning_effort": reasoning_effort,
        "reasoning_request_verified": True,
        "stop_policy": stop_policy,
        "stop_sequence_request_verified": True,
        "spec_sha256": sha256_file(SPEC_PATH),
        "item_manifest_sha256": sha256_file(manifest_path(args.profile)),
        "model_count": len(models),
        "record_count": sum(x["record_count"] for x in manifests),
        "api_call_count": sum(x["api_call_count"] for x in manifests),
        "transport_attempt_count": sum(x["transport_attempt_count"] for x in manifests),
        "transport_retry_count": sum(x["transport_retry_count"] for x in manifests),
        "terminal_metadata_fault_attempt_count": sum(
            x["terminal_metadata_fault_attempt_count"] for x in manifests
        ),
        "request_error_attempt_count": sum(
            x["request_error_attempt_count"] for x in manifests
        ),
        "transport_incomplete_logical_call_count": sum(
            x["transport_incomplete_logical_call_count"] for x in manifests
        ),
        "recovered_terminal_metadata_fault_logical_call_count": sum(
            x["recovered_terminal_metadata_fault_logical_call_count"]
            for x in manifests
        ),
        "usage_metadata_valid_logical_call_count": sum(
            x["usage_metadata_valid_logical_call_count"] for x in manifests
        ),
        "static_prompt_budget_verified_for_all_logical_calls": True,
        "response_unit_count": sum(x["response_unit_count"] for x in manifests),
        "empty_response_count": sum(x["empty_response_count"] for x in manifests),
        "reported_usage_context_budget_verified": True,
        "max_reported_prompt_tokens": max(
            x["max_reported_prompt_tokens"] for x in manifests
        ),
        "max_reported_completion_tokens": max(
            x["max_reported_completion_tokens"] for x in manifests
        ),
        "max_reported_total_tokens": max(
            x["max_reported_total_tokens"] for x in manifests
        ),
        "minimum_reported_prompt_reservation_margin_tokens": min(
            x["minimum_reported_prompt_reservation_margin_tokens"] for x in manifests
        ),
        "truncated_completion_count": sum(x["truncated_completion_count"] for x in manifests),
        "truncated_api_call_count": sum(x["truncated_api_call_count"] for x in manifests),
        "truncated_record_count": sum(x["truncated_record_count"] for x in manifests),
        "transport_protocol_invalid_record_count": sum(
            x["transport_protocol_invalid_record_count"] for x in manifests
        ),
        "invalid_record_count": sum(x["invalid_record_count"] for x in manifests),
        "recovered_terminal_metadata_fault_record_count": sum(
            x["recovered_terminal_metadata_fault_record_count"] for x in manifests
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
        "operation_span_nonparseable_count": sum(
            x["operation_span_nonparseable_count"] for x in manifests
        ),
        "operation_span_protocol_invalid_not_evaluated_count": sum(
            x["operation_span_protocol_invalid_not_evaluated_count"]
            for x in manifests
        ),
        "fully_gpu_served_model_count": sum(x["fully_gpu_served"] is True for x in manifests),
        "all_models_fully_gpu_served": True,
        "processor_requirement": "100% GPU",
        "execution_nodes": list(spec["execution_contract"]["eligible_inference_nodes"]),
        "required_gpu_name_fragment": spec["execution_contract"][
            "required_gpu_name_fragment"
        ],
        "model_manifest_tree_sha256": tree_hash(manifest_paths, profile_root),
        "execution_guard_count": len(guard_paths),
        "execution_guard_tree_sha256": tree_hash(guard_paths, profile_root),
        "profile_array_job_id": profile_array_job_id,
        "all_execution_guards_verified_complete": True,
        "record_reuse_allowed": False,
        "all_model_replays_passed": True,
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "status": "engineering_pilot_complete" if args.profile == "pilot" else "formal_raw_complete",
    }
    validate_reported_usage_summary(spec, closure)
    output = RESULTS_ROOT / f"RUN_MANIFEST_{args.profile}.json"
    atomic_write_json(output, closure)
    print(f"{args.profile}: models={len(models)} records={closure['record_count']} all gates PASS")


if __name__ == "__main__":
    main()
