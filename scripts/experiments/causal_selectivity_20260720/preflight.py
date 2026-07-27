#!/usr/bin/env python3
"""Zero-inference batch-node preflight; never invokes Git or a model server."""

from __future__ import annotations

import argparse
import os

from .common import (
    RESULTS_ROOT,
    ROOT,
    SPEC_PATH,
    format_user_prompt,
    load_json,
    load_spec,
    manifest_path,
    profile_models,
    request_reasoning_effort,
    request_stop_policy,
    response_terminator,
    require,
    sha256_file,
    sha256_text,
    system_prompt,
    validate_reported_usage_summary,
)


_CAPACITY_RELEASE_ORIGINAL_SHA256 = (
    "91d2b50bef88e7004f9ef76ffcbd37895b54f89c46e6f8d998f69716ac700e4f"
)
_CAPACITY_RELEASED_SHA256 = (
    "7d63e46146e4d86bbb8fd3b6daed69e1530043503c00efab5912cc41a3ffdb25"
)


def _verified_capacity_release_redaction(
    path, expected_sha256: object, actual_sha256: str
) -> bool:
    """Accept the one documented public-release rewrite, and nothing else.

    The frozen specification remains bound to the original execution artifact.
    A public archive may replace only the machine-local project-root prefix
    in the capacity-probe path.  The release ledger must bind both byte hashes,
    name the exact JSON field, and agree with the released payload on disk.
    """

    release_path = ROOT / "RELEASE_REDACTIONS.json"
    capacity_path = RESULTS_ROOT / "CAPACITY_GATE_MANIFEST.json"
    if not release_path.is_file() or path.resolve() != capacity_path.resolve():
        return False
    if (
        expected_sha256 != _CAPACITY_RELEASE_ORIGINAL_SHA256
        or actual_sha256 != _CAPACITY_RELEASED_SHA256
    ):
        return False

    try:
        release = load_json(release_path)
        capacity = load_json(path)
    except (OSError, ValueError, TypeError):
        return False

    if not isinstance(release, dict) or set(release) != {"schema_version", "redactions"}:
        return False
    if release.get("schema_version") != "cogarena.release_redactions.v1":
        return False
    redactions = release.get("redactions")
    if not isinstance(redactions, list) or len(redactions) != 1:
        return False
    redaction = redactions[0]
    required_fields = {
        "binding_note",
        "field",
        "file",
        "original_sha256",
        "reason",
        "released_sha256",
        "replacement",
    }
    if not isinstance(redaction, dict) or set(redaction) != required_fields:
        return False
    if redaction != {
        "file": "results/causal_selectivity_20260720/CAPACITY_GATE_MANIFEST.json",
        "field": "probes[0].path",
        "reason": "public-release removal of a machine-local absolute project root",
        "replacement": "${COGARENA_ROOT}",
        "original_sha256": expected_sha256,
        "released_sha256": actual_sha256,
        "binding_note": (
            "PREPILOT_SPEC.json retains the SHA-256 of the original execution "
            "manifest; the released copy differs only by this documented path redaction."
        ),
    }:
        return False

    probes = capacity.get("probes") if isinstance(capacity, dict) else None
    if not isinstance(probes, list) or len(probes) != 1:
        return False
    probe = probes[0]
    expected_released_path = (
        "${COGARENA_ROOT}/results/causal_selectivity_20260720/"
        "capacity/pro6000/CAPACITY_PROBE.json"
    )
    return (
        isinstance(probe, dict)
        and probe.get("path") == expected_released_path
        and sha256_file(path) == actual_sha256
    )


def validate_formal_freeze_gates(spec: dict) -> None:
    reasoning_effort = request_reasoning_effort(spec)
    stop_policy = request_stop_policy(spec)
    pilot_path = RESULTS_ROOT / "RUN_MANIFEST_pilot.json"
    capacity_path = RESULTS_ROOT / "CAPACITY_GATE_MANIFEST.json"
    bindings = (
        (pilot_path, "pilot_gate_manifest_sha256"),
        (capacity_path, "capacity_gate_manifest_sha256"),
    )
    for path, key in bindings:
        require(path.is_file(), f"formal freeze gate artifact missing: {path.name}")
        expected_sha256 = spec.get(key)
        actual_sha256 = sha256_file(path)
        require(expected_sha256 == actual_sha256 or _verified_capacity_release_redaction(
                    path, expected_sha256, actual_sha256),
                f"formal freeze gate hash mismatch: {path.name}")

    pilot = load_json(pilot_path)
    expected_pilot_models = len(spec["pilot_model_panel"])
    expected_pilot_items = spec["scope"]["pilot_items_per_paradigm"] * spec["scope"]["n_paradigms"]
    require(pilot.get("schema_version") == "cogarena.causal_selectivity.run_manifest.v3"
            and pilot.get("profile") == "pilot"
            and pilot.get("status") == "engineering_pilot_complete",
            "bound pilot gate is not a closed engineering pilot")
    require(pilot.get("model_count") == expected_pilot_models
            and pilot.get("record_count") == expected_pilot_models * expected_pilot_items * spec["scope"]["n_conditions"],
            "bound pilot gate has the wrong panel or record count")
    require(pilot.get("all_model_replays_passed") is True,
            "bound pilot gate failed scorer replay closure")
    validate_reported_usage_summary(spec, pilot)
    require(pilot.get("reasoning_effort") == reasoning_effort,
            "bound pilot gate used a different reasoning-effort request")
    require(pilot.get("reasoning_request_verified") is True,
            "bound pilot gate lacks replay verification of the reasoning request")
    require(pilot.get("stop_policy") == stop_policy
            and pilot.get("stop_sequence_request_verified") is True,
            "bound pilot gate lacks replay verification of the response-format policy")
    condition_ids = [condition["id"] for condition in spec["conditions"]]
    condition_counts = pilot.get("condition_record_counts")
    truncated_counts = pilot.get("condition_truncated_record_counts")
    truncation_rates = pilot.get("condition_truncation_rates")
    invalid_counts = pilot.get("condition_invalid_record_counts")
    invalid_rates = pilot.get("condition_invalid_record_rates")
    transport_invalid_counts = pilot.get(
        "condition_transport_protocol_invalid_record_counts"
    )
    transport_invalid_rates = pilot.get(
        "condition_transport_protocol_invalid_rates"
    )
    recovered_fault_counts = pilot.get(
        "condition_recovered_terminal_metadata_fault_record_counts"
    )
    recovered_fault_rates = pilot.get(
        "condition_recovered_terminal_metadata_fault_record_rates"
    )
    expected_per_condition = expected_pilot_models * expected_pilot_items
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
        and all(condition_counts[cid] == expected_per_condition for cid in condition_ids),
        "bound pilot gate has incomplete condition invalid accounting",
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
        and pilot.get("truncated_record_count") == sum(truncated_counts.values())
        and pilot.get("truncated_api_call_count") == pilot.get("truncated_completion_count"),
        "bound pilot gate truncation totals are inconsistent",
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
            and abs(
                float(invalid_rates[cid])
                - invalid_counts[cid] / condition_counts[cid]
            ) < 1e-15
            and abs(
                float(transport_invalid_rates[cid])
                - transport_invalid_counts[cid] / condition_counts[cid]
            ) < 1e-15
            and isinstance(recovered_fault_counts[cid], int)
            and not isinstance(recovered_fault_counts[cid], bool)
            and 0 <= recovered_fault_counts[cid] <= condition_counts[cid]
            and abs(
                float(recovered_fault_rates[cid])
                - recovered_fault_counts[cid] / condition_counts[cid]
            ) < 1e-15
            for cid in condition_ids
        )
        and pilot.get("invalid_record_count") == sum(invalid_counts.values())
        and pilot.get("transport_protocol_invalid_record_count")
        == sum(transport_invalid_counts.values())
        and pilot.get("recovered_terminal_metadata_fault_record_count")
        == sum(recovered_fault_counts.values())
        and isinstance(
            pilot.get("recovered_terminal_metadata_fault_logical_call_count"), int
        )
        and not isinstance(
            pilot.get("recovered_terminal_metadata_fault_logical_call_count"), bool
        )
        and 0 <= pilot["recovered_terminal_metadata_fault_logical_call_count"]
        <= pilot["usage_metadata_valid_logical_call_count"],
        "bound pilot gate generic invalid totals are inconsistent",
    )
    pilot_rate_limit = spec.get("execution_contract", {}).get(
        "pilot_maximum_condition_task_record_invalid_rate"
    )
    formal_rate_limit = spec.get("estimands", {}).get(
        "confirmatory_success_gate", {}
    ).get("numeric_thresholds", {}).get(
        "maximum_formal_condition_task_record_invalid_rate"
    )
    require(
        isinstance(pilot_rate_limit, (int, float))
        and not isinstance(pilot_rate_limit, bool)
        and 0 <= pilot_rate_limit <= 1,
        "pilot invalid-rate threshold is missing or invalid",
    )
    require(
        isinstance(formal_rate_limit, (int, float))
        and not isinstance(formal_rate_limit, bool)
        and formal_rate_limit == pilot_rate_limit,
        "pilot/formal invalid-rate thresholds differ",
    )
    require(
        max(float(invalid_rates[cid]) for cid in condition_ids) <= pilot_rate_limit,
        "bound pilot exceeds the frozen per-condition task-record invalid rate",
    )
    require(pilot.get("all_models_fully_gpu_served") is True
            and pilot.get("fully_gpu_served_model_count") == expected_pilot_models
            and pilot.get("processor_requirement") == "100% GPU",
            "bound pilot gate lacks exact full-GPU closure")
    guard_tree = pilot.get("execution_guard_tree_sha256")
    require(
        pilot.get("execution_guard_count") == expected_pilot_models
        and pilot.get("all_execution_guards_verified_complete") is True
        and pilot.get("record_reuse_allowed") is False
        and isinstance(pilot.get("profile_array_job_id"), str)
        and bool(pilot["profile_array_job_id"])
        and isinstance(guard_tree, str)
        and len(guard_tree) == 64
        and all(character in "0123456789abcdef" for character in guard_tree),
        "bound pilot gate lacks verified no-resume execution-guard closure",
    )
    require(
        pilot.get("execution_nodes") == spec["execution_contract"]["eligible_inference_nodes"]
        and pilot.get("required_gpu_name_fragment")
        == spec["execution_contract"]["required_gpu_name_fragment"],
        "bound pilot gate used hardware outside the frozen scope",
    )
    require("condition_paradigm_mean_accuracy" not in pilot,
            "pilot gate leaks outcome aggregates")

    capacity = load_json(capacity_path)
    expected_labels = list(spec["capacity_probe"]["required_hardware_labels"])
    require(capacity.get("schema_version") == "cogarena.causal_selectivity.capacity_gate.v1"
            and capacity.get("status") == "pass",
            "bound capacity artifact is not a closed capacity gate")
    require(capacity.get("model") == spec["capacity_probe"]["model"]
            and capacity.get("hardware_labels") == expected_labels
            and len(capacity.get("probes", [])) == len(expected_labels),
            "bound capacity gate identity/panel mismatch")
    require(capacity.get("actual_context_tokens") == spec["scope"]["served_context_tokens"],
            "bound capacity gate context mismatch")
    require(capacity.get("reasoning_effort") == reasoning_effort,
            "bound capacity gate used a different reasoning-effort request")
    require(capacity.get("reasoning_request_verified") is True,
            "bound capacity gate lacks verification of the reasoning request")
    require(capacity.get("stop_policy") == stop_policy
            and capacity.get("stop_sequence_request_verified") is True,
            "bound capacity gate lacks verification of the response-format policy")
    require(capacity.get("all_probes_fully_gpu_served") is True
            and capacity.get("processor_requirement") == "100% GPU",
            "bound capacity gate lacks exact full-GPU closure")
    require(
        capacity.get("execution_nodes") == spec["execution_contract"]["eligible_inference_nodes"]
        and capacity.get("required_gpu_name_fragment")
        == spec["execution_contract"]["required_gpu_name_fragment"],
        "bound capacity gate used hardware outside the frozen scope",
    )
    require(
        pilot.get("source_revision") == capacity.get("source_revision"),
        "pilot and capacity gates were produced from different source revisions",
    )
    require(
        pilot.get("spec_sha256") == capacity.get("spec_sha256"),
        "pilot and capacity gates were produced from different prepilot specifications",
    )
    require(
        pilot.get("item_manifest_sha256")
        == capacity.get("pilot_item_manifest_sha256"),
        "pilot and capacity gates were produced from different pilot item manifests",
    )


def validate_source_revision(spec: dict, profile: str, revision: str) -> None:
    require(
        len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
        "COGARENA_GIT_HEAD must be a submission-injected 40-hex revision",
    )
    if profile == "formal":
        require(spec.get("status") == "formal_frozen_after_pilot",
                "formal run refused until the engineering pilot is closed and the spec is frozen")
        gate_hash = spec.get("pilot_gate_manifest_sha256")
        require(isinstance(gate_hash, str) and len(gate_hash) == 64,
                "formal spec lacks a pilot gate manifest binding")
        capacity_hash = spec.get("capacity_gate_manifest_sha256")
        require(isinstance(capacity_hash, str) and len(capacity_hash) == 64,
                "formal spec lacks the frozen capacity gate binding")
        validate_formal_freeze_gates(spec)


def validate_manifest_sources(spec: dict, profile: str) -> dict:
    path = manifest_path(profile)
    manifest = load_json(path)
    require(manifest["profile"] == profile, "item-manifest profile mismatch")
    require(manifest["spec_sha256"] == sha256_file(SPEC_PATH), "item manifest binds stale spec")
    for relative, expected in manifest["source_sha256"].items():
        source = ROOT / relative
        require(source.is_file(), f"frozen source missing: {relative}")
        require(sha256_file(source) == expected, f"frozen source drift: {relative}")
    audit = {x["condition_id"]: x for x in manifest["prompt_length_audit"]["conditions"]}
    for condition in spec["conditions"]:
        prompt = system_prompt(spec, condition["id"])
        row = audit[condition["id"]]
        require(row["complete_system_sha256"] == sha256_text(prompt), "prompt hash drift")
        require(row["complete_system_characters"] == len(prompt), "prompt character count drift")
        require(row["complete_system_whitespace_tokens"] == len(prompt.split()),
                "prompt token count drift")
    format_audit = manifest.get("response_format_and_context_audit")
    require(isinstance(format_audit, dict), "missing response-format/context audit")
    terminator = response_terminator(spec)
    require(
        format_audit.get("transport_terminator") == terminator
        and format_audit.get("response_format_policy") == request_stop_policy(spec)
        and format_audit.get(
            "terminator_absent_from_all_item_payloads_and_condition_scaffolds"
        ) is True
        and format_audit.get("item_payload_terminator_collision_count") == 0,
        "item-manifest transport-marker audit failed",
    )
    require(
        format_audit.get("stop_sequence_collisions_with_declared_call_responses") == 0,
        "item-manifest response-format stop collides with a declared answer",
    )
    override_audit = format_audit.get("user_prompt_format_overrides")
    overrides = spec.get("response_format_overrides")
    require(
        isinstance(override_audit, dict)
        and isinstance(overrides, dict)
        and set(override_audit) == set(overrides),
        "item-manifest response-format override coverage mismatch",
    )
    for paradigm, instruction in overrides.items():
        row = override_audit[paradigm]
        require(
            row.get("instruction_sha256") == sha256_text(instruction)
            and row.get("instruction_characters") == len(instruction)
            and row.get("instruction_whitespace_tokens") == len(instruction.split())
            and format_user_prompt(spec, paradigm, "X").endswith(instruction),
            f"response-format override drift: {paradigm}",
        )
    completion = int(spec["scope"]["max_completion_tokens"])
    context = int(spec["scope"]["served_context_tokens"])
    stress = format_audit.get("context_stress")
    declared = format_audit.get("declared_response_global_max")
    require(
        format_audit.get("completion_budget_tokens") == completion
        and isinstance(declared, dict)
        and isinstance(declared.get("whitespace_tokens"), int)
        and declared["whitespace_tokens"] * 4 <= completion
        and isinstance(stress, dict)
        and stress.get("served_context_tokens") == context
        and isinstance(stress.get("max_reserved_prompt_plus_completion_tokens"), int)
        and stress["max_reserved_prompt_plus_completion_tokens"] <= context
        and stress.get("minimum_reserved_margin_tokens")
        == context - stress["max_reserved_prompt_plus_completion_tokens"],
        "item-manifest response/context budget audit failed",
    )
    return manifest


def validate(profile: str, model: str, revision: str) -> dict:
    spec = load_spec()
    validate_source_revision(spec, profile, revision)
    require(model in {x["model"] for x in profile_models(spec, profile)},
            f"{model} is not in the {profile} panel")
    other = "pilot" if profile == "formal" else "formal"
    require(model not in {x["model"] for x in profile_models(spec, other)},
            "pilot/formal model overlap")
    validate_manifest_sources(spec, profile)
    return {
        "profile": profile,
        "model": model,
        "revision": revision,
        "manifest": str(manifest_path(profile)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("formal", "pilot"), required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    revision = os.environ.get("COGARENA_GIT_HEAD", "")
    result = validate(args.profile, args.model, revision)
    print(f"PRE-INFERENCE GATES PASSED {result}")


if __name__ == "__main__":
    main()
