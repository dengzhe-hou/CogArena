from __future__ import annotations

import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from cogarena.core import (
    DifficultyLevel,
    EvalMode,
    ScoringConfig,
    TaskInstance,
    TaskMetadata,
)
from cogarena.generators.working_memory_gen import generate_wm_items

from ..build_item_manifests import build_manifest, generate_pool, validate_cross_profile
from ..common import (
    RESULTS_ROOT,
    SPEC_PATH,
    atomic_write_json,
    execution_guard_policy,
    format_user_prompt,
    load_json,
    load_spec,
    manifest_path,
    presentation_fingerprint,
    recovered_terminal_metadata_fault_exposure,
    request_stop_policy,
    request_stop_sequences,
    require,
    sha256_file,
    sha256_text,
    system_prompt,
    transport_retry_policy,
    validate_execution_guard,
)
from .. import preflight as preflight_module
from .. import run_model as run_model_module
from .. import scorer_adapter as scorer_adapter_module
from ..run_model import (
    LocalChatClient,
    TRANSPORT_INVALID_HISTORY_SENTINEL,
    TRANSPORT_INCOMPLETE_FINISH_REASON,
    TRUNCATED_HISTORY_SENTINEL,
    assert_slurm_gpu_context,
    acquire_execution_guard,
    complete_execution_guard,
    enforce_frozen_hardware,
    evaluate_item,
    normalize_model_content,
    normalize_completion_usage,
    parse_ollama_ps_context,
    parse_ollama_ps_serving_row,
    require_reported_usage_fits_context,
    require_request_fits_context,
    require_fully_gpu_served,
    reconstruct_items,
    validate_response_transcript,
)
from ..verify_model import outcome_aggregate_fields
from ..verify_run import (
    enforce_pilot_truncation_gate,
    require_single_profile_array_job_id,
)
from ..preflight import validate_formal_freeze_gates, validate_source_revision
from ..scorer_adapter import (
    STATIC_SCORERS,
    score_response,
    score_response_with_completion_contract,
)


def test_prepilot_spec_is_balanced_but_formal_is_hard_disabled():
    spec = load_spec()
    assert spec["status"] in {
        "prepilot_specification_not_formally_frozen",
        "formal_frozen_after_pilot",
    }
    if spec["status"] == "prepilot_specification_not_formally_frozen":
        assert "frozen_at" not in spec
    else:
        assert len(spec["pilot_gate_manifest_sha256"]) == 64
        assert len(spec["capacity_gate_manifest_sha256"]) == 64
        assert spec["frozen_at"]
    assert len(spec["conditions"]) == 7
    assert transport_retry_policy(spec)["maximum_total_attempts"] == 3
    assert execution_guard_policy(spec)["record_reuse"] is False
    drifted = copy.deepcopy(spec)
    drifted["execution_contract"]["transport_retry_policy"]["maximum_total_attempts"] = 4
    with pytest.raises(RuntimeError, match="retry policy drift"):
        transport_retry_policy(drifted)
    drifted_guard = copy.deepcopy(spec)
    drifted_guard["execution_contract"]["execution_guard_policy"]["record_reuse"] = True
    with pytest.raises(RuntimeError, match="execution-guard policy drift"):
        execution_guard_policy(drifted_guard)
    targeted = [x for x in spec["conditions"] if x["kind"] == "targeted"]
    assert {x["target_group"] for x in targeted} == set(spec["grouping"])
    placebo = next(x for x in spec["conditions"] if x["id"] == "neutral_placebo")
    assert max(abs(len(x["scaffold"].split()) - len(placebo["scaffold"].split())) for x in targeted) <= 2
    formal = spec["formal_model_panel"]
    assert Counter(x["family"] for x in formal) == {x: 2 for x in Counter(y["family"] for y in formal)}
    pilot_models = {x["model"] for x in spec["pilot_model_panel"]}
    assert len(pilot_models) == 3
    assert pilot_models == {"qwen2.5:1.5b", "llama3.2:3b", "gemma3:1b"}
    assert "qwen3:0.6b" not in pilot_models
    assert "smollm2:1.7b" not in pilot_models
    assert "tinyllama:1.1b" not in pilot_models
    assert not ({x["model"] for x in formal} & pilot_models)
    assert spec["capacity_probe"]["model"] not in {x["model"] for x in formal}
    assert list(spec["capacity_probe"]["required_hardware_labels"]) == ["pro6000"]
    assert spec["capacity_probe"]["model"] == "qwen2.5:32b"
    assert spec["scope"]["max_completion_tokens"] == 512
    assert spec["scope"]["served_context_tokens"] == 4096
    assert spec["scope"]["reasoning_effort"] == "none"
    assert spec["scope"]["response_terminator"] == "<END_COGARENA_RESPONSE>"
    policy = request_stop_policy(spec)
    assert request_stop_sequences(spec, "n_back") == [
        "\n",
        "<END_COGARENA_RESPONSE>",
    ]
    assert request_stop_sequences(spec, "drm_false_memory") == [
        "\n\n",
        "<END_COGARENA_RESPONSE>",
    ]
    assert request_stop_sequences(spec, "cvlt_word_list") == [
        "\n",
        "<END_COGARENA_RESPONSE>",
    ]
    cvlt_prompt = format_user_prompt(spec, "cvlt_word_list", "Your response:")
    assert cvlt_prompt.endswith(spec["response_format_overrides"]["cvlt_word_list"])
    assert "separating entries with commas" in cvlt_prompt
    assert format_user_prompt(spec, "n_back", "Your response:") == "Your response:"
    assert set(
        paradigm
        for row in policy.values()
        for paradigm in row["paradigms"]
    ) == {p for paradigms in spec["grouping"].values() for p in paradigms}
    assert all(
        "<END_COGARENA_RESPONSE>" in system_prompt(spec, condition["id"])
        for condition in spec["conditions"]
    )
    for condition in spec["conditions"]:
        prompt = system_prompt(spec, condition["id"])
        assert prompt.count(spec["scope"]["response_terminator"]) == 1
        assert prompt.endswith(spec["transport_instruction"])
    assert spec["scope"]["formal_task_records_total"] == 12 * 13 * 18 * 7
    thresholds = spec["estimands"]["confirmatory_success_gate"]["numeric_thresholds"]
    assert thresholds["maximum_formal_condition_task_record_invalid_rate"] == 0.01
    assert spec["execution_contract"][
        "pilot_maximum_condition_task_record_invalid_rate"
    ] == thresholds["maximum_formal_condition_task_record_invalid_rate"]


def test_execution_guard_atomic_acquire_and_records_complete(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "12000")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    model_root = tmp_path / "pilot" / "fixture__model"
    guard = acquire_execution_guard(
        model_root,
        study_id="fixture-study",
        profile="pilot",
        model="fixture:model",
        source_revision="a" * 40,
        spec_sha256="b" * 64,
        item_manifest_sha256="c" * 64,
        execution_node="c03",
    )
    validate_execution_guard(guard, expected_state="in_progress")
    with pytest.raises(RuntimeError, match="already exists"):
        acquire_execution_guard(
            model_root,
            study_id="fixture-study",
            profile="pilot",
            model="fixture:model",
            source_revision="a" * 40,
            spec_sha256="b" * 64,
            item_manifest_sha256="c" * 64,
            execution_node="c03",
        )
    record = model_root / "condition" / "record.json"
    serving = model_root / "serving_provenance.json"
    summary = model_root / "run_summary.json"
    atomic_write_json(record, {"record": 1})
    atomic_write_json(serving, {"serving": 1})
    atomic_write_json(summary, {"summary": 1})
    completed = complete_execution_guard(
        model_root,
        guard,
        record_paths=[record],
        serving_path=serving,
        run_summary_path=summary,
        records_completed_at="2026-07-20T00:00:00+00:00",
    )
    validate_execution_guard(completed, expected_state="records_complete")
    tampered = copy.deepcopy(completed)
    tampered["record_tree_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="records-complete hash mismatch"):
        validate_execution_guard(tampered, expected_state="records_complete")
    assert require_single_profile_array_job_id({"12000"}) == "12000"
    with pytest.raises(RuntimeError, match="multiple Slurm array attempts"):
        require_single_profile_array_job_id({"12000", "13000"})


def test_item_manifests_exact_balance_and_cross_profile_exclusion():
    spec = load_spec()
    formal = build_manifest(spec, "formal")
    forbidden = {x["presentation_sha256"] for x in formal["items"]}
    pilot = build_manifest(spec, "pilot", forbidden)
    validate_cross_profile(spec, formal, pilot)
    assert formal["item_count"] == 234
    assert pilot["item_count"] == 39
    assert set(formal["paradigm_counts"].values()) == {18}
    for paradigm in formal["paradigm_counts"]:
        cells = Counter(x["difficulty"] for x in formal["items"] if x["paradigm"] == paradigm)
        assert cells == {"easy": 6, "medium": 6, "hard": 6}
    assert formal["prompt_length_audit"]["targeted_vs_placebo_max_absolute_whitespace_token_delta"] <= 2
    audit = formal["response_format_and_context_audit"]
    assert audit["terminator_absent_from_all_item_payloads_and_condition_scaffolds"] is True
    assert audit["item_payload_terminator_collision_count"] == 0
    assert audit["stop_sequence_collisions_with_declared_call_responses"] == 0
    assert audit["response_format_policy"] == request_stop_policy(spec)
    assert set(audit["user_prompt_format_overrides"]) == {"cvlt_word_list"}
    assert audit["declared_response_global_max"] == {
        "lines": 35,
        "characters": 431,
        "whitespace_tokens": 71,
    }
    assert audit["by_paradigm"]["drm_false_memory"]["max_declared_lines"] == 35
    assert audit["by_paradigm"]["source_monitoring"]["max_declared_lines"] == 22
    assert audit["by_paradigm"]["confidence_calibration"]["max_declared_lines"] == 2
    assert audit["by_paradigm"]["post_decision_wagering"]["max_declared_lines"] == 2
    assert audit["completion_budget_tokens"] >= 4 * 71
    assert audit["context_stress"]["filler_response_characters"] == 1024
    assert audit["context_stress"]["max_reserved_prompt_plus_completion_tokens"] <= 4096
    assert audit["context_stress"]["minimum_reserved_margin_tokens"] > 0


def test_tracked_candidate_manifests_bind_spec_and_provenance():
    provenance = load_json(RESULTS_ROOT / "ITEM_PROVENANCE.json")
    assert provenance["spec_sha256"] == sha256_file(SPEC_PATH)
    assert provenance["formal_manifest_sha256"] == sha256_file(manifest_path("formal"))
    assert provenance["pilot_manifest_sha256"] == sha256_file(manifest_path("pilot"))
    assert provenance["cross_profile_disjoint"] is True


def test_ospan_gold_is_recovered_from_turn_stimuli_not_assumed_metadata():
    item = next(
        x for x in generate_wm_items(seed=991, n_per_paradigm=12, include_contamination_probes=False)
        if x.metadata.paradigm == "operation_span"
    )
    item = copy.deepcopy(item)
    letters = list(item.metadata.parameters.pop("letters"))
    responses = []
    for turn in item.metadata.parameters["turns"]:
        if turn["type"] == "operation_letter":
            responses.append({"response": turn["math_expected"]})
        else:
            responses.append({"response": "Final recalled letters: " + ", ".join(letters)})
    result = score_response(item, responses)
    assert result["primary_accuracy"] == 1.0
    assert result["metrics"]["math_accuracy"] == 1.0


def test_ospan_metadata_turn_disagreement_fails_closed():
    item = next(
        x for x in generate_wm_items(seed=992, n_per_paradigm=12, include_contamination_probes=False)
        if x.metadata.paradigm == "operation_span"
    )
    item = copy.deepcopy(item)
    item.metadata.parameters["letters"][0] = "Z" if item.metadata.parameters["letters"][0] != "Z" else "Y"
    responses = [{"response": "NO"} for _ in item.metadata.parameters["turns"]]
    with pytest.raises(RuntimeError, match="metadata letters disagree"):
        score_response(item, responses)


def test_cvlt_fixed_designated_turn_semantics_deduplicate_and_use_own_gold():
    turns = [
        {"type": "learning_trial", "stimulus": "main", "expected_words": ["a", "b", "c", "d"]},
        {"type": "filler_task", "stimulus": "filler"},
        {"type": "interference_recall", "stimulus": "new", "expected_words": ["w", "x", "y", "z"]},
    ]
    metadata = TaskMetadata(
        dimension="episodic_memory",
        paradigm="cvlt_word_list",
        mode=EvalMode.LLM_STATIC,
        parameters={"turns": turns, "multi_turn": True},
        scoring=ScoringConfig(method="custom", params={}),
        difficulty=DifficultyLevel.MEDIUM,
    )
    item = TaskInstance("cvlt_fixture", metadata, "fixture", ["a", "b", "c", "d"])
    responses = [
        {"response": "a, a, a, a, a, a"},
        {"response": "anything"},
        {"response": "w, x, y, z, z"},
    ]
    result = score_response(item, responses)
    assert result["primary_accuracy"] == 0.5
    assert result["metrics"]["n_scored_turns"] == 2
    assert result["metrics"]["turns"][0]["unique_hits"] == 1
    assert result["metrics"]["turns"][0]["thresholded_accuracy"] == 0.0
    assert result["metrics"]["turns"][1]["unique_hits"] == 4
    assert result["metrics"]["turns"][1]["thresholded_accuracy"] == 1.0


def test_all_static_paradigm_score_signatures_smoke():
    spec = load_spec()
    manifest = load_json(manifest_path("formal"))
    items = reconstruct_items(spec, manifest)
    for paradigm in STATIC_SCORERS:
        entry = next(x for x in manifest["items"] if x["paradigm"] == paradigm)
        item = items[entry["task_id"]]
        result = score_response(item, str(item.expected_response or ""))
        assert 0.0 <= result["primary_accuracy"] <= 1.0, paradigm


def test_pilot_manifest_reconstructs_despite_generator_local_id_reuse():
    spec = load_spec()
    manifest = load_json(manifest_path("pilot"))
    items = reconstruct_items(spec, manifest)
    assert set(items) == {entry["task_id"] for entry in manifest["items"]}
    assert len(items) == manifest["item_count"] == 39


def test_ollama_context_parser_and_slurm_login_gate(monkeypatch):
    table = (
        "NAME              ID              SIZE      PROCESSOR    CONTEXT    UNTIL\n"
        "qwen2.5:3b        abcdef123456    3.0 GB    100% GPU     8192       Forever\n"
    )
    assert parse_ollama_ps_context(table, "qwen2.5:3b") == 8192
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="forbidden outside a Slurm job"):
        assert_slurm_gpu_context()


def test_selected_ollama_row_must_be_exactly_100_percent_gpu():
    header = (
        f"{'NAME':<20}{'ID':<16}{'SIZE':<12}{'PROCESSOR':<24}"
        f"{'CONTEXT':<12}UNTIL"
    )
    mixed = (
        header + "\n"
        + f"{'qwen2.5:14b':<20}{'abc':<16}{'9.0 GB':<12}"
          f"{'48%/52% CPU/GPU':<24}{'8192':<12}Forever\n"
        + f"{'other:7b':<20}{'def':<16}{'5.0 GB':<12}"
          f"{'100% GPU':<24}{'8192':<12}Forever\n"
    )
    parsed = parse_ollama_ps_serving_row(mixed, "qwen2.5:14b")
    assert parsed == {
        "processor": "48%/52% CPU/GPU",
        "actual_context_tokens": 8192,
    }
    with pytest.raises(RuntimeError, match="not fully GPU-served"):
        require_fully_gpu_served(mixed, "qwen2.5:14b")

    full = mixed.replace("48%/52% CPU/GPU", "100% GPU          ", 1)
    assert require_fully_gpu_served(full, "qwen2.5:14b")["processor"] == "100% GPU"


def test_pilot_model_manifest_helper_emits_no_outcome_aggregates():
    cells = {
        ("neutral_placebo", "digit_span"): [0.25, 0.75],
        ("working_memory_ledger", "digit_span"): [1.0, 1.0],
    }
    pilot = outcome_aggregate_fields("pilot", cells)
    assert pilot == {}
    assert not any("accuracy" in key or "contrast" in key for key in pilot)
    formal = outcome_aggregate_fields("formal", cells)
    assert formal == {}


def test_formal_source_revision_gate_is_closed_before_pilot():
    spec = load_spec()
    if spec["status"] == "prepilot_specification_not_formally_frozen":
        with pytest.raises(RuntimeError, match="formal run refused"):
            validate_source_revision(spec, "formal", "a" * 40)
    else:
        validate_source_revision(spec, "formal", "a" * 40)


def test_anonymous_release_redaction_is_exact_and_fail_closed(tmp_path, monkeypatch):
    results_root = tmp_path / "results" / "causal_selectivity_20260720"
    capacity_path = results_root / "CAPACITY_GATE_MANIFEST.json"
    capacity_path.parent.mkdir(parents=True)
    source_capacity = RESULTS_ROOT / "CAPACITY_GATE_MANIFEST.json"
    source_bytes = source_capacity.read_bytes()
    source_sha256 = sha256_file(source_capacity)
    original_sha256 = preflight_module._CAPACITY_RELEASE_ORIGINAL_SHA256
    released_sha256 = preflight_module._CAPACITY_RELEASED_SHA256
    if source_sha256 == original_sha256:
        source_document = json.loads(source_bytes)
        suffix = (
            "/results/causal_selectivity_20260720/"
            "capacity/pro6000/CAPACITY_PROBE.json"
        )
        source_probe_path = source_document["probes"][0]["path"]
        assert source_probe_path.endswith(suffix)
        private_root = source_probe_path[:-len(suffix)].encode()
        released_bytes = source_bytes.replace(
            private_root, b"${COGARENA_ROOT}"
        )
    else:
        assert source_sha256 == released_sha256
        released_bytes = source_bytes
    capacity_path.write_bytes(released_bytes)
    assert sha256_file(capacity_path) == released_sha256
    release = {
        "schema_version": "cogarena.release_redactions.v1",
        "redactions": [{
            "file": "results/causal_selectivity_20260720/CAPACITY_GATE_MANIFEST.json",
            "field": "probes[0].path",
            "reason": "anonymous-release removal of a machine-local absolute project root",
            "replacement": "${COGARENA_ROOT}",
            "original_sha256": original_sha256,
            "released_sha256": released_sha256,
            "binding_note": (
                "PREPILOT_SPEC.json retains the SHA-256 of the original execution "
                "manifest; the released copy differs only by this documented path redaction."
            ),
        }],
    }
    atomic_write_json(tmp_path / "RELEASE_REDACTIONS.json", release)
    monkeypatch.setattr(preflight_module, "ROOT", tmp_path)
    monkeypatch.setattr(preflight_module, "RESULTS_ROOT", results_root)

    assert preflight_module._verified_capacity_release_redaction(
        capacity_path, original_sha256, released_sha256
    )

    release["redactions"][0]["field"] = "probes[1].path"
    atomic_write_json(tmp_path / "RELEASE_REDACTIONS.json", release)
    assert not preflight_module._verified_capacity_release_redaction(
        capacity_path, original_sha256, released_sha256
    )

    release["redactions"][0]["field"] = "probes[0].path"
    changed = load_json(capacity_path)
    changed["closed_at"] = "tampered"
    atomic_write_json(capacity_path, changed)
    changed_sha256 = sha256_file(capacity_path)
    release["redactions"][0]["released_sha256"] = changed_sha256
    atomic_write_json(tmp_path / "RELEASE_REDACTIONS.json", release)
    assert not preflight_module._verified_capacity_release_redaction(
        capacity_path, original_sha256, changed_sha256
    )


def test_formal_freeze_replays_bound_pilot_and_capacity_hashes(tmp_path, monkeypatch):
    spec = copy.deepcopy(load_spec())
    spec["status"] = "formal_frozen_after_pilot"
    pilot = {
        "schema_version": "cogarena.causal_selectivity.run_manifest.v3",
        "profile": "pilot",
        "status": "engineering_pilot_complete",
        "model_count": 3,
        "record_count": 3 * 39 * 7,
        "api_call_count": 1000,
        "transport_attempt_count": 1000,
        "transport_retry_count": 0,
        "terminal_metadata_fault_attempt_count": 0,
        "request_error_attempt_count": 0,
        "transport_incomplete_logical_call_count": 0,
        "recovered_terminal_metadata_fault_logical_call_count": 0,
        "usage_metadata_valid_logical_call_count": 1000,
        "static_prompt_budget_verified_for_all_logical_calls": True,
            "all_model_replays_passed": True,
            "execution_guard_count": 3,
            "execution_guard_tree_sha256": "d" * 64,
            "all_execution_guards_verified_complete": True,
            "record_reuse_allowed": False,
            "profile_array_job_id": "12345",
        "reported_usage_context_budget_verified": True,
        "max_reported_prompt_tokens": 100,
        "max_reported_completion_tokens": 20,
        "max_reported_total_tokens": 120,
        "minimum_reported_prompt_reservation_margin_tokens": 3484,
        "reasoning_effort": spec["scope"]["reasoning_effort"],
        "reasoning_request_verified": True,
        "stop_policy": request_stop_policy(spec),
        "stop_sequence_request_verified": True,
        "truncated_completion_count": 0,
        "truncated_api_call_count": 0,
        "truncated_record_count": 0,
        "transport_protocol_invalid_record_count": 0,
        "invalid_record_count": 0,
        "condition_record_counts": {
            condition["id"]: 3 * 39 for condition in spec["conditions"]
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
        "all_models_fully_gpu_served": True,
        "fully_gpu_served_model_count": 3,
        "processor_requirement": "100% GPU",
        "execution_nodes": list(spec["execution_contract"]["eligible_inference_nodes"]),
        "required_gpu_name_fragment": spec["execution_contract"][
            "required_gpu_name_fragment"
        ],
        "source_revision": "a" * 40,
        "spec_sha256": "b" * 64,
        "item_manifest_sha256": "c" * 64,
    }
    capacity = {
        "schema_version": "cogarena.causal_selectivity.capacity_gate.v1",
        "status": "pass",
        "model": spec["capacity_probe"]["model"],
        "hardware_labels": list(spec["capacity_probe"]["required_hardware_labels"]),
        "probes": [{"label": label} for label in spec["capacity_probe"]["required_hardware_labels"]],
        "actual_context_tokens": spec["scope"]["served_context_tokens"],
        "reasoning_effort": spec["scope"]["reasoning_effort"],
        "reasoning_request_verified": True,
        "stop_policy": request_stop_policy(spec),
        "stop_sequence_request_verified": True,
        "all_probes_fully_gpu_served": True,
        "processor_requirement": "100% GPU",
        "execution_nodes": list(spec["execution_contract"]["eligible_inference_nodes"]),
        "required_gpu_name_fragment": spec["execution_contract"][
            "required_gpu_name_fragment"
        ],
        "source_revision": "a" * 40,
        "spec_sha256": "b" * 64,
        "pilot_item_manifest_sha256": "c" * 64,
    }
    pilot_path = tmp_path / "RUN_MANIFEST_pilot.json"
    capacity_path = tmp_path / "CAPACITY_GATE_MANIFEST.json"
    atomic_write_json(pilot_path, pilot)
    atomic_write_json(capacity_path, capacity)
    spec["pilot_gate_manifest_sha256"] = sha256_file(pilot_path)
    spec["capacity_gate_manifest_sha256"] = sha256_file(capacity_path)
    monkeypatch.setattr(preflight_module, "RESULTS_ROOT", tmp_path)
    validate_formal_freeze_gates(spec)

    bad_attempt_partition = copy.deepcopy(pilot)
    bad_attempt_partition["request_error_attempt_count"] = 1
    atomic_write_json(pilot_path, bad_attempt_partition)
    spec["pilot_gate_manifest_sha256"] = sha256_file(pilot_path)
    with pytest.raises(RuntimeError, match="logical-call/usage accounting"):
        validate_formal_freeze_gates(spec)

    bad_exposure = copy.deepcopy(pilot)
    first_condition = spec["conditions"][0]["id"]
    bad_exposure["recovered_terminal_metadata_fault_record_count"] = 1
    atomic_write_json(pilot_path, bad_exposure)
    spec["pilot_gate_manifest_sha256"] = sha256_file(pilot_path)
    with pytest.raises(RuntimeError, match="generic invalid totals"):
        validate_formal_freeze_gates(spec)

    atomic_write_json(pilot_path, pilot)
    spec["pilot_gate_manifest_sha256"] = sha256_file(pilot_path)
    validate_formal_freeze_gates(spec)

    pilot["truncated_completion_count"] = 1
    pilot["truncated_api_call_count"] = 1
    pilot["truncated_record_count"] = 1
    pilot["condition_truncated_record_counts"][first_condition] = 1
    pilot["condition_truncation_rates"][first_condition] = 1 / (3 * 39)
    atomic_write_json(pilot_path, pilot)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_formal_freeze_gates(spec)

    pilot["truncated_completion_count"] = 0
    pilot["truncated_api_call_count"] = 0
    pilot["truncated_record_count"] = 0
    pilot["condition_truncated_record_counts"][first_condition] = 0
    pilot["condition_truncation_rates"][first_condition] = 0.0
    atomic_write_json(pilot_path, pilot)

    pilot_limit = spec["execution_contract"][
        "pilot_maximum_condition_task_record_invalid_rate"
    ]
    excess = int(3 * 39 * pilot_limit) + 1
    pilot["truncated_completion_count"] = excess
    pilot["truncated_api_call_count"] = excess
    pilot["truncated_record_count"] = excess
    pilot["condition_truncated_record_counts"][first_condition] = excess
    pilot["condition_truncation_rates"][first_condition] = excess / (3 * 39)
    pilot["invalid_record_count"] = excess
    pilot["condition_invalid_record_counts"][first_condition] = excess
    pilot["condition_invalid_record_rates"][first_condition] = excess / (3 * 39)
    atomic_write_json(pilot_path, pilot)
    spec["pilot_gate_manifest_sha256"] = sha256_file(pilot_path)
    with pytest.raises(RuntimeError, match="exceeds the frozen"):
        validate_formal_freeze_gates(spec)

    pilot["truncated_completion_count"] = 0
    pilot["truncated_api_call_count"] = 0
    pilot["truncated_record_count"] = 0
    pilot["condition_truncated_record_counts"][first_condition] = 0
    pilot["condition_truncation_rates"][first_condition] = 0.0
    pilot["invalid_record_count"] = 0
    pilot["condition_invalid_record_counts"][first_condition] = 0
    pilot["condition_invalid_record_rates"][first_condition] = 0.0
    atomic_write_json(pilot_path, pilot)
    spec["pilot_gate_manifest_sha256"] = sha256_file(pilot_path)
    capacity["source_revision"] = "d" * 40
    atomic_write_json(capacity_path, capacity)
    spec["pilot_gate_manifest_sha256"] = sha256_file(pilot_path)
    spec["capacity_gate_manifest_sha256"] = sha256_file(capacity_path)
    with pytest.raises(RuntimeError, match="different source revisions"):
        validate_formal_freeze_gates(spec)


def test_closure_and_analysis_jobs_are_c01_pinned_and_revision_bound():
    root = SPEC_PATH.parent
    for name in ("finalize.sbatch", "finalize_capacity.sbatch", "analyze.sbatch"):
        text = (root / name).read_text(encoding="utf-8")
        assert "#SBATCH --nodelist=c01" in text
        assert "COGARENA_GIT_HEAD" in text
        assert "srun python" in text


def test_gpu_inference_launchers_are_restricted_to_capacity_verified_nodes():
    root = SPEC_PATH.parent
    for name in ("pilot.sbatch", "full.sbatch", "capacity_probe.sbatch"):
        text = (root / name).read_text(encoding="utf-8")
        assert "#SBATCH --nodelist=c04" in text
        assert "#SBATCH --nodelist=c03,c04" not in text
        assert "#SBATCH --nodes=1" in text
        assert "#SBATCH --ntasks=1" in text
        assert "#SBATCH --gres=gpu:pro_6000:1" in text


def test_all_inference_requests_explicitly_disable_reasoning():
    root = SPEC_PATH.parent
    for name in ("run_one_model.sh", "run_capacity_probe.sh"):
        text = (root / name).read_text(encoding="utf-8")
        assert "export COGARENA_REASONING_EFFORT=none" in text
        assert "export COGARENA_STOP_MODE=format_routed" in text
        assert r'\"reasoning_effort\":\"${COGARENA_REASONING_EFFORT}\"' in text
        assert r'\"stop\":[\"<END_COGARENA_RESPONSE>\"]' in text
    capacity_probe = (root / "capacity_probe.py").read_text(encoding="utf-8")
    assert 'COGARENA_STOP_MODE") == "format_routed"' in capacity_probe
    assert 'COGARENA_STOP_MODE") == "single_line"' not in capacity_probe


def test_runtime_hardware_scope_rejects_wrong_node_or_gpu():
    spec = load_spec()
    gpu = {"nvidia_smi": "GPU-0, NVIDIA RTX PRO 6000 Blackwell, 97887 MiB"}
    assert enforce_frozen_hardware(spec, gpu, hostname="c04") == "c04"
    with pytest.raises(RuntimeError, match="outside frozen hardware scope"):
        enforce_frozen_hardware(spec, gpu, hostname="c03")
    with pytest.raises(RuntimeError, match="GPU class is outside"):
        enforce_frozen_hardware(spec, {"nvidia_smi": "NVIDIA A100"}, hostname="c04")


def test_request_budget_reserves_full_completion_inside_context():
    spec = load_spec()
    estimate = require_request_fits_context(spec, "short user prompt", "short system prompt")
    assert estimate > 0
    with pytest.raises(RuntimeError, match="cannot reserve"):
        require_request_fits_context(spec, "X" * 50000, "system")


def test_server_reported_usage_must_reserve_full_completion_budget():
    spec = load_spec()
    require_reported_usage_fits_context(
        spec,
        {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    )
    with pytest.raises(RuntimeError, match="cannot reserve"):
        require_reported_usage_fits_context(
            spec,
            {"prompt_tokens": 3600, "completion_tokens": 20, "total_tokens": 3620},
        )
    with pytest.raises(RuntimeError, match="exceeds the frozen"):
        require_reported_usage_fits_context(
            spec,
            {"prompt_tokens": 100, "completion_tokens": 513, "total_tokens": 613},
        )
    with pytest.raises(RuntimeError, match="real prompt/total"):
        require_reported_usage_fits_context(
            spec,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
    with pytest.raises(RuntimeError, match="zero reported completion"):
        require_reported_usage_fits_context(
            spec,
            {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
            "NONEMPTY",
        )


def test_pilot_closure_enforces_condition_truncation_limit():
    spec = load_spec()
    rates = {condition["id"]: 0.0 for condition in spec["conditions"]}
    enforce_pilot_truncation_gate(spec, "pilot", rates)
    enforce_pilot_truncation_gate(spec, "formal", {key: 1.0 for key in rates})
    rates["episodic_source_binding"] = 0.0100001
    with pytest.raises(RuntimeError, match="pilot closure refused"):
        enforce_pilot_truncation_gate(spec, "pilot", rates)


def test_empty_string_is_valid_data_but_nontext_is_protocol_failure():
    assert normalize_model_content("") == ""
    assert normalize_model_content("  \n") == ""
    with pytest.raises(RuntimeError, match="not a string"):
        normalize_model_content(None)


def test_openai_sdk_internal_retries_are_disabled(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:12345/v1")
    client = LocalChatClient("fixture:1b", load_spec())
    assert client.client.max_retries == 0


def test_nonzero_reported_reasoning_tokens_fail_closed():
    usage = SimpleNamespace(
        model_dump=lambda **_: {
            "prompt_tokens": 2,
            "completion_tokens": 4,
            "total_tokens": 6,
            "completion_tokens_details": {"reasoning_tokens": 4},
        }
    )
    with pytest.raises(RuntimeError, match="nonzero reasoning tokens"):
        normalize_completion_usage(usage)


def test_length_finish_is_returned_once_without_retry():
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            class FakeUsage:
                def model_dump(self, *, mode):
                    assert mode == "json"
                    return {
                        "prompt_tokens": 2,
                        "completion_tokens": 4,
                        "total_tokens": 6,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    }
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="partial repeated text"),
                    finish_reason="length",
                )],
                usage=FakeUsage(),
            )

    client = LocalChatClient.__new__(LocalChatClient)
    client.model = "fixture:1b"
    client.spec = load_spec()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    result = client.call("USER", "SYSTEM", "digit_span")
    assert len(calls) == 1
    assert calls[0]["extra_body"] == {"reasoning_effort": "none"}
    assert calls[0]["stop"] == ["\n", "<END_COGARENA_RESPONSE>"]
    assert result["finish_reason"] == "length"
    assert result["response"] == "partial repeated text"
    assert result["reasoning_effort"] == "none"
    assert result["reasoning_output_characters"] == 0
    assert result["stop_sequences"] == ["\n", "<END_COGARENA_RESPONSE>"]
    assert result["usage"]["completion_tokens"] == 4
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0


def test_three_terminal_metadata_faults_become_one_protocol_invalid_call(monkeypatch):
    requests = []
    bodies = ["BODY ONE", "BODY TWO", "BODY THREE"]

    class ZeroUsage:
        def model_dump(self, *, mode):
            assert mode == "json"
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            body = bodies[len(requests) - 1]
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=body),
                    finish_reason="",
                )],
                usage=ZeroUsage(),
            )

    monkeypatch.setattr(run_model_module.time, "sleep", lambda _: None)
    client = LocalChatClient.__new__(LocalChatClient)
    client.model = "fixture:1b"
    client.spec = load_spec()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    result = client.call("USER", "SYSTEM", "digit_span")
    assert len(requests) == 3
    assert requests[0] == requests[1] == requests[2]
    assert result["response"] == "BODY THREE"
    assert result["finish_reason"] == TRANSPORT_INCOMPLETE_FINISH_REASON
    assert result["server_finish_reason"] == ""
    assert result["transport_status"] == "protocol_invalid"
    assert result["terminal_metadata_complete"] is False
    assert result["usage_metadata_valid"] is False
    assert result["attempt_count"] == 3
    assert [row["status"] for row in result["attempts"]] == [
        "protocol_fault", "protocol_fault", "protocol_fault"
    ]
    assert all(
        row["faults"] == ["missing_finish_reason", "zero_usage"]
        for row in result["attempts"]
    )
    assert all("content" not in row for row in result["attempts"])
    metadata = TaskMetadata(
        dimension="working_memory",
        paradigm="digit_span",
        mode=EvalMode.LLM_STATIC,
        parameters={},
        scoring=ScoringConfig(method="exact_match"),
        difficulty=DifficultyLevel.EASY,
    )
    item = TaskInstance("transport_fixture", metadata, "USER", "A")
    stored_call = copy.deepcopy(result)
    response_body = stored_call.pop("response")
    stored_call["prompt_sha256"] = sha256_text("USER")
    validate_response_transcript(
        item,
        response_body,
        [stored_call],
        load_spec(),
        model="fixture:1b",
        sys_prompt="SYSTEM",
    )
    tampered = copy.deepcopy(stored_call)
    tampered["attempts"][0]["request_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="attempt evidence hash mismatch"):
        validate_response_transcript(
            item,
            response_body,
            [tampered],
            load_spec(),
            model="fixture:1b",
            sys_prompt="SYSTEM",
        )


def test_terminal_metadata_fault_then_valid_response_consumes_only_valid(monkeypatch):
    requests = []

    class FakeUsage:
        def __init__(self, valid):
            self.valid = valid

        def model_dump(self, *, mode):
            assert mode == "json"
            return (
                {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9}
                if self.valid
                else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            valid = len(requests) == 2
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="VALID" if valid else "REJECTED"),
                    finish_reason="stop" if valid else "",
                )],
                usage=FakeUsage(valid),
            )

    monkeypatch.setattr(run_model_module.time, "sleep", lambda _: None)
    client = LocalChatClient.__new__(LocalChatClient)
    client.model = "fixture:1b"
    client.spec = load_spec()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    result = client.call("USER", "SYSTEM", "digit_span")
    assert len(requests) == 2 and requests[0] == requests[1]
    assert result["response"] == "VALID"
    assert result["finish_reason"] == "stop"
    assert result["transport_status"] == "valid"
    assert result["attempt_count"] == 2
    assert [row["status"] for row in result["attempts"]] == [
        "protocol_fault", "accepted"
    ]
    assert recovered_terminal_metadata_fault_exposure(result) is True


def test_request_exception_then_valid_response_consumes_valid(monkeypatch):
    requests = []

    class ValidUsage:
        def model_dump(self, *, mode):
            return {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9}

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            if len(requests) == 1:
                raise ConnectionError("network")
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="VALID"), finish_reason="stop"
                )],
                usage=ValidUsage(),
            )

    monkeypatch.setattr(run_model_module.time, "sleep", lambda _: None)
    client = LocalChatClient.__new__(LocalChatClient)
    client.model = "fixture:1b"
    client.spec = load_spec()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    result = client.call("USER", "SYSTEM", "digit_span")
    assert len(requests) == 2 and requests[0] == requests[1]
    assert result["response"] == "VALID"
    assert [row["status"] for row in result["attempts"]] == [
        "request_error", "accepted"
    ]
    assert recovered_terminal_metadata_fault_exposure(result) is False


@pytest.mark.parametrize(
    ("finish_reason", "total_tokens", "message"),
    [
        ("content_filter", 9, "unsupported completion finish reason"),
        ("stop", 99, "reported total tokens disagree"),
        ("", 99, "reported total tokens disagree"),
    ],
)
def test_nonretryable_http_200_contract_errors_fail_immediately(
    monkeypatch, finish_reason, total_tokens, message
):
    calls = []

    class Usage:
        def model_dump(self, *, mode):
            return {
                "prompt_tokens": 8,
                "completion_tokens": 1,
                "total_tokens": total_tokens,
            }

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(copy.deepcopy(kwargs))
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="BODY"),
                    finish_reason=finish_reason,
                )],
                usage=Usage(),
            )

    monkeypatch.setattr(run_model_module.time, "sleep", lambda _: None)
    client = LocalChatClient.__new__(LocalChatClient)
    client.model = "fixture:1b"
    client.spec = load_spec()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    with pytest.raises(RuntimeError, match=message):
        client.call("USER", "SYSTEM", "digit_span")
    assert len(calls) == 1


@pytest.mark.parametrize("malformed_finish", [False, 0])
def test_malformed_finish_reason_type_fails_immediately(monkeypatch, malformed_finish):
    calls = []

    class Usage:
        def model_dump(self, *, mode):
            return {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(copy.deepcopy(kwargs))
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="BODY"),
                    finish_reason=malformed_finish,
                )],
                usage=Usage(),
            )

    monkeypatch.setattr(run_model_module.time, "sleep", lambda _: None)
    client = LocalChatClient.__new__(LocalChatClient)
    client.model = "fixture:1b"
    client.spec = load_spec()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    with pytest.raises(RuntimeError, match="finish reason has a malformed type"):
        client.call("USER", "SYSTEM", "digit_span")
    assert len(calls) == 1


def test_mixed_network_and_terminal_faults_remain_fatal(monkeypatch):
    count = 0

    class ZeroUsage:
        def model_dump(self, *, mode):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    class FakeCompletions:
        def create(self, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                raise ConnectionError("network")
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="BODY"), finish_reason=""
                )],
                usage=ZeroUsage(),
            )

    monkeypatch.setattr(run_model_module.time, "sleep", lambda _: None)
    client = LocalChatClient.__new__(LocalChatClient)
    client.model = "fixture:1b"
    client.spec = load_spec()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        client.call("USER", "SYSTEM", "digit_span")
    assert count == 3


def test_end_marker_transport_preserves_multiline_and_rejects_marker_leak():
    class FakeUsage:
        def model_dump(self, *, mode):
            assert mode == "json"
            return {
                "prompt_tokens": 20,
                "completion_tokens": 3,
                "total_tokens": 23,
                "completion_tokens_details": {"reasoning_tokens": 0},
            }

    class FakeCompletions:
        def __init__(self, content):
            self.content = content
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                    finish_reason="stop",
                )],
                usage=FakeUsage(),
            )

    good = FakeCompletions("A\nB\nC")
    client = LocalChatClient.__new__(LocalChatClient)
    client.model = "fixture:1b"
    client.spec = load_spec()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=good))
    result = client.call("USER", "SYSTEM", "drm_false_memory")
    assert result["response"] == "A\nB\nC"
    assert good.calls[0]["stop"] == ["\n\n", "<END_COGARENA_RESPONSE>"]
    assert good.calls[0]["max_tokens"] == 512

    leaked = FakeCompletions("A\n<END_COGARENA_RESPONSE>")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=leaked))
    with pytest.raises(RuntimeError, match="transport terminator leaked"):
        client.call("USER", "SYSTEM", "drm_false_memory")
    assert len(leaked.calls) == 1


def test_wording_replication_transport_shim_is_exactly_shape_and_model_bound():
    from scripts.experiments.causal_selectivity_20260720.common import (
        response_terminator,
    )
    from scripts.experiments.scaffold_wording_replication_20260725.run_model import (
        EXPECTED_MODEL,
        normalize_olmo2_transport_content,
    )

    spec = load_spec()
    marker = response_terminator(spec)
    request_sha256 = "0" * 64
    assert normalize_olmo2_transport_content(
        marker,
        model=EXPECTED_MODEL,
        request_sha256=request_sha256,
        spec=spec,
    ) == ("", True)
    assert normalize_olmo2_transport_content(
        f"ANSWER\n{marker}",
        model=EXPECTED_MODEL,
        request_sha256=request_sha256,
        spec=spec,
    ) == (f"ANSWER\n{marker}", False)
    assert normalize_olmo2_transport_content(
        marker,
        model="olmo2:13b",
        request_sha256=request_sha256,
        spec=spec,
    ) == (marker, False)


def test_completion_contract_scores_any_truncated_task_record_zero():
    spec = load_spec()
    manifest = load_json(manifest_path("formal"))
    items = reconstruct_items(spec, manifest)
    entry = next(x for x in manifest["items"] if x["paradigm"] == "digit_span")
    item = items[entry["task_id"]]
    response = str(item.expected_response or "")
    scored = score_response_with_completion_contract(
        item,
        response,
        [{"finish_reason": "length"}],
    )
    assert scored["primary_accuracy"] == 0.0
    contract = scored["metrics"]["completion_contract"]
    assert contract["task_invalidated"] is True
    assert contract["truncated_call_count"] == 1
    assert contract["native_scorer_evaluated"] is False
    assert "native_primary_accuracy_before_completion_contract" not in contract


def test_transport_incomplete_task_is_zero_without_native_scorer(monkeypatch):
    spec = load_spec()
    manifest = load_json(manifest_path("formal"))
    items = reconstruct_items(spec, manifest)
    entry = next(x for x in manifest["items"] if x["paradigm"] == "digit_span")
    item = items[entry["task_id"]]

    def forbidden_native(*args, **kwargs):
        raise AssertionError("native scorer must not receive protocol-invalid body")

    monkeypatch.setattr(scorer_adapter_module, "score_response", forbidden_native)
    scored = score_response_with_completion_contract(
        item,
        "PRIVATE UNTRUSTED BODY",
        [{"finish_reason": TRANSPORT_INCOMPLETE_FINISH_REASON}],
    )
    assert scored["primary_accuracy"] == 0.0
    contract = scored["metrics"]["completion_contract"]
    assert contract["invalid_call_count"] == 1
    assert contract["truncated_call_count"] == 0
    assert contract["transport_protocol_invalid_call_count"] == 1
    assert contract["native_scorer_evaluated"] is False

    ospan_entry = next(
        x for x in manifest["items"] if x["paradigm"] == "operation_span"
    )
    ospan_scored = score_response_with_completion_contract(
        items[ospan_entry["task_id"]],
        [],
        [{"finish_reason": TRANSPORT_INCOMPLETE_FINISH_REASON}],
    )
    assert ospan_scored["metrics"]["strict_parse_status"] == (
        "not_evaluated_protocol_invalid"
    )
    assert ospan_scored["metrics"]["completion_contract"][
        "native_scorer_evaluated"
    ] is False


def test_empty_responses_score_zero_in_every_paradigm():
    spec = load_spec()
    manifest = load_json(manifest_path("formal"))
    items = reconstruct_items(spec, manifest)
    for paradigm in sorted(manifest["paradigm_counts"]):
        entry = next(x for x in manifest["items"] if x["paradigm"] == paradigm)
        item = items[entry["task_id"]]
        payload = ([{"response": ""} for _ in range(entry["n_turns"])]
                   if entry["is_multiturn"] else "")
        assert score_response(item, payload)["primary_accuracy"] == 0.0, paradigm


def test_nback_strict_turn_scorer_matches_production_acceptance_and_empty_rule():
    turns = [
        {"stimulus": "A", "expected": "MATCH"},
        {"stimulus": "B", "expected": "NO MATCH"},
        {"stimulus": "C", "expected": "NO MATCH"},
    ]
    metadata = TaskMetadata(
        dimension="working_memory",
        paradigm="n_back",
        mode=EvalMode.LLM_STATIC,
        parameters={"turns": turns, "multi_turn": True},
        scoring=ScoringConfig(method="custom", params={}),
        difficulty=DifficultyLevel.EASY,
    )
    item = TaskInstance("nback_fixture", metadata, "fixture", None)
    result = score_response(
        item,
        [
            {"response": "MATCH."},
            {"response": "NO MATCH because it differs"},
            {"response": ""},
        ],
    )
    assert result["primary_accuracy"] == pytest.approx(2 / 3)
    assert result["metrics"]["unparseable_turns"] == 1


def test_complete_presented_hash_includes_multiturn_stimuli():
    items = [
        x for x in generate_wm_items(seed=994, n_per_paradigm=12, include_contamination_probes=False)
        if x.metadata.paradigm == "n_back"
    ]
    a, b = copy.deepcopy(items[0]), copy.deepcopy(items[0])
    b.metadata.parameters["turns"][0]["stimulus"] = "DIFFERENT"
    assert presentation_fingerprint(a) != presentation_fingerprint(b)


def test_multiturn_history_and_feedback_match_production_run_eval_contract():
    turns = [
        {"stimulus": "FIRST", "correct_answer": "GOLD1"},
        {"stimulus": "SECOND"},
    ]
    metadata = TaskMetadata(
        dimension="working_memory",
        paradigm="n_back",
        mode=EvalMode.LLM_STATIC,
        parameters={"turns": turns, "multi_turn": True},
        scoring=ScoringConfig(method="exact_match"),
        difficulty=DifficultyLevel.EASY,
    )
    item = TaskInstance("history_fixture", metadata, "TASK INSTRUCTIONS", None)

    class FakeClient:
        def __init__(self):
            self.prompts = []

        def call(self, prompt, system, paradigm):
            self.prompts.append((prompt, system))
            response = "R1" if len(self.prompts) == 1 else "R2"
            return {
                "response": response,
                "finish_reason": "stop",
                "reasoning_effort": "none",
                "reasoning_output_characters": 0,
                "stop_sequences": ["\n", "<END_COGARENA_RESPONSE>"],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
                "latency_seconds": 0.0,
                "attempt": 1,
            }

    client = FakeClient()
    payload, calls = evaluate_item(client, item, "SYSTEM", load_spec())
    assert client.prompts[0][0] == "TASK INSTRUCTIONS\nTrial 1: FIRST\nYour response:"
    assert client.prompts[1][0] == (
        "TASK INSTRUCTIONS\nTrial 1: FIRST\nYour response: R1\nFeedback: GOLD1"
        "\nTrial 2: SECOND\nYour response:"
    )
    assert [x["response"] for x in payload] == ["R1", "R2"]
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("invalid_reason", "sentinel"),
    [
        ("length", TRUNCATED_HISTORY_SENTINEL),
        (TRANSPORT_INCOMPLETE_FINISH_REASON, TRANSPORT_INVALID_HISTORY_SENTINEL),
    ],
)
def test_multiturn_history_uses_fixed_sentinel_after_invalid_call(
    invalid_reason, sentinel
):
    turns = [
        {"stimulus": "FIRST", "correct_answer": "GOLD1"},
        {"stimulus": "SECOND"},
    ]
    metadata = TaskMetadata(
        dimension="working_memory",
        paradigm="n_back",
        mode=EvalMode.LLM_STATIC,
        parameters={"turns": turns, "multi_turn": True},
        scoring=ScoringConfig(method="exact_match"),
        difficulty=DifficultyLevel.EASY,
    )
    item = TaskInstance("truncation_history_fixture", metadata, "TASK INSTRUCTIONS", None)

    class FakeClient:
        def __init__(self):
            self.prompts = []

        def call(self, prompt, system, paradigm):
            self.prompts.append((prompt, system))
            first = len(self.prompts) == 1
            return {
                "response": "PRIVATE PARTIAL" if first else "R2",
                "finish_reason": invalid_reason if first else "stop",
                "reasoning_effort": "none",
                "reasoning_output_characters": 0,
                "stop_sequences": ["\n", "<END_COGARENA_RESPONSE>"],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
                "latency_seconds": 0.0,
                "attempt": 1,
            }

    client = FakeClient()
    payload, calls = evaluate_item(client, item, "SYSTEM", load_spec())
    validate_response_transcript(item, payload, calls, load_spec())
    assert payload[0]["response"] == "PRIVATE PARTIAL"
    assert calls[0]["finish_reason"] == invalid_reason
    assert sentinel in client.prompts[1][0]
    assert "PRIVATE PARTIAL" not in client.prompts[1][0]

    tampered = copy.deepcopy(calls)
    tampered[1]["prompt_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="prompt/history hash mismatch"):
        validate_response_transcript(item, payload, tampered, load_spec())


def test_static_prompt_and_finish_reason_are_independently_replayed():
    metadata = TaskMetadata(
        dimension="working_memory",
        paradigm="digit_span",
        mode=EvalMode.LLM_STATIC,
        parameters={},
        scoring=ScoringConfig(method="exact_match"),
        difficulty=DifficultyLevel.EASY,
    )
    item = TaskInstance("static_transcript_fixture", metadata, "STATIC PROMPT", "A")
    calls = [{
        "finish_reason": "stop",
        "prompt_sha256": sha256_text(item.stimulus),
        "reasoning_effort": "none",
        "reasoning_output_characters": 0,
        "stop_sequences": ["\n", "<END_COGARENA_RESPONSE>"],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "total_tokens": 11,
        },
    }]
    validate_response_transcript(item, "A", calls, load_spec())
    validate_response_transcript(item, "A\nB\nC", calls, load_spec())

    bad_finish = copy.deepcopy(calls)
    bad_finish[0]["finish_reason"] = "content_filter"
    with pytest.raises(RuntimeError, match="unsupported completion finish reason"):
        validate_response_transcript(item, "A", bad_finish, load_spec())

    bad_reasoning = copy.deepcopy(calls)
    bad_reasoning[0]["reasoning_effort"] = "medium"
    with pytest.raises(RuntimeError, match="reasoning-effort request"):
        validate_response_transcript(item, "A", bad_reasoning, load_spec())

    bad_stop = copy.deepcopy(calls)
    bad_stop[0]["stop_sequences"] = []
    with pytest.raises(RuntimeError, match="stop sequence"):
        validate_response_transcript(item, "A", bad_stop, load_spec())

    with pytest.raises(RuntimeError, match="terminator leaked"):
        validate_response_transcript(
            item, "A\n<END_COGARENA_RESPONSE>", calls, load_spec()
        )

    bad_prompt = copy.deepcopy(calls)
    bad_prompt[0]["prompt_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="static transcript prompt hash mismatch"):
        validate_response_transcript(item, "A", bad_prompt, load_spec())
