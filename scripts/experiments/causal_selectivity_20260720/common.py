"""Shared, fail-closed utilities for the frozen causal-selectivity study."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("COGARENA_ROOT", HERE.parents[2])).resolve()
SPEC_PATH = Path(
    os.environ.get("COGARENA_CAUSAL_SPEC", HERE / "PREPILOT_SPEC.json")
).resolve()
RESULTS_ROOT = Path(
    os.environ.get(
        "COGARENA_CAUSAL_RESULTS_ROOT",
        ROOT / "results" / "causal_selectivity_20260720",
    )
).resolve()
PROMPT_TOKEN_SAFETY_OVERHEAD = 64
MULTITURN_HISTORY_LINES = 30
PROTOCOL_VALID_FINISH_REASONS = frozenset({"stop", "length"})
TRANSPORT_INCOMPLETE_FINISH_REASON = "transport_incomplete"
PROTOCOL_INVALID_FINISH_REASONS = frozenset(
    {"length", TRANSPORT_INCOMPLETE_FINISH_REASON}
)
STORED_FINISH_REASONS = (
    PROTOCOL_VALID_FINISH_REASONS | {TRANSPORT_INCOMPLETE_FINISH_REASON}
)
TRANSPORT_RETRYABLE_HTTP_200_FAULTS = (
    "missing_finish_reason",
    "missing_usage",
    "zero_usage",
    "nonpositive_prompt_tokens",
    "nonpositive_total_tokens",
    "nonempty_content_zero_completion_tokens",
)
FROZEN_TRANSPORT_RETRY_POLICY = {
    "maximum_total_attempts": 3,
    "request_identity": "canonical normalized request payload has identical SHA-256 across attempts",
    "retryable_http_200_faults": list(TRANSPORT_RETRYABLE_HTTP_200_FAULTS),
    "exhausted_status": TRANSPORT_INCOMPLETE_FINISH_REASON,
    "sdk_internal_retries": 0,
    "request_exception_policy": {
        "retry_identical_request": True,
        "accept_later_valid_response": True,
        "exhausted_sequence_with_request_exception": "fatal",
    },
    "native_endpoint_fallback": False,
}
EXECUTION_GUARD_FILENAME = "EXECUTION_GUARD.json"
EXECUTION_GUARD_SCHEMA = "cogarena.causal_selectivity.execution_guard.v1"
EXECUTION_GUARD_STATES = (
    "in_progress",
    "records_complete",
    "verified_complete",
)
FROZEN_EXECUTION_GUARD_POLICY = {
    "model_root_acquisition": "atomic_mkdir_must_not_exist",
    "record_reuse": False,
    "interrupted_attempt_policy": (
        "quarantine_entire_profile_attempt_and_restart_in_a_fresh_profile_root"
    ),
    "states": list(EXECUTION_GUARD_STATES),
    "same_slurm_job_model_verification": True,
    "profile_attempt_identity": "all model guards share one SLURM_ARRAY_JOB_ID",
}

_EXECUTION_GUARD_IDENTITY_FIELDS = (
    "schema_version",
    "study_id",
    "profile",
    "model",
    "model_safe",
    "source_revision",
    "spec_sha256",
    "item_manifest_sha256",
    "slurm_job_id",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "execution_node",
    "acquired_at",
    "no_resume",
)
_EXECUTION_GUARD_RECORDS_FIELDS = (
    *_EXECUTION_GUARD_IDENTITY_FIELDS,
    "guard_identity_sha256",
    "records_completed_at",
    "expected_record_count",
    "record_tree_sha256",
    "serving_provenance_sha256",
    "run_summary_sha256",
)
_EXECUTION_GUARD_BASE_KEYS = frozenset(
    (*_EXECUTION_GUARD_IDENTITY_FIELDS, "state", "guard_identity_sha256")
)
_EXECUTION_GUARD_RECORDS_KEYS = frozenset(
    (*_EXECUTION_GUARD_RECORDS_FIELDS, "state", "records_complete_sha256")
)
_EXECUTION_GUARD_VERIFIED_KEYS = frozenset(
    (
        *_EXECUTION_GUARD_RECORDS_KEYS,
        "verified_at",
        "verification_slurm_job_id",
        "verification_slurm_array_task_id",
        "verification_execution_node",
        "model_manifest_sha256",
    )
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def is_protocol_invalid_finish_reason(value: Any) -> bool:
    """Return whether a persisted logical call invalidates its whole task record."""
    return value in PROTOCOL_INVALID_FINISH_REASONS


def recovered_terminal_metadata_fault_exposure(call: dict[str, Any]) -> bool:
    """Flag a valid logical call recovered after an HTTP-200 metadata fault.

    Request exceptions alone do not qualify.  The caller first validates the
    full transport transcript, so the terminal accepted attempt and identical
    request hashes remain part of the fail-closed proof rather than being
    inferred from this descriptive flag.
    """
    attempts = call.get("attempts")
    return bool(
        call.get("transport_status") == "valid"
        and isinstance(attempts, list)
        and attempts
        and isinstance(attempts[-1], dict)
        and attempts[-1].get("status") == "accepted"
        and any(
            isinstance(attempt, dict) and attempt.get("status") == "protocol_fault"
            for attempt in attempts[:-1]
        )
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path | str) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_spec(path: Path | str = SPEC_PATH) -> dict[str, Any]:
    spec = load_json(path)
    require(spec.get("schema_version") == "cogarena.causal_selectivity.spec.v2", "bad spec")
    transport_retry_policy(spec)
    execution_guard_policy(spec)
    return spec


def transport_retry_policy(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the one structured retry policy, rejecting spec/runtime drift."""
    policy = spec.get("execution_contract", {}).get("transport_retry_policy")
    require(
        policy == FROZEN_TRANSPORT_RETRY_POLICY,
        "causal-selectivity transport retry policy drift",
    )
    return json.loads(json.dumps(policy))


def execution_guard_policy(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen no-resume policy, rejecting spec/runtime drift."""
    policy = spec.get("execution_contract", {}).get("execution_guard_policy")
    require(
        policy == FROZEN_EXECUTION_GUARD_POLICY,
        "causal-selectivity execution-guard policy drift",
    )
    return json.loads(json.dumps(policy))


def request_reasoning_effort(spec: dict[str, Any]) -> str:
    """Return the frozen OpenAI-compatible reasoning control, failing closed."""
    value = spec.get("scope", {}).get("reasoning_effort")
    require(
        value == "none",
        "causal-selectivity inference requires reasoning_effort='none'",
    )
    return value


def request_stop_policy(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the frozen paradigm-format stop policy, failing closed."""
    scope = spec.get("scope", {})
    terminator = scope.get("response_terminator")
    policy = scope.get("response_format_policy")
    expected = {
        "single_line": {
            "paradigms": [
                "digit_span",
                "n_back",
                "operation_span",
                "stroop",
                "flanker",
                "go_nogo",
                "cvlt_word_list",
                "false_belief",
                "epitome_tom",
            ],
            "stop_sequences": ["\n", "<END_COGARENA_RESPONSE>"],
        },
        "multiline": {
            "paradigms": [
                "drm_false_memory",
                "source_monitoring",
                "confidence_calibration",
                "post_decision_wagering",
            ],
            "stop_sequences": ["\n\n", "<END_COGARENA_RESPONSE>"],
        },
    }
    require(
        terminator == "<END_COGARENA_RESPONSE>" and policy == expected,
        "causal-selectivity inference requires the frozen response-format stop policy",
    )
    return json.loads(json.dumps(policy))


def request_stop_sequences(spec: dict[str, Any], paradigm: str) -> list[str]:
    """Return the exact stop list for one paradigm's declared answer format."""
    policy = request_stop_policy(spec)
    matches = [
        row["stop_sequences"]
        for row in policy.values()
        if paradigm in row["paradigms"]
    ]
    require(len(matches) == 1, f"unfrozen response format for paradigm: {paradigm}")
    return list(matches[0])


def response_terminator(spec: dict[str, Any]) -> str:
    request_stop_policy(spec)
    return str(spec["scope"]["response_terminator"])


def format_user_prompt(spec: dict[str, Any], paradigm: str, prompt: str) -> str:
    """Apply the one frozen paradigm-only response-format wrapper."""
    require(isinstance(prompt, str), "user prompt must be text")
    overrides = spec.get("response_format_overrides")
    expected = {
        "cvlt_word_list": (
            "Return the complete requested list on one line, separating entries "
            "with commas. Do not use line breaks or add an explanation."
        )
    }
    require(overrides == expected, "response-format override contract drift")
    instruction = overrides.get(paradigm)
    if instruction is None:
        return prompt
    return f"{prompt}\n\nResponse format: {instruction}"


def conservative_prompt_token_estimate(user_prompt: str, system_prompt_text: str) -> int:
    """Tokenizer-independent upper-budget proxy used by preparation and runtime.

    All frozen task prompts are English/ASCII-heavy. Three characters per token
    plus a fixed chat-template allowance is deliberately more conservative than
    the observed tokenizers while remaining model-family agnostic.
    """
    require(
        isinstance(user_prompt, str) and isinstance(system_prompt_text, str),
        "prompt components must be text",
    )
    return (
        (len(user_prompt) + len(system_prompt_text) + 2) // 3
        + PROMPT_TOKEN_SAFETY_OVERHEAD
    )


def require_prompt_budget(
    spec: dict[str, Any], user_prompt: str, system_prompt_text: str
) -> int:
    """Reserve the complete frozen generation budget inside served context."""
    estimated_prompt_tokens = conservative_prompt_token_estimate(
        user_prompt, system_prompt_text
    )
    context = int(spec["scope"]["served_context_tokens"])
    completion = int(spec["scope"]["max_completion_tokens"])
    require(
        estimated_prompt_tokens + completion <= context,
        "request cannot reserve the frozen completion budget within served context: "
        f"prompt_estimate={estimated_prompt_tokens} completion={completion} context={context}",
    )
    return estimated_prompt_tokens


def turn_shown_text(turn: dict[str, Any]) -> str:
    """Canonical runtime rendering of one presented multi-turn stimulus."""
    shown = turn.get("stimulus", turn.get("presented", turn.get("item", str(turn))))
    if isinstance(shown, dict):
        return json.dumps(shown, sort_keys=True, ensure_ascii=False)
    return str(shown)


def validate_reported_usage_summary(spec: dict[str, Any], summary: dict[str, Any]) -> None:
    """Fail closed on aggregate valid-usage evidence at closure boundaries."""
    require(
        summary.get("reported_usage_context_budget_verified") is True,
        "reported usage/context budget was not replay-verified",
    )
    if str(summary.get("schema_version", "")).endswith(".v3"):
        logical_calls = summary.get("api_call_count")
        valid_usage_calls = summary.get("usage_metadata_valid_logical_call_count")
        invalid_transport_calls = summary.get(
            "transport_incomplete_logical_call_count"
        )
        attempts = summary.get("transport_attempt_count")
        retries = summary.get("transport_retry_count")
        terminal_fault_attempts = summary.get(
            "terminal_metadata_fault_attempt_count"
        )
        request_error_attempts = summary.get("request_error_attempt_count")
        require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (
                    logical_calls,
                    valid_usage_calls,
                    invalid_transport_calls,
                    attempts,
                    retries,
                    terminal_fault_attempts,
                    request_error_attempts,
                )
            )
            and valid_usage_calls + invalid_transport_calls == logical_calls
            and attempts >= logical_calls
            and retries == attempts - logical_calls
            and attempts
            == valid_usage_calls + terminal_fault_attempts + request_error_attempts
            and summary.get("static_prompt_budget_verified_for_all_logical_calls") is True,
            "v3 logical-call/usage accounting is inconsistent",
        )
    keys = (
        "max_reported_prompt_tokens",
        "max_reported_completion_tokens",
        "max_reported_total_tokens",
        "minimum_reported_prompt_reservation_margin_tokens",
    )
    require(
        all(
            isinstance(summary.get(key), int)
            and not isinstance(summary.get(key), bool)
            and summary[key] >= 0
            for key in keys
        ),
        "reported usage/context summary is malformed",
    )
    context = int(spec["scope"]["served_context_tokens"])
    completion = int(spec["scope"]["max_completion_tokens"])
    require(
        summary["max_reported_prompt_tokens"] + completion <= context,
        "aggregate reported prompt cannot reserve the completion budget",
    )
    require(
        summary["max_reported_completion_tokens"] <= completion,
        "aggregate reported completion exceeds the completion budget",
    )
    require(
        summary["max_reported_total_tokens"] <= context,
        "aggregate reported total exceeds served context",
    )
    require(
        summary["minimum_reported_prompt_reservation_margin_tokens"]
        == context - completion - summary["max_reported_prompt_tokens"],
        "aggregate reported prompt-reservation margin is inconsistent",
    )


def atomic_write_bytes(path: Path | str, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path | str, value: Any, *, indent: int = 2) -> None:
    payload = json.dumps(
        value, indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite value: {value}")
        return value
    return str(value)


def item_payload(item: Any) -> dict[str, Any]:
    return {
        "task_id": item.task_id,
        "metadata": jsonable(item.metadata),
        "stimulus": item.stimulus,
        "image_path": item.image_path,
        "expected_response": jsonable(item.expected_response),
    }


def item_fingerprint(item: Any) -> str:
    return sha256_bytes(canonical_bytes(item_payload(item)))


def presentation_payload(item: Any) -> dict[str, Any]:
    """Only content shown to a model; deliberately excludes every gold field."""
    turns = item.metadata.parameters.get("turns", [])
    presented_turns = []
    for turn in turns:
        shown = turn.get("stimulus", turn.get("presented", turn.get("item", "")))
        presented_turns.append(jsonable(shown))
    return {"stimulus": item.stimulus, "turn_stimuli": presented_turns}


def presentation_fingerprint(item: Any) -> str:
    return sha256_bytes(canonical_bytes(presentation_payload(item)))


def scoring_gold_payload(item: Any) -> dict[str, Any]:
    """Gold fields consumed by scorers, separated from presented content."""
    turns = item.metadata.parameters.get("turns", [])
    turn_gold = []
    for turn in turns:
        turn_gold.append(
            {
                key: jsonable(turn[key])
                for key in ("expected", "expected_words", "math_expected", "recall_letter", "type")
                if key in turn
            }
        )
    return {
        "expected_response": jsonable(item.expected_response),
        "metadata_letters": jsonable(item.metadata.parameters.get("letters")),
        "turn_gold": turn_gold,
    }


def scoring_gold_fingerprint(item: Any) -> str:
    return sha256_bytes(canonical_bytes(scoring_gold_payload(item)))


def model_safe(model: str) -> str:
    return model.replace("/", "_").replace(":", "__")


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def execution_guard_identity_sha256(guard: dict[str, Any]) -> str:
    """Hash the immutable acquisition identity, independent of later states."""
    require(isinstance(guard, dict), "execution guard is not a mapping")
    return sha256_bytes(
        canonical_bytes({key: guard.get(key) for key in _EXECUTION_GUARD_IDENTITY_FIELDS})
    )


def execution_guard_records_complete_sha256(guard: dict[str, Any]) -> str:
    """Hash acquisition plus completed raw/serving/summary bindings.

    This semantic projection remains recomputable after the final verified
    transition and avoids a circular full-file hash between the guard and
    ``MODEL_MANIFEST.json``.
    """
    require(isinstance(guard, dict), "execution guard is not a mapping")
    return sha256_bytes(
        canonical_bytes({key: guard.get(key) for key in _EXECUTION_GUARD_RECORDS_FIELDS})
    )


def validate_execution_guard(
    guard: dict[str, Any], *, expected_state: str | None = None
) -> None:
    """Validate the exact three-state, no-resume execution-guard schema."""
    require(isinstance(guard, dict), "execution guard is not a mapping")
    state = guard.get("state")
    require(state in EXECUTION_GUARD_STATES, "execution guard has an invalid state")
    if expected_state is not None:
        require(state == expected_state, f"execution guard state is not {expected_state}")
    expected_keys = {
        "in_progress": _EXECUTION_GUARD_BASE_KEYS,
        "records_complete": _EXECUTION_GUARD_RECORDS_KEYS,
        "verified_complete": _EXECUTION_GUARD_VERIFIED_KEYS,
    }[state]
    require(set(guard) == expected_keys, "execution guard schema/field set mismatch")
    require(
        guard.get("schema_version") == EXECUTION_GUARD_SCHEMA
        and isinstance(guard.get("study_id"), str)
        and bool(guard["study_id"])
        and guard.get("profile") in {"pilot", "formal"}
        and isinstance(guard.get("model"), str)
        and bool(guard["model"])
        and guard.get("model_safe") == model_safe(guard["model"])
        and _is_lower_hex(guard.get("source_revision"), 40)
        and _is_lower_hex(guard.get("spec_sha256"), 64)
        and _is_lower_hex(guard.get("item_manifest_sha256"), 64)
        and isinstance(guard.get("slurm_job_id"), str)
        and bool(guard["slurm_job_id"])
        and isinstance(guard.get("slurm_array_job_id"), str)
        and bool(guard["slurm_array_job_id"])
        and isinstance(guard.get("slurm_array_task_id"), str)
        and bool(guard["slurm_array_task_id"])
        and isinstance(guard.get("execution_node"), str)
        and bool(guard["execution_node"])
        and isinstance(guard.get("acquired_at"), str)
        and bool(guard["acquired_at"])
        and guard.get("no_resume") is True,
        "execution guard identity is malformed",
    )
    require(
        guard.get("guard_identity_sha256") == execution_guard_identity_sha256(guard),
        "execution guard identity hash mismatch",
    )
    if state == "in_progress":
        return
    require(
        isinstance(guard.get("records_completed_at"), str)
        and bool(guard["records_completed_at"])
        and isinstance(guard.get("expected_record_count"), int)
        and not isinstance(guard["expected_record_count"], bool)
        and guard["expected_record_count"] > 0
        and _is_lower_hex(guard.get("record_tree_sha256"), 64)
        and _is_lower_hex(guard.get("serving_provenance_sha256"), 64)
        and _is_lower_hex(guard.get("run_summary_sha256"), 64),
        "execution guard records-complete binding is malformed",
    )
    require(
        guard.get("records_complete_sha256")
        == execution_guard_records_complete_sha256(guard),
        "execution guard records-complete hash mismatch",
    )
    if state == "records_complete":
        return
    require(
        isinstance(guard.get("verified_at"), str)
        and bool(guard["verified_at"])
        and guard.get("verification_slurm_job_id") == guard.get("slurm_job_id")
        and guard.get("verification_slurm_array_task_id")
        == guard.get("slurm_array_task_id")
        and guard.get("verification_execution_node") == guard.get("execution_node")
        and _is_lower_hex(guard.get("model_manifest_sha256"), 64),
        "execution guard verified-complete binding is malformed",
    )


def stable_seed(*parts: str, bits: int = 63) -> int:
    raw = sha256_text("\0".join(parts))
    return int(raw[:16], 16) & ((1 << bits) - 1)


def condition_map(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {c["id"]: c for c in spec["conditions"]}
    require(len(out) == len(spec["conditions"]), "duplicate condition IDs")
    return out


def system_prompt(spec: dict[str, Any], condition_id: str) -> str:
    cond = condition_map(spec)[condition_id]
    base = spec["system_base"].strip()
    scaffold = cond["scaffold"].strip()
    body = base if not scaffold else f"{base}\n\nAdditional procedure:\n{scaffold}"
    transport = str(spec.get("transport_instruction", "")).strip()
    terminator = response_terminator(spec)
    require(transport and terminator in transport,
            "transport instruction does not bind the frozen response terminator")
    return f"{body}\n\nTransport framing:\n{transport}"


def manifest_path(profile: str) -> Path:
    require(profile in {"formal", "pilot"}, f"unknown profile: {profile}")
    return RESULTS_ROOT / f"item_manifest_{profile}.json"


def profile_models(spec: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    key = "formal_model_panel" if profile == "formal" else "pilot_model_panel"
    return list(spec[key])


def ensure_finite_accuracy(value: Any, label: str = "accuracy") -> float:
    require(not isinstance(value, bool), f"{label} is bool")
    result = float(value)
    require(math.isfinite(result), f"{label} is non-finite")
    require(0.0 <= result <= 1.0, f"{label} outside [0,1]: {result}")
    return result


def tree_hash(paths: Iterable[Path], base: Path) -> str:
    entries = []
    for path in sorted(set(Path(p) for p in paths)):
        entries.append([path.relative_to(base).as_posix(), sha256_file(path)])
    return sha256_bytes(canonical_bytes(entries))
