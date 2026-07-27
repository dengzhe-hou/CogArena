#!/usr/bin/env python3
"""Run one frozen causal-selectivity model panel member on a Slurm GPU node."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .build_item_manifests import generate_pool
from .common import (
    EXECUTION_GUARD_FILENAME,
    EXECUTION_GUARD_SCHEMA,
    RESULTS_ROOT,
    SPEC_PATH,
    atomic_write_json,
    canonical_bytes,
    condition_map,
    require_prompt_budget,
    ensure_finite_accuracy,
    execution_guard_identity_sha256,
    execution_guard_records_complete_sha256,
    format_user_prompt,
    item_fingerprint,
    jsonable,
    load_json,
    load_spec,
    manifest_path,
    model_safe,
    MULTITURN_HISTORY_LINES,
    PROTOCOL_VALID_FINISH_REASONS,
    STORED_FINISH_REASONS,
    TRANSPORT_INCOMPLETE_FINISH_REASON,
    presentation_fingerprint,
    profile_models,
    request_reasoning_effort,
    request_stop_policy,
    request_stop_sequences,
    response_terminator,
    require,
    is_protocol_invalid_finish_reason,
    sha256_bytes,
    sha256_file,
    sha256_text,
    scoring_gold_fingerprint,
    stable_seed,
    system_prompt,
    tree_hash,
    transport_retry_policy,
    turn_shown_text,
    validate_execution_guard,
)
from .scorer_adapter import (
    SCORER_CONTRACT_VERSION,
    score_response_with_completion_contract,
)


RESULT_SCHEMA = "cogarena.causal_selectivity.result.v3"
TRUNCATED_HISTORY_SENTINEL = "[INVALID_COMPLETION_AT_TOKEN_LIMIT]"
TRANSPORT_INVALID_HISTORY_SENTINEL = "[INVALID_TRANSPORT_COMPLETION]"
ALLOWED_FINISH_REASONS = set(PROTOCOL_VALID_FINISH_REASONS)


def _current_slurm_guard_fields(execution_node: str | None = None) -> dict[str, str]:
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    require(job_id, "execution guard requires SLURM_JOB_ID")
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "").strip()
    require(array_job_id, "execution guard requires SLURM_ARRAY_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0").strip() or "0"
    node = (execution_node or socket.gethostname()).split(".", 1)[0]
    require(node, "execution guard requires an execution node")
    return {
        "slurm_job_id": job_id,
        "slurm_array_job_id": array_job_id,
        "slurm_array_task_id": task_id,
        "execution_node": node,
    }


def acquire_execution_guard(
    model_root: Path,
    *,
    study_id: str,
    profile: str,
    model: str,
    source_revision: str,
    spec_sha256: str,
    item_manifest_sha256: str,
    execution_node: str | None = None,
) -> dict[str, Any]:
    """Atomically acquire a fresh model root; any prior root is fatal."""
    model_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        model_root.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(
            "model output root already exists; per-record/model resume is forbidden. "
            "Quarantine the entire profile attempt and start in a fresh profile root: "
            f"{model_root}"
        ) from error
    acquired_at = datetime.now(timezone.utc).isoformat()
    guard = {
        "schema_version": EXECUTION_GUARD_SCHEMA,
        "state": "in_progress",
        "study_id": study_id,
        "profile": profile,
        "model": model,
        "model_safe": model_safe(model),
        "source_revision": source_revision,
        "spec_sha256": spec_sha256,
        "item_manifest_sha256": item_manifest_sha256,
        **_current_slurm_guard_fields(execution_node),
        "acquired_at": acquired_at,
        "no_resume": True,
    }
    guard["guard_identity_sha256"] = execution_guard_identity_sha256(guard)
    validate_execution_guard(guard, expected_state="in_progress")
    guard_path = model_root / EXECUTION_GUARD_FILENAME
    atomic_write_json(guard_path, guard)
    validate_execution_guard(load_json(guard_path), expected_state="in_progress")
    return guard


def complete_execution_guard(
    model_root: Path,
    acquired_guard: dict[str, Any],
    *,
    record_paths: list[Path],
    serving_path: Path,
    run_summary_path: Path,
    records_completed_at: str,
) -> dict[str, Any]:
    """Atomically bind the completed raw tree and immutable metadata."""
    guard_path = model_root / EXECUTION_GUARD_FILENAME
    current = load_json(guard_path)
    validate_execution_guard(current, expected_state="in_progress")
    require(
        canonical_bytes(current) == canonical_bytes(acquired_guard),
        "execution guard changed during model execution",
    )
    completed = dict(current)
    completed.update(
        {
            "state": "records_complete",
            "records_completed_at": records_completed_at,
            "expected_record_count": len(record_paths),
            "record_tree_sha256": tree_hash(record_paths, model_root),
            "serving_provenance_sha256": sha256_file(serving_path),
            "run_summary_sha256": sha256_file(run_summary_path),
        }
    )
    completed["records_complete_sha256"] = (
        execution_guard_records_complete_sha256(completed)
    )
    validate_execution_guard(completed, expected_state="records_complete")
    atomic_write_json(guard_path, completed)
    stored = load_json(guard_path)
    validate_execution_guard(stored, expected_state="records_complete")
    require(
        canonical_bytes(stored) == canonical_bytes(completed),
        "execution guard records-complete transition did not persist exactly",
    )
    return stored


def assert_slurm_gpu_context() -> dict[str, str]:
    require(os.environ.get("SLURM_JOB_ID"), "inference is forbidden outside a Slurm job")
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    require(cuda and cuda != "-1", "no CUDA_VISIBLE_DEVICES allocation")
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    require(proc.returncode == 0 and proc.stdout.strip(), f"nvidia-smi gate failed: {proc.stderr}")
    return {"cuda_visible_devices": cuda, "nvidia_smi": proc.stdout.strip()}


def enforce_frozen_hardware(
    spec: dict[str, Any], gpu: dict[str, str], hostname: str | None = None
) -> str:
    """Bind inference to the pre-freeze node and GPU class, not only sbatch text."""
    contract = spec.get("execution_contract", {})
    allowed = contract.get("eligible_inference_nodes")
    fragment = contract.get("required_gpu_name_fragment")
    require(isinstance(allowed, list) and allowed
            and all(isinstance(node, str) and node for node in allowed),
            "eligible inference-node contract is missing")
    require(isinstance(fragment, str) and fragment,
            "required GPU-name contract is missing")
    node = (hostname or socket.gethostname()).split(".", 1)[0]
    require(node in allowed, f"inference node {node} is outside frozen hardware scope {allowed}")
    require(fragment.lower() in gpu.get("nvidia_smi", "").lower(),
            f"GPU class is outside frozen hardware scope: require {fragment!r}")
    return node


def require_request_fits_context(
    spec: dict[str, Any], user_prompt: str, system_prompt_text: str
) -> int:
    """Conservatively reserve the full completion budget before every call."""
    return require_prompt_budget(spec, user_prompt, system_prompt_text)


def _local_api_urls() -> tuple[str, str]:
    base = os.environ.get("OPENAI_BASE_URL", "")
    parsed = urlparse(base)
    require(parsed.scheme in {"http", "https"}, "OPENAI_BASE_URL missing/invalid")
    require(parsed.hostname in {"127.0.0.1", "localhost"}, "only a node-local model server is allowed")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return base.rstrip("/"), f"{origin}/api/tags"


def _fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _match_tag(tags: dict[str, Any], model: str) -> dict[str, Any]:
    candidates = []
    for entry in tags.get("models", []):
        names = {str(entry.get("name", "")), str(entry.get("model", ""))}
        normalized = {name[:-7] if name.endswith(":latest") else name for name in names}
        if model in names or model in normalized:
            candidates.append(entry)
    require(len(candidates) == 1, f"expected one /api/tags entry for {model}, got {len(candidates)}")
    digest = str(candidates[0].get("digest", ""))
    require(len(digest) >= 32, f"missing model digest for {model}")
    return jsonable(candidates[0])


def parse_ollama_ps_serving_row(table: str, model: str) -> dict[str, Any]:
    """Parse the selected model row by header boundaries, never substring presence.

    Ollama reports split offload such as ``48%/52% CPU/GPU`` in the PROCESSOR
    column. A different model can also be fully GPU-loaded in the same table.
    Callers therefore must inspect the exact selected row and exact column.
    """
    lines = [line for line in table.splitlines() if line.strip()]
    header = lines[0] if lines else ""
    required_headers = ("NAME", "ID", "SIZE", "PROCESSOR", "CONTEXT", "UNTIL")
    require(len(lines) >= 2 and all(name in header for name in required_headers),
            "unparseable ollama ps table")
    starts = {name: header.index(name) for name in required_headers}
    require(
        [starts[name] for name in required_headers] == sorted(starts.values()),
        "ollama ps columns are out of order",
    )
    model_rows = [
        line for line in lines[1:]
        if line[starts["NAME"]:starts["ID"]].strip().removesuffix(":latest") == model
    ]
    require(len(model_rows) == 1, f"expected one loaded ollama ps row for {model}")
    row = model_rows[0]
    require(len(row) >= starts["UNTIL"], "truncated ollama ps model row")
    processor = " ".join(row[starts["PROCESSOR"]:starts["CONTEXT"]].split())
    actual_context_text = row[starts["CONTEXT"]:starts["UNTIL"]].strip().replace(",", "")
    require(processor, "empty ollama ps PROCESSOR column")
    require(actual_context_text.isdigit(), f"unparseable served context: {actual_context_text!r}")
    return {"processor": processor, "actual_context_tokens": int(actual_context_text)}


def parse_ollama_ps_context(table: str, model: str) -> int:
    """Compatibility wrapper for callers/tests that only need context."""
    return int(parse_ollama_ps_serving_row(table, model)["actual_context_tokens"])


def require_fully_gpu_served(table: str, model: str) -> dict[str, Any]:
    row = parse_ollama_ps_serving_row(table, model)
    require(row["processor"] == "100% GPU",
            f"selected model is not fully GPU-served: {model} PROCESSOR={row['processor']!r}")
    return row


def capture_serving_provenance(
    model: str,
    model_root: Path,
    gpu: dict[str, str],
    reasoning_effort: str,
    stop_policy: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    require(reasoning_effort == "none", "serving provenance requires reasoning_effort=none")
    require(os.environ.get("COGARENA_REASONING_EFFORT") == reasoning_effort,
            "serving provenance reasoning-effort environment mismatch")
    require(stop_policy == request_stop_policy(load_spec()),
            "serving provenance response-format policy mismatch")
    require(os.environ.get("COGARENA_STOP_MODE") == "format_routed",
            "serving provenance response-format environment mismatch")
    _, tags_url = _local_api_urls()
    tag = _match_tag(_fetch_json(tags_url), model)
    ps = subprocess.run(["ollama", "ps"], text=True, capture_output=True)
    require(ps.returncode == 0, f"ollama ps failed: {ps.stderr}")
    serving_row = require_fully_gpu_served(ps.stdout, model)
    processor = serving_row["processor"]
    actual_context = serving_row["actual_context_tokens"]
    expected_context = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "0"))
    require(expected_context > 0 and actual_context == expected_context,
            f"served context {actual_context} != frozen environment {expected_context}")
    version = subprocess.run(["ollama", "--version"], text=True, capture_output=True)
    require(version.returncode == 0, "ollama --version failed")
    session = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": gpu["cuda_visible_devices"],
        "nvidia_smi": gpu["nvidia_smi"],
        "ollama_ps": ps.stdout.strip(),
        "ollama_version": version.stdout.strip() or version.stderr.strip(),
        "processor": processor,
        "fully_gpu_served": True,
        "actual_context_tokens": actual_context,
        "reasoning_effort": reasoning_effort,
        "stop_policy": stop_policy,
    }
    path = model_root / "serving_provenance.json"
    existing = load_json(path) if path.exists() else None
    if existing is not None:
        require(existing.get("model") == model, "serving provenance model drift")
        require(existing.get("tag", {}).get("digest") == tag.get("digest"), "served digest drift")
        require(existing.get("fully_gpu_served") is True
                and existing.get("processor") == "100% GPU",
                "existing serving provenance lacks the full-GPU gate")
        require(existing.get("reasoning_effort") == reasoning_effort,
                "existing serving provenance used a different reasoning request")
        require(existing.get("stop_policy") == stop_policy,
                "existing serving provenance used a different response-format policy")
        old_sessions = existing.get("sessions")
        require(isinstance(old_sessions, list) and old_sessions
                and all(old.get("fully_gpu_served") is True
                        and old.get("processor") == "100% GPU"
                        and old.get("reasoning_effort") == reasoning_effort
                        and old.get("stop_policy") == stop_policy
                        for old in old_sessions),
                "an existing serving session lacks exact full-GPU/reasoning proof")
        sessions = list(old_sessions) + [session]
    else:
        sessions = [session]
    provenance = {
        "schema_version": "cogarena.causal_selectivity.serving.v1",
        "model": model,
        "tag": tag,
        "processor": processor,
        "fully_gpu_served": True,
        "served_context_tokens": int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "0")),
        "actual_context_tokens": actual_context,
        "reasoning_effort": reasoning_effort,
        "stop_policy": stop_policy,
        "sessions": sessions,
    }
    atomic_write_json(path, provenance)
    return provenance


def reconstruct_items(spec: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    pool = generate_pool(spec, manifest["profile"])
    # Sacrificial pilot variants may reuse a generator-local task ID while
    # changing the complete presented content. Reconstruct by the frozen
    # fingerprint tuple instead of assuming task IDs identify the whole pool.
    by_id: dict[str, dict[tuple[str, str, str], Any]] = {}
    for item in pool:
        if item.metadata.paradigm not in spec_grouped_paradigms(spec):
            continue
        key = (
            item_fingerprint(item),
            presentation_fingerprint(item),
            scoring_gold_fingerprint(item),
        )
        by_id.setdefault(item.task_id, {}).setdefault(key, item)
    selected = {}
    for entry in manifest["items"]:
        task_id = entry["task_id"]
        require(task_id in by_id, f"manifest task not regenerated: {task_id}")
        key = (
            entry["item_fingerprint_sha256"],
            entry["presentation_sha256"],
            entry["scoring_gold_sha256"],
        )
        require(key in by_id[task_id], f"manifest fingerprint not regenerated: {task_id}")
        require(task_id not in selected, f"selected manifest task ID collision: {task_id}")
        selected[task_id] = by_id[task_id][key]
    require(len(selected) == manifest["item_count"], "reconstruction count mismatch")
    return selected


def spec_grouped_paradigms(spec: dict[str, Any]) -> set[str]:
    return {p for paradigms in spec["grouping"].values() for p in paradigms}


class LocalChatClient:
    def __init__(self, model: str, spec: dict[str, Any]):
        import openai

        base, _ = _local_api_urls()
        self.model = model
        self.spec = spec
        self.client = openai.OpenAI(
            base_url=base,
            api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
            timeout=300.0,
            max_retries=0,
        )

    def call(self, user_prompt: str, sys_prompt: str, paradigm: str) -> dict[str, Any]:
        require_request_fits_context(self.spec, user_prompt, sys_prompt)
        reasoning_effort = request_reasoning_effort(self.spec)
        stop_sequences = request_stop_sequences(self.spec, paradigm)
        request_payload = completion_request_payload(
            self.model, self.spec, user_prompt, sys_prompt, paradigm
        )
        retry_policy = transport_retry_policy(self.spec)
        maximum_attempts = int(retry_policy["maximum_total_attempts"])
        retryable_faults = set(retry_policy["retryable_http_200_faults"])
        request_sha = sha256_bytes(canonical_bytes(request_payload))
        last_error: Exception | None = None
        hard_error_seen = False
        attempt_evidence: list[dict[str, Any]] = []
        last_protocol_content = ""
        for attempt in range(maximum_attempts):
            start = time.monotonic()
            attempt_payload = json.loads(json.dumps(request_payload))
            attempt_request_sha = sha256_bytes(canonical_bytes(attempt_payload))
            require(
                attempt_request_sha == request_sha,
                "canonical request payload changed across transport attempts",
            )
            try:
                response = self.client.chat.completions.create(**attempt_payload)
            except Exception as error:  # retried, then made fatal without writing a record
                last_error = error
                hard_error_seen = True
                elapsed = time.monotonic() - start
                attempt_evidence.append({
                    "attempt": attempt + 1,
                    "status": "request_error",
                    "faults": ["request_exception"],
                    "finish_reason": "",
                    "usage": None,
                    "content_characters": 0,
                    "content_sha256": sha256_text(""),
                    "latency_seconds": elapsed,
                    "request_sha256": request_sha,
                    "exception_type": type(error).__name__,
                })
                if attempt < maximum_attempts - 1:
                    time.sleep(2**attempt)
                continue

            # Only request exceptions and the six structured HTTP-200 terminal-
            # metadata faults are retryable. Every other protocol violation is
            # evaluated outside the request try/except and fails immediately.
            choice = response.choices[0]
            content = normalize_model_content(choice.message.content)
            require(
                response_terminator(self.spec) not in content,
                "transport terminator leaked through the model server",
            )
            exposed_reasoning = getattr(choice.message, "reasoning", None)
            if exposed_reasoning is None:
                exposed_reasoning = getattr(choice.message, "reasoning_content", None)
            require(
                exposed_reasoning in (None, ""),
                "server returned an exposed reasoning channel despite reasoning_effort=none",
            )
            raw_finish = choice.finish_reason
            require(
                raw_finish is None or isinstance(raw_finish, str),
                "completion finish reason has a malformed type",
            )
            finish = raw_finish or ""
            require(
                not finish or finish in ALLOWED_FINISH_REASONS,
                f"unsupported completion finish reason: {finish!r}",
            )
            usage = normalize_completion_usage(response.usage)
            require_reported_usage_structurally_consistent(self.spec, usage)
            faults = terminal_metadata_faults(content, finish, usage)
            require(
                set(faults).issubset(retryable_faults),
                "terminal-metadata classifier drifted outside the frozen retry policy",
            )
            if not faults:
                require_reported_usage_fits_context(self.spec, usage, content)
            elapsed = time.monotonic() - start
            evidence = transport_attempt_evidence(
                attempt + 1,
                "protocol_fault" if faults else "accepted",
                faults,
                finish,
                usage,
                content,
                elapsed,
                request_sha,
            )
            attempt_evidence.append(evidence)
            if faults:
                last_protocol_content = content
                if attempt < maximum_attempts - 1:
                    time.sleep(2**attempt)
                continue
            return {
                "response": content,
                "finish_reason": finish,
                "server_finish_reason": finish,
                "completion_status": finish,
                "usage": usage,
                "latency_seconds": sum(x["latency_seconds"] for x in attempt_evidence),
                "attempt": attempt + 1,
                "attempt_count": attempt + 1,
                "attempts": attempt_evidence,
                "attempt_evidence_sha256": sha256_bytes(
                    canonical_bytes(attempt_evidence)
                ),
                "request_sha256": request_sha,
                "transport_status": "valid",
                "terminal_metadata_complete": True,
                "usage_metadata_valid": True,
                "reasoning_effort": reasoning_effort,
                "reasoning_output_characters": 0,
                "stop_sequences": stop_sequences,
            }
        if (
            not hard_error_seen
            and len(attempt_evidence) == maximum_attempts
            and all(row["status"] == "protocol_fault" for row in attempt_evidence)
        ):
            return {
                "response": last_protocol_content,
                "finish_reason": TRANSPORT_INCOMPLETE_FINISH_REASON,
                "server_finish_reason": attempt_evidence[-1]["finish_reason"],
                "completion_status": TRANSPORT_INCOMPLETE_FINISH_REASON,
                "usage": None,
                "latency_seconds": sum(x["latency_seconds"] for x in attempt_evidence),
                "attempt": maximum_attempts,
                "attempt_count": maximum_attempts,
                "attempts": attempt_evidence,
                "attempt_evidence_sha256": sha256_bytes(
                    canonical_bytes(attempt_evidence)
                ),
                "request_sha256": request_sha,
                "transport_status": "protocol_invalid",
                "terminal_metadata_complete": False,
                "usage_metadata_valid": False,
                "reasoning_effort": reasoning_effort,
                "reasoning_output_characters": 0,
                "stop_sequences": stop_sequences,
            }
        raise RuntimeError(f"model call failed after {maximum_attempts} attempts: {last_error}")


def completion_request_payload(
    model: str,
    spec: dict[str, Any],
    user_prompt: str,
    sys_prompt: str,
    paradigm: str,
) -> dict[str, Any]:
    """Build the one frozen request payload used identically by every retry."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": spec["scope"]["temperature"],
        "max_tokens": spec["scope"]["max_completion_tokens"],
        "stop": request_stop_sequences(spec, paradigm),
        "extra_body": {"reasoning_effort": request_reasoning_effort(spec)},
    }


def terminal_metadata_faults(
    content: str, finish_reason: str, usage: dict[str, Any] | None
) -> list[str]:
    """Classify retryable HTTP-200 terminal-metadata faults without scoring content."""
    require(isinstance(content, str), "terminal-metadata classifier requires text content")
    require(
        not finish_reason or finish_reason in ALLOWED_FINISH_REASONS,
        f"unsupported completion finish reason: {finish_reason!r}",
    )
    faults: list[str] = []
    if not finish_reason:
        faults.append("missing_finish_reason")
    if usage is None:
        faults.append("missing_usage")
        return faults
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    total_tokens = usage["total_tokens"]
    if prompt_tokens == completion_tokens == total_tokens == 0:
        faults.append("zero_usage")
        return faults
    if prompt_tokens <= 0:
        faults.append("nonpositive_prompt_tokens")
    if total_tokens <= 0:
        faults.append("nonpositive_total_tokens")
    if content and completion_tokens <= 0:
        faults.append("nonempty_content_zero_completion_tokens")
    return faults


def transport_attempt_evidence(
    attempt: int,
    status: str,
    faults: list[str],
    finish_reason: str,
    usage: dict[str, Any] | None,
    content: str,
    latency_seconds: float,
    request_sha256: str,
) -> dict[str, Any]:
    """Persist outcome-blind transport evidence without duplicating response bodies."""
    return {
        "attempt": attempt,
        "status": status,
        "faults": list(faults),
        "finish_reason": finish_reason,
        "usage": usage,
        "content_characters": len(content),
        "content_sha256": sha256_text(content),
        "latency_seconds": latency_seconds,
        "request_sha256": request_sha256,
    }


def normalize_model_content(content: Any) -> str:
    """Protocol-valid empty text is data and scores zero; only non-text is fatal."""
    require(isinstance(content, str), "model response content is not a string")
    return content.strip()


def normalize_completion_usage(usage: Any) -> dict[str, Any] | None:
    """Persist SDK usage as structured JSON rather than an opaque repr string."""
    if usage is None:
        return None
    model_dump = getattr(usage, "model_dump", None)
    payload = model_dump(mode="json") if callable(model_dump) else jsonable(usage)
    require(isinstance(payload, dict), "completion usage is not structurally serializable")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = payload.get(key)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"completion usage has invalid {key}",
        )
    details = payload.get("completion_tokens_details")
    if details is not None:
        require(isinstance(details, dict), "completion token details are not a mapping")
        reasoning_tokens = details.get("reasoning_tokens")
        require(
            reasoning_tokens in (None, 0),
            "server reported nonzero reasoning tokens despite reasoning_effort=none",
        )
    return jsonable(payload)


def require_reported_usage_fits_context(
    spec: dict[str, Any], usage: dict[str, Any] | None, content: str | None = None
) -> None:
    """Bind the server-reported tokenizer count to the same full-budget gate."""
    require_reported_usage_structurally_consistent(spec, usage)
    require(isinstance(usage, dict), "server omitted structured completion usage")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    require(
        isinstance(prompt_tokens, int)
        and not isinstance(prompt_tokens, bool)
        and isinstance(completion_tokens, int)
        and not isinstance(completion_tokens, bool)
        and isinstance(total_tokens, int)
        and not isinstance(total_tokens, bool),
        "server usage token counts are malformed",
    )
    require(prompt_tokens > 0 and total_tokens > 0,
            "server omitted real prompt/total token usage")
    if content:
        require(completion_tokens > 0,
                "nonempty response has zero reported completion tokens")


def require_reported_usage_structurally_consistent(
    spec: dict[str, Any], usage: dict[str, Any] | None
) -> None:
    """Reject non-retryable usage/accounting drift even beside a listed fault."""
    if usage is None:
        return
    require(isinstance(usage, dict), "completion usage is not a mapping")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    require(
        isinstance(prompt_tokens, int)
        and not isinstance(prompt_tokens, bool)
        and isinstance(completion_tokens, int)
        and not isinstance(completion_tokens, bool)
        and isinstance(total_tokens, int)
        and not isinstance(total_tokens, bool),
        "server usage token counts are malformed",
    )
    completion_budget = int(spec["scope"]["max_completion_tokens"])
    context = int(spec["scope"]["served_context_tokens"])
    require(completion_tokens <= completion_budget,
            "reported completion exceeds the frozen completion budget")
    require(prompt_tokens + completion_budget <= context,
            "reported prompt cannot reserve the frozen completion budget")
    require(total_tokens == prompt_tokens + completion_tokens,
            "reported total tokens disagree with prompt plus completion")
    require(total_tokens <= context, "reported request exceeds served context")


def invalid_history_answer(finish_reason: str, response: str) -> str:
    if finish_reason == "length":
        return TRUNCATED_HISTORY_SENTINEL
    if finish_reason == TRANSPORT_INCOMPLETE_FINISH_REASON:
        return TRANSPORT_INVALID_HISTORY_SENTINEL
    return response


def validate_transport_call(
    call: dict[str, Any],
    response: str,
    expected_request_sha256: str,
    spec: dict[str, Any],
) -> None:
    """Replay the v3 logical-call and physical-attempt transport contract."""
    retry_policy = transport_retry_policy(spec)
    maximum_attempts = int(retry_policy["maximum_total_attempts"])
    retryable_faults = set(retry_policy["retryable_http_200_faults"])
    require(call.get("request_sha256") == expected_request_sha256,
            "stored API call request hash mismatch")
    attempt_count = call.get("attempt_count")
    attempts = call.get("attempts")
    require(
        isinstance(attempt_count, int)
        and not isinstance(attempt_count, bool)
        and 1 <= attempt_count <= maximum_attempts
        and call.get("attempt") == attempt_count
        and isinstance(attempts, list)
        and len(attempts) == attempt_count,
        "stored transport attempt accounting is malformed",
    )
    require(
        call.get("attempt_evidence_sha256")
        == sha256_bytes(canonical_bytes(attempts)),
        "stored transport attempt evidence hash mismatch",
    )
    for index, attempt in enumerate(attempts, start=1):
        require(isinstance(attempt, dict), "stored transport attempt is not a mapping")
        require(attempt.get("attempt") == index, "stored transport attempt order mismatch")
        require(attempt.get("request_sha256") == expected_request_sha256,
                "stored transport attempt request hash mismatch")
        status = attempt.get("status")
        faults = attempt.get("faults")
        finish = attempt.get("finish_reason")
        usage = attempt.get("usage")
        characters = attempt.get("content_characters")
        latency = attempt.get("latency_seconds")
        digest = attempt.get("content_sha256")
        require(
            status in {"protocol_fault", "request_error", "accepted"}
            and isinstance(faults, list)
            and all(isinstance(value, str) and value for value in faults)
            and isinstance(finish, str)
            and isinstance(characters, int)
            and not isinstance(characters, bool)
            and characters >= 0
            and isinstance(latency, (int, float))
            and not isinstance(latency, bool)
            and 0 <= float(latency)
            and isinstance(digest, str)
            and len(digest) == 64,
            "stored transport attempt evidence is malformed",
        )
        if status == "request_error":
            require(
                faults == ["request_exception"]
                and finish == ""
                and usage is None
                and characters == 0
                and digest == sha256_text("")
                and isinstance(attempt.get("exception_type"), str)
                and attempt["exception_type"],
                "stored request-error attempt evidence is malformed",
            )
            continue
        normalized_usage = normalize_completion_usage(usage)
        require_reported_usage_structurally_consistent(spec, normalized_usage)
        recomputed_faults = terminal_metadata_faults(
            "X" if characters else "", finish, normalized_usage
        )
        require(set(recomputed_faults).issubset(retryable_faults),
                "stored transport fault lies outside the frozen retry policy")
        require(faults == recomputed_faults,
                "stored transport attempt fault classification mismatch")
        require(
            (status == "protocol_fault") is bool(recomputed_faults),
            "stored transport attempt status mismatch",
        )
        if status == "accepted":
            require(index == attempt_count, "accepted attempt must terminate retry sequence")
            require_reported_usage_fits_context(
                spec, normalized_usage, "X" if characters else ""
            )

    require(
        isinstance(call.get("latency_seconds"), (int, float))
        and not isinstance(call.get("latency_seconds"), bool)
        and abs(
            float(call["latency_seconds"])
            - sum(float(row["latency_seconds"]) for row in attempts)
        ) < 1e-9,
        "stored logical-call latency does not equal attempt latencies",
    )
    final_attempt = attempts[-1]
    require(
        final_attempt["content_characters"] == len(response)
        and final_attempt["content_sha256"] == sha256_text(response),
        "stored terminal-attempt response evidence mismatch",
    )
    transport_status = call.get("transport_status")
    if transport_status == "valid":
        require(
            call.get("terminal_metadata_complete") is True
            and call.get("usage_metadata_valid") is True
            and call.get("finish_reason") in ALLOWED_FINISH_REASONS
            and call.get("completion_status") == call.get("finish_reason")
            and call.get("server_finish_reason") == call.get("finish_reason")
            and call.get("finish_reason") == final_attempt["finish_reason"]
            and call.get("usage") == final_attempt["usage"]
            and final_attempt["status"] == "accepted"
            and all(
                row["status"] in {"protocol_fault", "request_error"}
                for row in attempts[:-1]
            ),
            "stored valid transport-call contract mismatch",
        )
        require_reported_usage_fits_context(spec, call.get("usage"), response)
        return
    require(
        transport_status == "protocol_invalid"
        and attempt_count == maximum_attempts
        and call.get("finish_reason") == TRANSPORT_INCOMPLETE_FINISH_REASON
        and call.get("completion_status") == TRANSPORT_INCOMPLETE_FINISH_REASON
        and call.get("server_finish_reason") == final_attempt["finish_reason"]
        and call.get("usage") is None
        and call.get("terminal_metadata_complete") is False
        and call.get("usage_metadata_valid") is False
        and all(row["status"] == "protocol_fault" for row in attempts),
        "stored protocol-invalid transport-call contract mismatch",
    )


def _multiturn_prompt(
    history_lines: list[str], index: int, shown: str, item: Any, spec: dict[str, Any]
) -> str:
    kept = (
        [history_lines[0]] + history_lines[-MULTITURN_HISTORY_LINES:]
        if len(history_lines) > MULTITURN_HISTORY_LINES + 1
        else history_lines
    )
    prompt = "\n".join(kept) + f"\nTrial {index + 1}: {shown}\nYour response:"
    prompt = format_user_prompt(spec, item.metadata.paradigm, prompt)
    require(
        len(prompt) // 4 <= int(spec["scope"]["served_context_tokens"]),
        f"prompt exceeds frozen context: {item.task_id}",
    )
    return prompt


def validate_response_transcript(
    item: Any,
    response_payload: str | list[dict[str, Any]],
    api_calls: list[dict[str, Any]],
    spec: dict[str, Any],
    *,
    model: str | None = None,
    sys_prompt: str | None = None,
) -> None:
    """Independently replay exact prompts, attempts, and invalid-call history."""
    require(isinstance(api_calls, list) and api_calls, "missing API call transcript")
    require(all(isinstance(call, dict) for call in api_calls), "non-dict API call transcript")
    require(
        all(call.get("finish_reason") in STORED_FINISH_REASONS for call in api_calls),
        "unsupported completion finish reason in stored transcript",
    )
    reasoning_effort = request_reasoning_effort(spec)
    stop_sequences = request_stop_sequences(spec, item.metadata.paradigm)
    terminator = response_terminator(spec)
    require(
        all(call.get("reasoning_effort") == reasoning_effort for call in api_calls),
        "stored API call does not bind the frozen reasoning-effort request",
    )
    require(
        all(call.get("stop_sequences") == stop_sequences for call in api_calls),
        "stored API call does not bind the frozen stop sequence",
    )
    require(
        all(call.get("reasoning_output_characters") == 0 for call in api_calls),
        "stored API call contains an exposed reasoning channel",
    )
    require((model is None) is (sys_prompt is None),
            "transcript request replay needs both model and system prompt")
    turns = item.metadata.parameters.get("turns", [])
    if not turns:
        require(isinstance(response_payload, str), "static transcript has non-string response")
        require(terminator not in response_payload,
                "transport terminator leaked into stored static response")
        require(len(api_calls) == 1, "static transcript API-call count mismatch")
        static_prompt = format_user_prompt(spec, item.metadata.paradigm, item.stimulus)
        require(
            api_calls[0].get("prompt_sha256") == sha256_text(static_prompt),
            "static transcript prompt hash mismatch",
        )
        if model is not None and sys_prompt is not None:
            require_request_fits_context(spec, static_prompt, sys_prompt)
            expected_request_sha = sha256_bytes(canonical_bytes(completion_request_payload(
                model, spec, static_prompt, sys_prompt, item.metadata.paradigm
            )))
            validate_transport_call(
                api_calls[0], response_payload, expected_request_sha, spec
            )
        return

    require(
        isinstance(response_payload, list)
        and len(response_payload) == len(turns)
        and len(api_calls) == len(turns),
        "multi-turn transcript length mismatch",
    )
    history_lines = [item.stimulus]
    for index, (turn, response_entry, call) in enumerate(
        zip(turns, response_payload, api_calls)
    ):
        require(isinstance(response_entry, dict), "non-dict multi-turn response entry")
        response = response_entry.get("response")
        require(isinstance(response, str), "non-string multi-turn response")
        require(terminator not in response,
                "transport terminator leaked into stored multi-turn response")
        shown = turn_shown_text(turn)
        prompt = _multiturn_prompt(history_lines, index, shown, item, spec)
        require(response_entry.get("trial") == index + 1, "multi-turn trial index mismatch")
        require(
            response_entry.get("stimulus_sha256") == sha256_text(shown),
            "multi-turn stimulus hash mismatch",
        )
        require(
            call.get("prompt_sha256") == sha256_text(prompt),
            "multi-turn prompt/history hash mismatch",
        )
        if model is not None and sys_prompt is not None:
            require_request_fits_context(spec, prompt, sys_prompt)
            expected_request_sha = sha256_bytes(canonical_bytes(completion_request_payload(
                model, spec, prompt, sys_prompt, item.metadata.paradigm
            )))
            validate_transport_call(call, response, expected_request_sha, spec)
        history_lines.append(f"Trial {index + 1}: {shown}")
        history_answer = invalid_history_answer(call["finish_reason"], response)
        history_lines.append(f"Your response: {history_answer}")
        feedback = turn.get("feedback", turn.get("correct_answer", ""))
        if feedback:
            history_lines.append(f"Feedback: {feedback}")


def evaluate_item(client: LocalChatClient, item: Any, sys_prompt: str, spec: dict) -> tuple[Any, list]:
    turns = item.metadata.parameters.get("turns", [])
    if not turns:
        prompt = format_user_prompt(spec, item.metadata.paradigm, item.stimulus)
        call = client.call(prompt, sys_prompt, item.metadata.paradigm)
        call["prompt_sha256"] = sha256_text(prompt)
        return call.pop("response"), [call]

    history_lines = [item.stimulus]
    responses = []
    api_calls = []
    for index, turn in enumerate(turns):
        shown = turn_shown_text(turn)
        prompt = _multiturn_prompt(history_lines, index, shown, item, spec)
        call = client.call(prompt, sys_prompt, item.metadata.paradigm)
        answer = call.pop("response")
        call["prompt_sha256"] = sha256_text(prompt)
        api_calls.append(call)
        responses.append(
            {
                "trial": index + 1,
                "stimulus_sha256": sha256_text(str(shown)),
                "response": answer,
            }
        )
        feedback = turn.get("feedback", turn.get("correct_answer", ""))
        history_lines.append(f"Trial {index + 1}: {shown}")
        # A partial completion must not be copied into later-turn context. It
        # can be arbitrarily long and can contain unstable fragments. Keep the
        # private raw response, but condition every subsequent turn on one
        # fixed, predeclared invalid-response sentinel.
        history_answer = invalid_history_answer(call["finish_reason"], answer)
        history_lines.append(f"Your response: {history_answer}")
        if feedback:
            history_lines.append(f"Feedback: {feedback}")
    return responses, api_calls


def result_path(model_root: Path, condition_id: str, entry: dict[str, Any]) -> Path:
    return model_root / condition_id / entry["paradigm"] / f"{entry['task_id']}.json"


def validate_record(
    record: dict[str, Any],
    *,
    model: str,
    profile: str,
    condition: dict[str, Any],
    entry: dict[str, Any],
    item: Any,
    spec_sha: str,
    manifest_sha: str,
    served_digest: str,
    source_revision: str,
    spec: dict[str, Any],
) -> None:
    require(record.get("schema_version") == RESULT_SCHEMA, "result schema mismatch")
    expected_identity = {
        "study_id": spec["study_id"],
        "profile": profile,
        "model_id": model,
        "condition_id": condition["id"],
        "task_id": entry["task_id"],
        "paradigm": entry["paradigm"],
        "difficulty": entry["difficulty"],
        "item_fingerprint_sha256": entry["item_fingerprint_sha256"],
        "scoring_gold_sha256": entry["scoring_gold_sha256"],
        "spec_sha256": spec_sha,
        "item_manifest_sha256": manifest_sha,
        "served_model_digest": served_digest,
        "source_revision": source_revision,
        "reasoning_effort": request_reasoning_effort(spec),
        "stop_policy": request_stop_policy(spec),
        "scorer_contract": SCORER_CONTRACT_VERSION,
        "system_prompt_sha256": sha256_text(system_prompt(spec, condition["id"])),
        "system_prompt_characters": len(system_prompt(spec, condition["id"])),
        "system_prompt_whitespace_tokens": len(system_prompt(spec, condition["id"]).split()),
    }
    for key, expected in expected_identity.items():
        require(record.get(key) == expected, f"record {key} mismatch for {entry['task_id']}")
    payload = record.get("responses") if entry["is_multiturn"] else record.get("response")
    if entry["is_multiturn"]:
        require(isinstance(payload, list) and len(payload) == entry["n_turns"], "bad turn payload")
        require(all(isinstance(x.get("response"), str) for x in payload), "bad turn response")
    else:
        require(isinstance(payload, str), "bad static response")
    calls = record.get("api_calls")
    require(isinstance(calls, list) and len(calls) == (entry["n_turns"] or 1), "bad API call log")
    validate_response_transcript(
        item,
        payload,
        calls,
        spec,
        model=model,
        sys_prompt=system_prompt(spec, condition["id"]),
    )
    replay = jsonable(score_response_with_completion_contract(item, payload, calls))
    require(canonical_bytes(replay) == canonical_bytes(record.get("score")), "scorer replay mismatch")
    accuracy = ensure_finite_accuracy(replay["primary_accuracy"])
    truncated_calls = sum(call.get("finish_reason") == "length" for call in calls)
    transport_invalid_calls = sum(
        call.get("finish_reason") == TRANSPORT_INCOMPLETE_FINISH_REASON for call in calls
    )
    invalid_calls = sum(
        is_protocol_invalid_finish_reason(call.get("finish_reason")) for call in calls
    )
    completion_contract = replay["metrics"].get("completion_contract")
    require(
        isinstance(completion_contract, dict)
        and completion_contract.get("truncated_call_count") == truncated_calls
        and completion_contract.get("transport_protocol_invalid_call_count")
        == transport_invalid_calls
        and completion_contract.get("invalid_call_count") == invalid_calls
        and completion_contract.get("task_invalidated") is (invalid_calls > 0)
        and completion_contract.get("native_scorer_evaluated") is (invalid_calls == 0),
        "completion-contract replay mismatch",
    )
    require(not invalid_calls or accuracy == 0.0,
            "protocol-invalid task record did not receive zero primary accuracy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", choices=("formal", "pilot"), required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "raw")
    args = parser.parse_args()

    gpu = assert_slurm_gpu_context()
    spec = load_spec()
    reasoning_effort = request_reasoning_effort(spec)
    stop_policy = request_stop_policy(spec)
    require(
        os.environ.get("COGARENA_REASONING_EFFORT") == reasoning_effort,
        "launcher reasoning-effort environment differs from the frozen specification",
    )
    require(
        os.environ.get("COGARENA_STOP_MODE") == "format_routed",
        "launcher response-format environment differs from the frozen specification",
    )
    execution_node = enforce_frozen_hardware(spec, gpu)
    source_revision = os.environ.get("COGARENA_GIT_HEAD", "").strip()
    require(
        len(source_revision) == 40 and all(c in "0123456789abcdef" for c in source_revision),
        "COGARENA_GIT_HEAD must be injected at sbatch submission; batch-node git is forbidden",
    )
    if args.profile == "formal":
        require(
            spec.get("status") == "formal_frozen_after_pilot",
            "formal run refused: pre-pilot specification has not passed the freeze gate",
        )
        require(spec.get("pilot_gate_manifest_sha256"), "formal freeze lacks pilot gate binding")
        require(spec.get("capacity_gate_manifest_sha256"),
                "formal freeze lacks the frozen capacity gate binding")
    models = {x["model"]: x for x in profile_models(spec, args.profile)}
    require(args.model in models, f"model {args.model} is not in the {args.profile} panel")
    other_profile = "pilot" if args.profile == "formal" else "formal"
    require(
        args.model not in {x["model"] for x in profile_models(spec, other_profile)},
        "pilot/formal model overlap",
    )

    manifest_file = (args.manifest or manifest_path(args.profile)).resolve()
    require(manifest_file == manifest_path(args.profile).resolve(), "noncanonical item manifest refused")
    manifest = load_json(manifest_file)
    require(manifest.get("profile") == args.profile, "manifest profile mismatch")
    spec_sha = sha256_file(SPEC_PATH)
    manifest_sha = sha256_file(manifest_file)
    require(manifest.get("spec_sha256") == spec_sha, "manifest/spec hash mismatch")
    require(manifest["task_record_count_per_model"] == manifest["item_count"] * len(spec["conditions"]), "bad record count")
    items = reconstruct_items(spec, manifest)

    model_root = args.output_dir.resolve() / args.profile / model_safe(args.model)
    guard = acquire_execution_guard(
        model_root,
        study_id=spec["study_id"],
        profile=args.profile,
        model=args.model,
        source_revision=source_revision,
        spec_sha256=spec_sha,
        item_manifest_sha256=manifest_sha,
        execution_node=execution_node,
    )
    guard_path = model_root / EXECUTION_GUARD_FILENAME
    serving = capture_serving_provenance(
        args.model, model_root, gpu, reasoning_effort, stop_policy
    )
    serving_path = model_root / "serving_provenance.json"
    digest = serving["tag"]["digest"]
    client = LocalChatClient(args.model, spec)

    entries = {x["task_id"]: x for x in manifest["items"]}
    conditions = condition_map(spec)
    schedule = [(task_id, cid) for task_id in entries for cid in conditions]
    schedule_seed = stable_seed(spec["study_id"], args.profile, args.model, "schedule-v1")
    random.Random(schedule_seed).shuffle(schedule)
    completed = 0
    started = datetime.now(timezone.utc).isoformat()
    record_paths: list[Path] = []
    for schedule_index, (task_id, condition_id) in enumerate(schedule):
        entry = entries[task_id]
        item = items[task_id]
        condition = conditions[condition_id]
        path = result_path(model_root, condition_id, entry)
        require(not path.exists(), f"fresh execution produced a duplicate record path: {path}")

        prompt = system_prompt(spec, condition_id)
        payload, api_calls = evaluate_item(client, item, prompt, spec)
        score = jsonable(score_response_with_completion_contract(item, payload, api_calls))
        record = {
            "schema_version": RESULT_SCHEMA,
            "study_id": spec["study_id"],
            "profile": args.profile,
            "model_id": args.model,
            "family": models[args.model]["family"],
            "served_model_digest": digest,
            "source_revision": source_revision,
            "reasoning_effort": reasoning_effort,
            "stop_policy": stop_policy,
            "condition_id": condition_id,
            "condition_kind": condition["kind"],
            "target_group": condition["target_group"],
            "system_prompt_sha256": sha256_text(prompt),
            "system_prompt_characters": len(prompt),
            "system_prompt_whitespace_tokens": len(prompt.split()),
            "task_id": task_id,
            "dimension": entry["dimension"],
            "group": entry["group"],
            "paradigm": entry["paradigm"],
            "difficulty": entry["difficulty"],
            "item_fingerprint_sha256": entry["item_fingerprint_sha256"],
            "scoring_gold_sha256": entry["scoring_gold_sha256"],
            "presentation_sha256": entry["presentation_sha256"],
            "spec_sha256": spec_sha,
            "item_manifest_sha256": manifest_sha,
            "scorer_contract": SCORER_CONTRACT_VERSION,
            "schedule_seed": schedule_seed,
            "schedule_index": schedule_index,
            "response" if not entry["is_multiturn"] else "responses": payload,
            "api_calls": api_calls,
            "score": score,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        validate_record(
            record, model=args.model, profile=args.profile, condition=condition, entry=entry,
            item=item, spec_sha=spec_sha, manifest_sha=manifest_sha, served_digest=digest, spec=spec,
            source_revision=source_revision,
        )
        atomic_write_json(path, record)
        record_paths.append(path)
        completed += 1
        if completed % 25 == 0:
            print(f"{args.model}: {completed}/{len(schedule)} (fresh-only)", flush=True)

    require(completed == len(schedule), "fresh model execution did not complete its schedule")
    require(set(record_paths) == {
        result_path(model_root, condition_id, entries[task_id])
        for task_id, condition_id in schedule
    }, "fresh model execution record set mismatch")
    record_tree_sha256 = tree_hash(record_paths, model_root)
    serving_sha256 = sha256_file(serving_path)
    completed_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": "cogarena.causal_selectivity.run_summary.v3",
        "study_id": spec["study_id"],
        "profile": args.profile,
        "model": args.model,
        "family": models[args.model]["family"],
        "served_model_digest": digest,
        "source_revision": source_revision,
        "reasoning_effort": reasoning_effort,
        "stop_policy": stop_policy,
        "spec_sha256": spec_sha,
        "item_manifest_sha256": manifest_sha,
        "schedule_seed": schedule_seed,
        "expected_records": len(schedule),
        "new_records": completed,
        "record_reuse_allowed": False,
        "execution_guard_identity_sha256": guard["guard_identity_sha256"],
        "record_tree_sha256": record_tree_sha256,
        "serving_provenance_sha256": serving_sha256,
        "started_at": started,
        "completed_at": completed_at,
        "status": "records_complete_pending_independent_replay",
    }
    summary_path = model_root / "run_summary.json"
    atomic_write_json(summary_path, summary)
    expected_json = set(record_paths) | {guard_path, serving_path, summary_path}
    actual_json = set(model_root.rglob("*.json"))
    require(
        actual_json == expected_json,
        "fresh model output contains missing/extra JSON before guard completion",
    )
    temporary = list(model_root.rglob("*.tmp")) + list(model_root.rglob(".*.tmp"))
    require(not temporary, "temporary files remain before guard completion")
    complete_execution_guard(
        model_root,
        guard,
        record_paths=record_paths,
        serving_path=serving_path,
        run_summary_path=summary_path,
        records_completed_at=completed_at,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
