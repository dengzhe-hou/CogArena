"""Replication runner with a model-bound transport compatibility shim.

Repeated execution established that OLMo2 7B can return only the requested
delimiter while reporting ``finish_reason=stop`` and leaving the stop sequence
in the OpenAI-compatible response body. The semantic response is empty.

The parent runner remains fail-closed. This shim normalizes only an exact
delimiter-only body from OLMo2 7B and writes a separate incident record for
each affected request. Any body containing both answer text and the delimiter,
and every delimiter occurrence from another model, still reaches the parent
leakage guard and fails.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scripts.experiments.causal_selectivity_20260720 import run_model as parent
from scripts.experiments.causal_selectivity_20260720.common import (
    RESULTS_ROOT,
    atomic_write_json,
    canonical_bytes,
    load_json,
    response_terminator,
    sha256_bytes,
    sha256_text,
)


EXPECTED_MODEL = "olmo2:7b"
INCIDENT_SCHEMA = "cogarena.scaffold_wording.transport_incident.v4"


def normalize_olmo2_transport_content(
    content: str,
    *,
    model: str,
    request_sha256: str,
    spec: dict[str, Any],
) -> tuple[str, bool]:
    """Map an OLMo2 7B delimiter-only body to an empty response."""
    terminator = response_terminator(spec)
    if model != EXPECTED_MODEL:
        return content, False
    if content == terminator:
        return "", True
    return content, False


def incident_path(request_sha256: str) -> Path:
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "").strip()
    if not array_job_id:
        raise RuntimeError("transport incident provenance requires SLURM_ARRAY_JOB_ID")
    return (
        RESULTS_ROOT
        / "transport_incidents"
        / f"array_{array_job_id}"
        / f"olmo2__7b_{request_sha256[:16]}.json"
    )


def write_incident(
    raw_content: str,
    normalized_content: str,
    request_sha256: str,
) -> None:
    payload = {
        "schema_version": INCIDENT_SCHEMA,
        "study_id": "scaffold_wording_replication_20260725",
        "model": EXPECTED_MODEL,
        "request_sha256": request_sha256,
        "raw_content_sha256": sha256_text(raw_content),
        "raw_content_characters": len(raw_content),
        "raw_shape": "exact_transport_terminator_only",
        "normalized_content_sha256": sha256_text(normalized_content),
        "normalized_content_characters": len(normalized_content),
        "normalization": "exact_delimiter_only_to_empty_response",
        "slurm_array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    path = incident_path(request_sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if load_json(path) != payload:
            raise RuntimeError("transport incident provenance drift")
        return
    atomic_write_json(path, payload)


def main() -> None:
    base_normalize = parent.normalize_model_content
    base_call = parent.LocalChatClient.call
    active: dict[str, Any] = {}

    def normalize_with_shim(value: Any) -> str:
        content = base_normalize(value)
        normalized, changed = normalize_olmo2_transport_content(
            content,
            model=str(active.get("model", "")),
            request_sha256=str(active.get("request_sha256", "")),
            spec=active["spec"],
        )
        if changed:
            write_incident(
                content,
                normalized,
                str(active["request_sha256"]),
            )
        return normalized

    def call_with_context(
        self: parent.LocalChatClient,
        user_prompt: str,
        sys_prompt: str,
        paradigm: str,
    ) -> dict[str, Any]:
        request = parent.completion_request_payload(
            self.model, self.spec, user_prompt, sys_prompt, paradigm
        )
        active.clear()
        active.update(
            {
                "model": self.model,
                "request_sha256": sha256_bytes(canonical_bytes(request)),
                "spec": self.spec,
            }
        )
        try:
            return base_call(self, user_prompt, sys_prompt, paradigm)
        finally:
            active.clear()

    parent.normalize_model_content = normalize_with_shim
    parent.LocalChatClient.call = call_with_context
    parent.main()


if __name__ == "__main__":
    main()
