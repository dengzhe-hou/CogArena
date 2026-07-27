#!/usr/bin/env python3
"""Close the frozen serving-capacity gate over every eligible GPU class."""

from __future__ import annotations

import os
import socket
from datetime import datetime, timezone

from .common import (
    RESULTS_ROOT,
    SPEC_PATH,
    atomic_write_json,
    load_json,
    load_spec,
    manifest_path,
    request_reasoning_effort,
    request_stop_policy,
    require,
    sha256_file,
)


def main() -> None:
    require(os.environ.get("SLURM_JOB_ID"), "capacity closure must run in Slurm")
    require(socket.gethostname().split(".", 1)[0].startswith("c01"),
            "capacity closure must execute on c01")
    injected_revision = os.environ.get("COGARENA_GIT_HEAD", "")
    require(len(injected_revision) == 40
            and all(c in "0123456789abcdef" for c in injected_revision),
            "COGARENA_GIT_HEAD must be a full frozen revision")
    spec = load_spec()
    reasoning_effort = request_reasoning_effort(spec)
    stop_policy = request_stop_policy(spec)
    required = spec["capacity_probe"]["required_hardware_labels"]
    capacity_root = RESULTS_ROOT / "capacity"
    require(capacity_root.is_dir(), "missing capacity-probe root")
    actual_labels = {path.name for path in capacity_root.iterdir() if path.is_dir()}
    require(actual_labels == set(required),
            f"unexpected or missing capacity hardware directories: {actual_labels ^ set(required)}")
    probes = []
    revisions = set()
    digests = set()
    execution_nodes = set()
    allowed_nodes = spec["execution_contract"]["eligible_inference_nodes"]
    required_gpu = spec["execution_contract"]["required_gpu_name_fragment"]
    for label, expected_name in required.items():
        path = RESULTS_ROOT / "capacity" / label / "CAPACITY_PROBE.json"
        require(path.is_file(), f"missing capacity probe: {label}")
        probe = load_json(path)
        require(probe.get("schema_version") == "cogarena.causal_selectivity.capacity_probe.v1"
                and probe.get("status") == "pass" and probe.get("hardware_label") == label,
                f"bad capacity probe identity: {label}")
        require(probe.get("model") == spec["capacity_probe"]["model"], "capacity model drift")
        require(expected_name.lower() in probe.get("nvidia_smi", "").lower(),
                f"capacity GPU class mismatch: {label}")
        require(probe.get("actual_context_tokens") == spec["scope"]["served_context_tokens"],
                f"capacity context mismatch: {label}")
        require(probe.get("fully_gpu_served") is True and probe.get("processor") == "100% GPU",
                f"capacity probe was not fully GPU-served: {label}")
        require(probe.get("execution_node") in allowed_nodes
                and probe.get("required_gpu_name_fragment") == required_gpu,
                f"capacity probe used hardware outside the frozen scope: {label}")
        require(probe.get("spec_sha256") == sha256_file(SPEC_PATH), "capacity spec hash drift")
        require(probe.get("pilot_item_manifest_sha256") == sha256_file(manifest_path("pilot")),
                "capacity source-manifest hash drift")
        require(probe.get("reasoning_effort") == reasoning_effort,
                f"capacity reasoning-effort contract mismatch: {label}")
        require(probe.get("reasoning_request_verified") is True,
                f"capacity reasoning-effort request was not verified: {label}")
        require(probe.get("stop_policy") == stop_policy
                and probe.get("stop_sequence_request_verified") is True,
                f"capacity response-format policy was not verified: {label}")
        serving_path = RESULTS_ROOT / "capacity" / label / "serving_provenance.json"
        require(probe.get("serving_provenance_sha256") == sha256_file(serving_path),
                f"capacity serving provenance drift: {label}")
        serving = load_json(serving_path)
        require(serving.get("fully_gpu_served") is True
                and serving.get("processor") == "100% GPU"
                and serving.get("reasoning_effort") == reasoning_effort
                and serving.get("stop_policy") == stop_policy
                and serving.get("actual_context_tokens") == spec["scope"]["served_context_tokens"],
                f"capacity serving provenance lacks exact full-GPU/reasoning proof: {label}")
        sessions = serving.get("sessions")
        require(isinstance(sessions, list) and sessions
                and all(session.get("fully_gpu_served") is True
                        and session.get("processor") == "100% GPU"
                        and session.get("reasoning_effort") == reasoning_effort
                        and session.get("stop_policy") == stop_policy
                        and session.get("actual_context_tokens") == spec["scope"]["served_context_tokens"]
                        and str(session.get("hostname", "")).split(".", 1)[0] in allowed_nodes
                        and required_gpu.lower() in str(session.get("nvidia_smi", "")).lower()
                        for session in sessions),
                f"capacity serving session is not fully GPU-served: {label}")
        execution_nodes.update(
            str(session["hostname"]).split(".", 1)[0] for session in sessions
        )
        revisions.add(probe["source_revision"])
        digests.add(probe["served_model_digest"])
        probes.append({"label": label, "path": str(path), "sha256": sha256_file(path)})
    require(len(revisions) == 1, f"mixed capacity source revisions: {revisions}")
    require(revisions == {injected_revision},
            "capacity probes do not match the closure source revision")
    require(len(digests) == 1, f"mixed capacity model digests: {digests}")
    gate = {
        "schema_version": "cogarena.causal_selectivity.capacity_gate.v1",
        "study_id": spec["study_id"],
        "model": spec["capacity_probe"]["model"],
        "source_revision": next(iter(revisions)),
        "reasoning_effort": reasoning_effort,
        "reasoning_request_verified": True,
        "stop_policy": stop_policy,
        "stop_sequence_request_verified": True,
        "spec_sha256": sha256_file(SPEC_PATH),
        "pilot_item_manifest_sha256": sha256_file(manifest_path("pilot")),
        "served_model_digest": next(iter(digests)),
        "actual_context_tokens": spec["scope"]["served_context_tokens"],
        "processor_requirement": "100% GPU",
        "execution_nodes": sorted(execution_nodes),
        "required_gpu_name_fragment": required_gpu,
        "all_probes_fully_gpu_served": True,
        "hardware_labels": list(required),
        "probes": probes,
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
    }
    atomic_write_json(RESULTS_ROOT / "CAPACITY_GATE_MANIFEST.json", gate)
    print("CAPACITY GATE PASS: " + " + ".join(required))


if __name__ == "__main__":
    main()
