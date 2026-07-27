#!/usr/bin/env python3
"""Record the frozen nonformal GPU/context capacity probe."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from .common import (
    RESULTS_ROOT,
    SPEC_PATH,
    atomic_write_json,
    load_spec,
    manifest_path,
    request_reasoning_effort,
    request_stop_policy,
    require,
    sha256_file,
)
from .preflight import validate_manifest_sources, validate_source_revision
from .run_model import (
    assert_slurm_gpu_context,
    capture_serving_provenance,
    enforce_frozen_hardware,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-label", required=True)
    args = parser.parse_args()
    spec = load_spec()
    reasoning_effort = request_reasoning_effort(spec)
    stop_policy = request_stop_policy(spec)
    require(
        os.environ.get("COGARENA_REASONING_EFFORT") == reasoning_effort,
        "capacity launcher reasoning-effort environment differs from specification",
    )
    require(
        os.environ.get("COGARENA_STOP_MODE") == "format_routed",
        "capacity launcher response-format environment differs from specification",
    )
    require(args.hardware_label in spec["capacity_probe"]["required_hardware_labels"],
            f"hardware label is not formally eligible: {args.hardware_label}")
    revision = os.environ.get("COGARENA_GIT_HEAD", "")
    validate_source_revision(spec, "pilot", revision)
    validate_manifest_sources(spec, "pilot")
    model = spec["capacity_probe"]["model"]
    require(model not in {x["model"] for x in spec["formal_model_panel"]},
            "capacity probe model leaked into formal panel")
    gpu = assert_slurm_gpu_context()
    enforce_frozen_hardware(spec, gpu)
    expected_name = spec["capacity_probe"]["required_hardware_labels"][args.hardware_label]
    require(expected_name.lower() in gpu["nvidia_smi"].lower(),
            f"allocated GPU does not match {args.hardware_label}: {gpu['nvidia_smi']}")

    root = RESULTS_ROOT / "capacity" / args.hardware_label
    serving = capture_serving_provenance(
        model, root, gpu, reasoning_effort, stop_policy
    )
    probe = {
        "schema_version": "cogarena.causal_selectivity.capacity_probe.v1",
        "study_id": spec["study_id"],
        "hardware_label": args.hardware_label,
        "expected_gpu_name_fragment": expected_name,
        "nvidia_smi": gpu["nvidia_smi"],
        "model": model,
        "served_model_digest": serving["tag"]["digest"],
        "processor": serving["processor"],
        "fully_gpu_served": serving["fully_gpu_served"],
        "actual_context_tokens": serving["actual_context_tokens"],
        "execution_node": serving["sessions"][-1]["hostname"].split(".", 1)[0],
        "required_gpu_name_fragment": spec["execution_contract"][
            "required_gpu_name_fragment"
        ],
        "source_revision": revision,
        "reasoning_effort": reasoning_effort,
        "stop_policy": stop_policy,
        "reasoning_request_verified": True,
        "stop_sequence_request_verified": True,
        "spec_sha256": sha256_file(SPEC_PATH),
        "pilot_item_manifest_sha256": sha256_file(manifest_path("pilot")),
        "serving_provenance_sha256": sha256_file(root / "serving_provenance.json"),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
    }
    require(probe["actual_context_tokens"] == spec["scope"]["served_context_tokens"],
            "capacity probe context mismatch")
    require(probe["fully_gpu_served"] is True and probe["processor"] == "100% GPU",
            "capacity probe did not fully offload the selected model to GPU")
    atomic_write_json(root / "CAPACITY_PROBE.json", probe)
    print(f"CAPACITY PASS {args.hardware_label} {model} {gpu['nvidia_smi']}")


if __name__ == "__main__":
    main()
