#!/usr/bin/env python3
"""Build the frozen exploratory scaffold-wording replication specification.

The replication changes only the five targeted scaffold wordings. It retains
the original baseline, neutral placebo, model panel, held-out items, transport
contract, scorers, and analysis implementation. Results are exploratory and
cannot update the original all-nine-gate confirmation decision.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PARENT_SPEC = (
    ROOT
    / "scripts"
    / "experiments"
    / "causal_selectivity_20260720"
    / "PREPILOT_SPEC.json"
)
DEFAULT_OUTPUT = Path(__file__).with_name("SPEC.json")
EXPECTED_PARENT_SPEC_SHA256 = (
    "8367e0db2e9bd03a4fa3d3dc283b0be955203f938b889550e3a61a7b45b0ff40"
)
PARENT_STUDY_ID = "causal_selectivity_20260720"
STUDY_ID = "scaffold_wording_replication_20260725"

ALTERNATIVE_SCAFFOLDS = {
    "working_memory_ledger": (
        "Before each response, silently track the active symbols, positions, "
        "and intermediate values in their exact order. Refresh this ordered "
        "record after every trial, prevent later material from overwriting "
        "earlier positions, and check the record before answering. Preserve "
        "the requested output format and never reveal the record or hidden reasoning."
    ),
    "control_rule_rehearsal": (
        "Before each response, silently identify the active rule and the feature "
        "that determines the answer. Carefully ignore conflicting or habitual "
        "cues, verify the candidate response against the current rule, and answer "
        "only after this check. Preserve the requested output format and never "
        "reveal this procedure or hidden reasoning."
    ),
    "episodic_source_binding": (
        "Before each response, silently link every presented element to where "
        "and when it appeared across trials. Carefully separate similar events, "
        "distinguish genuinely presented content from merely familiar content, "
        "and retrieve the answer from these links. Preserve the requested output "
        "format and never reveal the links or hidden reasoning."
    ),
    "belief_state_ledger": (
        "Before each response, silently represent each agent's observations, "
        "goals, and beliefs separately. Change an agent's state only when that "
        "agent receives new evidence, and answer from the perspective named in "
        "the question rather than your own. Preserve the requested output format "
        "and never reveal these representations or hidden reasoning."
    ),
    "metacognitive_forecast": (
        "Before each response, silently estimate how strongly the available "
        "evidence supports the candidate answer. Separate knowing from guessing, "
        "align confidence or wagering with that estimate and the stated payoff, "
        "and make one check for unsupported certainty. Preserve the requested "
        "output format and never reveal the estimate or hidden reasoning."
    ),
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build() -> dict[str, Any]:
    actual_parent_sha = _sha256_file(PARENT_SPEC)
    if actual_parent_sha != EXPECTED_PARENT_SPEC_SHA256:
        raise RuntimeError(
            "parent causal-selectivity specification drifted: "
            f"{actual_parent_sha} != {EXPECTED_PARENT_SPEC_SHA256}"
        )
    with PARENT_SPEC.open(encoding="utf-8") as handle:
        parent = json.load(handle)
    if parent.get("study_id") != PARENT_STUDY_ID:
        raise RuntimeError("unexpected parent study identity")

    spec = copy.deepcopy(parent)
    spec["study_id"] = STUDY_ID
    spec["authored_on"] = "2026-07-25"
    spec["frozen_at"] = "2026-07-25T05:45:00Z"
    spec["repository_base_revision"] = (
        "482b9d73349441df15761efcc97334972b07ece2"
    )
    spec["purpose"] = (
        "Exploratorily test whether the matched-scaffold tendency replicates "
        "under a second, semantically equivalent wording of every targeted scaffold."
    )
    spec["analysis_role"] = (
        "post_hoc_exploratory_wording_replication_not_part_of_the_parent_"
        "confirmation_decision"
    )
    spec["parent_study"] = {
        "study_id": PARENT_STUDY_ID,
        "spec_sha256": EXPECTED_PARENT_SPEC_SHA256,
        "formal_item_manifest_sha256": (
            "0b8a54402104bcb266d06b7ee3c281d225876107e4b41e9c50c51bf8917202a0"
        ),
        "pilot_manifest_sha256": (
            "eadfc1c52eab95a40f40d8fcba965f6d6b83d2982e2d27061d0ed60c42cf55c1"
        ),
        "released_capacity_manifest_sha256": (
            "7d63e46146e4d86bbb8fd3b6daed69e1530043503c00efab5912cc41a3ffdb25"
        ),
        "operational_gate_inheritance": (
            "The parent pilot and capacity gate are reused only for transport, "
            "response-format, context-window, and full-GPU feasibility because "
            "the panel, items, response formats, hardware, context, and prompt "
            "length envelope are unchanged. No parent outcome is used as a gate."
        ),
    }
    spec["result_interpretation"] = {
        "confirmatory_status": "none",
        "paper_inclusion": "optional_after_separate_review",
        "permitted_claim": (
            "wording-replication sensitivity for the aggregate matched tendency"
        ),
        "forbidden_claim": (
            "confirmation, preregistration, or replacement of the frozen parent decision"
        ),
    }

    conditions = {row["id"]: row for row in spec["conditions"]}
    if set(ALTERNATIVE_SCAFFOLDS) != {
        row["id"] for row in spec["conditions"] if row["kind"] == "targeted"
    }:
        raise RuntimeError("targeted-condition identity drift")
    for condition_id, scaffold in ALTERNATIVE_SCAFFOLDS.items():
        conditions[condition_id]["scaffold"] = scaffold
        conditions[condition_id]["wording_replication_of"] = {
            "parent_condition_id": condition_id,
            "parent_scaffold_sha256": hashlib.sha256(
                next(
                    row["scaffold"]
                    for row in parent["conditions"]
                    if row["id"] == condition_id
                ).encode("utf-8")
            ).hexdigest(),
            "version": "alternative_wording_1",
        }

    parent_conditions = {row["id"]: row for row in parent["conditions"]}
    for control_id in ("baseline", "neutral_placebo"):
        if conditions[control_id] != parent_conditions[control_id]:
            raise RuntimeError(f"{control_id} changed in wording replication")

    placebo_tokens = len(conditions["neutral_placebo"]["scaffold"].split())
    for condition_id in ALTERNATIVE_SCAFFOLDS:
        delta = abs(len(conditions[condition_id]["scaffold"].split()) - placebo_tokens)
        if delta > 2:
            raise RuntimeError(
                f"{condition_id} differs from placebo by {delta} whitespace tokens"
            )
    return spec


def main() -> None:
    spec = build()
    _atomic_json(DEFAULT_OUTPUT, spec)
    print(
        f"wrote {DEFAULT_OUTPUT} study={spec['study_id']} "
        f"conditions={len(spec['conditions'])} role={spec['analysis_role']}"
    )


if __name__ == "__main__":
    main()
