#!/usr/bin/env python3
"""Pair the A100 authority run with the RTX PRO 6000 diagnostic arm."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import MODELS, atomic_write_json, model_safe, read_json, sha256_file


def _load(root: Path, model_id: str) -> dict[str, dict[str, Any]]:
    record_root = root / "raw" / model_safe(model_id) / "records"
    records = {
        record["task_id"]: record
        for record in (read_json(path) for path in sorted(record_root.glob("*.json")))
    }
    if len(records) != 250:
        raise RuntimeError(f"{root} {model_id} has {len(records)} records")
    return records


def compare(authority: Path, diagnostic: Path, output: Path) -> None:
    manifests = []
    for root in (authority, diagnostic):
        manifest = read_json(root / "VLM_RERUN_MANIFEST.json")
        if manifest.get("status") != "final" or manifest.get("record_count") != 1500:
            raise RuntimeError(f"arm is not final: {root}")
        manifests.append(manifest)
    if (
        manifests[0].get("scorer_id") != manifests[1].get("scorer_id")
        or manifests[0].get("image_manifest", {}).get("sha256")
        != manifests[1].get("image_manifest", {}).get("sha256")
        or manifests[0].get("scoring_contract", {}).get("sha256")
        != manifests[1].get("scoring_contract", {}).get("sha256")
    ):
        raise RuntimeError("hardware arms do not share identical frozen inputs")

    models: dict[str, Any] = {}
    total = 0
    exact = 0
    score_equal = 0
    for model_id in MODELS:
        left = _load(authority, model_id)
        right = _load(diagnostic, model_id)
        if set(left) != set(right):
            raise RuntimeError(f"task mismatch for {model_id}")
        paradigm_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "n": 0,
                "response_exact": 0,
                "score_equal": 0,
                "authority_correct": 0,
                "diagnostic_correct": 0,
                "authority_blank": 0,
                "diagnostic_blank": 0,
            }
        )
        for task_id in sorted(left):
            a = left[task_id]
            b = right[task_id]
            if (
                a["expected"] != b["expected"]
                or a["paradigm"] != b["paradigm"]
                or a["image_sha256"] != b["image_sha256"]
            ):
                raise RuntimeError(f"input mismatch for {model_id} {task_id}")
            stats = paradigm_stats[a["paradigm"]]
            stats["n"] += 1
            stats["response_exact"] += int(a["response"] == b["response"])
            stats["score_equal"] += int(a["correct"] == b["correct"])
            stats["authority_correct"] += int(a["correct"])
            stats["diagnostic_correct"] += int(b["correct"])
            stats["authority_blank"] += int(a["blank"])
            stats["diagnostic_blank"] += int(b["blank"])
            total += 1
            exact += int(a["response"] == b["response"])
            score_equal += int(a["correct"] == b["correct"])
        models[model_id] = {"paradigms": dict(paradigm_stats)}

    result = {
        "schema_version": 1,
        "authority_root": authority.name,
        "diagnostic_root": diagnostic.name,
        "authority_manifest_sha256": sha256_file(
            authority / "VLM_RERUN_MANIFEST.json"
        ),
        "diagnostic_manifest_sha256": sha256_file(
            diagnostic / "VLM_RERUN_MANIFEST.json"
        ),
        "paired_records": total,
        "response_exact_count": exact,
        "response_exact_rate": exact / total,
        "score_equal_count": score_equal,
        "score_equal_rate": score_equal / total,
        "models": models,
    }
    atomic_write_json(output, result)
    print(
        f"paired={total} response_exact={exact}/{total} "
        f"score_equal={score_equal}/{total}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.authority.resolve(), args.diagnostic.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
