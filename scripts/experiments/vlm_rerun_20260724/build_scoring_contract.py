#!/usr/bin/env python3
"""Freeze task-specific label choices without changing stimulus hashes."""

from __future__ import annotations

import subprocess

from cogarena.image_gen.false_belief_images import (
    CONTAINER_OPTIONS,
)
from cogarena.image_gen.stroop_images import COLOR_NAMES

from .common import (
    ROOT,
    SCORING_CONTRACT,
    SEED,
    atomic_write_json,
    load_image_manifest,
    sha256_file,
)
from .scoring import SCORER_ID


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build() -> None:
    manifest, manifest_sha256 = load_image_manifest(verify_images=True)
    tasks = {}
    container_space = [name for name, _ in CONTAINER_OPTIONS]
    for task in manifest["tasks"]:
        paradigm = task["paradigm"]
        if paradigm == "stroop":
            allowed = list(COLOR_NAMES)
            label_space = list(COLOR_NAMES)
        elif paradigm == "flanker":
            allowed = ["left", "right"]
            label_space = ["left", "right"]
        elif paradigm == "false_belief":
            factors = task.get("factors", {})
            allowed = list(factors.get("containers", []))
            characters = list(factors.get("characters", []))
            if len(allowed) != 2 or len(characters) != 2:
                raise RuntimeError(
                    f"{task['task_id']} lacks frozen false-belief factors"
                )
            label_space = container_space
        else:
            raise RuntimeError(f"unexpected image paradigm: {paradigm}")
        if task["expected"] not in allowed:
            raise RuntimeError(
                f"{task['task_id']} expected {task['expected']} outside {allowed}"
            )
        entry = {
            "paradigm": paradigm,
            "expected": task["expected"],
            "allowed_labels": allowed,
            "label_space": label_space,
        }
        if paradigm == "false_belief":
            entry["query_subject"] = characters[1]
        tasks[task["task_id"]] = entry
    if len(tasks) != 250:
        raise RuntimeError("scoring-contract coverage mismatch")
    contract = {
        "schema_version": 1,
        "status": "frozen",
        "scorer_id": SCORER_ID,
        "source_revision": _git_head(),
        "image_manifest_sha256": manifest_sha256,
        "source_files": {
            "scripts/experiments/vlm_rerun_20260724/scoring.py": sha256_file(
                ROOT / "scripts" / "experiments" / "vlm_rerun_20260724" / "scoring.py"
            ),
            "scripts/experiments/vlm_rerun_20260724/build_scoring_contract.py": sha256_file(
                __import__("pathlib").Path(__file__).resolve()
            ),
        },
        "tasks": tasks,
    }
    atomic_write_json(SCORING_CONTRACT, contract)
    print(f"frozen label contract for {len(tasks)} tasks at {SCORING_CONTRACT}")


if __name__ == "__main__":
    build()
