#!/usr/bin/env python3
"""Fail-closed replay and aggregation for the frozen VLM rerun."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    EXPECTED_ITEM_COUNTS,
    EXPECTED_ITEMS_PER_MODEL,
    MODELS,
    ROOT,
    RUN_ROOT,
    atomic_write_json,
    load_image_manifest,
    load_scoring_contract,
    model_safe,
    raw_tree_hash,
    read_json,
    request_fingerprint,
    sha256_file,
    sha256_json,
)
from .scoring import SCORER_ID, parse_response
from .verify_cache import EXPECTED_MANIFEST_SHA256


FINAL_SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).with_name("common.py"),
    Path(__file__).with_name("scoring.py"),
    Path(__file__).with_name("run_model.py"),
    Path(__file__).with_name("run_array.sbatch"),
    Path(__file__).with_name("finalize.sbatch"),
    Path(__file__).with_name("models.txt"),
    Path(__file__).with_name("verify_cache.py"),
)


def _git_head() -> str:
    declared = os.environ.get("COGARENA_GIT_HEAD")
    if declared:
        return declared
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_model(
    model_id: str,
    tasks: dict[str, dict[str, Any]],
    image_manifest_sha256: str,
    scoring_tasks: dict[str, dict[str, Any]],
    scoring_contract_sha256: str,
) -> tuple[dict[str, Any], list[Path]]:
    safe = model_safe(model_id)
    model_root = RUN_ROOT / "raw" / safe
    record_root = model_root / "records"
    cache = read_json(model_root / "cache.json")
    if cache.get("model_id") != model_id:
        raise RuntimeError(f"{model_id} cache identity mismatch")
    if cache.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256[model_id]:
        raise RuntimeError(f"{model_id} cache manifest is not the pinned digest")
    session_root = model_root / "sessions"
    session_files = sorted(session_root.glob("*.json"))
    if not session_files:
        raise RuntimeError(f"{model_id} has no immutable serving session")
    sessions: dict[str, tuple[dict[str, Any], Path, str]] = {}
    for session_path in session_files:
        session = read_json(session_path)
        session_id = session.get("session_id")
        if (
            not isinstance(session_id, str)
            or session_path.name != f"{session_id}.json"
            or session_id in sessions
            or session.get("model_id") != model_id
            or session.get("image_manifest_sha256") != image_manifest_sha256
            or session.get("scoring_contract_sha256")
            != scoring_contract_sha256
            or session.get("scorer_id") != SCORER_ID
            or session.get("cache_manifest_sha256")
            != cache.get("manifest_sha256")
            or session.get("readiness", {}).get("status") != "pass"
            or session.get("readiness", {}).get("output_is_not_scored") is not True
            or session.get("serving", {}).get("model_id") != model_id
            or session.get("serving", {}).get("tag", {}).get("digest")
            != cache.get("manifest_sha256")
        ):
            raise RuntimeError(f"invalid serving session: {session_path}")
        source_files = session.get("source_files")
        if not isinstance(source_files, dict) or not source_files:
            raise RuntimeError(f"session lacks source hashes: {session_path}")
        for rel, expected_sha256 in source_files.items():
            source_path = ROOT / rel
            if not source_path.is_file() or sha256_file(source_path) != expected_sha256:
                raise RuntimeError(f"session source drift: {rel}")
        sessions[session_id] = (
            session,
            session_path,
            sha256_file(session_path),
        )

    files = sorted(record_root.glob("*.json"))
    extras = sorted(path.name for path in record_root.iterdir() if path.suffix != ".json")
    if extras:
        raise RuntimeError(f"{model_id} has unexpected record files: {extras}")
    if len(files) != EXPECTED_ITEMS_PER_MODEL:
        raise RuntimeError(f"{model_id} has {len(files)} records, expected 250")

    by_paradigm: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "count": 0, "blank": 0, "non_stop": 0}
    )
    by_condition: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0, "count": 0, "blank": 0})
    )
    usage_by_paradigm: dict[str, list[dict[str, int]]] = defaultdict(list)
    seen: set[str] = set()
    record_session_counts: Counter[str] = Counter()
    for path in files:
        record = read_json(path)
        task_id = record.get("task_id")
        if task_id not in tasks or task_id in seen:
            raise RuntimeError(f"{model_id} unexpected/duplicate task: {task_id}")
        seen.add(task_id)
        task = tasks[task_id]
        if record.get("model_id") != model_id:
            raise RuntimeError(f"{path} model identity mismatch")
        if record.get("api_model") != model_id.split("/", 1)[-1]:
            raise RuntimeError(f"{path} API model identity mismatch")
        if record.get("schema_version") != 2:
            raise RuntimeError(f"{path} unsupported record schema")
        for key in ("paradigm", "dimension", "expected", "congruent"):
            if record.get(key) != task.get(key):
                raise RuntimeError(f"{path} task field mismatch: {key}")
        if record.get("image_manifest_sha256") != image_manifest_sha256:
            raise RuntimeError(f"{path} manifest hash mismatch")
        if record.get("scoring_contract_sha256") != scoring_contract_sha256:
            raise RuntimeError(f"{path} scoring-contract hash mismatch")
        if record.get("scorer_id") != SCORER_ID:
            raise RuntimeError(f"{path} scorer identity mismatch")
        if record.get("image_sha256") != [
            entry["sha256"] for entry in task["images"]
        ]:
            raise RuntimeError(f"{path} image hash mismatch")
        response = record.get("response")
        if not isinstance(response, str):
            raise RuntimeError(f"{path} response is not text")
        scoring = scoring_tasks[task_id]
        expected_parse = parse_response(response=response, **scoring)
        expected_correct = expected_parse["correct"]
        if record.get("parse") != expected_parse:
            raise RuntimeError(f"{path} stored parse mismatch")
        if record.get("correct") is not expected_correct:
            raise RuntimeError(f"{path} stored score mismatch")
        if record.get("blank") is not (not response.strip()):
            raise RuntimeError(f"{path} stored blank flag mismatch")
        attempts = record.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise RuntimeError(f"{path} lacks API attempt metadata")
        last = attempts[-1]
        if last.get("transport_status") != "ok":
            raise RuntimeError(f"{path} has no successful API completion")
        finish_reason = record.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise RuntimeError(f"{path} lacks finish_reason")
        raw_choices = (last.get("raw_response") or {}).get("choices") or []
        if len(raw_choices) != 1:
            raise RuntimeError(f"{path} raw response choice mismatch")
        raw_choice = raw_choices[0]
        usage = (last.get("raw_response") or {}).get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError(f"{path} lacks API usage metadata")
        usage_values: dict[str, int] = {}
        for usage_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(usage_key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"{path} invalid API usage: {usage_key}")
            usage_values[usage_key] = value
        if (
            usage_values["prompt_tokens"] + usage_values["completion_tokens"]
            != usage_values["total_tokens"]
        ):
            raise RuntimeError(f"{path} inconsistent API token accounting")
        usage_by_paradigm[task["paradigm"]].append(usage_values)
        raw_content = raw_choice.get("message", {}).get("content")
        if raw_content is None:
            raw_content = ""
        if (
            last.get("content") != response
            or last.get("blank") is not (not response.strip())
            or raw_content != response
            or last.get("finish_reason") != finish_reason
            or raw_choice.get("finish_reason") != finish_reason
            or (last.get("raw_response") or {}).get("model")
            != record.get("api_model")
        ):
            raise RuntimeError(f"{path} completion replay mismatch")
        if record.get("prompt_sha256") != sha256_json(task["prompt"]):
            raise RuntimeError(f"{path} prompt hash mismatch")
        expected_fingerprint = request_fingerprint(
            model_id,
            task,
            image_manifest_sha256,
            scoring_contract_sha256,
        )
        if record.get("request_fingerprint") != expected_fingerprint:
            raise RuntimeError(f"{path} request fingerprint mismatch")
        session_id = record.get("session_id")
        if session_id not in sessions:
            raise RuntimeError(f"{path} references an unknown serving session")
        if record.get("session_sha256") != sessions[session_id][2]:
            raise RuntimeError(f"{path} serving-session hash mismatch")
        record_session_counts[session_id] += 1

        stats = by_paradigm[task["paradigm"]]
        stats["count"] += 1
        stats["correct"] += int(expected_correct)
        stats["blank"] += int(not response.strip())
        stats["non_stop"] += int(finish_reason != "stop")
        if task["paradigm"] in {"stroop", "flanker"}:
            condition = "congruent" if task["congruent"] else "incongruent"
            condition_stats = by_condition[task["paradigm"]][condition]
            condition_stats["count"] += 1
            condition_stats["correct"] += int(expected_correct)
            condition_stats["blank"] += int(not response.strip())

    if seen != set(tasks):
        raise RuntimeError(f"{model_id} task coverage mismatch")
    if {key: value["count"] for key, value in by_paradigm.items()} != EXPECTED_ITEM_COUNTS:
        raise RuntimeError(f"{model_id} paradigm counts mismatch")
    if any(
        by_condition[paradigm][condition]["count"] != 50
        for paradigm in ("stroop", "flanker")
        for condition in ("congruent", "incongruent")
    ):
        raise RuntimeError(f"{model_id} condition counts are not 50/50")
    usage_summary = {
        paradigm: {
            "min_prompt_tokens": min(row["prompt_tokens"] for row in rows),
            "max_prompt_tokens": max(row["prompt_tokens"] for row in rows),
            "max_completion_tokens": max(
                row["completion_tokens"] for row in rows
            ),
        }
        for paradigm, rows in sorted(usage_by_paradigm.items())
    }
    if (
        model_id == "openai/moondream:1.8b"
        and usage_summary["false_belief"]["max_prompt_tokens"] >= 2048
    ):
        raise RuntimeError("Moondream false-belief prompt still saturates 2048 tokens")

    summary = {
        "model_id": model_id,
        "n_records": len(files),
        "paradigms": {
            paradigm: {
                **stats,
                "accuracy": stats["correct"] / stats["count"],
                "blank_rate": stats["blank"] / stats["count"],
            }
            for paradigm, stats in sorted(by_paradigm.items())
        },
        "conditions": {
            paradigm: {
                condition: {
                    **stats,
                    "accuracy": stats["correct"] / stats["count"],
                    "blank_rate": stats["blank"] / stats["count"],
                }
                for condition, stats in sorted(conditions.items())
            }
            for paradigm, conditions in sorted(by_condition.items())
        },
        "usage": usage_summary,
        "model_digest": cache["manifest_sha256"],
        "session_ids": sorted(sessions),
        "record_counts_by_session": dict(sorted(record_session_counts.items())),
        "serving_sessions": {
            session_id: {
                "sha256": values[2],
                "node": values[0]["serving"]["node"],
                "slurm_job_id": values[0]["serving"]["slurm_job_id"],
                "source_revision": values[0]["source_revision"],
            }
            for session_id, values in sorted(sessions.items())
        },
    }
    return summary, files + [model_root / "cache.json", *session_files]


def verify(models: tuple[str, ...]) -> None:
    manifest, manifest_sha256 = load_image_manifest(verify_images=True)
    scoring_contract, scoring_contract_sha256 = load_scoring_contract(
        manifest, manifest_sha256
    )
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    summaries: dict[str, Any] = {}
    raw_paths: list[Path] = []
    for model_id in models:
        summary, paths = verify_model(
            model_id,
            tasks,
            manifest_sha256,
            scoring_contract["tasks"],
            scoring_contract_sha256,
        )
        summaries[model_id] = summary
        raw_paths.extend(paths)
        atomic_write_json(
            RUN_ROOT / "aggregates" / f"{model_safe(model_id)}.json", summary
        )
        compact = ", ".join(
            f"{name}={stats['accuracy']:.3f} blank={stats['blank']}"
            for name, stats in summary["paradigms"].items()
        )
        print(f"{model_id}: {compact}")

    if models == MODELS:
        total = sum(summary["n_records"] for summary in summaries.values())
        if total != len(MODELS) * EXPECTED_ITEMS_PER_MODEL:
            raise RuntimeError(f"global result count mismatch: {total}")
        all_finish = [
            stats["non_stop"]
            for summary in summaries.values()
            for stats in summary["paradigms"].values()
        ]
        if any(value < 0 or not math.isfinite(float(value)) for value in all_finish):
            raise RuntimeError("invalid finish-reason counts")
        aggregate_path = RUN_ROOT / "VLM_RERUN_SUMMARY.json"
        atomic_write_json(
            aggregate_path,
            {
                "schema_version": 1,
                "models": summaries,
                "record_count": total,
                "scorer_id": SCORER_ID,
                "scoring_contract_sha256": scoring_contract_sha256,
            },
        )
        output_paths = [
            aggregate_path,
            *sorted((RUN_ROOT / "aggregates").glob("*.json")),
        ]
        manifest_path = (
            (RUN_ROOT / "stimuli" / "IMAGE_MANIFEST.json")
            .relative_to(RUN_ROOT)
            .as_posix()
        )
        final_manifest = {
            "schema_version": 1,
            "status": "final",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "source_revision": _git_head(),
            "models": list(MODELS),
            "record_count": total,
            "replayed_score_count": total,
            "scorer_id": SCORER_ID,
            "stimulus_design_id": manifest["design_id"],
            "source_files": {
                path.relative_to(ROOT).as_posix(): sha256_file(path)
                for path in FINAL_SOURCE_PATHS
            },
            "image_manifest": {
                "path": manifest_path,
                "sha256": manifest_sha256,
            },
            "scoring_contract": {
                "path": (
                    (RUN_ROOT / "stimuli" / "SCORING_LABELS.json")
                    .relative_to(RUN_ROOT)
                    .as_posix()
                ),
                "sha256": scoring_contract_sha256,
                "source_files": scoring_contract["source_files"],
            },
            "raw_tree_sha256": raw_tree_hash(raw_paths),
            "outputs": {
                path.relative_to(RUN_ROOT).as_posix(): sha256_file(path)
                for path in output_paths
            },
            "blank_counts": {
                model: {
                    paradigm: stats["blank"]
                    for paradigm, stats in summary["paradigms"].items()
                }
                for model, summary in summaries.items()
            },
        }
        if not manifest_path:
            raise AssertionError("unreachable")
        atomic_write_json(RUN_ROOT / "VLM_RERUN_MANIFEST.json", final_manifest)
        print(f"ALL GATES PASSED records={total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    verify((args.model,) if args.model else MODELS)


if __name__ == "__main__":
    main()
