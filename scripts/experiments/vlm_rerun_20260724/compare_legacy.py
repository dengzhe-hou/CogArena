#!/usr/bin/env python3
"""Compare the provenance-hardened rerun with corrected legacy responses."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import (
    MODELS,
    ROOT,
    RUN_ROOT,
    atomic_write_json,
    model_safe,
    read_json,
)
from .scoring import SCORER_ID, parse_response
from cogarena.image_gen.false_belief_images import (
    CONTAINER_OPTIONS,
    NAMES,
    OBJECT_OPTIONS,
)
from cogarena.image_gen.stroop_images import COLOR_NAMES


LEGACY_ROOT = ROOT / "results" / "full_eval_20260526_2208"


def _legacy_false_belief_contracts() -> list[dict[str, Any]]:
    rng = random.Random(42)
    label_space = [name for name, _ in CONTAINER_OPTIONS]
    output = []
    for _ in range(50):
        characters = rng.sample(NAMES, 2)
        rng.choice(OBJECT_OPTIONS)
        containers = [name for name, _ in rng.sample(CONTAINER_OPTIONS, 2)]
        output.append(
            {
                "allowed_labels": containers,
                "label_space": label_space,
                "query_subject": characters[1],
            }
        )
    return output


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, object], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        cells[(record["paradigm"], record.get("congruent"))].append(record)

    paradigms: dict[str, Any] = {}
    for paradigm in ("stroop", "flanker", "false_belief"):
        relevant = [
            record for record in records if record["paradigm"] == paradigm
        ]
        paradigms[paradigm] = {
            "n": len(relevant),
            "accuracy": sum(record["correct"] for record in relevant) / len(relevant),
            "blank": sum(not record["response"].strip() for record in relevant),
        }
        if paradigm in {"stroop", "flanker"}:
            congruent = cells[(paradigm, True)]
            incongruent = cells[(paradigm, False)]
            congruent_accuracy = sum(record["correct"] for record in congruent) / len(
                congruent
            )
            incongruent_accuracy = sum(
                record["correct"] for record in incongruent
            ) / len(incongruent)
            paradigms[paradigm].update(
                {
                    "congruent_accuracy": congruent_accuracy,
                    "incongruent_accuracy": incongruent_accuracy,
                    "congruency_advantage": congruent_accuracy
                    - incongruent_accuracy,
                }
            )
    return paradigms


def _legacy(model_id: str) -> list[dict[str, Any]]:
    path = LEGACY_ROOT / model_safe(model_id) / "image" / "details.json"
    rows = read_json(path)
    if len(rows) != 250:
        raise RuntimeError(f"legacy result count drift for {model_id}")
    output = []
    false_belief_contracts = _legacy_false_belief_contracts()
    for row in rows:
        response = row.get("response")
        if not isinstance(response, str):
            raise RuntimeError(f"legacy response is not text for {row.get('task_id')}")
        paradigm = row["paradigm"]
        if paradigm == "stroop":
            labels = {
                "allowed_labels": list(COLOR_NAMES),
                "label_space": list(COLOR_NAMES),
            }
        elif paradigm == "flanker":
            labels = {
                "allowed_labels": ["left", "right"],
                "label_space": ["left", "right"],
            }
        elif paradigm == "false_belief":
            index = int(row["task_id"].rsplit("_", 1)[1])
            labels = false_belief_contracts[index]
        else:
            raise RuntimeError(f"unexpected legacy paradigm: {paradigm}")
        parsed = parse_response(
            paradigm=paradigm,
            response=response,
            expected=str(row.get("expected", "")),
            **labels,
        )
        output.append({**row, "parse": parsed, "correct": parsed["correct"]})
    return output


def _rerun(model_id: str) -> list[dict[str, Any]]:
    record_root = RUN_ROOT / "raw" / model_safe(model_id) / "records"
    rows = [read_json(path) for path in sorted(record_root.glob("*.json"))]
    if len(rows) != 250:
        raise RuntimeError(f"rerun result count drift for {model_id}")
    return rows


def compare() -> None:
    final = read_json(RUN_ROOT / "VLM_RERUN_MANIFEST.json")
    if final.get("status") != "final" or final.get("record_count") != 1500:
        raise RuntimeError("VLM rerun is not final")

    models: dict[str, Any] = {}
    for model_id in MODELS:
        legacy = _summarize(_legacy(model_id))
        rerun = _summarize(_rerun(model_id))
        models[model_id] = {
            "legacy_corrected": legacy,
            "rerun": rerun,
            "accuracy_change": {
                paradigm: rerun[paradigm]["accuracy"]
                - legacy[paradigm]["accuracy"]
                for paradigm in rerun
            },
        }

    paradigm_means: dict[str, Any] = {}
    for paradigm in ("stroop", "flanker", "false_belief"):
        old_values = [
            models[model]["legacy_corrected"][paradigm]["accuracy"]
            for model in MODELS
        ]
        new_values = [
            models[model]["rerun"][paradigm]["accuracy"] for model in MODELS
        ]
        paradigm_means[paradigm] = {
            "legacy_corrected_mean": sum(old_values) / len(old_values),
            "rerun_mean": sum(new_values) / len(new_values),
            "mean_change": sum(new_values) / len(new_values)
            - sum(old_values) / len(old_values),
        }

    output = {
        "schema_version": 1,
        "legacy_note": (
            f"Legacy responses are replayed with {SCORER_ID}. "
            "Their exact stimulus PNG bytes were not archived."
        ),
        "rerun_note": (
            "Rerun responses consume the frozen, hash-pinned readable stimulus set."
        ),
        "models": models,
        "paradigm_means": paradigm_means,
    }
    atomic_write_json(RUN_ROOT / "VLM_LEGACY_COMPARISON.json", output)
    for paradigm, values in paradigm_means.items():
        print(
            f"{paradigm}: old={values['legacy_corrected_mean']:.3f} "
            f"new={values['rerun_mean']:.3f} "
            f"change={values['mean_change']:+.3f}"
        )


if __name__ == "__main__":
    compare()
