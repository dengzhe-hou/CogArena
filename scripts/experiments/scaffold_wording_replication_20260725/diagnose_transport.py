"""Locate the second deterministic OLMo2 transport-delimiter failure.

The formal runner reports progress every 25 completed task records. It failed
after printing 1075/1638, so this diagnostic replays schedule positions
1075--1099 without writing into the formal result tree. It stops at the first
delimiter leak and prints the exact request identity and raw response shape.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from scripts.experiments.causal_selectivity_20260720 import run_model as parent
from scripts.experiments.causal_selectivity_20260720.common import (
    condition_map,
    load_json,
    load_spec,
    manifest_path,
    response_terminator,
    stable_seed,
    system_prompt,
)


MODEL = "olmo2:7b"
START_INDEX = int(os.environ.get("COGARENA_DIAG_START_INDEX", "1075"))
STOP_INDEX = int(os.environ.get("COGARENA_DIAG_STOP_INDEX", "1100"))


def main() -> None:
    spec = load_spec()
    manifest = load_json(manifest_path("formal"))
    entries = {row["task_id"]: row for row in manifest["items"]}
    items = parent.reconstruct_items(spec, manifest)
    conditions = condition_map(spec)
    schedule = [
        (task_id, condition_id)
        for task_id in entries
        for condition_id in conditions
    ]
    random.Random(
        stable_seed(spec["study_id"], "formal", MODEL, "schedule-v1")
    ).shuffle(schedule)

    terminator = response_terminator(spec)
    base_normalize = parent.normalize_model_content
    active: dict[str, Any] = {}
    leaked: dict[str, Any] = {}

    def capture_content(value: Any) -> str:
        content = base_normalize(value)
        if terminator in content:
            leaked.update({
                **active,
                "content": content,
                "content_characters": len(content),
                "terminator_count": content.count(terminator),
                "ends_with_terminator": content.endswith(terminator),
            })
        return content

    parent.normalize_model_content = capture_content
    client = parent.LocalChatClient(MODEL, spec)
    checked = []
    for schedule_index in range(START_INDEX, STOP_INDEX):
        task_id, condition_id = schedule[schedule_index]
        item = items[task_id]
        active.clear()
        active.update({
            "schedule_index": schedule_index,
            "task_id": task_id,
            "condition_id": condition_id,
            "paradigm": item.metadata.paradigm,
        })
        try:
            parent.evaluate_item(
                client,
                item,
                system_prompt(spec, condition_id),
                spec,
            )
        except RuntimeError as error:
            if leaked:
                leaked["error"] = str(error)
                break
            raise
        checked.append(schedule_index)

    print(json.dumps({
        "checked_without_leak": checked,
        "leak": leaked or None,
    }, indent=2, ensure_ascii=False))
    if not leaked:
        raise RuntimeError("diagnostic window did not reproduce delimiter leakage")


if __name__ == "__main__":
    main()
