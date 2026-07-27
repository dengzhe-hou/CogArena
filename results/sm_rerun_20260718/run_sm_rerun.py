#!/usr/bin/env python3
"""Rerun the 11 dedup-fixed Source Monitoring episodes for one model.

The frozen battery carried 11 SM episodes whose statement lists contained
cross-source duplicates (see
results/reanalysis/sm_20260718/source_monitoring_exclusions.json).  The
generator now enforces episode-wide statement uniqueness, and same-seed
regeneration reproduces the other 39 episodes byte-identically while the
11 affected episodes regenerate clean under their frozen task_ids.  This
script regenerates those 11 episodes in-process and evaluates them with
the production static runner (same system prompt, temperature=0,
max_tokens=1024, same on-disk result schema).

Usage: run_sm_rerun.py <ollama_model_tag> [output_root]
Results: <output_root>/openai_<tag>/text/<dim>/<paradigm>/<task_id>.json
(output_root defaults to this script's directory; the default-context
serving-alignment arm passes results/sm_rerun_default_ctx_20260718/)
"""
import collections
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cogarena.generators.episodic_memory_gen import generate_em_items  # noqa: E402

spec = importlib.util.spec_from_file_location("run_eval", ROOT / "scripts" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)

MANIFEST = ROOT / "results" / "reanalysis" / "sm_20260718" / "rerun_manifest.json"
OUT = Path(__file__).resolve().parent


def _valid_accuracy(acc):
    import math
    return (isinstance(acc, (int, float)) and not isinstance(acc, bool)
            and math.isfinite(acc) and 0.0 <= acc <= 1.0)


def _existing_result_ok(path, model_id, task_id):
    """A stored result is reusable only if readable, not an API error, scored
    with a finite in-range accuracy, and carrying the expected identity
    (model_id/task_id/dimension/paradigm)."""
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    resp = d.get("response")
    if not isinstance(resp, str) or resp.startswith("ERROR:"):
        return False
    if d.get("model_id") != model_id or d.get("task_id") != task_id:
        return False
    if d.get("dimension") != "episodic_memory" or d.get("paradigm") != "source_monitoring":
        return False
    return _valid_accuracy(d.get("score", {}).get("accuracy"))


def _run_item(model_id, item, results_dir):
    """run_static_item semantics (same call, scorer, and schema) with an
    atomic write and self-healing of truncated or errored files."""
    import datetime
    import os

    result_path = results_dir / item.metadata.dimension / item.metadata.paradigm / f"{item.task_id}.json"
    if result_path.exists():
        if _existing_result_ok(result_path, model_id, item.task_id):
            return json.loads(result_path.read_text())
        result_path.unlink()

    response = run_eval.call_llm(model_id, item.stimulus, run_eval.SYSTEM_PROMPT)
    score = run_eval.score_static_item(item, response)
    result = {
        "task_id": item.task_id,
        "model_id": model_id,
        "dimension": item.metadata.dimension,
        "paradigm": item.metadata.paradigm,
        "difficulty": run_eval._get_difficulty(item),
        "response": response,
        "score": score,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(result_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp, result_path)
    return result


def main():
    tag = sys.argv[1]
    task_ids = set(json.load(open(MANIFEST))["task_ids"])
    assert len(task_ids) == 11, f"manifest has {len(task_ids)} task_ids"

    # The frozen battery was generated with include_contamination_probes=False;
    # the flag shifts the master-RNG sub-seed sequence, so it must match.
    items = [
        it
        for it in generate_em_items(seed=42, n_per_paradigm=50, include_contamination_probes=False)
        if it.metadata.paradigm == "source_monitoring" and it.task_id in task_ids
    ]
    assert len(items) == 11, f"regenerated {len(items)} of 11 affected episodes"
    for it in items:
        stmts = [s["statement"] for s in it.metadata.parameters["all_statements"]]
        assert len(set(stmts)) == len(stmts), f"duplicate statements in {it.task_id}"
        by_stmt = collections.defaultdict(set)
        for s in it.metadata.parameters["all_statements"]:
            by_stmt[s["statement"]].add(s["source"])
        for probe in it.metadata.parameters["test_items"]:
            assert len(by_stmt[probe["statement"]]) == 1, f"ambiguous probe in {it.task_id}"

    out_root = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else OUT
    results_dir = out_root / f"openai_{tag}" / "text"
    for it in sorted(items, key=lambda x: x.task_id):
        r = _run_item(f"openai/{tag}", it, results_dir)
        acc = r.get("score", {}).get("accuracy")
        print(f"{tag} {it.task_id} accuracy={acc}", flush=True)


if __name__ == "__main__":
    main()
