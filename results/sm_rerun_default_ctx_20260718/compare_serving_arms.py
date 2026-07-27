#!/usr/bin/env python3
"""Pairwise comparison of the two SM rerun serving arms, with hard gates.

Arms:
  4096 arm     results/sm_rerun_20260718/      (job 6805, OLLAMA_CONTEXT_LENGTH=4096)
  default arm  results/sm_rerun_default_ctx_20260718/  (this dir, no override)

The default arm matches the original evaluation's serving path for the 21
ordinary models, so it is the PRIMARY source for them in the final overlay;
the 4096 arm remains a serving-sensitivity record. llama3.1:70b and
mixtral:8x22b ran 4096 originally and stay with the 4096 arm.

Gates (nonzero exit on failure): 21 x 11 default-arm files present, valid
(reusing the runner's validator), rescore with the production scorer
reproduces each stored score.

Output: serving_arm_comparison.json with per-model response-identity and
score-delta stats.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARM4096 = ROOT / "results" / "sm_rerun_20260718"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ARM4096))

from cogarena.generators.episodic_memory_gen import generate_em_items  # noqa: E402
from cogarena.dimensions.episodic_memory import SourceMonitoringGenerator  # noqa: E402
from run_sm_rerun import _existing_result_ok  # noqa: E402

MANIFEST = ROOT / "results" / "reanalysis" / "sm_20260718" / "rerun_manifest.json"
TOL = 1e-9


def atomic_dump(obj, path):
    import os
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    task_ids = sorted(json.load(open(MANIFEST))["task_ids"])
    models = [l.strip() for l in open(OUT / "modellist_default_ctx.txt") if l.strip()]
    items = {
        it.task_id: it
        for it in generate_em_items(seed=42, n_per_paradigm=50, include_contamination_probes=False)
        if it.metadata.paradigm == "source_monitoring"
    }
    failures = []
    comparison = {}
    for m in models:
        rows = []
        for t in task_ids:
            fd = OUT / f"openai_{m}" / "text" / "episodic_memory" / "source_monitoring" / f"{t}.json"
            f4 = ARM4096 / f"openai_{m}" / "text" / "episodic_memory" / "source_monitoring" / f"{t}.json"
            if not (fd.exists() and _existing_result_ok(fd, f"openai/{m}", t)):
                failures.append(f"{m}: default-arm file missing or invalid {t}")
                continue
            if not (f4.exists() and _existing_result_ok(f4, f"openai/{m}", t)):
                failures.append(f"{m}: 4096-arm file missing or invalid {t}")
                continue
            dd = json.loads(fd.read_text())
            d4 = json.loads(f4.read_text())
            mismatch = False
            for tag, rec in (("default", dd), ("4096", d4)):
                re_acc = SourceMonitoringGenerator.score(items[t], rec["response"])["accuracy"]
                if abs(float(re_acc) - float(rec["score"]["accuracy"])) > TOL:
                    failures.append(f"{m}: {tag}-arm rescore mismatch {t}")
                    mismatch = True
            if mismatch:
                continue
            rows.append({
                "task_id": t,
                "acc_default": float(dd["score"]["accuracy"]),
                "acc_4096": float(d4["score"]["accuracy"]),
                "response_identical": dd["response"] == d4["response"],
            })
        if len(rows) == len(task_ids):
            comparison[m] = {
                "n_identical_responses": sum(r["response_identical"] for r in rows),
                "mean_acc_default": sum(r["acc_default"] for r in rows) / len(rows),
                "mean_acc_4096": sum(r["acc_4096"] for r in rows) / len(rows),
                "max_abs_score_delta": max(abs(r["acc_default"] - r["acc_4096"]) for r in rows),
                "rows": rows,
            }

    print(f"models complete: {len(comparison)}/{len(models)}")
    if failures:
        print(f"GATE FAILURES ({len(failures)}):")
        for x in failures[:30]:
            print("  " + x)
        sys.exit(1)

    ident = sum(c["n_identical_responses"] for c in comparison.values())
    grand_d = sum(c["mean_acc_default"] for c in comparison.values()) / len(comparison)
    grand_4 = sum(c["mean_acc_4096"] for c in comparison.values()) / len(comparison)
    atomic_dump(
        {
            "spec": "SM serving-arm comparison: default context (primary for these 21) "
                    "vs 4096 override (sensitivity)",
            "n_models": len(models),
            "identical_responses": f"{ident}/{len(models) * len(task_ids)}",
            "grand_affected_mean_default": grand_d,
            "grand_affected_mean_4096": grand_4,
            "models": comparison,
        },
        OUT / "serving_arm_comparison.json",
    )
    print(f"identical responses: {ident}/{len(models) * len(task_ids)}")
    print(f"affected-11 grand mean default={grand_d:.4f} vs 4096={grand_4:.4f}")
    n_files = len(models) * len(task_ids)
    print(f"symmetric validation: {2 * n_files} files ({n_files} per arm), "
          f"identity + schema + production-scorer rescore")
    print("ALL GATES PASSED")


if __name__ == "__main__":
    main()
