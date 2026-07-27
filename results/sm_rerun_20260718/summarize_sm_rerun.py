#!/usr/bin/env python3
"""Collate the SM rerun into a corrected per-item overlay, with hard gates.

The 55x50 final SM scores combine two corrected sources:
  - the 39 unaffected frozen episodes: corrected-scorer values from the
    results/rescore_20260702/new_scores/ overlays (the raw May per-item
    files carry the buggy scorer and are never averaged here);
  - the 11 dedup-fixed episodes: the rerun outputs, which the production
    runner scored with the same corrected scorer at inference time.

Hard gates (any failure -> nonzero exit, no summary trusted):
  G1  605 rerun files present (11 x 55), all JSON-readable, no ERROR:
      responses, each with a finite non-bool accuracy in [0, 1].
  G2  model set == modellist.txt exactly; per-model task set == the 11
      manifest task_ids exactly; per-model old SM set == the other 39.
  G3  overlay files cover all 50 frozen episodes per model with finite
      in-range values.
  G4  rescoring every stored response with the current production scorer
      against the regenerated battery reproduces the recorded value
      (overlay for old 39, stored score for new 11) within 1e-9.
  G5  after substitution every model has exactly 50 SM scores.
  G6  no extra files anywhere: unknown model dirs under the rerun root,
      files beyond the 11 expected per model (including *.tmp leftovers),
      and identity-field mismatches inside any JSON
      (model_id/task_id/dimension/paradigm) are all rejected.

Outputs (atomic writes):
  sm_scores_overlay.json   {model: {task_id: accuracy}} full precision
  sm_rerun_summary.json    per-model corrected-old vs corrected-new means
"""
import collections
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))

from cogarena.generators.episodic_memory_gen import generate_em_items  # noqa: E402
from cogarena.dimensions.episodic_memory import SourceMonitoringGenerator  # noqa: E402
from run_sm_rerun import _valid_accuracy  # noqa: E402  (single-source validity)

MANIFEST = ROOT / "results" / "reanalysis" / "sm_20260718" / "rerun_manifest.json"
RESCORE = ROOT / "results" / "rescore_20260702" / "new_scores"
TOL = 1e-9

# Serving-arm source map (inert until the file exists): after the
# default-context alignment arm (results/sm_rerun_default_ctx_20260718/)
# completes and is adopted, serving_source_map.json records, per model,
# which arm directory supplies its 11 rerun episodes in the final overlay
# (default arm for the 21 ordinary models; this 4096 arm for llama3.1:70b
# and mixtral:8x22b, matching the original evaluation's serving contract).
SOURCE_MAP_PATH = OUT / "serving_source_map.json"
SOURCE_MAP = (json.load(open(SOURCE_MAP_PATH))
              if SOURCE_MAP_PATH.exists() else None)


def rerun_dir_for(model):
    if SOURCE_MAP is not None:
        return ROOT / SOURCE_MAP[model]
    return OUT


def atomic_dump(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
    os.replace(tmp, path)


def build_overlay():
    """Single-source overlay computation with all gates; returns
    (overlay_out, summary, failures, n_rescore_checked) without writing.
    Both main() and the SM manifest builder consume this, so 'final'
    status is backed by exactly the gated computation."""
    rerun_ids = set(json.load(open(MANIFEST))["task_ids"])
    models = [l.strip() for l in open(OUT / "modellist.txt") if l.strip()]
    failures = []

    items = {
        it.task_id: it
        for it in generate_em_items(seed=42, n_per_paradigm=50, include_contamination_probes=False)
        if it.metadata.paradigm == "source_monitoring"
    }
    assert len(items) == 50, f"regenerated {len(items)} SM episodes"
    assert rerun_ids <= set(items), "manifest task_ids missing from regeneration"
    frozen_ids = set(items) - rerun_ids

    # G6a: no unknown model dirs and no files beyond the 11 expected per
    # model, in every consumed arm root
    model_set = set(models)
    roots = {OUT}
    if SOURCE_MAP is not None:
        roots |= {ROOT / p for p in SOURCE_MAP.values()}
    for root in sorted(roots):
        for md in sorted(root.glob("openai_*")):
            tag = md.name[len("openai_"):]
            if tag not in model_set:
                failures.append(f"unknown model dir {root.name}/{md.name}")
                continue
            expected = {
                md / "text" / "episodic_memory" / "source_monitoring" / f"{t}.json"
                for t in rerun_ids
            }
            for f in md.rglob("*"):
                if f.is_dir():
                    continue
                if f not in expected:
                    failures.append(f"unexpected file {root.name}/{f.relative_to(root)}")

    overlay_out = {}
    summary = {}
    n_rescore_checked = 0
    for m in models:
        errs = []
        old_dirs = glob.glob(
            str(ROOT / f"results/full_eval_*/openai_{m}/text/episodic_memory/source_monitoring")
        )
        if len(old_dirs) != 1:
            failures.append(f"{m}: {len(old_dirs)} frozen SM dirs")
            continue
        old_dir = Path(old_dirs[0])
        set_name = old_dir.parts[len(ROOT.parts) + 1]
        ov_path = RESCORE / f"{set_name}__openai_{m}.json"
        if not ov_path.exists():
            failures.append(f"{m}: missing overlay {ov_path.name}")
            continue
        overlay = json.load(open(ov_path))

        old_raw = {}
        for f in old_dir.glob("*.json"):
            d = json.load(open(f))
            if d.get("task_id") != f.stem:
                errs.append(f"frozen file identity mismatch {f.name}: task_id={d.get('task_id')}")
                continue
            if d.get("model_id") != f"openai/{m}":
                errs.append(f"frozen file model_id mismatch {f.name}: {d.get('model_id')}")
                continue
            if d.get("dimension") != "episodic_memory" or d.get("paradigm") != "source_monitoring":
                errs.append(f"frozen file wrong dimension/paradigm {f.name}")
                continue
            if not isinstance(d.get("response"), str):
                errs.append(f"frozen file response not a string {f.name}")
                continue
            old_raw[d["task_id"]] = d
        if set(old_raw) != frozen_ids | rerun_ids:
            errs.append(f"frozen task set mismatch ({len(old_raw)} files)")

        # G3: the overlay must cover ALL 50 episodes with valid values (the
        # 11 void entries feed the old-corrected comparison means, so an
        # invalid value there would poison the summary too)
        for t in sorted(items):
            if t not in overlay:
                errs.append(f"overlay missing {t}")
            elif not _valid_accuracy(overlay[t]):
                errs.append(f"overlay value invalid {t}: {overlay[t]!r}")

        # corrected old 39 (the 11 frozen ambiguous episodes are void)
        corrected = {}
        for t in frozen_ids:
            if t in overlay and _valid_accuracy(overlay[t]):
                corrected[t] = float(overlay[t])

        # G4 on old 39: current scorer x regenerated (byte-identical) items
        for t in sorted(frozen_ids):
            if t not in old_raw or t not in overlay:
                continue
            re_acc = SourceMonitoringGenerator.score(items[t], old_raw[t].get("response", ""))["accuracy"]
            n_rescore_checked += 1
            if abs(float(re_acc) - float(overlay[t])) > TOL:
                errs.append(f"old rescore mismatch {t}: {re_acc} vs overlay {overlay[t]}")

        # G1 + G4 on new 11
        new_acc = {}
        for t in sorted(rerun_ids):
            f = rerun_dir_for(m) / f"openai_{m}" / "text" / "episodic_memory" / "source_monitoring" / f"{t}.json"
            if not f.exists():
                errs.append(f"rerun file missing {t}")
                continue
            try:
                d = json.load(open(f))
            except Exception as e:
                errs.append(f"rerun file unreadable {t}: {e}")
                continue
            if d.get("model_id") != f"openai/{m}" or d.get("task_id") != t:
                errs.append(f"rerun identity mismatch {t}: "
                            f"model_id={d.get('model_id')} task_id={d.get('task_id')}")
                continue
            if d.get("dimension") != "episodic_memory" or d.get("paradigm") != "source_monitoring":
                errs.append(f"rerun wrong dimension/paradigm {t}")
                continue
            resp = d.get("response")
            if not isinstance(resp, str):
                errs.append(f"rerun response missing or not a string {t}")
                continue
            if resp.startswith("ERROR:"):
                errs.append(f"rerun API error {t}")
                continue
            acc = d.get("score", {}).get("accuracy")
            if not _valid_accuracy(acc):
                errs.append(f"rerun score invalid {t}: {acc!r}")
                continue
            re_acc = SourceMonitoringGenerator.score(items[t], resp)["accuracy"]
            n_rescore_checked += 1
            if abs(float(re_acc) - float(acc)) > TOL:
                errs.append(f"new rescore mismatch {t}: {re_acc} vs stored {acc}")
                continue
            new_acc[t] = float(acc)

        corrected.update(new_acc)
        if len(corrected) != 50:
            errs.append(f"substituted set has {len(corrected)} scores, want 50")

        if errs:
            failures.extend(f"{m}: {e}" for e in errs)
            continue

        overlay_out[m] = {t: corrected[t] for t in sorted(corrected)}
        old_corrected_full = [float(overlay[t]) for t in sorted(set(items) & set(overlay))]
        summary[m] = {
            "old_corrected_paradigm_mean": sum(old_corrected_full) / len(old_corrected_full),
            "new_paradigm_mean": sum(corrected.values()) / 50,
            "old_corrected_affected_mean": sum(float(overlay[t]) for t in rerun_ids) / len(rerun_ids),
            "new_affected_mean": sum(new_acc.values()) / len(new_acc),
        }

    return overlay_out, summary, failures, n_rescore_checked, sorted(rerun_ids), models


def summary_doc(overlay_out, summary, rerun_ids, models):
    grand_old = sum(s["old_corrected_paradigm_mean"] for s in summary.values()) / len(summary)
    grand_new = sum(s["new_paradigm_mean"] for s in summary.values()) / len(summary)
    return {
        "spec": "SM corrected overlay: rescore_20260702 corrected scores for the 39 "
                "unaffected frozen episodes + dedup-fixed rerun for the 11 affected",
        "serving_source_map": SOURCE_MAP,
        "task_ids_rerun": rerun_ids,
        "n_models": len(models),
        "grand_sm_mean_old_corrected": grand_old,
        "grand_sm_mean_new": grand_new,
        "models": summary,
    }


def main():
    overlay_out, summary, failures, n_rescore_checked, rerun_ids, models = build_overlay()
    print(f"models passing all gates: {len(overlay_out)}/{len(models)}")
    print(f"rescore checks run: {n_rescore_checked}")
    if failures:
        print(f"GATE FAILURES ({len(failures)}):")
        for x in failures[:40]:
            print("  " + x)
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")
        sys.exit(1)

    doc = summary_doc(overlay_out, summary, rerun_ids, models)
    atomic_dump(overlay_out, OUT / "sm_scores_overlay.json")
    atomic_dump(doc, OUT / "sm_rerun_summary.json")
    print(f"grand SM mean old(corrected)={doc['grand_sm_mean_old_corrected']:.4f} "
          f"new={doc['grand_sm_mean_new']:.4f}")
    print("ALL GATES PASSED")


if __name__ == "__main__":
    main()
