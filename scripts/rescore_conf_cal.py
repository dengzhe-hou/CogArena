#!/usr/bin/env python3
"""Re-score confidence_calibration with the FIXED metacognition scorer
(diacritic + numeric-format normalization) for all 55 models, using the STORED
responses (no model re-run). Writes the corrected per-model accuracy map to
results/reanalysis/conf_cal_corrected.json, which the analysis scripts pick up
to propagate the scorer fix. Run on a compute node via Slurm (not the login node).
"""
import sys, os, json, glob, statistics
import os
ROOT = os.environ.get("COGARENA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
from cogarena.generators.metacognition_gen import generate_mc_items
from cogarena.dimensions.metacognition import ConfidenceCalibrationGenerator as CC
import compute_b2_expanded as B2

# Reproduce the eval's confidence_calibration items (seed=42, probes=False) to
# recover each item's gold answer keyed by task_id.
items = generate_mc_items(seed=42, n_per_paradigm=50, include_contamination_probes=False)
itmap = {it.task_id: it for it in items}


def rescore(text_dir):
    accs = []
    for f in glob.glob(f"{text_dir}/metacognition/confidence_calibration/*.json"):
        d = json.load(open(f))
        t = d.get("task_id")
        if t in itmap:
            try:
                accs.append(CC.score(itmap[t], d.get("response", "")).get("accuracy"))
            except Exception:
                pass
    return statistics.mean(accs) if accs else None


corrected = {}
for m in B2.OLD_MODELS:
    a = rescore(f"{ROOT}/results/full_eval_20260526_2208/openai_{m}/text")
    if a is not None:
        corrected[m] = a
for m in [l.strip() for l in open(f"{ROOT}/results/reanalysis/expansion_modellist.txt") if l.strip()]:
    a = rescore(f"{ROOT}/results/full_eval_expansion/openai_{m}/text")
    if a is not None:
        corrected[m] = a

out = f"{ROOT}/results/reanalysis/conf_cal_corrected.json"
json.dump(corrected, open(out, "w"), indent=1)
print(f"wrote {out} for {len(corrected)} models")
print("mean corrected conf_cal:", round(sum(corrected.values()) / len(corrected) * 100, 1), "%")
