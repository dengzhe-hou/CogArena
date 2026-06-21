#!/usr/bin/env python3
"""fp16-vs-default-quantization de-confound for the predictive-validity check
(paper Section 5.6 and limitation 7).

CogArena scores are produced under Ollama's default per-tag quantization
(~4-bit Q4_K_M for the 7B class), while the published external benchmarks used
in Section 5.6 are full precision. To show the Section 5.6 associations are not
a precision artifact, 6 models were re-run at fp16 (single-turn battery only)
and compared to their default-quant scores.

This script re-aggregates BOTH the default-quant and the fp16 single-turn
batteries from per-item details.json (continuous per-item mean, since the
inline aggregator under-counts dict-scored paradigms), builds the per-paradigm
table, and reports the per-paradigm and per-grouping agreement (Pearson r and
mean |delta|).

Reproduces results/reanalysis/fp16_deconfound.json
(paper: per-paradigm Pearson r=0.86, mean|delta|=7.2pp; grouping r=0.79).

CPU-only, numpy/scipy. Run on a Slurm node, not the login node.
"""
import json, os
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr

import os
ROOT = os.environ.get("COGARENA_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
Q4_DIR = os.path.join(ROOT, "results/full_eval_20260526_2208")
FP16_DIR = os.path.join(ROOT, "results/full_eval_fp16")
OUT = os.path.join(ROOT, "results/reanalysis/fp16_deconfound.json")

# 6 models re-run at fp16. Default-quant dir = openai_<tag>; fp16 dir uses the
# Ollama fp16 tag naming openai_<tag>-instruct-fp16.
MODELS = ["qwen2.5:0.5b", "qwen2.5:7b", "qwen2.5:14b", "gemma2:9b", "llama3.1:8b", "mistral:7b"]
FP16_SUFFIX = "-instruct-fp16"

PARADIGMS = [
    'digit_span', 'stroop', 'flanker', 'go_nogo',
    'drm_false_memory', 'source_monitoring',
    'false_belief', 'epitome_tom',
    'confidence_calibration', 'post_decision_wagering',
]

# Single-turn grouping membership (n_back/operation_span/cvlt are multi-turn,
# absent from this battery, so WM here is digit_span only).
GROUP = {
    'digit_span': 'WM',
    'stroop': 'Control', 'flanker': 'Control', 'go_nogo': 'Control',
    'drm_false_memory': 'Episodic', 'source_monitoring': 'Episodic',
    'false_belief': 'ToM', 'epitome_tom': 'ToM',
    'confidence_calibration': 'Meta', 'post_decision_wagering': 'Meta',
}
GROUPS = ["WM", "Control", "Episodic", "ToM", "Meta"]


def item_accuracy(score):
    if isinstance(score, dict):
        if 'accuracy' in score:
            return float(score['accuracy'])
        if 'score' in score:
            return float(score['score'])
        if 'correct' in score:
            return 1.0 if score['correct'] else 0.0
        return 0.0
    return float(score)


def reaggregate(details_path):
    """Per-paradigm continuous mean accuracy from per-item details.json."""
    if not os.path.exists(details_path):
        return None
    items = json.load(open(details_path))
    by_par = defaultdict(list)
    for it in items:
        par = it.get('paradigm')
        if par in PARADIGMS:
            by_par[par].append(item_accuracy(it.get('score')))
    return {p: float(np.mean(v)) for p, v in by_par.items() if v}


def main():
    table = {}
    q4_cells, fp16_cells = [], []          # per-paradigm (60 cells)
    g_q4_cells, g_fp16_cells = [], []      # per-grouping (30 cells)

    for m in MODELS:
        q4 = reaggregate(os.path.join(Q4_DIR, "openai_" + m, "text", "details.json"))
        fp16 = reaggregate(os.path.join(FP16_DIR, "openai_" + m + FP16_SUFFIX, "text", "details.json"))
        if q4 is None or fp16 is None:
            print(f"WARNING missing data for {m} (q4={q4 is not None}, fp16={fp16 is not None})")
            continue
        table[m] = {}
        for p in PARADIGMS:
            if p in q4 and p in fp16:
                table[m][p] = [round(q4[p], 3), round(fp16[p], 3)]
                q4_cells.append(q4[p])
                fp16_cells.append(fp16[p])
        # grouping means (per model)
        for g in GROUPS:
            ps = [p for p in PARADIGMS if GROUP[p] == g and p in q4 and p in fp16]
            if ps:
                g_q4_cells.append(float(np.mean([q4[p] for p in ps])))
                g_fp16_cells.append(float(np.mean([fp16[p] for p in ps])))

    per_par_r, _ = pearsonr(q4_cells, fp16_cells)
    per_par_mad = float(np.mean(np.abs(np.array(q4_cells) - np.array(fp16_cells)))) * 100
    grp_r, _ = pearsonr(g_q4_cells, g_fp16_cells)
    grp_mad = float(np.mean(np.abs(np.array(g_q4_cells) - np.array(g_fp16_cells)))) * 100

    out = {
        "models": MODELS,
        "n_models": len(MODELS),
        "static_only": True,
        "per_paradigm": {
            "pearson_r": round(per_par_r, 3),
            "n_cells": len(q4_cells),
            "mean_abs_delta_pp": round(per_par_mad, 1),
        },
        "domain": {
            "pearson_r": round(grp_r, 3),
            "n_cells": len(g_q4_cells),
            "mean_abs_delta_pp": round(grp_mad, 1),
        },
        "note": ("fp16 re-run of single-turn battery vs default-quant (Q4); re-aggregated "
                 "from per-item details.json (the inline aggregate undercounts dict-scored "
                 "paradigms). Each cell is [Q4, fp16] per-paradigm continuous-mean accuracy. "
                 "'domain' = the 5 theory groupings; no systematic precision reordering."),
        "table_q4_vs_fp16": table,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"per-paradigm: Pearson r={out['per_paradigm']['pearson_r']} "
          f"n={out['per_paradigm']['n_cells']} mean|d|={out['per_paradigm']['mean_abs_delta_pp']}pp")
    print(f"grouping:     Pearson r={out['domain']['pearson_r']} "
          f"n={out['domain']['n_cells']} mean|d|={out['domain']['mean_abs_delta_pp']}pp")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
