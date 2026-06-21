#!/usr/bin/env python3
"""Compute predictive validity: CogArena domain scores vs external benchmarks.
Uses bivariate Spearman correlations and partial correlations controlling for log(size)."""
import json, glob, csv, os, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr
from numpy.linalg import lstsq

STATIC_DIR = "results/full_eval_20260526_2208"
MT_DIR = "results/multiturn_eval_v3"

DOMAIN_MAP = {
    'n_back': 'WM', 'digit_span': 'WM', 'operation_span': 'WM',
    'stroop': 'Control', 'flanker': 'Control', 'go_nogo': 'Control',
    'cvlt_word_list': 'Episodic', 'drm_false_memory': 'Episodic', 'source_monitoring': 'Episodic',
    'false_belief': 'ToM', 'epitome_tom': 'ToM',
    'confidence_calibration': 'Meta', 'post_decision_wagering': 'Meta'
}

SIZE_MAP = {
    'tinyllama:1.1b': 1.1, 'qwen2.5:0.5b': 0.5, 'qwen2.5:1.5b': 1.5,
    'gemma2:2b': 2, 'llama3.2:1b': 1, 'qwen2.5:3b': 3, 'llama3.2:3b': 3,
    'qwen2.5:7b': 7, 'mistral:7b': 7, 'llama3.1:8b': 8, 'deepseek-r1:7b': 7,
    'gemma2:9b': 9, 'qwen2.5:14b': 14, 'phi3:14b': 14, 'deepseek-r1:14b': 14,
    'gemma2:27b': 27, 'qwen2.5:32b': 32, 'mixtral:8x7b': 47, 'yi:34b': 34,
    'command-r:35b': 35
}

def main():
    # Load static paradigms
    static = {}
    for f in sorted(glob.glob(f"{STATIC_DIR}/*/text/aggregate.json")):
        d = json.load(open(f))
        m = d["model"].replace("openai/", "")
        static[m] = {p: v["accuracy"] for p, v in d["paradigms"].items()}

    # conf_cal scorer-fix override (fixed metacognition scorer); no-op if map absent
    _ccp = "results/reanalysis/conf_cal_corrected.json"
    _cc = json.load(open(_ccp)) if os.path.exists(_ccp) else {}
    for _m in static:
        if _m in _cc and "confidence_calibration" in static[_m]:
            static[_m]["confidence_calibration"] = _cc[_m]

    # Load multi-turn paradigms
    mt = {}
    for f in sorted(glob.glob(f"{MT_DIR}/*/aggregate.json")):
        d = json.load(open(f))
        m = d["model"].replace("openai/", "")
        mt[m] = {k.split("/")[-1]: v["accuracy"] for k, v in d.get("dimensions", {}).items()}

    # Merge
    merged = {}
    for m in static:
        merged[m] = dict(static[m])
        if m in mt:
            for p in ["n_back", "operation_span", "cvlt_word_list"]:
                if p in mt[m]:
                    merged[m][p] = mt[m][p]

    # Domain scores
    domain_scores = {}
    for m in merged:
        ds = {}
        for dom in ["WM", "Control", "Episodic", "ToM", "Meta"]:
            ps = [merged[m][p] for p in merged[m] if DOMAIN_MAP.get(p) == dom]
            ds[dom] = float(np.mean(ps)) if ps else 0.0
        domain_scores[m] = ds

    # Load benchmarks
    benchmarks = {}
    with open("data/published_benchmarks.csv") as f:
        for row in csv.DictReader(f):
            m = row["model"]
            benchmarks[m] = {col: float(row[col]) if row.get(col, "N/A") != "N/A" else None
                             for col in ["mmlu", "arc_challenge", "gsm8k"]}

    # Bivariate Spearman
    results = {"bivariate_spearman": {}, "partial_correlations": {}}
    print(f"{'':15s} {'MMLU':>15s} {'ARC':>15s} {'GSM8K':>15s}")
    for dom in ["WM", "Control", "Episodic", "ToM", "Meta"]:
        line = f"  {dom:13s}"
        for bm in ["mmlu", "arc_challenge", "gsm8k"]:
            pairs = [(domain_scores[m][dom], benchmarks[m][bm])
                     for m in domain_scores if m in benchmarks and benchmarks[m].get(bm) is not None]
            if len(pairs) >= 5:
                x, y = zip(*pairs)
                rho, p = spearmanr(x, y)
                results["bivariate_spearman"][f"{dom}_vs_{bm}"] = {"rho": round(rho, 3), "p": round(p, 3), "n": len(pairs)}
                line += f"  ρ={rho:.2f} p={p:.2f}{'*' if p < 0.05 else ' '}"
            else:
                line += f"  {'N/A':>13s}"
        print(line)

    # Partial correlations controlling for log(size)
    print(f"\nPartial correlations (controlling for log size):")
    for dom in ["WM", "Control", "Episodic", "ToM", "Meta"]:
        for bm in ["mmlu", "arc_challenge"]:
            mods = [m for m in domain_scores if m in benchmarks and benchmarks[m].get(bm) is not None and m in SIZE_MAP]
            if len(mods) < 8:
                continue
            x = np.array([domain_scores[m][dom] for m in mods])
            y = np.array([benchmarks[m][bm] for m in mods])
            z = np.array([np.log(SIZE_MAP[m]) for m in mods])
            Z = np.column_stack([z, np.ones(len(z))])
            x_res = x - Z @ lstsq(Z, x, rcond=None)[0]
            y_res = y - Z @ lstsq(Z, y, rcond=None)[0]
            r_partial, _ = pearsonr(x_res, y_res)
            results["partial_correlations"][f"{dom}_vs_{bm}"] = {"partial_r": round(r_partial, 3), "n": len(mods)}
            print(f"  {dom:10s} vs {bm:15s} partial_r={r_partial:.3f} (n={len(mods)})")

    json.dump(results, open("results/predictive_validity.json", "w"), indent=2)
    print(f"\nSaved to results/predictive_validity.json")

if __name__ == "__main__":
    main()
