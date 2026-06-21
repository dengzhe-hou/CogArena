#!/usr/bin/env python3
"""
TASK P0-3: Signature significance.

Paper section 5.2 reports per-model replication of behavioral signatures as raw
fractions (Stroop 9/20, False Belief 12/20, Flanker 17/20) with NO statistical
test. This script computes, for each condition-split paradigm:

  1. For each of the 20 text LLMs, whether the expected DIRECTIONAL contrast holds
     (e.g. congruent > incongruent for Stroop/Flanker; order1 > order2 for False
     Belief; 1-back > 2-back load effect for n-back; non-belief > belief for EPITOME).
  2. k/N models in the expected direction.
  3. A ONE-SIDED binomial test (alternative='greater') under H0: p=0.5
     (a model with no real effect is equally likely to land either side; we test
     whether models land in the expected direction MORE often than chance).
  4. BH (Benjamini-Hochberg) correction across the paradigms.
  5. Classification: replicates / weak / does_not_replicate.

Data sources (reusing validated plumbing conventions from
scripts/compute_predictive_validity.py; model keys drop the "openai/" prefix):
  - Static per-item JSONs:  results/full_eval_20260526_2208/<model>/text/<dim>/<paradigm>/*.json
      score.condition in {congruent, incongruent} (Stroop, Flanker), score.correct
      score.order in {1.0, 2.0}, score.accuracy (False Belief)
  - n-back load effect:     results/multiturn_eval_v3/<model>/working_memory/n_back/*.json
      filename / task_id encodes load nN (n1/n2/n3), score.accuracy per episode
  - EPITOME:                results/epitome_v3_rerun/<model>.json
      sub_acc[capacity].accuracy  (belief vs desire/intention/emotion)

CPU-only, numpy/scipy/statsmodels.
"""
import json, glob, os, re
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

import os
ROOT = os.environ.get("COGARENA_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STATIC_DIR = os.path.join(ROOT, "results/full_eval_20260526_2208")
MT_DIR = os.path.join(ROOT, "results/multiturn_eval_v3")
EPITOME_DIR = os.path.join(ROOT, "results/epitome_v3_rerun")
OUT = os.path.join(ROOT, "results/reanalysis/signature_significance.json")

# Canonical 20 text LLMs (SIZE_MAP from compute_predictive_validity.py)
MODELS = [
    'tinyllama:1.1b', 'qwen2.5:0.5b', 'qwen2.5:1.5b', 'gemma2:2b', 'llama3.2:1b',
    'qwen2.5:3b', 'llama3.2:3b', 'qwen2.5:7b', 'mistral:7b', 'llama3.1:8b',
    'deepseek-r1:7b', 'gemma2:9b', 'qwen2.5:14b', 'phi3:14b', 'deepseek-r1:14b',
    'gemma2:27b', 'qwen2.5:32b', 'mixtral:8x7b', 'yi:34b', 'command-r:35b',
]

def model_dirname(m):
    """full_eval / multiturn use 'openai_<tag>' directory naming."""
    return "openai_" + m

def epitome_filename(m):
    """epitome_v3_rerun uses '<family>_<size>.json' (':' -> '_')."""
    return m.replace(":", "_") + ".json"


# ---------------------------------------------------------------------------
# Per-paradigm: return (acc_expected_easy, acc_expected_hard) condition means
# for a single model, where the EXPECTED contrast is easy > hard.
# Return None if data missing.
# ---------------------------------------------------------------------------

def stroop_flanker_split(model, paradigm):
    """congruent (easy) > incongruent (hard). Static per-item files."""
    d = os.path.join(STATIC_DIR, model_dirname(model), "text", "cognitive_control", paradigm)
    files = glob.glob(os.path.join(d, "*.json"))
    if not files:
        return None
    cong, incong = [], []
    for f in files:
        j = json.load(open(f))
        sc = j.get("score", {})
        cond = sc.get("condition")
        cor = sc.get("correct")
        if cor is None or cond is None:
            continue
        v = 1.0 if cor else 0.0
        if cond == "congruent":
            cong.append(v)
        elif cond == "incongruent":
            incong.append(v)
    if not cong or not incong:
        return None
    return float(np.mean(cong)), float(np.mean(incong)), len(cong), len(incong)


def false_belief_split(model):
    """1st order (easy) > 2nd order (hard). Static per-item files, score.order, score.accuracy."""
    d = os.path.join(STATIC_DIR, model_dirname(model), "text", "theory_of_mind", "false_belief")
    files = glob.glob(os.path.join(d, "*.json"))
    if not files:
        return None
    o1, o2 = [], []
    for f in files:
        j = json.load(open(f))
        sc = j.get("score", {})
        order = sc.get("order")
        acc = sc.get("accuracy")
        if order is None or acc is None:
            continue
        if abs(order - 1.0) < 1e-9:
            o1.append(float(acc))
        elif abs(order - 2.0) < 1e-9:
            o2.append(float(acc))
    if not o1 or not o2:
        return None
    return float(np.mean(o1)), float(np.mean(o2)), len(o1), len(o2)


def nback_load_split(model):
    """Load effect: 1-back (easy) > 2-back (hard). Multi-turn per-episode files.
    Uses the paper's stated signature (1-back > 2-back). Each episode contributes its
    score.accuracy; load read from task_id / filename 'nN'."""
    d = os.path.join(MT_DIR, model_dirname(model), "working_memory", "n_back")
    files = glob.glob(os.path.join(d, "*.json"))
    if not files:
        return None
    by_load = defaultdict(list)
    for f in files:
        j = json.load(open(f))
        tid = j.get("task_id", os.path.basename(f))
        mobj = re.search(r"_n(\d+)_", tid) or re.search(r"_n(\d+)\b", tid)
        if not mobj:
            mobj = re.search(r"n(\d+)", os.path.basename(f))
        if not mobj:
            continue
        load = int(mobj.group(1))
        acc = j.get("score", {}).get("accuracy")
        if acc is None:
            continue
        by_load[load].append(float(acc))
    if 1 not in by_load or 2 not in by_load:
        return None
    return float(np.mean(by_load[1])), float(np.mean(by_load[2])), len(by_load[1]), len(by_load[2])


def epitome_split(model):
    """Canonical ToM signature: belief (false belief) is the hardest sub-capacity.
    Expected contrast: mean(desire,intention,emotion) (easy) > belief (hard).
    epitome_v3_rerun/<file>.json sub_acc[cap].accuracy."""
    f = os.path.join(EPITOME_DIR, epitome_filename(model))
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    sub = j.get("sub_acc", {})
    if "belief" not in sub:
        return None
    belief = sub["belief"]["accuracy"]
    others = [sub[k]["accuracy"] for k in ("desire", "intention", "emotion") if k in sub]
    if not others:
        return None
    n_belief = sub["belief"].get("total")
    n_others = sum(sub[k].get("total", 0) for k in ("desire", "intention", "emotion") if k in sub)
    return float(np.mean(others)), float(belief), int(n_others or 0), int(n_belief or 0)


PARADIGMS = [
    # name, contrast description, expected-easy label, expected-hard label, extractor
    ("stroop", "congruent > incongruent", "congruent", "incongruent",
     lambda m: stroop_flanker_split(m, "stroop")),
    ("flanker", "congruent > incongruent", "congruent", "incongruent",
     lambda m: stroop_flanker_split(m, "flanker")),
    ("false_belief", "1st-order > 2nd-order", "order1", "order2",
     false_belief_split),
    ("n_back_load", "1-back > 2-back", "1-back", "2-back",
     nback_load_split),
    ("epitome", "non-belief (desire/intention/emotion) > belief", "non_belief", "belief",
     epitome_split),
]


def classify(k, n, p_bh):
    """replicates / weak / does_not_replicate.
    - replicates: significant after BH (p_bh < 0.05) AND majority in expected direction.
    - weak: majority in expected direction (k/n > 0.5) but not significant.
    - does_not_replicate: not a majority in expected direction (k/n <= 0.5)."""
    frac = k / n
    if frac <= 0.5:
        return "does_not_replicate"
    if p_bh < 0.05:
        return "replicates"
    return "weak"


def main():
    rows = []
    for name, desc, easy_lab, hard_lab, fn in PARADIGMS:
        k = 0
        n = 0
        dropped = []
        per_model = {}
        easy_means, hard_means = [], []
        for m in MODELS:
            res = fn(m)
            if res is None:
                dropped.append(m)
                continue
            easy_acc, hard_acc, n_easy, n_hard = res
            n += 1
            in_dir = easy_acc > hard_acc  # STRICT: ties do NOT count as expected direction
            if in_dir:
                k += 1
            per_model[m] = {
                "easy_acc": round(easy_acc, 6),
                "hard_acc": round(hard_acc, 6),
                "delta": round(easy_acc - hard_acc, 6),
                "n_easy_items": n_easy,
                "n_hard_items": n_hard,
                "in_expected_direction": bool(in_dir),
            }
            easy_means.append(easy_acc)
            hard_means.append(hard_acc)
        bt = binomtest(k, n, 0.5, alternative="greater")
        rows.append({
            "paradigm": name,
            "contrast": desc,
            "expected_easy": easy_lab,
            "expected_hard": hard_lab,
            "k_models_expected_dir": k,
            "n": n,
            "fraction": f"{k}/{n}",
            "p_binom_onesided": float(bt.pvalue),
            "aggregate_easy_mean": round(float(np.mean(easy_means)), 6),
            "aggregate_hard_mean": round(float(np.mean(hard_means)), 6),
            "aggregate_delta": round(float(np.mean(easy_means) - np.mean(hard_means)), 6),
            "models_dropped_missing_data": dropped,
            "per_model": per_model,
        })

    # BH correction across paradigms
    pvals = [r["p_binom_onesided"] for r in rows]
    rej, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    for r, pa, rj in zip(rows, p_adj, rej):
        r["p_bh_adjusted"] = float(pa)
        r["bh_significant"] = bool(rj)
        r["classification"] = classify(r["k_models_expected_dir"], r["n"], r["p_bh_adjusted"])

    out = {
        "task": "P0-3 signature significance",
        "description": (
            "One-sided binomial test (H0: p=0.5, alternative=greater) on per-model "
            "directional replication of behavioral signatures, BH-corrected across "
            "the 5 condition-split paradigms. A model counts in the expected direction "
            "iff easy-condition accuracy STRICTLY exceeds hard-condition accuracy."
        ),
        "n_models_total": len(MODELS),
        "bh_method": "fdr_bh (Benjamini-Hochberg) across paradigms, alpha=0.05",
        "binom_null": "p=0.5, one-sided (alternative='greater')",
        "paradigms": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    # Console report
    print(f"{'paradigm':16s} {'k/n':>7s} {'p_binom':>10s} {'p_BH':>10s}  {'class':<18s} contrast")
    print("-" * 90)
    for r in rows:
        print(f"{r['paradigm']:16s} {r['fraction']:>7s} {r['p_binom_onesided']:10.5f} "
              f"{r['p_bh_adjusted']:10.5f}  {r['classification']:<18s} {r['contrast']}")
        if r["models_dropped_missing_data"]:
            print(f"    dropped (missing data): {r['models_dropped_missing_data']}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
