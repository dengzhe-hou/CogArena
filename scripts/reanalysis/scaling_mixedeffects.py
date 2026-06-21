#!/usr/bin/env python3
"""TASK P0-2: Fix scaling pseudoreplication via family-aware mixed-effects models.

For each of the 13 cognitive paradigms we have accuracy x 20 models. Model size
(log scale) is the predictor. Because most models cluster into a few families
(qwen2.5 x6, gemma2 x3, llama3.2 x2, deepseek-r1 x2 plus 7 singletons), treating
the 20 models as independent for a Pearson scaling correlation is
pseudoreplication: within-family points are not independent draws.

We fit, per paradigm:
    accuracy ~ log(size)   with random intercept (1 | family)
using statsmodels MixedLM. We report the fixed-effect log(size) slope, its 95%
CI and p-value, and compare against the paper's bootstrap Pearson r. We also
compute the pure OLS Pearson r on the same 20 points (to reproduce the
"naive" estimate) and pure within-Qwen2.5 (n=6) / within-Gemma2 (n=3) OLS slopes
as a sensitivity check.

CPU-only, numpy/scipy/statsmodels.
"""
import json
import glob
import os
import warnings
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

STATIC_DIR = "results/full_eval_20260526_2208"
MT_DIR = "results/multiturn_eval_v3"

# Reused verbatim from scripts/compute_predictive_validity.py
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

# 13 paradigms in canonical order (matches DOMAIN_MAP grouping)
PARADIGMS = [
    'n_back', 'digit_span', 'operation_span',
    'stroop', 'flanker', 'go_nogo',
    'cvlt_word_list', 'drm_false_memory', 'source_monitoring',
    'false_belief', 'epitome_tom',
    'confidence_calibration', 'post_decision_wagering',
]

# Paper-reported bootstrap Pearson r (from task description)
PAPER_R = {
    'stroop': 0.71, 'epitome_tom': 0.70, 'drm_false_memory': 0.70, 'false_belief': 0.62,
    'n_back': 0.13, 'cvlt_word_list': 0.10, 'go_nogo': 0.12,
}
# Strong paradigms per paper narrative
STRONG_PARADIGMS = {'stroop', 'epitome_tom', 'drm_false_memory', 'false_belief'}


def family_of(model_key):
    """Map a model key (e.g. 'qwen2.5:7b') to a base family label."""
    base = model_key.split(':')[0]
    return base


def load_merged():
    """Reproduce the exact domain-score input construction: static paradigms
    overridden by multi-turn n_back/operation_span/cvlt_word_list."""
    static = {}
    for f in sorted(glob.glob(f"{STATIC_DIR}/*/text/aggregate.json")):
        d = json.load(open(f))
        m = d["model"].replace("openai/", "")
        static[m] = {p: v["accuracy"] for p, v in d["paradigms"].items()}

    mt = {}
    for f in sorted(glob.glob(f"{MT_DIR}/*/aggregate.json")):
        d = json.load(open(f))
        m = d["model"].replace("openai/", "")
        mt[m] = {k.split("/")[-1]: v["accuracy"] for k, v in d.get("dimensions", {}).items()}

    merged = {}
    for m in static:
        merged[m] = dict(static[m])
        if m in mt:
            for p in ["n_back", "operation_span", "cvlt_word_list"]:
                if p in mt[m]:
                    merged[m][p] = mt[m][p]
    return merged


def ols_slope(x, y):
    """Simple OLS slope of y on x; returns (slope, n). Needs >=2 distinct x."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) == 0:
        return None, len(x)
    X = np.column_stack([x, np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), len(x)


def main():
    merged = load_merged()
    # Restrict to the 20 models that have a size
    models = sorted([m for m in merged if m in SIZE_MAP])

    # Sanity: all 13 paradigms present for all models
    dropped_info = {}
    for p in PARADIGMS:
        miss = [m for m in models if p not in merged[m]]
        if miss:
            dropped_info[p] = miss

    results = {
        "method": (
            "statsmodels MixedLM: accuracy ~ log10(size) with random intercept "
            "(1|family). Fixed-effect log10(size) slope, 95% CI (fe_params +/- "
            "1.96*bse), and Wald p. Compared to paper bootstrap Pearson r and to "
            "naive OLS Pearson r on the same 20 points. Sensitivity: within-Qwen2.5 "
            "(n=6) and within-Gemma2 (n=3) OLS log10(size) slopes."
        ),
        "n_models": len(models),
        "models": models,
        "family_counts": {},
        "log_base": "log10",
        "dropped_for_missing_data": dropped_info,
        "paradigms": {},
        "within_family_slopes": {},
    }

    # Family counts
    fams = [family_of(m) for m in models]
    from collections import Counter
    results["family_counts"] = dict(Counter(fams))

    # Precompute size / log-size / family per model
    log_size = {m: float(np.log10(SIZE_MAP[m])) for m in models}
    fam = {m: family_of(m) for m in models}

    # Within-family subsets
    qwen_models = [m for m in models if fam[m] == 'qwen2.5']
    gemma_models = [m for m in models if fam[m] == 'gemma2']

    for p in PARADIGMS:
        acc = {m: float(merged[m][p]) for m in models if p in merged[m]}
        mods_p = [m for m in models if m in acc]
        x = np.array([log_size[m] for m in mods_p])
        y = np.array([acc[m] for m in mods_p])
        n = len(mods_p)

        # Naive OLS Pearson r on the 20 points (reproduces naive estimate)
        naive_r, naive_p = pearsonr(y, x)
        naive_slope, _ = ols_slope(x, y)

        # Mixed effects: accuracy ~ log10(size), random intercept by family
        df = pd.DataFrame({
            "acc": y,
            "logsize": x,
            "family": [fam[m] for m in mods_p],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm("acc ~ logsize", df, groups=df["family"])
            try:
                mdf = md.fit(method="lbfgs", reml=True)
                fe_slope = float(mdf.fe_params["logsize"])
                bse = float(mdf.bse_fe["logsize"]) if "logsize" in mdf.bse_fe else float(mdf.bse["logsize"])
                pval = float(mdf.pvalues["logsize"])
                ci_lo = fe_slope - 1.96 * bse
                ci_hi = fe_slope + 1.96 * bse
                # random-intercept variance
                try:
                    group_var = float(mdf.cov_re.iloc[0, 0])
                except Exception:
                    group_var = float(np.asarray(mdf.cov_re).ravel()[0])
                resid_var = float(mdf.scale)
                converged = bool(mdf.converged)
            except Exception as e:
                fe_slope = bse = pval = ci_lo = ci_hi = group_var = resid_var = float("nan")
                converged = False

        paper_r = PAPER_R.get(p)
        # still_strong: only meaningful for paradigms the paper called strong.
        # True iff family-aware LMM keeps the strong-scaling conclusion:
        # positive slope AND significant (p<0.05).
        is_strong_claim = p in STRONG_PARADIGMS
        still_strong = bool(is_strong_claim and (fe_slope > 0) and (pval < 0.05))

        results["paradigms"][p] = {
            "domain": DOMAIN_MAP[p],
            "n": n,
            "paper_bootstrap_pearson_r": paper_r,
            "naive_pearson_r": float(naive_r),
            "naive_pearson_p": float(naive_p),
            "naive_ols_slope_log10size": naive_slope,
            "lmm_fixed_slope_log10size": fe_slope,
            "lmm_slope_se": bse,
            "lmm_slope_ci95": [float(ci_lo), float(ci_hi)],
            "lmm_slope_p": pval,
            "lmm_group_intercept_var": group_var,
            "lmm_resid_var": resid_var,
            "lmm_converged": converged,
            "is_paper_strong_claim": is_strong_claim,
            "still_strong": still_strong,
        }

    # Within-family sensitivity slopes (OLS on log10 size)
    for fam_name, fam_mods in [("qwen2.5", qwen_models), ("gemma2", gemma_models)]:
        per_par = {}
        for p in PARADIGMS:
            mm = [m for m in fam_mods if p in merged[m]]
            x = [log_size[m] for m in mm]
            yv = [float(merged[m][p]) for m in mm]
            slope, nn = ols_slope(x, yv)
            # Pearson r too (if enough points & variance)
            if nn >= 2 and np.ptp(x) > 0 and np.ptp(yv) > 0:
                r, pr = pearsonr(yv, x)
                r = float(r); pr = float(pr)
            else:
                r = pr = None
            per_par[p] = {
                "n": nn,
                "ols_slope_log10size": slope,
                "pearson_r": r,
                "pearson_p": pr,
            }
        results["within_family_slopes"][fam_name] = {
            "models": fam_mods,
            "n": len(fam_mods),
            "per_paradigm": per_par,
        }

    out_path = "results/reanalysis/scaling_mixedeffects.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Console summary
    print(f"n_models = {len(models)}")
    print(f"family_counts = {results['family_counts']}")
    print(f"dropped_for_missing_data = {dropped_info}")
    print()
    hdr = f"{'paradigm':22s} {'paperR':>7s} {'naiveR':>7s} {'lmmSlope':>9s} {'ci95':>22s} {'p':>9s} {'strong?':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for p in PARADIGMS:
        r = results["paradigms"][p]
        pr = r["paper_bootstrap_pearson_r"]
        prs = f"{pr:.2f}" if pr is not None else "  -"
        ci = r["lmm_slope_ci95"]
        cis = f"[{ci[0]:+.4f},{ci[1]:+.4f}]"
        strong = ("YES" if r["still_strong"] else ("no" if r["is_paper_strong_claim"] else "n/a"))
        print(f"{p:22s} {prs:>7s} {r['naive_pearson_r']:>+7.3f} {r['lmm_fixed_slope_log10size']:>+9.4f} {cis:>22s} {r['lmm_slope_p']:>9.2e} {strong:>8s}")

    print("\n=== Within-family sensitivity (OLS slope per log10 size) ===")
    for fam_name in ["qwen2.5", "gemma2"]:
        wf = results["within_family_slopes"][fam_name]
        print(f"\n{fam_name} (n={wf['n']}): {wf['models']}")
        for p in PARADIGMS:
            d = wf["per_paradigm"][p]
            rs = f"{d['pearson_r']:+.3f}" if d["pearson_r"] is not None else "  -"
            ss = f"{d['ols_slope_log10size']:+.4f}" if d["ols_slope_log10size"] is not None else "  -"
            print(f"  {p:22s} slope={ss:>9s}  r={rs:>7s}  (n={d['n']})")

    print(f"\nSaved to {out_path}")
    return results


if __name__ == "__main__":
    main()
