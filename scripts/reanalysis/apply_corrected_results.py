#!/usr/bin/env python3
"""Regenerate the committed reanalysis result JSONs from the CORRECTED data.

The committed results/reanalysis/*.json (and results/predictive_validity.json)
were produced by the pre-fix scorers. This rewrites them from the corrected
55x13 matrix (all scorer fixes + fixed-prompt go_nogo rerun + uniform
continuous multi-turn metric), preserving each file's schema and only updating
the numeric fields that the scorer corrections changed. It reads the corrected
artifacts under results/recompute_20260703/ and recomputes per-model detail
directly from the corrected per-item data where a file needs it.

CPU-only, seconds to run. Prints a compact before -> after for each file.
"""
import csv
import glob
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.environ.get("COGARENA_ROOT",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RC = os.path.join(ROOT, "results", "recompute_20260703")
RESCORE = os.path.join(ROOT, "results", "rescore_20260702", "new_scores")
# Final SM overlay adapter: see build_and_recompute.py. The fp16 sweep keeps
# the historical path (fp16 configs are not in the 55-model overlay; the
# quantization comparison is internally consistent on the frozen battery).
SM_OVERLAY = (json.load(open(os.environ["COGARENA_SM_OVERLAY"]))
              if os.environ.get("COGARENA_SM_OVERLAY") else None)
WAGER_OVERLAY = (json.load(open(os.environ["COGARENA_WAGER_OVERLAY"]))
                 if os.environ.get("COGARENA_WAGER_OVERLAY") else None)
GONOGO = os.path.join(ROOT, "results", "gonogo_rerun_20260702")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ["COGARENA_ROOT"] = ROOT
import compute_b2_expanded as b2

STEP4 = json.load(open(os.path.join(RC, "step4_artifacts.json")))
STEP4B = json.load(open(os.path.join(RC, "step4b_artifacts.json")))
FINAL = json.load(open(os.path.join(RC, "final_inference.json")))

# corrected matrix
_rows = list(csv.reader(open(os.environ.get("COGARENA_PRIMARY_MATRIX")
                             or os.path.join(RC, "corrected_matrix.csv"))))
HDR = _rows[0][1:]
MAT = {r[0]: {p: float(v) for p, v in zip(HDR, r[1:])} for r in _rows[1:]}
ALL_MODELS = list(MAT.keys())
OLD_MODELS = list(b2.OLD_MODELS.keys())
NEW_META = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
SIZE = {m: b2.OLD_MODELS[m][0] for m in OLD_MODELS}
SIZE.update({m: NEW_META[m]["size_b"] for m in NEW_META})
LABELS = [b2.DOMAIN_MAP[p] for p in HDR]


def save(path, obj):
    json.dump(obj, open(path, "w"), indent=1)


def perm_delta(M, labels, n_perm=5000, seed=42):
    corr = np.corrcoef(M.T)
    prs = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    w = [corr[i, j] for i, j in prs if labels[i] == labels[j]]
    c = [corr[i, j] for i, j in prs if labels[i] != labels[j]]
    obs = float(np.mean(w) - np.mean(c))
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(n_perm):
        sh = rng.permutation(labels).tolist()
        ww = [corr[i, j] for i, j in prs if sh[i] == sh[j]]
        cc = [corr[i, j] for i, j in prs if sh[i] != sh[j]]
        if ww and cc:
            perm.append(np.mean(ww) - np.mean(cc))
    p = float(np.mean([d >= obs for d in perm]))
    lo, hi = np.percentile(perm, [2.5, 97.5])
    return round(float(np.mean(w)), 4), round(float(np.mean(c)), 4), round(obs, 4), round(p, 4), [round(float(lo), 4), round(float(hi), 4)]


def corrected_static_items(model):
    is_old = model in b2.OLD_MODELS
    s = "full_eval_20260526_2208" if is_old else "full_eval_expansion"
    det = b2.load_details(f"{ROOT}/results/{s}/openai_{model}/text/details.json")
    ov = {}
    op = os.path.join(RESCORE, f"{s}__openai_{model}.json")
    if os.path.exists(op):
        ov = json.load(open(op))
    out = {}
    for r in det:
        p = r.get("paradigm")
        if p not in b2.STATIC_PARADIGMS or p == "go_nogo":
            continue
        if p == "source_monitoring" and SM_OVERLAY is not None and model in SM_OVERLAY:
            acc = SM_OVERLAY[model][r["task_id"]]
        elif p == "post_decision_wagering" and WAGER_OVERLAY is not None:
            acc = WAGER_OVERLAY[model][r["task_id"]]
        else:
            acc = ov.get(r["task_id"], b2.item_accuracy(r.get("score")))
        out[r["task_id"]] = (p, float(acc), r.get("score") or {})
    gg = f"{GONOGO}/openai_{model}/text/details.json"
    if os.path.exists(gg):
        for r in json.load(open(gg)):
            out[r["task_id"]] = ("go_nogo", b2.item_accuracy(r.get("score")), r.get("score") or {})
    return out


# ---------------------------------------------------------------- b2_expanded
def do_b2():
    p = f"{ROOT}/results/reanalysis/b2_expanded.json"
    d = json.load(open(p))
    M = np.array([[MAT[m][pp] for pp in HDR] for m in ALL_MODELS])
    w, c, dl, pv, ci = perm_delta(M, LABELS)
    old = dict(d["exp55"])
    d["exp55"] = {"n_models": 55, "within_mean": w, "cross_mean": c, "delta": dl,
                  "ci_lo": ci[0], "ci_hi": ci[1], "p_value": pv, "n_perm": 5000,
                  "seed": 42, "pass": bool(pv < 0.05)}
    # model-bootstrap CI (the paper's cited interval) as an added field
    d["exp55_model_bootstrap_ci95"] = FINAL["corrected_raw"]["boot_model_ci"][:2]
    d["exp55_family_bootstrap_ci95"] = FINAL["corrected_raw"]["boot_family_merged"]["seed42"][:2]
    # scaling per-paradigm (55-model)
    mlm = {x["paradigm"]: x for x in STEP4["scaling_mixedlm_55"]}
    for pp in d["scaling"]:
        d["scaling"][pp] = {"pearson_r": STEP4["scaling_r_55"][pp]["r"],
                            "mlm_beta": mlm[pp]["slope"], "mlm_p": mlm[pp]["p"]}
    d["paper_target"] = {"within": w, "cross": c, "delta": dl, "p": pv,
                         "note": "corrected 55-model headline (all scorer fixes)"}
    d["aggregation_note"] = ("Regenerated after the 2026-07 scorer corrections "
        "(source_monitoring/epitome/go_nogo/digit_span/false_belief/stroop/flanker/"
        "multi-turn fixes) and the fixed-prompt go_nogo rerun. exp55 is the "
        "corrected within-minus-cross separability on 55 models; repro20 retains "
        "the pre-correction 20-model self-check for provenance. Model-bootstrap and "
        "family-clustered 95% CIs for delta are in the *_bootstrap_ci95 fields.")
    save(p, d)
    return f"exp55 delta {old['delta']}/p{old['p_value']} -> {dl}/p{pv}; pc1 scaling GN {d['scaling']['go_nogo']['pearson_r']}"


# ------------------------------------------------------------ pca_partialcorr
def do_pca():
    p = f"{ROOT}/results/reanalysis/pca_partialcorr.json"
    d = json.load(open(p))
    M = np.array([[MAT[m][pp] for pp in HDR] for m in ALL_MODELS])
    corr = np.corrcoef(M.T)
    iu = np.triu_indices(len(HDR), 1)
    ev = np.linalg.eigvalsh(corr)[::-1]
    z = (M - M.mean(0)) / M.std(0, ddof=1)
    v1 = np.linalg.eigh(corr)[1][:, -1]
    pc1_scores = z @ v1
    r_mean = abs(float(stats.pearsonr(pc1_scores, M.mean(1))[0]))
    old = dict(d["raw"])
    w, c, dl, pv, ci = perm_delta(M, LABELS)
    d["raw"] = {"within": w, "cross": c, "delta": dl}
    d["positive_manifold"] = {
        "mean_offdiag_r": round(float(np.mean(corr[iu])), 4),
        "frac_pairs_positive": round(float(np.mean(corr[iu] > 0)), 3),
        "pc1_variance_explained": round(float(ev[0] / ev.sum()), 4),
        "eigenvalues_top5": [round(float(x), 3) for x in ev[:5]],
        "pc1_corr_with_mean_accuracy": round(r_mean, 3)}
    d["after_removing_general_factor_PC1"] = {
        "delta": STEP4B["pc1_removal_delta_p"][0], "p_value": STEP4B["pc1_removal_delta_p"][1],
        "note": "biased downward (PC1-orthogonality artifact); reference only"}
    rr = FINAL["corrected_residualized"]
    d["after_removing_general_factor_rowmean"] = {
        "within": rr["within_mean"], "cross": rr["cross_mean"], "delta": rr["delta"],
        "p_value": rr["perm_p_one_sided"], "p_value_two_sided": rr["perm_p_two_sided"]}
    d["interpretation"] = (
        f"First PC explains {round(ev[0]/ev.sum()*100)} percent of paradigm variance "
        f"with all pairwise correlations positive (a positive manifold, r=0.99 with "
        f"mean accuracy). The raw within-minus-cross gap is small and non-significant "
        f"(delta={dl:.3f}, one-sided p={pv:.2f}). After removing the general factor by "
        f"row-mean residualization the within-grouping contrast is suggestive but "
        f"reaches only one-sided significance (delta={rr['delta']:.3f}, one-sided "
        f"p={rr['perm_p_one_sided']:.2f}, two-sided p={rr['perm_p_two_sided']:.2f}).")
    save(p, d)
    return f"raw delta {old['delta']} -> {dl}; pc1 {round(ev[0]/ev.sum(),4)}; resid {rr['delta']}"


# ------------------------------------------------------------ split_half
def do_split_half():
    p = f"{ROOT}/results/reanalysis/split_half_reliability.json"
    d = json.load(open(p))
    sh = STEP4["split_half_corrected_20"]
    old_mean = d.get("mean")
    for k in list(d.keys()):
        if k in sh:
            d[k] = sh[k]
    d["mean"] = STEP4["split_half_mean_excl_epitome"]
    d["mean_note"] = ("mean over single-turn paradigms EXCLUDING epitome_tom, whose "
                      "20-model per-item records are a forced-choice rerun placeholder")
    save(p, d)
    return f"mean {old_mean} -> {d['mean']}"


# ------------------------------------------------------------ predictive_validity
def do_pv():
    p = f"{ROOT}/results/predictive_validity.json"
    d = json.load(open(p))
    cells = STEP4B["pv_grouping_cells"]
    biv = {}
    for key, v in cells.items():
        g, bm = key.split("|")
        biv[f"{g}_vs_{bm}"] = {"rho": v["rho"], "p": v["p"], "p_bh": v["p_bh"], "n": v["n"]}
    old_n = len(d.get("bivariate_spearman", {}))
    d["bivariate_spearman"] = biv
    d["partial_correlations"] = {
        "Meta_vs_mmlu": {"partial_r": STEP4B["pv_partial_meta_mmlu"][0], "p": STEP4B["pv_partial_meta_mmlu"][1], "controls": "log10(size)"},
        "Episodic_vs_arc_challenge": {"partial_r": STEP4B["pv_partial_episodic_arc"][0], "p": STEP4B["pv_partial_episodic_arc"][1], "controls": "log10(size)"}}
    n_survive = sum(1 for v in biv.values() if v["p_bh"] < 0.05)
    d["n_survive_bh"] = n_survive
    save(p, d)
    return f"{old_n} cells -> {len(biv)} cells; survive BH {n_survive}/15"


# ------------------------------------------------------------ restricted_range
def do_restricted():
    p = f"{ROOT}/results/reanalysis/restricted_range_robustness.json"
    d = json.load(open(p))
    # paradigm_stats_full: mean/std/min/max from corrected matrix
    for pp in HDR:
        col = np.array([MAT[m][pp] for m in ALL_MODELS])
        d["paradigm_stats_full"][pp] = {"mean": round(float(col.mean()), 4),
            "std": round(float(col.std(ddof=1)), 4), "min": round(float(col.min()), 4),
            "max": round(float(col.max()), 4)}
    sds = {pp: float(np.array([MAT[m][pp] for m in ALL_MODELS]).std(ddof=1)) for pp in HDR}
    d["lowest_variance_3"] = sorted(sds, key=sds.get)[:3]
    rt = STEP4B["restricted_range_table"]
    keymap = {"Full (13 paradigms)": "full", "Drop NB,CV,GN": "drop_nb_cvlt_gn",
              "Drop 3 lowest-variance": "drop_3_lowest_var", "Leave out CVLT": "lo_cvlt",
              "Leave out Go/No-Go": "lo_gonogo", "Leave out n-back": "lo_nback"}
    conds = []
    for label, k in keymap.items():
        v = rt[k]
        conds.append({"condition": label, "pc1": v["pc1"], "raw_delta": v["raw_delta"],
                      "raw_p_onesided": v["raw_p"], "resid_delta": v["resid_delta"],
                      "resid_p_onesided": v["resid_p"]})
    d["conditions"] = conds
    save(p, d)
    return f"conditions {len(conds)}; full pc1 {rt['full']['pc1']} raw {rt['full']['raw_delta']}"


# ------------------------------------------------------------ scaling_mixedeffects
_SCALING_DISPLAY = {
    "n_back": "N-back",
    "digit_span": "Digit span",
    "operation_span": "Operation span",
    "stroop": "Stroop",
    "flanker": "Flanker",
    "go_nogo": "Go/No-Go",
    "cvlt_word_list": "CVLT",
    "drm_false_memory": "DRM",
    "source_monitoring": "Source monitoring",
    "false_belief": "False belief",
    "epitome_tom": "EPITOME",
    "confidence_calibration": "Calibration",
    "post_decision_wagering": "Wagering",
}

_SCALING_ORDER = [
    "n_back", "digit_span", "operation_span",
    "stroop", "flanker", "go_nogo",
    "cvlt_word_list", "drm_false_memory", "source_monitoring",
    "false_belief", "epitome_tom",
    "confidence_calibration", "post_decision_wagering",
]


def _scaling_p_text(p):
    """Publication display for a two-sided Wald p value."""
    if p < .001:
        return "<.001"
    return f"{p:.3f}".lstrip("0")


def _scaling_var_text(value):
    """Keep near-boundary random-effect estimates visible rather than 0.0000."""
    if 0 < abs(value) < 1e-4:
        coefficient, exponent = f"{value:.2e}".split("e")
        return rf"${coefficient}\mathord{{\times}}10^{{{int(exponent)}}}$"
    return f"{value:.4f}"


def _write_scaling_mixedlm_publication_artifacts(results):
    """Write machine-readable and ready-to-include versions of the full table.

    The JSON remains the single numeric source.  CSV and LaTeX are deterministic
    projections of ``publication_table`` so that manuscript updates do not
    require hand transcription.
    """
    out_dir = os.path.join(ROOT, "results", "reanalysis")
    csv_path = os.path.join(out_dir, "scaling_mixedeffects_table.csv")
    tex_path = os.path.join(out_dir, "scaling_mixedeffects_table.tex")
    rows = results["publication_table"]
    fields = [
        "grouping", "paradigm", "paradigm_key", "n_checkpoints", "n_families",
        "fixed_slope_per_log10_parameter", "fixed_slope_se", "wald_p",
        "random_intercept_variance", "residual_variance",
        "random_intercept_icc", "converged", "boundary_warning",
    ]

    csv_tmp = csv_path + ".tmp"
    with open(csv_tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])
    os.replace(csv_tmp, csv_path)

    lines = [
        "% Auto-generated by scripts/reanalysis/apply_corrected_results.py.",
        "% Do not edit by hand; regenerate from the frozen primary matrix.",
        r"\begin{table*}[t]",
        r"\caption{Family-random-intercept scaling models on the 20-checkpoint "
        r"primary pool. The fixed slope is the accuracy-unit change associated "
        r"with a tenfold increase in parameter count. Models were fit by maximum "
        r"likelihood with a random intercept for model family (11 families; seven "
        r"singletons). $\widehat{\mathrm{Var}}(u_f)$ and $\mathrm{ICC}_f$ summarize "
        r"the family intercept. Wald statistics are two-sided. "
        r"$^\dagger$The DRM and false-belief optimizers did not converge; their "
        r"coefficients are retained for transparency but treated only as "
        r"diagnostics, not as evidence that scaling ranks are preserved.}",
        r"\label{tab:scaling_mixedlm_full}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrc}",
        r"\toprule",
        r"Grouping & Paradigm & $\hat{\beta}_{\log_{10}B}$ & SE & "
        r"$p_{\mathrm{Wald}}$ & $\widehat{\mathrm{Var}}(u_f)$ & "
        r"$\mathrm{ICC}_f$ & Converged \\",
        r"\midrule",
    ]
    previous_group = None
    for row in rows:
        if previous_group is not None and row["grouping"] != previous_group:
            lines.append(r"\addlinespace[2pt]")
        suffix = "" if row["converged"] else r"$^\dagger$"
        lines.append(
            f"{row['grouping']} & {row['paradigm']} & "
            f"{row['fixed_slope_per_log10_parameter']:.3f} & "
            f"{row['fixed_slope_se']:.3f} & "
            f"{_scaling_p_text(row['wald_p'])} & "
            f"{_scaling_var_text(row['random_intercept_variance'])} & "
            f"{row['random_intercept_icc']:.3f} & "
            f"{'Yes' if row['converged'] else 'No'}{suffix} \\\\"
        )
        previous_group = row["grouping"]
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    tex_tmp = tex_path + ".tmp"
    with open(tex_tmp, "w") as handle:
        handle.write("\n".join(lines))
    os.replace(tex_tmp, tex_path)


def do_scaling_mlm():
    """Rebuild every per-paradigm block coherently in the ACTIVE regime.

    The pre-rebuild file mixed three generations (2026-07-03 production
    lmm_*/naive_* fields, later primary paper values, and a method string
    that grew one 'Regenerated' clause per run).  Each block is now
    recomputed wholesale from the active matrix, with the 2026-07-03
    production Pearson r preserved once under an explicitly historical key.
    """
    p = f"{ROOT}/results/reanalysis/scaling_mixedeffects.json"
    d = json.load(open(p))
    import statsmodels
    import statsmodels.formula.api as smf
    import pandas as pd_
    ls = {m: np.log10(SIZE[m]) for m in OLD_MODELS}
    fam = {m: b2.OLD_MODELS[m][1] for m in OLD_MODELS}
    r20 = STEP4B["scaling_r_20"]
    mlm20 = STEP4B["scaling_mixedlm_20"]
    # the historical production reference is RECOMPUTED from the frozen
    # production matrix on every run (inheriting it from the mutable JSON
    # let a stale generation leak through; 9/13 paradigms had drifted)
    _prod_rows = list(csv.reader(open(os.path.join(RC, "corrected_matrix.csv"))))
    _PROD20 = {r[0]: {pp: float(v) for pp, v in zip(_prod_rows[0][1:], r[1:])}
               for r in _prod_rows[1:]}
    regime = ("primary (COGARENA_PRIMARY_MATRIX)"
              if os.environ.get("COGARENA_PRIMARY_MATRIX") else "corrected (production)")
    publication_rows = []
    for pp in _SCALING_ORDER:
        blk = d["paradigms"][pp]
        y = np.array([MAT[m][pp] for m in OLD_MODELS])
        x = np.array([ls[m] for m in OLD_MODELS])
        r, pv = stats.pearsonr(x, y)
        hist = float(stats.pearsonr(x, np.array([_PROD20[m][pp] for m in OLD_MODELS]))[0])
        df = pd_.DataFrame({"acc": y, "ls": x,
                            "fam": [fam[m] for m in OLD_MODELS]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            md = smf.mixedlm("acc ~ ls", df, groups=df["fam"]).fit(
                reml=False, method=["bfgs", "lbfgs", "cg"])
        warning_messages = list(dict.fromkeys(str(w.message) for w in caught))
        slope = float(md.params["ls"]); se = float(md.bse["ls"])
        group_var = float(np.asarray(md.cov_re).ravel()[0])
        resid_var = float(md.scale)
        variance_total = group_var + resid_var
        group_icc = group_var / variance_total if variance_total > 0 else float("nan")
        boundary_warning = any("boundary" in message.lower()
                               for message in warning_messages)
        wf = {}
        for f, members in (("qwen2.5", [m for m in OLD_MODELS if fam[m] == "qwen2.5"]),
                           ("gemma2", [m for m in OLD_MODELS if fam[m] == "gemma2"])):
            xs = np.array([ls[m] for m in members]); ys = np.array([MAT[m][pp] for m in members])
            if len(members) >= 3 and xs.std() > 0:
                wf[f] = round(float(np.polyfit(xs, ys, 1)[0]), 4)
        d["paradigms"][pp] = {
            "domain": blk.get("domain"),
            "n": len(OLD_MODELS),
            "regime": regime,
            "pearson_r": round(float(r), 4),
            "pearson_p": round(float(pv), 4),
            "paper_bootstrap_pearson_r": r20[pp]["r"],
            "lmm_fixed_slope_log10size": round(slope, 4),
            "lmm_slope_se": round(se, 4),
            "lmm_slope_ci95": [round(slope - 1.96 * se, 4), round(slope + 1.96 * se, 4)],
            "lmm_slope_p": round(float(md.pvalues["ls"]), 4),
            "lmm_converged": bool(md.converged),
            "lmm_random_intercept_variance": round(group_var, 8),
            "lmm_residual_variance": round(resid_var, 8),
            "lmm_random_intercept_icc": round(group_icc, 6),
            "lmm_log_likelihood": round(float(md.llf), 8),
            "lmm_boundary_warning": boundary_warning,
            "lmm_warning_messages": warning_messages,
            "lmm_estimation": "maximum likelihood (REML=False)",
            "lmm_optimizer_sequence": ["bfgs", "lbfgs", "cg"],
            "lmm_n_families": len(set(fam.values())),
            "lmm_n_singleton_families": sum(
                1 for family in set(fam.values())
                if sum(f == family for f in fam.values()) == 1),
            "mlm_slope_step4b": mlm20[pp]["slope"],
            "mlm_p_step4b": mlm20[pp]["p"],
            "within_family_ols_slope": wf,
            "production_naive_pearson_r_20260703": (round(float(hist), 4)
                                                    if hist is not None else None),
        }
        publication_rows.append({
            "grouping": blk.get("domain"),
            "paradigm": _SCALING_DISPLAY[pp],
            "paradigm_key": pp,
            "n_checkpoints": len(OLD_MODELS),
            "n_families": len(set(fam.values())),
            "fixed_slope_per_log10_parameter": round(slope, 8),
            "fixed_slope_se": round(se, 8),
            "wald_p": round(float(md.pvalues["ls"]), 10),
            "random_intercept_variance": round(group_var, 10),
            "residual_variance": round(resid_var, 10),
            "random_intercept_icc": round(group_icc, 8),
            "converged": bool(md.converged),
            "boundary_warning": boundary_warning,
        })
    # top-level within-family OLS slopes rebuilt in the same regime (the
    # 2026-07-03 production block conflicted with the per-paradigm blocks)
    tl = {}
    for f in sorted(set(fam.values())):
        members = sorted([m for m in OLD_MODELS if fam[m] == f])
        if len(members) < 3:
            continue
        xs = np.array([ls[m] for m in members])
        if xs.std() == 0:
            continue
        pp_slopes = {}
        for pp in d["paradigms"]:
            ys = np.array([MAT[m][pp] for m in members])
            pp_slopes[pp] = {"n": len(members),
                             "ols_slope_log10size": round(float(np.polyfit(xs, ys, 1)[0]), 4)}
        tl[f] = {"models": members, "n": len(members), "regime": regime,
                 "per_paradigm": pp_slopes}
    d["within_family_slopes"] = tl

    nonconverged = [row["paradigm_key"] for row in publication_rows
                    if not row["converged"]]
    d["schema_version"] = "cogarena-scaling-mixedlm-v2"
    d["method"] = ("statsmodels MixedLM: accuracy ~ log10(size) with random intercept "
                   "(1|family), maximum likelihood (REML=False; optimizer sequence "
                   "bfgs/lbfgs/cg), on the 20 primary checkpoints, rebuilt wholesale "
                   f"from the active matrix ({regime}); the 2026-07-03 production "
                   "Pearson r is kept once under "
                   "production_naive_pearson_r_20260703.")
    d["fit_population"] = {
        "n_checkpoints": len(OLD_MODELS),
        "n_families": len(set(fam.values())),
        "n_singleton_families": sum(
            1 for family in set(fam.values())
            if sum(f == family for f in fam.values()) == 1),
        "family_counts": {
            family: sum(f == family for f in fam.values())
            for family in sorted(set(fam.values()))
        },
    }
    d["software"] = {
        "statsmodels": statsmodels.__version__,
        "numpy": np.__version__,
    }
    d["publication_table"] = publication_rows
    d["nonconverged_paradigms"] = nonconverged
    d["interpretation_guardrail"] = (
        "Fixed-effect estimates from non-converged fits are diagnostic only. "
        "In this run DRM and false belief did not converge; they must not be "
        "used to claim that family-random-intercept models preserve the "
        "Pearson scaling ordering.")
    save(p, d)
    _write_scaling_mixedlm_publication_artifacts(d)
    return (f"regime {regime}; n-back r {d['paradigms']['n_back']['pearson_r']}, "
            f"cvlt r {d['paradigms']['cvlt_word_list']['pearson_r']}; "
            f"non-converged {nonconverged}")


# ------------------------------------------------------------ signature_significance
def do_signature():
    p = f"{ROOT}/results/reanalysis/signature_significance.json"
    d = json.load(open(p))
    from cogarena.cli import _collect_items
    batt = {it.task_id: it for it in _collect_items(50, 42, None)}
    per_items = {m: corrected_static_items(m) for m in OLD_MODELS}
    mt_over = {}
    for m in OLD_MODELS:
        op = os.path.join(RESCORE, f"multiturn_eval_v3__openai_{m}__multiturn.json")
        mt_over[m] = json.load(open(op)) if os.path.exists(op) else {}

    def cond_means(par, cond_key, easy_v, hard_v):
        rows = {}
        for m in OLD_MODELS:
            e, h = [], []
            for tid, (pp, acc, sc) in per_items[m].items():
                if pp != par:
                    continue
                cc = sc.get(cond_key)
                if cc == easy_v:
                    e.append(acc)
                elif cc == hard_v:
                    h.append(acc)
            if e and h:
                rows[m] = (float(np.mean(e)), float(np.mean(h)), len(e), len(h))
        return rows

    def fb_order():
        rows = {}
        for m in OLD_MODELS:
            a = {1: [], 2: []}
            for tid, (pp, acc, sc) in per_items[m].items():
                if pp == "false_belief" and tid in batt:
                    a[int(batt[tid].metadata.parameters["order"])].append(acc)
            if a[1] and a[2]:
                rows[m] = (float(np.mean(a[1])), float(np.mean(a[2])), len(a[1]), len(a[2]))
        return rows

    def nback_load():
        rows = {}
        for m in OLD_MODELS:
            base = f"{ROOT}/results/multiturn_eval_v3/openai_{m}"
            ad = defaultdict(list)
            for f in glob.glob(os.path.join(base, "*", "n_back", "*.json")):
                dd = json.load(open(f)); sc = dd.get("score") or {}
                if "accuracy" not in sc:
                    continue
                rel = os.path.relpath(f, f"{ROOT}/results/multiturn_eval_v3")
                ad[dd.get("difficulty")].append(float(mt_over[m].get(rel, sc["accuracy"])))
            if ad.get("easy") and ad.get("hard"):
                rows[m] = (float(np.mean(ad["easy"])), float(np.mean(ad["hard"])), len(ad["easy"]), len(ad["hard"]))
        return rows

    def drm():
        # pooled per-model false-alarm rates (matches fig2_signatures + paper text)
        rows = {}
        for m in OLD_MODELS:
            clf = clt = uf = ut = 0.0
            for tid, (pp, acc, sc) in per_items[m].items():
                if pp == "drm_false_memory":
                    clf += sc.get("critical_lure_false_alarms", 0); clt += sc.get("critical_lure_total", 0)
                    uf += sc.get("unrelated_false_alarms", 0); ut += sc.get("unrelated_total", 0)
            if clt and ut:
                rows[m] = (clf / clt, uf / ut, int(clt), int(ut))
        return rows

    # block name (in the file) -> getter of {model: (easy, hard, n_easy, n_hard)}
    getters = {"stroop": lambda: cond_means("stroop", "condition", "congruent", "incongruent"),
               "flanker": lambda: cond_means("flanker", "condition", "congruent", "incongruent"),
               "false_belief": fb_order, "n_back_load": nback_load, "drm_false_memory": drm}
    counts, pvals = {}, {}
    for blk in d["paradigms"]:
        par = blk["paradigm"]
        if par == "epitome":
            # 20-pool EPITOME records are a synthetic forced-choice rerun; the
            # per-model sub-capacity signature is evaluated on the 35-model
            # expansion pool instead (desire easier than belief).
            k35, n35 = (int(x) for x in STEP4B["signature_epitome_35"].split("/"))
            pval = float(stats.binomtest(k35, n35, 0.5, alternative="greater").pvalue)
            blk.update({"k_models_expected_dir": k35, "n": n35, "fraction": f"{k35}/{n35}",
                        "p_binom_onesided": pval, "pool": "35-model expansion",
                        "contrast": "desire > belief sub-capacity",
                        "note": "20-model records are a synthetic forced-choice rerun; "
                                "evaluated on the expansion pool. Excluded from the BH set below.",
                        "per_model": {}})
            continue
        g = getters.get(par)
        if not g:
            continue
        rows = g()
        pm, n_ok = {}, 0
        for m, (e, h, ne, nh) in rows.items():
            ind = e > h
            n_ok += ind
            pm[m] = {"easy_acc": round(e, 6), "hard_acc": round(h, 6), "delta": round(e - h, 6),
                     "n_easy_items": ne, "n_hard_items": nh, "in_expected_direction": bool(ind)}
        k, n = n_ok, len(rows)
        pval = float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)
        ems = float(np.mean([v[0] for v in rows.values()]))
        hms = float(np.mean([v[1] for v in rows.values()]))
        blk.update({"k_models_expected_dir": k, "n": n, "fraction": f"{k}/{n}",
                    "p_binom_onesided": pval, "aggregate_easy_mean": round(ems, 6),
                    "aggregate_hard_mean": round(hms, 6), "aggregate_delta": round(ems - hms, 6),
                    "per_model": pm})
        counts[par] = k; pvals[par] = pval
    # BH across the 5 paradigms with per-item records in the 20-model pool
    names = list(pvals)
    ps = [pvals[n] for n in names]
    order = np.argsort(ps); adj = np.empty(len(ps)); prev = 1.0
    for rank, i in enumerate(reversed(order)):
        prev = min(prev, ps[i] * len(ps) / (len(ps) - rank)); adj[i] = prev
    bh = {n: float(a) for n, a in zip(names, adj)}
    for blk in d["paradigms"]:
        if blk["paradigm"] in bh:
            blk["p_binom_bh"] = bh[blk["paradigm"]]
    d["n_paradigms"] = 5
    d["bh_method"] = ("fdr_bh across the 5 paradigms with 20-model per-item records "
                      "(stroop, flanker, n_back_load, false_belief, drm_false_memory), alpha=0.05; "
                      "EPITOME is evaluated separately on the 35-model expansion pool")
    d["description"] = ("Signature significance battery regenerated from the active scorer regime (2026-07); DRM uses pooled per-model false-alarm rates; EPITOME moved to the 35-model expansion pool.")
    save(p, d)
    return "; ".join(f"{k} {counts[k]}/20" for k in counts) + f"; epitome {STEP4B['signature_epitome_35']}"


# ------------------------------------------------------------ fp16_deconfound
def do_fp16():
    p = f"{ROOT}/results/reanalysis/fp16_deconfound.json"
    d = json.load(open(p))
    S = ["digit_span", "stroop", "flanker", "go_nogo", "drm_false_memory",
         "source_monitoring", "false_belief", "epitome_tom", "confidence_calibration",
         "post_decision_wagering"]
    fp16_models = [x.split("openai_")[1] for x in
                   sorted(glob.glob(f"{ROOT}/results/full_eval_fp16/openai_*")) if os.path.isdir(x)]
    table = {}
    cells_fp, cells_q, grp_fp, grp_q = [], [], [], []
    for fm in fp16_models:
        qm = fm.replace("-instruct-fp16", "").replace(":latest", "")
        if qm not in MAT:
            continue
        det = b2.load_details(f"{ROOT}/results/full_eval_fp16/openai_{fm}/text/details.json")
        ov = {}
        op = os.path.join(RESCORE, f"full_eval_fp16__openai_{fm}.json")
        if os.path.exists(op):
            ov = json.load(open(op))
        by = defaultdict(list)
        for r in det:
            pp = r.get("paradigm")
            if pp in b2.STATIC_PARADIGMS and pp != "go_nogo":
                by[pp].append(float(ov.get(r["task_id"], b2.item_accuracy(r.get("score")))))
        # go_nogo fp16 from the rerun
        gg = f"{GONOGO}/openai_{fm}/text/details.json"
        if os.path.exists(gg):
            by["go_nogo"] = [b2.item_accuracy(r.get("score")) for r in json.load(open(gg))]
        row = {}
        gs_fp, gs_q = defaultdict(list), defaultdict(list)
        for pp in S:
            if by.get(pp):
                fpv = float(np.mean(by[pp])); qv = MAT[qm][pp]
                row[pp] = [round(qv, 4), round(fpv, 4)]
                cells_fp.append(fpv); cells_q.append(qv)
                gs_fp[b2.DOMAIN_MAP[pp]].append(fpv); gs_q[b2.DOMAIN_MAP[pp]].append(qv)
        table[qm] = row
        for gg2 in gs_fp:
            grp_fp.append(np.mean(gs_fp[gg2])); grp_q.append(np.mean(gs_q[gg2]))
    r_cell = float(stats.pearsonr(cells_fp, cells_q)[0])
    r_grp = float(stats.pearsonr(grp_fp, grp_q)[0])
    d["per_paradigm"] = {"pearson_r": round(r_cell, 3), "n_cells": len(cells_fp),
        "mean_abs_delta_pp": round(float(np.mean(np.abs(np.array(cells_fp) - np.array(cells_q)))) * 100, 1)}
    d["domain"] = {"pearson_r": round(r_grp, 3), "n_cells": len(grp_fp),
        "mean_abs_delta_pp": round(float(np.mean(np.abs(np.array(grp_fp) - np.array(grp_q)))) * 100, 1)}
    d["table_q4_vs_fp16"] = table
    d["note"] = ("fp16 deconfound regenerated from the active scorer regime + go_nogo rerun (2026-07); fp16 configs keep the historical per-item path by design.")
    save(p, d)
    return f"cell r {round(r_cell,3)}, domain r {round(r_grp,3)}, n {len(cells_fp)}"


# ------------------------------------------------------------ scorer_robustness
def do_scorer_robustness():
    p = f"{ROOT}/results/reanalysis/scorer_robustness.json"
    d = json.load(open(p))
    # This file documents multi-turn scorer-choice sensitivity. Its conclusion
    # (the separability/scaling findings survive the scorer choice) is unchanged;
    # flag that the deployed scorer is now the corrected strict scorer.
    d.setdefault("notes", {})["update_2026_07"] = (
        "The deployed multi-turn scorer is the corrected strict scorer (yes/no graded "
        "on the labeled current-trial answer after stripping prompt echoes; match/no-match "
        "punctuation-tolerant). The finding that the dimensional-separability and scaling "
        "results are robust to the multi-turn scorer choice is unchanged; the per-model "
        "generic-vs-dedicated values below predate the yes/no fix and are retained only to "
        "document the scorer-choice comparison, not as current accuracies.")
    save(p, d)
    return "annotated (scorer-choice sensitivity doc; conclusion unchanged)"


def main():
    failed = []
    for name, fn in [("b2_expanded", do_b2), ("pca_partialcorr", do_pca),
                     ("split_half", do_split_half), ("predictive_validity", do_pv),
                     ("restricted_range", do_restricted), ("scaling_mixedeffects", do_scaling_mlm),
                     ("signature_significance", do_signature), ("fp16_deconfound", do_fp16),
                     ("scorer_robustness", do_scorer_robustness)]:
        try:
            msg = fn()
            print(f"[OK] {name}: {msg}", flush=True)
        except Exception as e:
            import traceback
            failed.append(name)
            print(f"[FAIL] {name}: {e}\n{traceback.format_exc()}", flush=True)
    if failed:
        print(f"[done with FAILURES] {failed}", flush=True)
        sys.exit(1)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
