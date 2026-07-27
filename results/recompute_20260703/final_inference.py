#!/usr/bin/env python3
"""FINAL inference battery on the fully corrected 55x13 matrix.

Runs after build_and_recompute.py has rebuilt corrected_matrix.csv with ALL
scorer fixes (round-1 + round-2 strictness). Implements the inference exactly
as the adversarial verification demanded:
  - permutation p reported one- AND two-sided (50k perms)
  - naive model bootstrap AND family-clustered bootstrap (raw + merged
    family labels, 5 seeds each) for delta
  - row-mean residualized within-vs-cross contrast with the same battery
  - Spearman variants
  - sensitivity: drop-Meta-pair, exclude-go_nogo, leave-one-group-out
All of it on the corrected matrix, with the uncorrected matrix as reference.
"""
import csv
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.environ.get("COGARENA_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "recompute_20260703")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ["COGARENA_ROOT"] = ROOT
import compute_b2_expanded as b2

# Primary-estimand propagation: when COGARENA_PRIMARY_MATRIX points at the
# frozen selected matrix (strict-v4 in the reported analysis), all inference
# runs on it; unset, the historical corrected_matrix path is reproduced.
PRIMARY_MATRIX = os.environ.get("COGARENA_PRIMARY_MATRIX")

N_PERM = 50000
BOOT = 5000
SEEDS = [42, 0, 1, 7, 123]


def load_corrected():
    rows = list(csv.reader(open(PRIMARY_MATRIX or os.path.join(OUT, "corrected_matrix.csv"))))
    hdr = rows[0][1:]
    models = [r[0] for r in rows[1:]]
    M = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    return hdr, models, M


def families(models):
    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    fam = {}
    for m in models:
        if m in b2.OLD_MODELS:
            fam[m] = b2.OLD_MODELS[m][1]
        else:
            fam[m] = new_meta[m].get("family", m)
    raw = [fam[m] for m in models]
    merged = [re.sub(r"[\d.]+$", "", f) for f in raw]
    return raw, merged


def pair_index(labels):
    n = len(labels)
    prs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    within = [(i, j) for i, j in prs if labels[i] == labels[j]]
    cross = [(i, j) for i, j in prs if labels[i] != labels[j]]
    return prs, within, cross


def delta_of(corr, within, cross):
    w = [corr[i, j] for i, j in within]
    c = [corr[i, j] for i, j in cross]
    return float(np.mean(w) - np.mean(c)), float(np.mean(w)), float(np.mean(c))


def perm_p(corr, labels, obs, n_perm=N_PERM, seed=42, drop_pairs=frozenset()):
    prs = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels))
           if (i, j) not in drop_pairs]
    rng = np.random.default_rng(seed)
    ge = 0; abs_ge = 0; n_valid = 0
    for _ in range(n_perm):
        sh = rng.permutation(labels).tolist()
        w = [corr[i, j] for i, j in prs if sh[i] == sh[j]]
        c = [corr[i, j] for i, j in prs if sh[i] != sh[j]]
        if not w or not c:
            continue
        d = np.mean(w) - np.mean(c)
        n_valid += 1
        ge += d >= obs
        abs_ge += abs(d) >= abs(obs)
    return round(ge / n_valid, 4), round(abs_ge / n_valid, 4)


def boot_ci(M, labels, cluster_ids, n_boot=BOOT, seed=42):
    """Bootstrap CI for delta; cluster_ids=None -> model-level resampling."""
    prs, within, cross = pair_index(labels)
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    if cluster_ids is None:
        groups = [[i] for i in range(n)]
    else:
        by = {}
        for i, c in enumerate(cluster_ids):
            by.setdefault(c, []).append(i)
        groups = list(by.values())
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = [i for g in pick for i in groups[g]]
        sub = M[idx]
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(sub.T)
        w = [corr[i, j] for i, j in within if np.isfinite(corr[i, j])]
        c = [corr[i, j] for i, j in cross if np.isfinite(corr[i, j])]
        if w and c:
            deltas.append(np.mean(w) - np.mean(c))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_le0 = float(np.mean([d <= 0 for d in deltas]))
    return round(float(lo), 4), round(float(hi), 4), round(p_le0, 4)


def residualize(M):
    row_mean = M.mean(axis=1, keepdims=True)
    R = np.empty_like(M)
    x = np.column_stack([np.ones(M.shape[0]), row_mean.ravel()])
    for j in range(M.shape[1]):
        beta, *_ = np.linalg.lstsq(x, M[:, j], rcond=None)
        R[:, j] = M[:, j] - x @ beta
    return R


def battery(M, labels, models, raw_fam, merged_fam, tag):
    prs, within, cross = pair_index(labels)
    corr = np.corrcoef(M.T)
    obs, wm, cm = delta_of(corr, within, cross)
    p1, p2 = perm_p(corr, labels, obs)
    rho = spearmanr(M).statistic
    obs_s, _, _ = delta_of(rho, within, cross)
    sp1, sp2 = perm_p(rho, labels, obs_s)
    res = {
        "within_mean": round(wm, 4), "cross_mean": round(cm, 4),
        "delta": round(obs, 4), "perm_p_one_sided": p1, "perm_p_two_sided": p2,
        "spearman_delta": round(obs_s, 4), "spearman_p1": sp1, "spearman_p2": sp2,
        "boot_model_ci": boot_ci(M, labels, None),
        "boot_family_raw": {}, "boot_family_merged": {},
    }
    for s in SEEDS:
        res["boot_family_raw"][f"seed{s}"] = boot_ci(M, labels, raw_fam, seed=s)
        res["boot_family_merged"][f"seed{s}"] = boot_ci(M, labels, merged_fam, seed=s)
    ev = np.linalg.eigvalsh(corr)[::-1]
    res["pc1_share"] = round(float(ev[0] / ev.sum()), 4)
    print(f"[{tag}] delta={obs:+.4f} p1={p1} p2={p2} | spearman {obs_s:+.4f} "
          f"p1={sp1} | famCI(merged,s42)={res['boot_family_merged']['seed42']}", flush=True)
    return res, corr


def main():
    hdr, models, M = load_corrected()
    labels = [b2.DOMAIN_MAP[p] for p in hdr]
    raw_fam, merged_fam = families(models)
    print(f"[pools] {len(models)} models; clusters raw={len(set(raw_fam))} "
          f"merged={len(set(merged_fam))}", flush=True)

    out = {"n_models": len(models),
           "n_clusters": {"raw": len(set(raw_fam)), "merged": len(set(merged_fam))}}

    out["corrected_raw"], corr = battery(M, labels, models, raw_fam, merged_fam, "corrected raw")
    R = residualize(M)
    out["corrected_residualized"], _ = battery(R, labels, models, raw_fam, merged_fam, "corrected resid")

    # reference: uncorrected matrix through the published pipeline
    unc_rows = {}
    for m in models:
        row, _ = b2.build_model_row(m, m in b2.OLD_MODELS)
        unc_rows[m] = row
    U = np.array([[unc_rows[m].get(p, 0) for p in hdr] for m in models])
    out["uncorrected_raw"], _ = battery(U, labels, models, raw_fam, merged_fam, "uncorrected raw")
    out["uncorrected_residualized"], _ = battery(residualize(U), labels, models,
                                                 raw_fam, merged_fam, "uncorrected resid")

    # sensitivity on corrected raw
    i_cc = hdr.index("confidence_calibration"); i_pdw = hdr.index("post_decision_wagering")
    pair = (min(i_cc, i_pdw), max(i_cc, i_pdw))
    prs, within, cross = pair_index(labels)
    w2 = [p for p in within if p != pair]
    obs2 = float(np.mean([corr[i, j] for i, j in w2]) - np.mean([corr[i, j] for i, j in cross]))
    p1, p2 = perm_p(corr, labels, obs2, drop_pairs=frozenset([pair]))
    out["sens_drop_meta_pair"] = {"delta": round(obs2, 4), "p1": p1, "p2": p2}

    keep = [i for i, h in enumerate(hdr) if h != "go_nogo"]
    subM = M[:, keep]; sub_labels = [labels[i] for i in keep]
    sc = np.corrcoef(subM.T)
    _, sw, scr = pair_index(sub_labels)
    obs3, _, _ = delta_of(sc, sw, scr)
    p1, p2 = perm_p(sc, sub_labels, obs3)
    out["sens_exclude_gonogo"] = {"delta": round(obs3, 4), "p1": p1, "p2": p2}

    out["sens_leave_one_group_out"] = {}
    for g in sorted(set(labels)):
        keep = [i for i, l in enumerate(labels) if l != g]
        sc = np.corrcoef(M[:, keep].T)
        sl = [labels[i] for i in keep]
        _, sw, scr = pair_index(sl)
        obsg, _, _ = delta_of(sc, sw, scr)
        p1, p2 = perm_p(sc, sl, obsg)
        out["sens_leave_one_group_out"][g] = {"delta": round(obsg, 4), "p1": p1, "p2": p2}

    json.dump(out, open(os.path.join(OUT, "final_inference.json"), "w"), indent=1)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
