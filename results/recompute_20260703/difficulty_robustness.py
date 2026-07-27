#!/usr/bin/env python3
"""Difficulty-stratified robustness on the frozen estimand (round-ten design).

Two deliverables:

1. `all_selfcheck`: a 55x13 matrix rebuilt from the SAME per-item pools the
   two-level bootstrap validates (fixed confidence-calibration scorer,
   go_nogo rerun, SM overlay, and - under the primary regime - the selected
   OSpan estimator / fixed CVLT), gated CELLWISE against the active matrix
   (COGARENA_PRIMARY_MATRIX, else the corrected matrix) at 5e-4.  The 20
   primary models' EPITOME cells have no per-item records and are
   backfilled from the matrix (trivially equal, count reported); every
   other cell is a real reconciliation.  Failure exits nonzero.

2. `tiers`: the primary sensitivity panel on a FIXED 11-paradigm set -
   go_nogo (easy-only items) and epitome_tom (aggregate-only for the 20
   primary models) are excluded from every tier, so each tier matrix is
   built purely from that tier's items with no fallback.  Per tier:
   delta, one-sided label-permutation p (50k MC), PC1 share, and a
   merged-family bootstrap 95% CI (20k, seed 42) with its position
   relative to the prespecified .15 threshold.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("COGARENA_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "results", "twolevel_bootstrap_20260712"))
sys.path.insert(0, os.path.join(ROOT, "results", "recompute_20260703"))
sys.path.insert(0, os.path.join(ROOT, "results", "construct_native_20260711"))
sys.path.insert(0, ROOT)
os.environ.setdefault("COGARENA_ROOT", ROOT)

import compute_b2_expanded as b2  # noqa: E402
import two_level_bootstrap as tl  # noqa: E402
from build_construct_matrix import load_conf_cal_gold  # noqa: E402

N_PERM = 50000
N_BOOT = 20000
SEED = 42
CELL_TOL = tl.TOL  # authoritative pool-vs-matrix tolerance (6e-5)
DROP = ("go_nogo", "epitome_tom")


def req(cond, msg):
    if not cond:
        raise SystemExit(f"DIFFICULTY GATE FAILED: {msg}")


def delta_p_pc1(M, labels, rng):
    corr = np.corrcoef(M.T)
    n = len(labels)
    prs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    w = [corr[i, j] for i, j in prs if labels[i] == labels[j]]
    c = [corr[i, j] for i, j in prs if labels[i] != labels[j]]
    obs = float(np.mean(w) - np.mean(c))
    hits = 0
    for _ in range(N_PERM):
        sh = rng.permutation(labels).tolist()
        ww = [corr[i, j] for i, j in prs if sh[i] == sh[j]]
        cc = [corr[i, j] for i, j in prs if sh[i] != sh[j]]
        if np.mean(ww) - np.mean(cc) >= obs:
            hits += 1
    ev = np.linalg.eigvalsh(np.corrcoef(((M - M.mean(0)) / M.std(0, ddof=0)).T))[::-1]
    return (float(np.mean(w)), float(np.mean(c)), obs, hits / N_PERM,
            float(ev[0] / ev.sum()))


def fam_boot_ci(M, labels, models, fam_merged, rng):
    fam_ids = sorted(set(fam_merged[m] for m in models))
    idx_by = {f: [i for i, m in enumerate(models) if fam_merged[m] == f] for f in fam_ids}
    n = len(labels)
    prs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    wm = np.array([labels[i] == labels[j] for i, j in prs])
    ds = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(fam_ids), len(fam_ids), replace=True)
        Xb = M[[i for k in pick for i in idx_by[fam_ids[k]]]]
        if np.any(Xb.std(0) == 0):
            continue
        corr = np.corrcoef(Xb.T)
        v = np.array([corr[i, j] for i, j in prs])
        ds.append(float(v[wm].mean() - v[~wm].mean()))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)], len(ds)


def main():
    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    old_models = list(b2.OLD_MODELS.keys())
    all_models = old_models + sorted(new_meta.keys())

    from cogarena.cli import _collect_items
    batt = {it.task_id: it for it in _collect_items(50, 42, None)}

    def diff_of(tid):
        it = batt.get(tid)
        if it is None:
            return None
        d = it.metadata.difficulty
        return getattr(d, "value", str(d))

    primary_path = os.environ.get("COGARENA_PRIMARY_MATRIX")
    mrows = __import__("csv").reader(open(primary_path or os.path.join(HERE, "corrected_matrix.csv")))
    mrows = list(mrows)
    mhdr = mrows[0][1:]
    mat = {r[0]: {p: float(v) for p, v in zip(mhdr, r[1:])} for r in mrows[1:]}
    req(set(mat) == set(all_models), "matrix model set mismatch")

    # Primary-regime episode overrides.  The reported freeze uses strict-v4
    # OSpan; the optional adjudicated branch remains available only when
    # explicitly selected.  CVLT always uses the fixed episode estimator.
    aplus_dir = f"{ROOT}/results/reanalysis/aplus_20260718"
    ospan_over = cvlt_over = None
    primary_config = os.environ.get("COGARENA_PRIMARY_CONFIG", "aplus_strict")
    if primary_path:
        req(primary_config in ("aplus_strict", "aplus_adjudicated"),
            f"unsupported primary config {primary_config!r}")
        ospan_field = ("strict" if primary_config == "aplus_strict" else "adj_strict")
        osp = json.load(open(f"{aplus_dir}/ospan_recall_scores.json"))
        req(all(ospan_field in e for eps in osp.values() for e in eps),
            f"primary regime requires OSpan score field {ospan_field!r}")
        ospan_over = {m: [(e['task_id'], e[ospan_field]) for e in eps]
                      for m, eps in osp.items()}
        cv = json.load(open(f"{aplus_dir}/cvlt_fixed_scores.json"))
        cvlt_over = {m: [(e['task_id'], e['fixed_accuracy']) for e in eps]
                     for m, eps in cv.items()}

    cc_gold = load_conf_cal_gold()
    fam_merged = json.load(open(f"{aplus_dir}/family_map.json"))['merged']

    # per-model pools (the same per-item layer the two-level bootstrap
    # self-checks against the matrix), plus difficulty lookup
    pools = {}
    mt_diff_cache = {}
    for m in all_models:
        pl = tl.item_pools(m, m in b2.OLD_MODELS, cc_gold)
        req(pl is not None, f"pools missing for {m}")
        if ospan_over is not None:
            pl['operation_span'] = ospan_over[m]
            pl['cvlt_word_list'] = cvlt_over[m]
        pools[m] = pl
        mt_set, base = tl.mt_base(m, m in b2.OLD_MODELS)
        for par in sorted(b2.MT_PARADIGMS):
            for f in glob.glob(os.path.join(base, "*", par, "*.json")):
                tid = os.path.basename(f)[:-5]
                if tid not in mt_diff_cache:
                    d = diff_of(tid)
                    if d is None:
                        d = json.load(open(f)).get("difficulty", "?")
                    mt_diff_cache[tid] = str(d)

    def item_diff(par, key):
        tid = os.path.basename(str(key))
        if tid.endswith(".json"):
            tid = tid[:-5]
        if par in b2.MT_PARADIGMS:
            got = mt_diff_cache.get(tid) or diff_of(tid)
        else:
            got = diff_of(tid)
        req(got is not None, f"no difficulty for {par}/{tid}")
        return got

    # ---- deliverable 1: 55x13 cellwise self-check vs the active matrix ----
    # The 20 primary models' EPITOME pools are SYNTHETIC v3_rerun
    # placeholder records (archived scores, no raw responses); they are
    # counted honestly rather than passed off as real item records.  The
    # tier panel is unaffected (epitome_tom is excluded from every tier).
    n_direct_backfill = 0
    n_synthetic = 0
    max_dev = 0.0
    worst = None
    M_all = np.empty((len(all_models), len(mhdr)))
    for mi, m in enumerate(all_models):
        for pi, p in enumerate(mhdr):
            pool = pools[m].get(p)
            if not pool:
                req(p == "epitome_tom" and m in b2.OLD_MODELS,
                    f"unexpected empty pool {m}/{p}")
                M_all[mi, pi] = mat[m][p]
                n_direct_backfill += 1
                continue
            if p == "epitome_tom" and m in b2.OLD_MODELS:
                n_synthetic += 1
            M_all[mi, pi] = float(np.mean([a for _, a in pool]))
            dev = abs(M_all[mi, pi] - mat[m][p])
            if dev > max_dev:
                max_dev, worst = dev, (m, p)
    req(max_dev <= CELL_TOL,
        f"all-tier rebuild deviates from the active matrix: {worst} by {max_dev:.6f}")

    labels13 = [b2.DOMAIN_MAP[p] for p in mhdr]
    w, c, d, p1, pc1 = delta_p_pc1(M_all, labels13, np.random.default_rng(SEED))
    out = {"all_selfcheck": {
        "n": len(all_models), "matrix": primary_path or "corrected_matrix.csv",
        "max_cell_dev": round(max_dev, 6), "cell_tol": CELL_TOL,
        "n_synthetic_placeholder_cells": n_synthetic,
        "n_direct_backfill_cells": n_direct_backfill,
        "within": round(w, 3), "cross": round(c, 3), "delta": round(d, 4),
        "p_onesided": round(p1, 3), "pc1": round(pc1, 3)}}
    print("all_selfcheck", json.dumps(out["all_selfcheck"]), flush=True)

    # ---- deliverable 2: fixed 11-paradigm tier panel ----------------------
    panel = [p for p in mhdr if p not in DROP]
    labels11 = [b2.DOMAIN_MAP[p] for p in panel]
    out["panel_paradigms"] = panel
    out["dropped"] = list(DROP)
    out["tiers"] = {}
    for tier in ("easy", "medium", "hard"):
        M = np.empty((len(all_models), len(panel)))
        for mi, m in enumerate(all_models):
            for pi, p in enumerate(panel):
                vals = [a for k, a in pools[m][p] if item_diff(p, k) == tier]
                req(bool(vals), f"empty tier cell {m}/{p}/{tier}")
                M[mi, pi] = float(np.mean(vals))
        w, c, d, p1, pc1 = delta_p_pc1(M, labels11, np.random.default_rng(SEED))
        ci, neff = fam_boot_ci(M, labels11, all_models, fam_merged,
                               np.random.default_rng(SEED))
        out["tiers"][tier] = {
            "n": len(all_models), "within": round(w, 3), "cross": round(c, 3),
            "delta": round(d, 4), "p_onesided": round(p1, 4), "pc1": round(pc1, 3),
            "fam_merged_ci95": ci, "ci_n_eff": neff,
            "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0),
            "ci_excludes_015": bool(ci[1] < 0.15 or ci[0] > 0.15)}
        print(tier, json.dumps(out["tiers"][tier]), flush=True)

    out["meta"] = {"n_perm": N_PERM, "n_boot": N_BOOT, "seed": SEED,
                   "sm_overlay": os.environ.get("COGARENA_SM_OVERLAY"),
                   "primary_matrix_regime": bool(primary_path),
                   "primary_config": primary_config if primary_path else None,
                   "ospan_score_field": ospan_field if primary_path else None}
    path = os.path.join(HERE, "difficulty_robustness.json")
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
    print("written:", path)


if __name__ == "__main__":
    main()
