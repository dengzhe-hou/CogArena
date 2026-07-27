#!/usr/bin/env python3
"""PC1-interpretation validation battery (reviewer request, 2026-07-11).

Five components, one script:

  1. SELF-CHECK: reproduce the committed numbers on the real 55x13 accuracy
     matrix with THIS script's implementations before trusting anything:
     raw delta=+0.055 (perm p1~.11), row-mean-residual delta~0.148 (p1~.04),
     PC1-removal delta~0.20 (p1~.008), PC1 share~0.50.
  2. GENERATIVE SIMULATIONS: pure-g (A), g + five group factors over a strength
     grid (B), g + text-method factor (C, observationally equivalent to A by
     construction). Each simulated dataset runs the SAME pipeline (raw /
     row-mean-residual / PC1-removal deltas + label-permutation tests).
     Answers: (Q1) does pure-g alone reproduce the observed pattern triple,
     i.e. is the PC1-removal significance an orthogonalization artifact?
     (Q2) what group-factor strength would our raw test detect (power curve)?
  3. PARALLEL ANALYSIS (Horn, column-permutation variant) + model-bootstrap
     CIs for the eigenvalue-1 share and PC1 loadings, for BOTH the accuracy
     and the construct-native matrices.
  4. FAMILY ROBUSTNESS: equal-family weighting (family-mean matrix) and
     leave-one-family-out (merged families) for the raw delta.
  5. EQUIVALENCE-THRESHOLD SWEEP tau in [0.05, 0.20]: pass iff the
     family-clustered bootstrap CI upper bound < tau (both matrices), plus
     row-mean residualization of the construct matrix (the "weak factors get
     their chance when g is weaker" check).

CPU-only; run via Slurm, never the login node.
"""
import csv
import json
import os
import re
import sys

import numpy as np

ROOT = os.environ.get("COGARENA_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "results", "construct_native_20260711"))
os.environ["COGARENA_ROOT"] = ROOT

import compute_b2_expanded as b2
from build_construct_matrix import family_boot_ci

PARADIGMS = b2.PARADIGMS_ORDER
LABELS = [b2.DOMAIN_MAP[p] for p in PARADIGMS]
NP_ = len(PARADIGMS)
PAIRS = [(i, j) for i in range(NP_) for j in range(i + 1, NP_)]
WITHIN_MASK = np.array([LABELS[i] == LABELS[j] for i, j in PAIRS])

N_PERM = 5000
N_REPS = 1000
SEED = 42


# --------------------------------------------------------------------------- #
# core pipeline pieces
# --------------------------------------------------------------------------- #
def load_matrix(path):
    rows = list(csv.reader(open(path)))
    hdr = rows[0][1:]
    order = [hdr.index(p) for p in PARADIGMS]
    models = [r[0] for r in rows[1:]]
    M = np.array([[float(r[1 + k]) for k in order] for r in rows[1:]])
    return models, M


def pair_values(corr):
    return np.array([corr[i, j] for i, j in PAIRS])


def perm_test(corr, rng, n_perm=N_PERM):
    """Label-permutation p (one- and two-sided, +1 Monte Carlo correction)."""
    v = pair_values(corr)
    obs = float(v[WITHIN_MASK].mean() - v[~WITHIN_MASK].mean())
    labs = np.array(LABELS)
    deltas = np.empty(n_perm)
    for k in range(n_perm):
        sh = labs[rng.permutation(NP_)]
        w = np.array([sh[i] == sh[j] for i, j in PAIRS])
        deltas[k] = v[w].mean() - v[~w].mean()
    p1 = float((np.sum(deltas >= obs) + 1) / (n_perm + 1))
    p2 = float((np.sum(np.abs(deltas) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, p1, p2


def zscore(M):
    return (M - M.mean(0)) / M.std(0, ddof=1)


def rowmean_residual(M, zfirst=False):
    """Residualize each paradigm on overall competence (row mean), replicating
    final_inference.py's committed control: RAW columns regressed (with
    intercept) on the RAW row mean. zfirst=True variant standardizes columns
    first; needed for mixed-metric matrices, where the raw row mean is
    dominated by high-variance d' columns."""
    X = zscore(M) if zfirst else M
    g = X.mean(1)
    gc = g - g.mean()
    denom = float(gc @ gc)
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        beta = float(gc @ (col - col.mean())) / denom
        out[:, j] = col - col.mean() - beta * gc
    return out


def pc1_removed(M):
    """Remove the leading principal component (rank-1) from the z-scored
    matrix, the analysis the paper discounts as artifact-prone."""
    Z = zscore(M)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    return Z - np.outer(U[:, 0] * S[0], Vt[0])


def pc1_share(M):
    ev = np.linalg.eigvalsh(np.corrcoef(zscore(M).T))[::-1]
    return float(ev[0] / ev.sum()), ev


def pipeline(M, rng, n_perm=N_PERM):
    """The observed pattern triple for one matrix."""
    corr = np.corrcoef(M.T)
    v = pair_values(corr)
    raw = perm_test(corr, rng, n_perm)
    res = perm_test(np.corrcoef(rowmean_residual(M).T), rng, n_perm)
    pc1rm = perm_test(np.corrcoef(pc1_removed(M).T), rng, n_perm)
    share, _ = pc1_share(M)
    return {"raw": raw, "rowmean_resid": res, "pc1_removed": pc1rm,
            "pc1_share": round(share, 4),
            "mean_within_r": round(float(v[WITHIN_MASK].mean()), 4),
            "mean_cross_r": round(float(v[~WITHIN_MASK].mean()), 4)}


# --------------------------------------------------------------------------- #
# generative worlds
# --------------------------------------------------------------------------- #
def calibrated_loadings(M):
    """Per-paradigm g-loadings lambda_p fit by iterated least squares on the
    off-diagonal correlations (minimizes sum_{i!=j} (r_ij - lam_i lam_j)^2),
    initialized from the first eigenvector. This avoids the PCA-loading bias
    that bakes a nonzero within-cross delta into the pure-g world."""
    corr = np.corrcoef(zscore(M).T)
    w, V = np.linalg.eigh(corr)
    lam = np.sqrt(max(w[-1], 0)) * np.abs(V[:, -1])
    for _ in range(200):
        new = lam.copy()
        for j in range(NP_):
            others = [i for i in range(NP_) if i != j]
            num = sum(corr[i, j] * lam[i] for i in others)
            den = sum(lam[i] ** 2 for i in others)
            new[j] = num / den if den > 0 else lam[j]
        if np.max(np.abs(new - lam)) < 1e-10:
            lam = new
            break
        lam = new
    return np.clip(lam, 0.05, 0.98)


def simulate(rng, lam, n_models, group_w=0.0, method_split=0.0):
    """One simulated 55x13 score matrix.

    score = lam_p * common + group_w * G_{dom(p)} + unique noise, where
    'common' is either a single g (method_split=0) or an equal mix of a
    cognitive g and a text-method factor (method_split=0.5) -- the latter is
    observationally identical by construction (World C).
    """
    g = rng.standard_normal(n_models)
    if method_split > 0:
        m = rng.standard_normal(n_models)
        common = np.sqrt(1 - method_split) * g[:, None] + \
            np.sqrt(method_split) * m[:, None]
    else:
        common = g[:, None]
    groups = sorted(set(LABELS))
    Gf = {d: rng.standard_normal(n_models) for d in groups}
    X = np.empty((n_models, NP_))
    for j, p in enumerate(PARADIGMS):
        lam_j = lam[j]
        gvar = group_w ** 2
        uvar = max(1.0 - lam_j ** 2 - gvar, 0.02)
        X[:, j] = lam_j * common[:, 0] + group_w * Gf[LABELS[j]] + \
            np.sqrt(uvar) * rng.standard_normal(n_models)
    return X


def run_world(lam, n_models, group_w, method_split, n_reps, seed, n_perm=2000,
              obs_resid=None, obs_pc1rm=None):
    """obs_resid / obs_pc1rm are the OBSERVED pattern deltas of the active
    (primary) analysis matrix; the tail-event probabilities are computed
    against these, not against historical hardcoded values."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_reps):
        X = simulate(rng, lam, n_models, group_w, method_split)
        rows.append(pipeline(X, rng, n_perm))
    def stat(key, idx):
        return np.array([r[key][idx] for r in rows])
    out = {"observed_thresholds": {"rowmean_resid_delta": round(float(obs_resid), 4),
                                   "pc1_removed_delta": round(float(obs_pc1rm), 4)}}
    for key in ("raw", "rowmean_resid", "pc1_removed"):
        d, p1 = stat(key, 0), stat(key, 1)
        out[key] = {
            "delta_mean": round(float(d.mean()), 4),
            "delta_q05_q95": [round(float(np.quantile(d, q)), 4) for q in (.05, .95)],
            "P(p1<.05)": round(float(np.mean(p1 < .05)), 3),
            "P(p1<.01)": round(float(np.mean(p1 < .01)), 3),
            "P(delta>=obs_pc1rm & p1<=.01)": round(float(np.mean((d >= obs_pc1rm) & (p1 <= .01))), 3),
            "P(delta>=obs_resid)": round(float(np.mean(d >= obs_resid)), 3),
        }
    share = np.array([r["pc1_share"] for r in rows])
    out["pc1_share_mean"] = round(float(share.mean()), 3)
    out["realized_mean_within_r"] = round(float(np.mean(
        [r["mean_within_r"] for r in rows])), 4)
    out["realized_mean_cross_r"] = round(float(np.mean(
        [r["mean_cross_r"] for r in rows])), 4)
    return out


# --------------------------------------------------------------------------- #
# parallel analysis + eigen bootstrap
# --------------------------------------------------------------------------- #
def parallel_analysis(M, rng, n_iter=2000):
    """Horn's parallel analysis, column-permutation variant (preserves
    marginals, breaks cross-column structure)."""
    obs = np.linalg.eigvalsh(np.corrcoef(zscore(M).T))[::-1]
    null = np.empty((n_iter, NP_))
    for k in range(n_iter):
        Xp = np.column_stack([M[rng.permutation(M.shape[0]), j]
                              for j in range(NP_)])
        null[k] = np.linalg.eigvalsh(np.corrcoef(zscore(Xp).T))[::-1]
    thr95 = np.quantile(null, .95, axis=0)
    retained = 0
    for k in range(NP_):          # Horn: sequential, stop at first failure
        if obs[k] > thr95[k]:
            retained += 1
        else:
            break
    return {"observed_eigs_top4": [round(float(x), 3) for x in obs[:4]],
            "null95_top4": [round(float(x), 3) for x in thr95[:4]],
            "n_retained": retained}


def eigen_bootstrap(M, rng, n_boot=5000):
    n = M.shape[0]
    shares, load0 = [], []
    wo, Vo = np.linalg.eigh(np.corrcoef(zscore(M).T))
    ref = Vo[:, -1] * (1 if Vo[:, -1].sum() >= 0 else -1)   # observed PC1
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Z = zscore(M[idx])
        corr = np.corrcoef(Z.T)
        w, V = np.linalg.eigh(corr)
        shares.append(w[-1] / w.sum())
        v = V[:, -1]
        v = v if (v @ ref) >= 0 else -v
        load0.append(v)
    shares = np.array(shares); L = np.array(load0)
    return {"pc1_share_ci": [round(float(np.quantile(shares, q)), 3)
                             for q in (.025, .975)],
            "min_loading_ci_lo": round(float(np.quantile(L.min(1), .025)), 3),
            "all_loadings_positive_frac": round(float(np.mean((L > 0).all(1))), 3)}


# --------------------------------------------------------------------------- #
# family robustness
# --------------------------------------------------------------------------- #
def merged_families(models):
    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    fam = {m: (b2.OLD_MODELS[m][1] if m in b2.OLD_MODELS
               else new_meta[m].get("family", m)) for m in models}
    return {m: re.sub(r"[\d.]+$", "", f) for m, f in fam.items()}


def family_mean_delta(M, models, fams, rng):
    fam_ids = sorted(set(fams[m] for m in models))
    FM = np.array([M[[i for i, m in enumerate(models) if fams[m] == f]].mean(0)
                   for f in fam_ids])
    return perm_test(np.corrcoef(FM.T), rng), len(fam_ids)


def leave_one_family_out(M, models, fams, rng):
    out = {}
    for f in sorted(set(fams.values())):
        keep = [i for i, m in enumerate(models) if fams[m] != f]
        obs, p1, _ = perm_test(np.corrcoef(M[keep].T), rng, 2000)
        out[f] = {"delta": round(obs, 4), "p1": round(p1, 3), "n": len(keep)}
    ds = [v["delta"] for v in out.values()]
    return {"range": [round(min(ds), 4), round(max(ds), 4)],
            "min_p1": round(min(v["p1"] for v in out.values()), 3),
            "per_family": out}


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)
    # The self-check reproduces the PRODUCTION baseline and therefore always
    # runs on the base corrected matrix; the analyses run on the primary
    # (adjudicated) matrices when the COGARENA_PRIMARY_* variables are set.
    models_b, MB = load_matrix(f"{ROOT}/results/recompute_20260703/corrected_matrix.csv")
    models_a, MA = load_matrix(os.environ.get("COGARENA_PRIMARY_MATRIX")
                            or f"{ROOT}/results/recompute_20260703/corrected_matrix.csv")
    models_c, MC = load_matrix(os.environ.get("COGARENA_PRIMARY_CONSTRUCT_MATRIX")
                            or f"{ROOT}/results/construct_native_20260711/construct_matrix.csv")
    assert models_a == models_c == models_b
    fams = merged_families(models_a)
    out = {"n_models": MA.shape[0],
           "primary_matrix": os.environ.get("COGARENA_PRIMARY_MATRIX"),
           "primary_construct_matrix": os.environ.get("COGARENA_PRIMARY_CONSTRUCT_MATRIX")}

    # ---- 1. self-check on the base corrected accuracy matrix ---------------
    sc = pipeline(MB, np.random.default_rng(SEED))
    out["self_check_accuracy"] = {k: ([round(x, 4) for x in v] if isinstance(v, tuple)
                                      else v) for k, v in sc.items()}
    print("[self-check]", json.dumps(out["self_check_accuracy"]), flush=True)
    ok = (abs(sc["raw"][0] - 0.0548) < 0.005 and abs(sc["pc1_share"] - 0.50) < 0.02
          and abs(sc["rowmean_resid"][0] - 0.148) < 0.02
          and abs(sc["pc1_removed"][0] - 0.20) < 0.03 and sc["pc1_removed"][1] < 0.03)
    if not ok:
        sys.exit("SELF-CHECK FAILED: implementations do not reproduce the "
                 "committed numbers; aborting before any simulation.")

    # construct-matrix pipeline (incl. its row-mean residualization = the
    # "weak factors get their chance" check). Mixed-metric matrix: the raw row
    # mean is dominated by high-variance d' columns, so the z-scored variant
    # is the meaningful one; both are reported, labeled.
    out["construct_pipeline"] = {k: ([round(x, 4) for x in v] if isinstance(v, tuple)
                                     else v)
                                 for k, v in pipeline(MC, np.random.default_rng(SEED)).items()}
    rz = perm_test(np.corrcoef(rowmean_residual(MC, zfirst=True).T),
                   np.random.default_rng(SEED))
    out["construct_pipeline"]["rowmean_resid_zscored"] = [round(x, 4) for x in rz]
    print("[construct]", json.dumps(out["construct_pipeline"]), flush=True)

    # ---- 2. generative simulations -----------------------------------------
    lam = calibrated_loadings(MA)
    implied = np.array([lam[i] * lam[j] for i, j in PAIRS])
    out["calibration"] = {"loadings": [round(float(x), 3) for x in lam],
                          "implied_mean_r": round(float(implied.mean()), 3),
                          "observed_mean_r": round(float(
                              pair_values(np.corrcoef(MA.T)).mean()), 3),
                          "baked_in_delta_pure_g": round(float(
                              implied[WITHIN_MASK].mean() -
                              implied[~WITHIN_MASK].mean()), 4)}
    print("[calibration]", json.dumps(out["calibration"]), flush=True)

    # observed pattern of the ACTIVE (primary) analysis matrix: the world
    # summaries' tail-event thresholds derive from these, not from the
    # historical production values
    obs_primary = pipeline(MA, np.random.default_rng(SEED))
    out["observed_primary_pattern"] = {
        k: ([round(x, 4) for x in v] if isinstance(v, tuple) else v)
        for k, v in obs_primary.items()}
    obs_resid = obs_primary["rowmean_resid"][0]
    obs_pc1rm = obs_primary["pc1_removed"][0]
    print("[observed primary]", json.dumps(out["observed_primary_pattern"]), flush=True)

    worlds = {}
    worlds["A_pure_g"] = run_world(lam, MA.shape[0], 0.0, 0.0, N_REPS, SEED + 1,
                                   obs_resid=obs_resid, obs_pc1rm=obs_pc1rm)
    print("[world A]", json.dumps(worlds["A_pure_g"]), flush=True)
    for k, w in enumerate((0.15, 0.22, 0.32, 0.39, 0.45)):
        tag = f"B_group_w{w}"  # within-pair correlation increment ~= w^2
        worlds[tag] = run_world(lam, MA.shape[0], w, 0.0, 500, SEED + 20 + k,
                                obs_resid=obs_resid, obs_pc1rm=obs_pc1rm)
        worlds[tag]["implied_within_increment"] = round(w * w, 3)
        print(f"[world {tag}]", json.dumps(worlds[tag]["raw"]), flush=True)
    worlds["C_g_plus_method"] = run_world(lam, MA.shape[0], 0.0, 0.5, N_REPS, SEED + 3,
                                          obs_resid=obs_resid, obs_pc1rm=obs_pc1rm)
    print("[world C]", json.dumps(worlds["C_g_plus_method"]), flush=True)
    out["worlds"] = worlds
    out["rng_note"] = ("one rng per world drives both data generation and the "
                       "permutation tests (reproducible as a unit; changing "
                       "n_perm changes the simulated datasets); each World-B "
                       "grid point has its own seed")

    # ---- 3. parallel analysis + eigen bootstrap ----------------------------
    out["parallel_analysis"] = {
        "accuracy": parallel_analysis(MA, np.random.default_rng(SEED + 4)),
        "construct": parallel_analysis(MC, np.random.default_rng(SEED + 5))}
    out["eigen_bootstrap"] = {
        "accuracy": eigen_bootstrap(MA, np.random.default_rng(SEED + 6)),
        "construct": eigen_bootstrap(MC, np.random.default_rng(SEED + 7))}
    print("[PA]", json.dumps(out["parallel_analysis"]), flush=True)

    # what IS the construct matrix's 2nd retained component? (metric-type
    # method factor suspicion: d'-scored paradigms clustering together)
    wC, VC = np.linalg.eigh(np.corrcoef(zscore(MC).T))
    pc1v = VC[:, -1] * (1 if VC[:, -1].sum() >= 0 else -1)
    pc2v = VC[:, -2]
    out["construct_pc_loadings"] = {
        "eig_shares_top3": [round(float(x / wC.sum()), 3) for x in wC[::-1][:3]],
        "PC1": {p: round(float(pc1v[j]), 2) for j, p in enumerate(PARADIGMS)},
        "PC2": {p: round(float(pc2v[j]), 2) for j, p in enumerate(PARADIGMS)},
    }
    print("[construct PCs]", json.dumps(out["construct_pc_loadings"]), flush=True)

    # ---- 4. family robustness ----------------------------------------------
    (fm_obs, fm_p1, fm_p2), n_fam = family_mean_delta(MA, models_a, fams,
                                                      np.random.default_rng(SEED + 8))
    out["equal_family_weighting"] = {"n_families": n_fam,
                                     "delta": round(fm_obs, 4),
                                     "p1": round(fm_p1, 4), "p2": round(fm_p2, 4)}
    out["leave_one_family_out"] = leave_one_family_out(MA, models_a, fams,
                                                       np.random.default_rng(SEED + 9))
    lofo_c = leave_one_family_out(MC, models_c, fams, np.random.default_rng(SEED + 10))
    out["leave_one_family_out_construct"] = {k: lofo_c[k] for k in ("range", "min_p1")}
    print("[family]", json.dumps({k: out[k] for k in
          ("equal_family_weighting",)}), flush=True)

    # ---- 5. equivalence threshold sweep ------------------------------------
    rows_a = {m: dict(zip(PARADIGMS, MA[i])) for i, m in enumerate(models_a)}
    rows_c = {m: dict(zip(PARADIGMS, MC[i])) for i, m in enumerate(models_c)}
    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    fams_raw = {m: (b2.OLD_MODELS[m][1] if m in b2.OLD_MODELS
                    else new_meta[m].get("family", m)) for m in models_a}
    cis = {}
    for cname, ff in (("merged", fams), ("raw", fams_raw)):
        cis[cname] = {"accuracy": family_boot_ci(rows_a, models_a, ff),
                      "construct": family_boot_ci(rows_c, models_c, ff)}
    sweep = {}
    for tau in [round(0.05 + 0.01 * k, 2) for k in range(16)]:
        sweep[str(tau)] = {
            f"{mat}_{cname}_pass": bool(cis[cname][mat][1] < tau)
            for cname in cis for mat in ("accuracy", "construct")}
    out["threshold_sweep"] = {
        "famCIs": cis,
        "min_tau": {f"{mat}_{cname}": round(cis[cname][mat][1], 3)
                    for cname in cis for mat in ("accuracy", "construct")},
        "sweep": sweep}
    print("[sweep] min tau:", json.dumps(out["threshold_sweep"]["min_tau"]), flush=True)

    json.dump(out, open(os.path.join(HERE, "pc1_validation.json"), "w"),
              indent=1, default=str)
    print("[done] wrote pc1_validation.json", flush=True)


if __name__ == "__main__":
    main()
