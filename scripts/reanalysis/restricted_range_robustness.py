#!/usr/bin/env python3
"""Restricted-range robustness for the positive-manifold / dimensional-null result.

A reviewer can argue the positive manifold and the within<=cross null are driven
by restricted-range paradigms (floor OR ceiling): low-variance columns attenuate
and distort correlations. This script tests whether the two headline claims
survive removing such paradigms.

Pipeline is IDENTICAL to the headline analysis (reuses compute_b2_expanded's
matrix builder + permutation machinery), so the `full` condition is a built-in
self-check: it must reproduce
    raw within-cross delta ~= -0.005, p ~= 0.52 ; PC1 ~= 0.44 ;
    overall-competence-residualized delta ~= 0.04, p ~= 0.29.

For each condition we report, on COMPLETE-CASE models for that paradigm subset:
  - n_models, n_paradigms
  - PC1 variance explained, mean off-diagonal r, fraction positive pairs
  - raw within / cross / delta / perm-p
  - residualized-on-overall-competence (row mean) within / cross / delta / perm-p
    [the DEFENSIBLE residualization]
  - residualized-on-PC1 delta / perm-p  [biased; reference only, do NOT headline]

Conditions:
  full              all 13 paradigms
  drop_floor3       drop cvlt + go_nogo + n_back (the 3 near-zero-scaling ones)
  drop_lowvar3      drop the 3 lowest-variance paradigms (data-driven)
  loo_cvlt          leave CVLT out (ceiling)
  loo_go_nogo       leave Go/No-Go out
  loo_n_back        leave n-back out
"""
import sys, os, json, warnings
import numpy as np

warnings.filterwarnings("ignore")

ROOT = os.environ.get("COGARENA_ROOT",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compute_b2_expanded as B2  # noqa: E402

P_ALL = B2.PARADIGMS_ORDER          # 13 paradigms, sorted
DOM = B2.DOMAIN_MAP
SEED, N_PERM = 42, 5000


def build_raw():
    """All model rows (paradigm->accuracy), including partially-complete rows."""
    data = {}
    for m in B2.OLD_MODELS:
        row, _ = B2.build_model_row(m, is_old=True)
        if row:
            data[m] = row
    new_names = [l.strip() for l in open(f"{ROOT}/results/reanalysis/expansion_modellist.txt") if l.strip()]
    for m in new_names:
        row, _ = B2.build_model_row(m, is_old=False)
        if row:
            data[m] = row
    return data


def _wc(corr, P):
    w, c = [], []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            (w if DOM[P[i]] == DOM[P[j]] else c).append(corr[i, j])
    return np.array(w), np.array(c)


def _perm_p(corr, delta, P):
    rng = np.random.default_rng(SEED)
    labels = [DOM[p] for p in P]
    deltas = []
    for _ in range(N_PERM):
        sh = rng.permutation(labels).tolist()
        w, c = [], []
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                (w if sh[i] == sh[j] else c).append(corr[i, j])
        if w and c:
            deltas.append(np.mean(w) - np.mean(c))
    deltas = np.array(deltas)
    return float(np.mean(deltas >= delta))


def _residualize(M, regressor):
    resid = np.empty_like(M, dtype=float)
    for j in range(M.shape[1]):
        b1, b0 = np.polyfit(regressor, M[:, j], 1)
        resid[:, j] = M[:, j] - (b1 * regressor + b0)
    return resid


def analyze(data, P):
    models = [m for m in sorted(data) if all(p in data[m] for p in P)]
    M = np.array([[data[m][p] for p in P] for m in models], dtype=float)
    corr = np.corrcoef(M.T)

    w0, c0 = _wc(corr, P)
    raw_delta = float(w0.mean() - c0.mean())
    raw_p = _perm_p(corr, raw_delta, P)

    eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
    pc1 = float(eig[0] / eig.sum())
    iu = np.triu_indices(len(P), 1)
    mean_r = float(np.mean(corr[iu]))
    frac_pos = float(np.mean(corr[iu] > 0))

    # DEFENSIBLE: residualize on overall competence (row mean)
    rm = M.mean(1)
    rc_rm = np.corrcoef(_residualize(M, rm).T)
    wrm, crm = _wc(rc_rm, P)
    rm_delta = float(wrm.mean() - crm.mean())
    rm_p = _perm_p(rc_rm, rm_delta, P)

    # REFERENCE ONLY (biased): residualize on first PC score
    Z = (M - M.mean(0)) / M.std(0, ddof=1)
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    g = U[:, 0] * S[0]
    if np.corrcoef(g, M.mean(1))[0, 1] < 0:
        g = -g
    rc_pc1 = np.corrcoef(_residualize(Z, g).T)
    wpc, cpc = _wc(rc_pc1, P)
    pc_delta = float(wpc.mean() - cpc.mean())
    pc_p = _perm_p(rc_pc1, pc_delta, P)

    return {
        "n_models": len(models), "n_paradigms": len(P),
        "pc1_var_explained": round(pc1, 4),
        "mean_offdiag_r": round(mean_r, 4), "frac_pairs_positive": round(frac_pos, 3),
        "raw_within": round(float(w0.mean()), 4), "raw_cross": round(float(c0.mean()), 4),
        "raw_delta": round(raw_delta, 4), "raw_p": round(raw_p, 4),
        "resid_overallcomp_within": round(float(wrm.mean()), 4),
        "resid_overallcomp_cross": round(float(crm.mean()), 4),
        "resid_overallcomp_delta": round(rm_delta, 4), "resid_overallcomp_p": round(rm_p, 4),
        "resid_pc1_delta_REF": round(pc_delta, 4), "resid_pc1_p_REF": round(pc_p, 4),
    }


def main():
    data = build_raw()
    full_models = [m for m in sorted(data) if all(p in data[m] for p in P_ALL)]
    M = np.array([[data[m][p] for p in P_ALL] for m in full_models], dtype=float)

    # per-paradigm mean & std (range/variance) on the full complete-case set
    stats = {P_ALL[j]: {"mean": round(float(M[:, j].mean()), 4),
                        "std": round(float(M[:, j].std(ddof=1)), 4),
                        "min": round(float(M[:, j].min()), 4),
                        "max": round(float(M[:, j].max()), 4)}
             for j in range(len(P_ALL))}
    by_std = sorted(P_ALL, key=lambda p: stats[p]["std"])
    lowvar3 = by_std[:3]

    floor3 = ["cvlt_word_list", "go_nogo", "n_back"]
    conditions = {
        "full": P_ALL,
        "drop_floor3 (cvlt+go_nogo+n_back)": [p for p in P_ALL if p not in floor3],
        f"drop_lowvar3 ({'+'.join(lowvar3)})": [p for p in P_ALL if p not in lowvar3],
        "loo_cvlt": [p for p in P_ALL if p != "cvlt_word_list"],
        "loo_go_nogo": [p for p in P_ALL if p != "go_nogo"],
        "loo_n_back": [p for p in P_ALL if p != "n_back"],
    }

    results = {name: analyze(data, P) for name, P in conditions.items()}

    out = {
        "method": ("Restricted-range robustness. Same matrix builder + 5000-perm test "
                   "(seed 42) as the headline B2/PCA analysis. Complete-case models per "
                   "paradigm subset. resid_overallcomp = residualize each paradigm on the "
                   "model's overall-competence (row mean) then recompute within vs cross "
                   "(the defensible general-factor removal). resid_pc1 = residualize on the "
                   "first PC score (biased downward; reference only, not for headline)."),
        "paradigm_stats_full": stats,
        "lowest_variance_3": lowvar3,
        "conditions": results,
    }
    os.makedirs(f"{ROOT}/results/reanalysis", exist_ok=True)
    outp = f"{ROOT}/results/reanalysis/restricted_range_robustness.json"
    json.dump(out, open(outp, "w"), indent=2)

    # ---- console table ----
    print("\n=== per-paradigm mean / std (full complete-case set, n=%d) ===" % len(full_models))
    for p in by_std:
        s = stats[p]
        flag = "  <-- low var" if p in lowvar3 else ""
        print(f"  {p:22s} mean={s['mean']:.3f} std={s['std']:.3f} range=[{s['min']:.2f},{s['max']:.2f}]{flag}")

    cols = ["n_models", "n_paradigms", "pc1_var_explained", "mean_offdiag_r",
            "raw_delta", "raw_p", "resid_overallcomp_delta", "resid_overallcomp_p",
            "resid_pc1_delta_REF", "resid_pc1_p_REF"]
    print("\n=== robustness table ===")
    hdr = f"{'condition':38s} " + " ".join(f"{c:>12s}" for c in cols)
    print(hdr)
    for name, r in results.items():
        print(f"{name:38s} " + " ".join(f"{r[c]:>12}" for c in cols))
    print("\nSaved to", outp)
    print("\n[self-check] full should be ~ raw_delta -0.005 / raw_p 0.52 / pc1 0.44 / "
          "resid_overallcomp_delta 0.04 / resid_overallcomp_p 0.29")


if __name__ == "__main__":
    main()
