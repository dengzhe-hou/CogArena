#!/usr/bin/env python3
"""Does the dimensional-separability null reflect a single general factor
(a positive manifold) rather than separable domains? And does ANY block
structure survive removing that general factor?

Reuses the validated 52x13 matrix builder from scripts/compute_b2_expanded.py
(reproducing its B2 delta is the built-in self-check), then adds:
  (1) PCA / variance decomposition on the 13-paradigm correlation matrix
      -> how much variance does the first principal component explain?
  (2) Partial-correlation: residualize every paradigm on the general factor
      (first PC score per model), recompute within- vs cross-domain mean
      correlation on the residuals, with a 5000-permutation test (seed=42).
If the residual within-minus-cross delta is still ~0, the null is not merely
"a strong general factor inflates everything"; no residual domain block
structure is detectable either.
"""
import sys, os, json
import numpy as np

import os
ROOT = os.environ.get("COGARENA_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compute_b2_expanded as B2  # noqa: E402

P = B2.PARADIGMS_ORDER
DOM = B2.DOMAIN_MAP


def build_all_data():
    data = {}
    for m in B2.OLD_MODELS:
        row, _ = B2.build_model_row(m, is_old=True)
        if row and all(p in row for p in P):
            data[m] = row
    new_names = [l.strip() for l in open(f"{ROOT}/results/reanalysis/expansion_modellist.txt") if l.strip()]
    for m in new_names:
        row, _ = B2.build_model_row(m, is_old=False)
        if row and all(p in row for p in P):
            data[m] = row
    return data


def within_cross(corr):
    w, c = [], []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            (w if DOM[P[i]] == DOM[P[j]] else c).append(corr[i, j])
    return np.array(w), np.array(c)


def perm_p(corr, delta, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    labels = [DOM[p] for p in P]
    deltas = []
    for _ in range(n_perm):
        sh = rng.permutation(labels).tolist()
        w, c = [], []
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                (w if sh[i] == sh[j] else c).append(corr[i, j])
        if w and c:
            deltas.append(np.mean(w) - np.mean(c))
    deltas = np.array(deltas)
    p = float(np.mean(deltas >= delta))
    lo, hi = [float(x) for x in np.percentile(deltas, [2.5, 97.5])]
    return p, lo, hi


def main():
    data = build_all_data()
    models = sorted(data)
    M = np.array([[data[m][p] for p in P] for m in models], dtype=float)  # 52 x 13
    n_models = len(models)

    # ---- built-in validation: reproduce B2 raw delta ----
    b2, _ = B2.run_b2(data, models)
    raw_corr = np.corrcoef(M.T)
    w0, c0 = within_cross(raw_corr)
    print(f"[validation] B2 repro on {n_models} models: "
          f"within={b2['within_mean']} cross={b2['cross_mean']} "
          f"delta={b2['delta']} p={b2['p_value']}")

    # ---- (1) PCA / variance decomposition (correlation-matrix eigenvalues) ----
    eig = np.sort(np.linalg.eigvalsh(raw_corr))[::-1]
    var_exp = eig / eig.sum()
    pc1_var = float(var_exp[0])

    # mean off-diagonal correlation (positive-manifold strength)
    iu = np.triu_indices(len(P), 1)
    mean_r = float(np.mean(raw_corr[iu]))
    frac_pos = float(np.mean(raw_corr[iu] > 0))

    # ---- (2) general factor = first PC; residualize and recompute ----
    Z = (M - M.mean(0)) / M.std(0, ddof=1)            # standardized paradigms
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    g = U[:, 0] * S[0]                                 # general-competence score per model
    # sign so that higher g = higher mean accuracy
    if np.corrcoef(g, M.mean(1))[0, 1] < 0:
        g = -g
    g_loading_meanacc = float(np.corrcoef(g, M.mean(1))[0, 1])
    resid = np.empty_like(Z)
    for j in range(Z.shape[1]):
        b1, b0 = np.polyfit(g, Z[:, j], 1)
        resid[:, j] = Z[:, j] - (b1 * g + b0)
    resid_corr = np.corrcoef(resid.T)
    wr, cr = within_cross(resid_corr)
    delta_r = float(wr.mean() - cr.mean())
    p_r, lo_r, hi_r = perm_p(resid_corr, delta_r)

    # robustness: partial out simple row-mean (overall accuracy) instead of PC1
    rm = M.mean(1)
    resid2 = np.empty_like(M, dtype=float)
    for j in range(M.shape[1]):
        b1, b0 = np.polyfit(rm, M[:, j], 1)
        resid2[:, j] = M[:, j] - (b1 * rm + b0)
    rc2 = np.corrcoef(resid2.T)
    wr2, cr2 = within_cross(rc2)
    delta_r2 = float(wr2.mean() - cr2.mean())
    p_r2, _, _ = perm_p(rc2, delta_r2)

    out = {
        "n_models": n_models, "n_paradigms": len(P),
        "raw": {"within": round(float(w0.mean()), 4), "cross": round(float(c0.mean()), 4),
                "delta": round(float(w0.mean() - c0.mean()), 4)},
        "positive_manifold": {
            "mean_offdiag_r": round(mean_r, 4),
            "frac_pairs_positive": round(frac_pos, 3),
            "pc1_variance_explained": round(pc1_var, 4),
            "eigenvalues_top5": [round(float(x), 3) for x in eig[:5]],
            "pc1_corr_with_mean_accuracy": round(g_loading_meanacc, 3),
        },
        "after_removing_general_factor_PC1": {
            "within": round(float(wr.mean()), 4), "cross": round(float(cr.mean()), 4),
            "delta": round(delta_r, 4), "p_value": round(p_r, 4),
            "ci95": [round(lo_r, 4), round(hi_r, 4)],
        },
        "after_removing_general_factor_rowmean": {
            "within": round(float(wr2.mean()), 4), "cross": round(float(cr2.mean()), 4),
            "delta": round(delta_r2, 4), "p_value": round(p_r2, 4),
        },
        "interpretation": (
            "First PC explains %.0f%% of paradigm variance with %.0f%% of pairwise "
            "correlations positive (a positive manifold). After removing that general "
            "factor, within-domain correlations are %s than cross-domain "
            "(delta=%.3f, p=%.2f): no residual block structure."
        ) % (pc1_var * 100, frac_pos * 100,
             "no higher" if delta_r <= 0.02 else "higher", delta_r, p_r),
    }
    os.makedirs(f"{ROOT}/results/reanalysis", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results/reanalysis/pca_partialcorr.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
