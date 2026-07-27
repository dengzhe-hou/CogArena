#!/usr/bin/env python3
"""Joint exclusion of the three validity-threatened paradigms (reviewer request):
text Stroop (ceiling, no text interference signature), Go/No-Go (response-bias
vulnerable accuracy), CVLT (availability, list visible at recall). Drop all
three jointly from BOTH the accuracy and the construct-native matrix; report
PC1 share, raw within-cross delta, and row-mean-residual delta with the same
5000-permutation label test (seed 42) used by Table S3."""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("COGARENA_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ["COGARENA_ROOT"] = ROOT
import compute_b2_expanded as b2

DROP = {"stroop", "go_nogo", "cvlt_word_list"}
KEEP = [p for p in b2.PARADIGMS_ORDER if p not in DROP]
LABELS = [b2.DOMAIN_MAP[p] for p in KEEP]
NP_ = len(KEEP)
PAIRS = [(i, j) for i in range(NP_) for j in range(i + 1, NP_)]
WMASK = np.array([LABELS[i] == LABELS[j] for i, j in PAIRS])


def load(path):
    rows = list(csv.reader(open(path)))
    hdr = rows[0][1:]
    idx = [hdr.index(p) for p in KEEP]
    return np.array([[float(r[1 + k]) for k in idx] for r in rows[1:]])


def perm_test(corr, seed=42, n_perm=5000):
    v = np.array([corr[i, j] for i, j in PAIRS])
    obs = float(v[WMASK].mean() - v[~WMASK].mean())
    rng = np.random.default_rng(seed)
    labs = np.array(LABELS)
    deltas = np.empty(n_perm)
    for k in range(n_perm):
        sh = labs[rng.permutation(NP_)]
        w = np.array([sh[i] == sh[j] for i, j in PAIRS])
        deltas[k] = v[w].mean() - v[~w].mean()
    p1 = float((np.sum(deltas >= obs) + 1) / (n_perm + 1))
    p2 = float((np.sum(np.abs(deltas) >= abs(obs)) + 1) / (n_perm + 1))
    return round(obs, 4), round(p1, 4), round(p2, 4)


def zscore(M):
    return (M - M.mean(0)) / M.std(0, ddof=1)


def rowmean_residual(M, zfirst=False):
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


def pc1_share(M):
    ev = np.linalg.eigvalsh(np.corrcoef(zscore(M).T))[::-1]
    return round(float(ev[0] / ev.sum()), 4)


def block(M, zfirst_resid):
    return {
        "pc1": pc1_share(M),
        "raw (d, p1, p2)": perm_test(np.corrcoef(M.T)),
        "rowmean_resid (d, p1, p2)": perm_test(
            np.corrcoef(rowmean_residual(M, zfirst=zfirst_resid).T)),
        "n_within_pairs": int(WMASK.sum()), "n_cross_pairs": int((~WMASK).sum()),
    }


MA = load(os.environ.get("COGARENA_PRIMARY_MATRIX")
          or f"{ROOT}/results/recompute_20260703/corrected_matrix.csv")
MC = load(os.environ.get("COGARENA_PRIMARY_CONSTRUCT_MATRIX")
          or f"{ROOT}/results/construct_native_20260711/construct_matrix.csv")
out = {"dropped": sorted(DROP), "kept": KEEP,
       "accuracy": block(MA, zfirst_resid=False),
       "construct": block(MC, zfirst_resid=True)}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
          "joint_exclusion.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
