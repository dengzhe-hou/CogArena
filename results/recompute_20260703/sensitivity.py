#!/usr/bin/env python3
"""Sensitivity of the corrected separability delta to known artifacts.

Variants on the corrected 55-model matrix (see build_and_recompute.py):
  A. metacognition item-dedup: wagering cells recomputed EXCLUDING items whose
     question text also appears in that model's confidence_calibration items
     (round-1 audit: 15/50 verbatim shared questions mechanically couple the
     only Meta within-pair).
  B. drop the Meta within-pair from the delta (treat its 2 paradigms as
     cross-only), keeping everything else.
  C. leave-one-group-out deltas.
  D. pairwise correlation table uncorrected vs corrected (which pairs moved).
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.environ.get("COGARENA_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "recompute_20260703")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ["COGARENA_ROOT"] = ROOT
import compute_b2_expanded as b2

RESCORE = os.path.join(ROOT, "results", "rescore_20260702", "new_scores")
WAGER_OVERLAY = (json.load(open(os.environ["COGARENA_WAGER_OVERLAY"]))
                 if os.environ.get("COGARENA_WAGER_OVERLAY") else None)


def norm_q(stim):
    """Normalize an item's question text for cross-paradigm matching."""
    s = str(stim).lower()
    # strip instruction boilerplate lines, keep question-ish content
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    qlines = [l for l in lines if "?" in l]
    core = qlines[0] if qlines else (lines[-1] if lines else s)
    return re.sub(r"[^a-z0-9]+", " ", core).strip()


def load_matrix():
    rows = list(csv.reader(open(os.environ.get("COGARENA_PRIMARY_MATRIX")
                                or os.path.join(OUT, "corrected_matrix.csv"))))
    hdr = rows[0][1:]
    models = [r[0] for r in rows[1:]]
    M = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    return hdr, models, M


def delta_stats(corr, labels, drop_pairs=frozenset(), n_perm=5000, seed=42):
    idx_pairs = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    within = [corr[i, j] for i, j in idx_pairs
              if labels[i] == labels[j] and (i, j) not in drop_pairs]
    cross = [corr[i, j] for i, j in idx_pairs
             if labels[i] != labels[j] and (i, j) not in drop_pairs]
    delta = float(np.mean(within) - np.mean(cross))
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(n_perm):
        sh = rng.permutation(labels).tolist()
        w = [corr[i, j] for i, j in idx_pairs if sh[i] == sh[j] and (i, j) not in drop_pairs]
        c = [corr[i, j] for i, j in idx_pairs if sh[i] != sh[j] and (i, j) not in drop_pairs]
        if w and c:
            perm.append(np.mean(w) - np.mean(c))
    p = float(np.mean([d >= delta for d in perm]))
    return round(float(np.mean(within)), 4), round(float(np.mean(cross)), 4), round(delta, 4), round(p, 4)


def main():
    hdr, models, M = load_matrix()
    labels = [b2.DOMAIN_MAP[p] for p in hdr]
    i_cc, i_pdw = hdr.index("confidence_calibration"), hdr.index("post_decision_wagering")

    # ---- A. metacognition item-dedup ---------------------------------------
    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    shared_report = None
    M_dedup = M.copy()
    n_excluded = []
    for k, m in enumerate(models):
        is_old = m in b2.OLD_MODELS
        set_name = "full_eval_20260526_2208" if is_old else "full_eval_expansion"
        det = b2.load_details(f"{ROOT}/results/{set_name}/openai_{m}/text/details.json")
        # regenerated battery not needed: match on details' own stimuli is not
        # stored, so match via the battery items (same for all models).
        n_excluded.append(None)
    # stimuli live in the battery, not details.json -> build the shared-item
    # task_id set ONCE from the regenerated battery.
    from cogarena.cli import _collect_items
    items = _collect_items(50, 42, None)
    cc_q = {}
    pdw_items = []
    for it in items:
        if it.metadata.paradigm == "confidence_calibration":
            cc_q.setdefault(norm_q(it.stimulus), []).append(it.task_id)
        elif it.metadata.paradigm == "post_decision_wagering":
            pdw_items.append((it.task_id, norm_q(it.stimulus)))
    shared_pdw_ids = {tid for tid, q in pdw_items if q in cc_q}
    shared_report = f"{len(shared_pdw_ids)}/50 wagering items share their question with confidence_calibration"
    print("[dedup]", shared_report, flush=True)

    for k, m in enumerate(models):
        is_old = m in b2.OLD_MODELS
        set_name = "full_eval_20260526_2208" if is_old else "full_eval_expansion"
        det = b2.load_details(f"{ROOT}/results/{set_name}/openai_{m}/text/details.json")
        ov_p = os.path.join(RESCORE, f"{set_name}__openai_{m}.json")
        ov = json.load(open(ov_p)) if os.path.exists(ov_p) else {}
        vals = [(WAGER_OVERLAY[m][r["task_id"]] if WAGER_OVERLAY is not None
                 else ov.get(r["task_id"], b2.item_accuracy(r.get("score"))))
                for r in det if r.get("paradigm") == "post_decision_wagering"
                and r["task_id"] not in shared_pdw_ids]
        if vals:
            M_dedup[k, i_pdw] = float(np.mean(vals))

    corr = np.corrcoef(M.T)
    corr_dedup = np.corrcoef(M_dedup.T)
    res = {"shared_items": shared_report}
    res["corrected_full"] = delta_stats(corr, labels)
    res["A_meta_dedup"] = delta_stats(corr_dedup, labels)
    res["A_meta_pair_r"] = {"before": round(float(corr[i_cc, i_pdw]), 4),
                            "after_dedup": round(float(corr_dedup[i_cc, i_pdw]), 4)}

    # ---- B. drop the Meta within-pair --------------------------------------
    pair = (min(i_cc, i_pdw), max(i_cc, i_pdw))
    res["B_drop_meta_pair"] = delta_stats(corr, labels, drop_pairs=frozenset([pair]))

    # ---- C. leave-one-group-out ---------------------------------------------
    res["C_leave_one_group_out"] = {}
    for g in sorted(set(labels)):
        keep = [i for i, l in enumerate(labels) if l != g]
        sub_corr = corr[np.ix_(keep, keep)]
        sub_labels = [labels[i] for i in keep]
        res["C_leave_one_group_out"][f"minus_{g}"] = delta_stats(sub_corr, sub_labels)

    # ---- D. pair table uncorrected vs corrected -----------------------------
    # rebuild uncorrected matrix through the published pipeline
    unc = {}
    for m in models:
        row, err = b2.build_model_row(m, m in b2.OLD_MODELS)
        unc[m] = row
    U = np.array([[unc[m].get(p, 0) for p in hdr] for m in models])
    corr_u = np.corrcoef(U.T)
    pairs = []
    for i in range(len(hdr)):
        for j in range(i + 1, len(hdr)):
            kind = labels[i] if labels[i] == labels[j] else "cross"
            pairs.append({"pair": f"{hdr[i]} x {hdr[j]}", "kind": kind,
                          "r_uncorrected": round(float(corr_u[i, j]), 3),
                          "r_corrected": round(float(corr[i, j]), 3),
                          "shift": round(float(corr[i, j] - corr_u[i, j]), 3)})
    pairs.sort(key=lambda x: -abs(x["shift"]))
    res["D_pair_shifts_top15"] = pairs[:15]

    json.dump(res, open(os.path.join(OUT, "sensitivity.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "D_pair_shifts_top15"}, indent=1), flush=True)
    print("[top pair shifts]", flush=True)
    for p in pairs[:10]:
        print(f"  {p['pair']:55s} {p['kind']:9s} {p['r_uncorrected']:+.3f} -> {p['r_corrected']:+.3f}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
