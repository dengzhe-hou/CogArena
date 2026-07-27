#!/usr/bin/env python3
"""Step-3 recompute: corrected 55-model x 13-paradigm matrix + headline stats.

Data corrections applied on top of the published pipeline (methodology is
IDENTICAL - the analysis functions are imported from
scripts/compute_b2_expanded.py; only the data layer changes):
  1. Scorer-fix overlays from results/rescore_20260702/new_scores/
     (source_monitoring all models; epitome_tom expansion models - the 20
     primary models' EPITOME per-item records are synthetic v3_rerun
     placeholders with no responses to re-score, kept as archived).
  2. go_nogo cells replaced by the fixed-prompt rerun
     (results/gonogo_rerun_20260702/, rule visible on every trial).
  3. Multiturn paradigms recomputed UNIFORMLY as continuous turn-mean accuracy
     from the per-item files for BOTH cohorts (the archived aggregates mixed
     continuous turn-mean for the 20 primary with binarized pass-rate for the
     35 expansion models), with operation_span file accuracies taken from the
     rescore overlay (fixed yes/no rule).
  4. Post-decision wagering accuracy taken from the full 55x50 current-scorer
     replay overlay when COGARENA_WAGER_OVERLAY is set.

Self-check: the UNCORRECTED pipeline is rerun first and must reproduce the
published 20-model numbers (within .4145 / cross .4250 / delta -.0106 / p .53)
before any corrected number is trusted.

Adds a model-bootstrap CI for delta (the published "equivalence CI" was
permutation-null percentiles) and PC1 variance share.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = __import__("os").environ.get("COGARENA_ROOT") or __import__("os").path.abspath(
    __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "recompute_20260703")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ["COGARENA_ROOT"] = ROOT

import compute_b2_expanded as b2

RESCORE = os.path.join(ROOT, "results", "rescore_20260702", "new_scores")
GONOGO = os.path.join(ROOT, "results", "gonogo_rerun_20260702")

# Final SM overlay adapter (2026-07-18): when COGARENA_SM_OVERLAY points at
# results/sm_rerun_20260718/sm_scores_overlay.json (or a successor), every
# source_monitoring cell and item pool takes its 55x50 corrected scores
# (rescore_20260702 values for the 39 unaffected frozen episodes plus the
# dedup-fixed reruns for the 11 affected ones).  Unset, the historical
# rescore_20260702 path is reproduced so pre-existing artifacts stay
# verifiable.  A missing model or task under the override fails loudly.
SM_OVERLAY_PATH = os.environ.get("COGARENA_SM_OVERLAY")
SM_OVERLAY = json.load(open(SM_OVERLAY_PATH)) if SM_OVERLAY_PATH else None

# Full 55x50 wagering replay overlay.  This is deliberately separate from the
# July rescore bundle: it was produced by replaying every archived response
# through the current scorer, so the two comma-normalisation corrections are
# applied without mutating the frozen historical artifacts.  Missing keys fail
# closed whenever the adapter is active.
WAGER_OVERLAY_PATH = os.environ.get("COGARENA_WAGER_OVERLAY")
WAGER_OVERLAY = json.load(open(WAGER_OVERLAY_PATH)) if WAGER_OVERLAY_PATH else None


def static_paths(model, is_old):
    set_name = "full_eval_20260526_2208" if is_old else "full_eval_expansion"
    return set_name, f"{ROOT}/results/{set_name}/openai_{model}/text/details.json"


def mt_base(model, is_old):
    if is_old:
        return "multiturn_eval_v3", f"{ROOT}/results/multiturn_eval_v3/openai_{model}"
    return "multiturn_expansion", f"{ROOT}/results/multiturn_expansion/openai_{model}/text"


def corrected_row(model, is_old):
    """Paradigm->accuracy dict with all frozen data corrections applied."""
    set_name, det_path = static_paths(model, is_old)
    if not os.path.exists(det_path):
        return None, f"missing static {det_path}"
    details = b2.load_details(det_path)

    # -- static paradigms: continuous per-item mean + rescore overlay --------
    ov_path = os.path.join(RESCORE, f"{set_name}__openai_{model}.json")
    overlay = json.load(open(ov_path)) if os.path.exists(ov_path) else {}
    by_par = {}
    for it in details:
        p = it.get("paradigm")
        if p not in b2.STATIC_PARADIGMS or p == "go_nogo":
            continue
        if p == "source_monitoring" and SM_OVERLAY is not None:
            acc = SM_OVERLAY[model][it["task_id"]]
        elif p == "post_decision_wagering" and WAGER_OVERLAY is not None:
            acc = WAGER_OVERLAY[model][it["task_id"]]
        else:
            acc = overlay.get(it["task_id"], b2.item_accuracy(it.get("score")))
        by_par.setdefault(p, []).append(acc)
    row = {p: float(np.mean(v)) for p, v in by_par.items() if v}

    # confidence_calibration corrected values (same override the pipeline uses)
    _cc = b2._cc_corrected().get(model)
    if _cc is not None:
        row["confidence_calibration"] = _cc

    # -- go_nogo: fixed-prompt rerun -----------------------------------------
    gg = os.path.join(GONOGO, f"openai_{model}", "text", "details.json")
    if not os.path.exists(gg):
        return None, f"missing gonogo rerun {gg}"
    gg_items = json.load(open(gg))
    row["go_nogo"] = float(np.mean([b2.item_accuracy(r.get("score")) for r in gg_items]))

    # -- multiturn: uniform continuous turn-mean + ospan overlay -------------
    mt_set, base = mt_base(model, is_old)
    mt_ov_path = os.path.join(RESCORE, f"{mt_set}__openai_{model}__multiturn.json")
    mt_overlay = json.load(open(mt_ov_path)) if os.path.exists(mt_ov_path) else {}
    set_root = f"{ROOT}/results/{mt_set}"
    for par in sorted(b2.MT_PARADIGMS):
        accs = []
        for f in sorted(glob.glob(os.path.join(base, "*", par, "*.json"))):
            d = json.load(open(f))
            sc = d.get("score") or {}
            if "accuracy" not in sc:
                continue
            rel = os.path.relpath(f, set_root)
            accs.append(float(mt_overlay.get(rel, sc["accuracy"])))
        if accs:
            row[par] = float(np.mean(accs))
    return row, None


def bootstrap_delta_ci(models_data, models, n_boot=5000, seed=42):
    """Model-bootstrap CI for delta (resample models with replacement)."""
    matrix = np.array([[models_data[m].get(p, 0) for p in b2.PARADIGMS_ORDER]
                       for m in models])
    labels = [b2.DOMAIN_MAP[p] for p in b2.PARADIGMS_ORDER]
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(models)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sub = matrix[idx]
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(sub.T)
        w, c = [], []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                r = corr[i, j]
                if not np.isfinite(r):
                    continue
                (w if labels[i] == labels[j] else c).append(r)
        if w and c:
            deltas.append(np.mean(w) - np.mean(c))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi), len(deltas)


def pc1_share(models_data, models):
    matrix = np.array([[models_data[m].get(p, 0) for p in b2.PARADIGMS_ORDER]
                       for m in models])
    z = (matrix - matrix.mean(0)) / matrix.std(0, ddof=1)
    ev = np.linalg.eigvalsh(np.corrcoef(matrix.T))[::-1]
    return float(ev[0] / ev.sum()), [float(x) for x in ev[:3]]


def main():
    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    old_models = list(b2.OLD_MODELS.keys())
    new_models = sorted(new_meta.keys())
    print(f"[pools] old={len(old_models)} new={len(new_models)}", flush=True)

    # ---- self-check: UNCORRECTED pipeline must reproduce published numbers --
    unc, drop = {}, []
    for m in old_models + new_models:
        row, err = b2.build_model_row(m, m in b2.OLD_MODELS)
        if row is None:
            drop.append((m, err)); continue
        unc[m] = row
    models_all = [m for m in old_models + new_models if m in unc]
    old_present = [m for m in old_models if m in unc]
    repro20, _ = b2.run_b2(unc, old_present)
    print("[self-check 20 uncorrected]", {k: repro20[k] for k in
          ("within_mean", "cross_mean", "delta", "p_value")}, flush=True)
    target = {"within_mean": 0.4145, "cross_mean": 0.425, "delta": -0.0106}
    ok = all(abs(repro20[k] - v) < 5e-4 for k, v in target.items())
    if not ok:
        sys.exit(f"SELF-CHECK FAILED vs published 20-model numbers: {repro20}")
    unc55, _ = b2.run_b2(unc, models_all)
    print("[self-check 55 uncorrected]", {k: unc55[k] for k in
          ("within_mean", "cross_mean", "delta", "p_value")}, flush=True)

    # ---- corrected matrix ----------------------------------------------------
    cor, drop2 = {}, []
    for m in models_all:
        row, err = corrected_row(m, m in b2.OLD_MODELS)
        if row is None:
            drop2.append((m, err)); continue
        missing = [p for p in b2.PARADIGMS_ORDER if p not in row]
        if missing:
            print(f"[warn] {m} missing paradigms: {missing}", flush=True)
        cor[m] = row
    models_c = [m for m in models_all if m in cor]
    print(f"[corrected] {len(models_c)} models, dropped: {drop2}", flush=True)

    cor55, matrix = b2.run_b2(cor, models_c)
    cor20, _ = b2.run_b2(cor, [m for m in old_present if m in cor])
    blo, bhi, nb = bootstrap_delta_ci(cor, models_c)
    ublo, ubhi, _ = bootstrap_delta_ci(unc, models_all)
    pc1_c, ev_c = pc1_share(cor, models_c)
    pc1_u, ev_u = pc1_share(unc, models_all)

    # per-paradigm means old vs corrected
    par_means = {}
    for p in b2.PARADIGMS_ORDER:
        par_means[p] = {
            "uncorrected": round(float(np.mean([unc[m].get(p, np.nan) for m in models_all])), 4),
            "corrected": round(float(np.mean([cor[m].get(p, np.nan) for m in models_c])), 4),
        }

    # biggest cell shifts
    shifts = []
    for m in models_c:
        for p in b2.PARADIGMS_ORDER:
            a, bb = unc[m].get(p), cor[m].get(p)
            if a is not None and bb is not None and abs(bb - a) > 1e-6:
                shifts.append((round(bb - a, 4), m, p, round(a, 4), round(bb, 4)))
    shifts.sort(key=lambda x: -abs(x[0]))

    out = {
        "method": "same as compute_b2_expanded.py (imported); data layer corrected: "
                  "rescore overlays (srcmon/epitome/ospan), go_nogo fixed-prompt rerun, "
                  "full-pool current-scorer wagering replay overlay, "
                  "uniform continuous turn-mean for multiturn paradigms in both cohorts",
        "self_check_20_uncorrected": repro20,
        "uncorrected_55": unc55,
        "uncorrected_55_bootstrap_delta_ci": [round(ublo, 4), round(ubhi, 4)],
        "corrected_55": cor55,
        "corrected_20": cor20,
        "corrected_55_bootstrap_delta_ci": [round(blo, 4), round(bhi, 4)],
        "bootstrap_n": nb,
        "equivalence_vs_0.15": bool(bhi < 0.15),
        "pc1_share": {"uncorrected": round(pc1_u, 4), "corrected": round(pc1_c, 4)},
        "eigenvalues_top3": {"uncorrected": ev_u, "corrected": ev_c},
        "paradigm_means": par_means,
        "n_cells_shifted": len(shifts),
        "top_cell_shifts": shifts[:30],
        "dropped_models": drop + drop2,
    }
    json.dump(out, open(os.path.join(OUT, "recompute_summary.json"), "w"), indent=1)

    import csv
    with open(os.path.join(OUT, "corrected_matrix.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model"] + b2.PARADIGMS_ORDER)
        for m in models_c:
            w.writerow([m] + [round(cor[m].get(p, float("nan")), 4) for p in b2.PARADIGMS_ORDER])

    print(json.dumps({k: out[k] for k in ("corrected_55", "corrected_55_bootstrap_delta_ci",
          "equivalence_vs_0.15", "pc1_share", "n_cells_shifted")}, indent=1), flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
