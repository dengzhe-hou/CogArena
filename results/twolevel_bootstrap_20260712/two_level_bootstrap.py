#!/usr/bin/env python3
"""Family x item TWO-LEVEL bootstrap for the headline within-minus-cross delta
(reviewer request: "family x item 双层 bootstrap", 2026-07-12).

The paper reports (a) a family-clustered bootstrap CI over 24 merged model
families and (b) item-level split-half reliabilities, but no JOINT resampling
of both variance sources. This script adds three percentile CIs for delta on
the corrected ACCURACY matrix (the confirmatory analysis):

  family-only  : resample families with replacement (anchor: must reproduce the
                 committed family CI [-0.02, 0.09] up to MC error)
  item-only    : models fixed; resample items with replacement per paradigm
                 (crossed: the battery is identical across models, so one item
                 resample per paradigm is applied to every model)
  two-level    : both levels resampled in the same replicate

Discipline mirrored from build_and_recompute.py / build_construct_matrix.py:
  SELF-CHECK first -- the per-item pools assembled here must reproduce
  results/recompute_20260703/corrected_matrix.csv cell-by-cell (tol 6e-5,
  the CSV is rounded to 4 dp) before any bootstrap number is trusted.

CPU-only; run via Slurm (batch partition), never the login node.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = __import__("os").environ.get("COGARENA_ROOT") or __import__("os").path.abspath(
    __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "results", "recompute_20260703"))
sys.path.insert(0, ROOT)
os.environ["COGARENA_ROOT"] = ROOT

import compute_b2_expanded as b2
import build_and_recompute as bar
from build_and_recompute import corrected_row, static_paths, mt_base

RESCORE = bar.RESCORE if hasattr(bar, "RESCORE") else os.path.join(
    ROOT, "results", "rescore_20260702", "new_scores")
GONOGO = os.path.join(ROOT, "results", "gonogo_rerun_20260702")
CSV = os.path.join(ROOT, "results", "recompute_20260703", "corrected_matrix.csv")
OUT_JSON = os.path.join(HERE, "twolevel_bootstrap.json")
SEED = 42
N_BOOT = 5000
TOL = 6e-5

sys.path.insert(0, os.path.join(ROOT, "results", "construct_native_20260711"))
from build_construct_matrix import load_conf_cal_gold  # noqa: E402
from cogarena.dimensions.metacognition import ConfidenceCalibrationGenerator as CC  # noqa: E402


# --------------------------------------------------------------------------- #
# per-item pools (mirror corrected_row exactly)
# --------------------------------------------------------------------------- #
def item_pools(model, is_old, cc_gold):
    """-> {paradigm: [(item_key, acc), ...]} mirroring corrected_row cell means."""
    set_name, det_path = static_paths(model, is_old)
    if not os.path.exists(det_path):
        return None
    details = b2.load_details(det_path)
    ov_path = os.path.join(RESCORE, f"{set_name}__openai_{model}.json")
    overlay = json.load(open(ov_path)) if os.path.exists(ov_path) else {}

    pools = {}
    for it in details:
        p = it.get("paradigm")
        if p not in b2.STATIC_PARADIGMS or p == "go_nogo":
            continue
        if p == "source_monitoring" and bar.SM_OVERLAY is not None:
            acc = bar.SM_OVERLAY[model][it["task_id"]]
        elif p == "post_decision_wagering" and bar.WAGER_OVERLAY is not None:
            acc = bar.WAGER_OVERLAY[model][it["task_id"]]
        else:
            acc = overlay.get(it["task_id"], b2.item_accuracy(it.get("score")))
        pools.setdefault(p, []).append((it["task_id"], float(acc)))

    # confidence_calibration: corrected_row overrides the mean with the fixed-
    # scorer value; rebuild the per-item pool with the same fixed scorer.
    cc_target = b2._cc_corrected().get(model)
    if cc_target is not None:
        cc_pool = []
        text_dir = os.path.dirname(det_path)
        for f in sorted(glob.glob(os.path.join(
                text_dir, "metacognition", "confidence_calibration", "*.json"))):
            d = json.load(open(f))
            t = d.get("task_id")
            if t in cc_gold:
                try:
                    s = CC.score(cc_gold[t], d.get("response", "") or "")
                    cc_pool.append((t, float(s.get("is_correct", 0.0))))
                except Exception:
                    pass
        if cc_pool and abs(np.mean([a for _, a in cc_pool]) - cc_target) < TOL:
            pools["confidence_calibration"] = cc_pool
        # else keep the overlay pool; self-check below decides.

    gg = os.path.join(GONOGO, f"openai_{model}", "text", "details.json")
    if not os.path.exists(gg):
        return None
    gg_items = json.load(open(gg))
    pools["go_nogo"] = [(it.get("task_id", str(i)),
                         float(b2.item_accuracy(it.get("score"))))
                        for i, it in enumerate(gg_items)]

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
            # key by episode basename pattern so keys align across models
            key = os.path.join(os.path.basename(os.path.dirname(os.path.dirname(f))),
                               par, os.path.basename(f))
            accs.append((key, float(mt_overlay.get(rel, sc["accuracy"]))))
        if accs:
            pools[par] = accs
    return pools


def delta_of(matrix, labels):
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(matrix.T)
    w, c = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            r = corr[i, j]
            if np.isfinite(r):
                (w if labels[i] == labels[j] else c).append(r)
    return (np.mean(w) - np.mean(c)) if (w and c) else np.nan


def pc1_of(matrix):
    z = (matrix - matrix.mean(0)) / matrix.std(0, ddof=1)
    ev = np.linalg.eigvalsh(np.cov(z.T))
    return float(ev[-1] / ev.sum())


def main():
    cc_gold = load_conf_cal_gold()
    csv_rows = {}
    with open(CSV) as fh:
        import csv as _csv
        rd = _csv.DictReader(fh)
        cols = [c for c in rd.fieldnames if c != "model"]
        for r in rd:
            csv_rows[r["model"]] = {p: float(r[p]) for p in cols}
    assert cols == b2.PARADIGMS_ORDER, "column order mismatch vs PARADIGMS_ORDER"
    models = list(csv_rows.keys())
    print(f"[csv] {len(models)} models x {len(cols)} paradigms", flush=True)

    import re as _re
    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    fams_raw = {m: (b2.OLD_MODELS[m][1] if m in b2.OLD_MODELS
                    else new_meta[m].get("family", m)) for m in models}
    # paper canon: merged families (validate_pc1.merged_families) -- strip
    # trailing version digits so e.g. qwen2.5/qwen3 merge.
    fams = {m: _re.sub(r"[\d.]+$", "", f) for m, f in fams_raw.items()}
    print(f"[fams] raw={len(set(fams_raw.values()))} "
          f"merged={len(set(fams.values()))} families", flush=True)

    # ---- pools + SELF-CHECK -------------------------------------------------
    pools, bad = {}, []
    for m in models:
        pl = item_pools(m, m in b2.OLD_MODELS, cc_gold)
        if pl is None:
            sys.exit(f"SELF-CHECK IMPOSSIBLE: pools missing for {m}")
        pools[m] = pl
        for p in b2.PARADIGMS_ORDER:
            if p not in pl or not pl[p]:
                sys.exit(f"SELF-CHECK IMPOSSIBLE: empty pool {m}/{p}")
            mean = float(np.mean([a for _, a in pl[p]]))
            if abs(mean - csv_rows[m][p]) > TOL:
                bad.append((m, p, round(mean, 6), csv_rows[m][p]))
    if bad:
        for b_ in bad[:20]:
            print("MISMATCH", b_, flush=True)
        sys.exit(f"SELF-CHECK FAILED: {len(bad)} cells deviate from corrected_matrix.csv")
    print("[self-check] all cells reproduce corrected_matrix.csv  OK", flush=True)

    # ---- crossed-id audit ---------------------------------------------------
    labels = [b2.DOMAIN_MAP[p] for p in b2.PARADIGMS_ORDER]
    crossed, item_mats, per_model_pools = {}, {}, {}
    for p in b2.PARADIGMS_ORDER:
        idsets = [frozenset(k for k, _ in pools[m][p]) for m in models]
        crossed[p] = all(s == idsets[0] for s in idsets)
        if crossed[p]:
            ids = sorted(idsets[0])
            item_mats[p] = np.array([[dict(pools[m][p])[k] for k in ids] for m in models])
        else:
            per_model_pools[p] = [np.array([a for _, a in pools[m][p]]) for m in models]
    n_crossed = sum(crossed.values())
    print(f"[items] crossed paradigms: {n_crossed}/13; "
          f"fallback(independent per-model): "
          f"{[p for p in b2.PARADIGMS_ORDER if not crossed[p]]}", flush=True)

    fam_ids = sorted(set(fams.values()))
    members = {f: [i for i, m in enumerate(models) if fams[m] == f] for f in fam_ids}
    base_matrix = np.array([[csv_rows[m][p] for p in b2.PARADIGMS_ORDER] for m in models])
    print(f"[base] delta = {delta_of(base_matrix, labels):+.4f}  "
          f"PC1 = {pc1_of(base_matrix):.4f}", flush=True)

    def resampled_means(rng):
        """One item-level resample -> (n_models x 13) matrix of paradigm means."""
        cols_out = np.empty((len(models), len(b2.PARADIGMS_ORDER)))
        for j, p in enumerate(b2.PARADIGMS_ORDER):
            if crossed[p]:
                M = item_mats[p]
                idx = rng.integers(0, M.shape[1], M.shape[1])
                cols_out[:, j] = M[:, idx].mean(axis=1)
            else:
                for i in range(len(models)):
                    arr = per_model_pools[p][i]
                    idx = rng.integers(0, arr.size, arr.size)
                    cols_out[i, j] = arr[idx].mean()
        return cols_out

    fam_ids_raw = sorted(set(fams_raw.values()))
    members_raw = {f: [i for i, m in enumerate(models) if fams_raw[m] == f]
                   for f in fam_ids_raw}

    results = {}
    modes = (("family_only_merged", fam_ids, members),
             ("family_only_raw", fam_ids_raw, members_raw),
             ("item_only", None, None),
             ("two_level_merged", fam_ids, members),
             ("two_level_raw", fam_ids_raw, members_raw))
    for mode, f_ids, f_members in modes:
        rng = np.random.default_rng(SEED)
        deltas, pc1s = [], []
        for _ in range(N_BOOT):
            mat = base_matrix if mode.startswith("family_only") else resampled_means(rng)
            if mode == "item_only":
                rows = mat
            else:
                idx = []
                for f in rng.choice(f_ids, len(f_ids), replace=True):
                    idx.extend(f_members[f])
                rows = mat[idx]
            d = delta_of(rows, labels)
            if np.isfinite(d):
                deltas.append(d)
                if mode.startswith("two_level"):
                    pc1s.append(pc1_of(rows))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        results[mode] = {
            "n_reps": len(deltas),
            "delta_mean": round(float(np.mean(deltas)), 4),
            "delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
        }
        if pc1s:
            plo, phi = np.percentile(pc1s, [2.5, 97.5])
            results[mode]["pc1_ci95"] = [round(float(plo), 4), round(float(phi), 4)]
        print(f"[{mode}] delta CI95 = {results[mode]['delta_ci95']} "
              f"(mean {results[mode]['delta_mean']}, n={results[mode]['n_reps']})",
              flush=True)

    out = {
        "seed": SEED, "n_boot": N_BOOT, "n_models": len(models),
        "n_families": len(fam_ids),
        "crossed_paradigms": {p: bool(crossed[p]) for p in b2.PARADIGMS_ORDER},
        "base_delta": round(float(delta_of(base_matrix, labels)), 4),
        "base_pc1": round(pc1_of(base_matrix), 4),
        "committed_family_ci_anchor": [-0.02, 0.09],
        "results": results,
        "note": ("two_level resamples families and items jointly; item resampling "
                 "is crossed (one item draw per paradigm applied to all models) "
                 "where the item-id sets are identical across models, otherwise "
                 "independent per model. Accuracy matrix (confirmatory analysis)."),
    }
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(f"[done] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
