#!/usr/bin/env python3
"""Construct-native robustness matrix (reviewer request, 2026-07-11).

Rebuilds the 55-model x 13-paradigm matrix with construct-native metrics for the
7 paradigms where raw accuracy confounds the construct with general answering
ability, keeping corrected accuracy for the other 6 (mixed-metric, disclosed):

  stroop / flanker  interference resistance = acc_incongruent - acc_congruent
  go_nogo           d' (log-linear); secondary: criterion, commission, omission
  n_back            d' over match/no-match turns; secondary: load slope n1->n3
  drm_false_memory  -(FA_critical_lure - FA_unrelated)   [higher = less false memory]
  confidence_calibration  1 - Brier on (confidence, CORRECTED correctness)
  post_decision_wagering  type-2 d' = z(P(bet|correct)) - z(P(bet|incorrect))

All metrics oriented higher = more of the construct ability.

Discipline mirrored from build_and_recompute.py:
  SELF-CHECK first: the v1 (corrected accuracy) matrix assembled here must match
  results/recompute_20260703/corrected_matrix.csv cell-by-cell before any v2
  number is trusted.

Outputs (this directory): construct_matrix.csv, construct_summary.json,
reliability.csv, v2_stats printed to stdout.

CPU-only; run via Slurm (batch partition), never the login node.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import norm

ROOT = __import__("os").environ.get("COGARENA_ROOT") or __import__("os").path.abspath(
    __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "results", "recompute_20260703"))
sys.path.insert(0, ROOT)
os.environ["COGARENA_ROOT"] = ROOT

import compute_b2_expanded as b2
from build_and_recompute import (WAGER_OVERLAY, corrected_row, static_paths,
                                 mt_base, pc1_share)

V2_PARADIGMS = {"stroop", "flanker", "go_nogo", "n_back", "drm_false_memory",
                "confidence_calibration", "post_decision_wagering"}
GONOGO = os.path.join(ROOT, "results", "gonogo_rerun_20260702")
RNG_SEED = 42
N_SPLITS = 100


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def loglin_dprime(n_hit, n_go, n_fa, n_nogo):
    """d' and criterion with the log-linear (Hautus 1995) correction."""
    h = (n_hit + 0.5) / (n_go + 1.0)
    f = (n_fa + 0.5) / (n_nogo + 1.0)
    zh, zf = norm.ppf(h), norm.ppf(f)
    return float(zh - zf), float(-(zh + zf) / 2.0)


def item_acc(it, overlay):
    return float(overlay.get(it["task_id"], b2.item_accuracy(it.get("score"))))


def brier_parts(pairs):
    """pairs = [(confidence, correct01), ...] -> 1-Brier, ECE(10 bins)."""
    if not pairs:
        return None, None
    conf = np.array([p[0] for p in pairs], float)
    corr = np.array([p[1] for p in pairs], float)
    brier = float(np.mean((conf - corr) ** 2))
    bins = np.clip((conf * 10).astype(int), 0, 9)
    ece = 0.0
    for bidx in range(10):
        m = bins == bidx
        if m.sum():
            ece += (m.sum() / len(conf)) * abs(conf[m].mean() - corr[m].mean())
    return 1.0 - brier, float(ece)


# --------------------------------------------------------------------------- #
# per-model construct metrics (each returns metric dict + reusable trial pools)
# --------------------------------------------------------------------------- #
def interference(items):
    """items: list of (condition, acc). Returns acc_incong - acc_cong."""
    by = collections.defaultdict(list)
    for cond, a in items:
        by[cond].append(a)
    if not by.get("congruent") or not by.get("incongruent"):
        return None
    return float(np.mean(by["incongruent"]) - np.mean(by["congruent"]))


def gonogo_metrics(trials):
    """trials: list of (condition, correct01). d', criterion, rates."""
    go = [c for cond, c in trials if cond == "go"]
    nogo = [c for cond, c in trials if cond == "nogo"]
    if not go or not nogo:
        return None
    n_hit = sum(go)                      # correct on GO = responded GO = hit
    n_fa = len(nogo) - sum(nogo)         # incorrect on NOGO = responded GO = FA
    d, c = loglin_dprime(n_hit, len(go), n_fa, len(nogo))
    return {"d_prime": d, "criterion": c,
            "commission": n_fa / len(nogo), "omission": (len(go) - n_hit) / len(go)}


def nback_code_response(resp):
    """3-way response coding mirroring the strict scorer's tolerance
    (scripts/run_eval.py:240-247): trailing-punctuation strip, exact or
    'exp + space' prefix, and the standalone-'no' guard for 'match'.
    Returns True (responded match), False (responded no match), or None
    (unparseable -> turn dropped from the SDT pool)."""
    act = str(resp or "").strip().lower()
    core = act.strip().strip('."\'!').strip()
    if core == "no match" or core.startswith("no match "):
        return False
    if (core == "match" or core.startswith("match ")) and "no" not in act:
        return True
    return None


NBACK_MIN_TRIALS = 10   # min parseable go AND no-match turns for a defined d'


def nback_pool(files):
    """-> list of (expected_is_match, coded_response_or_None, load, stored_correct)."""
    pool = []
    for f in files:
        m = re.search(r"_n(\d)_", os.path.basename(f))
        load = int(m.group(1)) if m else None
        d = json.load(open(f))
        for t in (d.get("score") or {}).get("turn_scores", []):
            exp = str(t.get("expected", "")).strip().lower()
            if exp not in ("match", "no match"):
                continue
            pool.append((exp == "match", nback_code_response(t.get("response")),
                         load, 1.0 if t.get("correct") else 0.0))
    return pool


def nback_metrics(pool):
    parse = [(e, r, load) for e, r, load, _ in pool if r is not None]
    n_go = sum(1 for e, _, _ in parse if e)
    n_hit = sum(1 for e, r, _ in parse if e and r)
    n_nogo = sum(1 for e, _, _ in parse if not e)
    n_fa = sum(1 for e, r, _ in parse if (not e) and r)
    if n_go < NBACK_MIN_TRIALS or n_nogo < NBACK_MIN_TRIALS:
        return None
    d, c = loglin_dprime(n_hit, n_go, n_fa, n_nogo)
    # load slope on per-load STRICT-scorer accuracy (stored correct, all turns)
    slope = None
    byload = collections.defaultdict(list)
    for _, _, load, corr01 in pool:
        if load:
            byload[load].append(corr01)
    if len(byload) >= 2:
        xs = sorted(byload)
        ys = [np.mean(byload[x]) for x in xs]
        slope = float(np.polyfit(xs, ys, 1)[0])
    parse_rate = len(parse) / len(pool) if pool else 0.0
    return {"d_prime": d, "criterion": c, "load_slope": slope,
            "parse_rate": round(parse_rate, 3)}


def t2_dprime(pairs):
    """pairs = [(did_bet01, correct01), ...] -> type-2 d'."""
    cor = [b for b, c in pairs if c >= 0.5]
    inc = [b for b, c in pairs if c < 0.5]
    if not cor or not inc:
        return None
    d, _ = loglin_dprime(sum(cor), len(cor), sum(inc), len(inc))
    return d


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #
def load_conf_cal_gold():
    from cogarena.generators.metacognition_gen import generate_mc_items
    items = generate_mc_items(seed=42, n_per_paradigm=50,
                              include_contamination_probes=False)
    return {it.task_id: it for it in items}


def collect_model(model, is_old, cc_gold, CC):
    """-> (v1_row, v2_row, pools) where pools hold per-item data for split-half."""
    v1, err = corrected_row(model, is_old)
    if v1 is None:
        return None, None, None, err
    set_name, det_path = static_paths(model, is_old)
    details = b2.load_details(det_path)
    ov_path = os.path.join(ROOT, "results", "rescore_20260702", "new_scores",
                           f"{set_name}__openai_{model}.json")
    overlay = json.load(open(ov_path)) if os.path.exists(ov_path) else {}

    pools = {}
    # stroop / flanker
    for par in ("stroop", "flanker"):
        pools[par] = [(it["score"].get("condition"), item_acc(it, overlay))
                      for it in details if it.get("paradigm") == par
                      and isinstance(it.get("score"), dict)]
    # drm per-item fields
    pools["drm_false_memory"] = [
        (float(it["score"].get("false_alarm_to_critical_lures", np.nan)),
         float(it["score"].get("false_alarm_to_unrelated", np.nan)),
         float(it["score"].get("d_prime", np.nan)))
        for it in details if it.get("paradigm") == "drm_false_memory"
        and isinstance(it.get("score"), dict)]
    # pdw
    pools["post_decision_wagering"] = [
        (float(it["score"].get("did_bet", 0.0)),
         float(WAGER_OVERLAY[model][it["task_id"]]) if WAGER_OVERLAY is not None
         else float(it["score"].get("is_correct", 0.0)))
        for it in details if it.get("paradigm") == "post_decision_wagering"
        and isinstance(it.get("score"), dict)]
    # conf_cal: re-score stored responses with the FIXED scorer
    cc_pairs = []
    text_dir = os.path.dirname(det_path)
    for f in glob.glob(os.path.join(text_dir, "metacognition",
                                    "confidence_calibration", "*.json")):
        d = json.load(open(f))
        t = d.get("task_id")
        if t in cc_gold:
            try:
                s = CC.score(cc_gold[t], d.get("response", "") or "")
                cc_pairs.append((float(s.get("confidence", 0.5)),
                                 float(s.get("is_correct", 0.0))))
            except Exception:
                pass
    pools["confidence_calibration"] = cc_pairs
    # go_nogo rerun trials
    gg = os.path.join(GONOGO, f"openai_{model}", "text", "details.json")
    gg_items = json.load(open(gg))
    pools["go_nogo"] = [(it["score"].get("condition"),
                         1.0 if it["score"].get("correct") else 0.0)
                        for it in gg_items if isinstance(it.get("score"), dict)]
    # n_back turn pool
    base = mt_base(model, is_old)[1]
    pools["n_back"] = nback_pool(sorted(glob.glob(
        os.path.join(base, "*", "n_back", "*.json"))))

    # ---- v2 row ----
    v2 = dict(v1)          # keep-acc paradigms inherit the corrected v1 cell
    v2["stroop"] = interference(pools["stroop"])
    v2["flanker"] = interference(pools["flanker"])
    gg_m = gonogo_metrics(pools["go_nogo"])
    v2["go_nogo"] = gg_m["d_prime"] if gg_m else None
    nb_m = nback_metrics(pools["n_back"])
    v2["n_back"] = nb_m["d_prime"] if nb_m else None
    drm = pools["drm_false_memory"]
    v2["drm_false_memory"] = (
        -(float(np.nanmean([x[0] for x in drm])) - float(np.nanmean([x[1] for x in drm])))
        if drm else None)
    ib, ece = brier_parts(pools["confidence_calibration"])
    v2["confidence_calibration"] = ib
    v2["post_decision_wagering"] = t2_dprime(pools["post_decision_wagering"])

    # sanitize: any non-finite v2 cell becomes an explicit None (imputed+reported)
    for p in V2_PARADIGMS:
        v = v2.get(p)
        v2[p] = float(v) if (v is not None and np.isfinite(v)) else None

    aux = {"go_nogo": gg_m, "n_back": nb_m, "conf_cal_ece": ece,
           "drm_mean_dprime": float(np.nanmean([x[2] for x in drm])) if drm else None}
    return v1, v2, pools, aux


# --------------------------------------------------------------------------- #
# split-half reliability
# --------------------------------------------------------------------------- #
def metric_from_half(par, half):
    if par in ("stroop", "flanker"):
        return interference(half)
    if par == "go_nogo":
        m = gonogo_metrics(half)
        return m["d_prime"] if m else None
    if par == "n_back":
        m = nback_metrics(half)
        return m["d_prime"] if m else None
    if par == "drm_false_memory":
        if not half:
            return None
        return -(float(np.nanmean([x[0] for x in half])) -
                 float(np.nanmean([x[1] for x in half])))
    if par == "confidence_calibration":
        return brier_parts(half)[0]
    if par == "post_decision_wagering":
        return t2_dprime(half)
    return None


def acc_from_half(par, half):
    """v1-style accuracy on the same split units, for a fair reliability baseline."""
    if par in ("stroop", "flanker"):
        return float(np.mean([a for _, a in half])) if half else None
    if par == "go_nogo":
        return float(np.mean([c for _, c in half])) if half else None
    if par == "n_back":
        return float(np.mean([c for _, _, _, c in half])) if half else None
    if par == "drm_false_memory":
        return None            # v1 cell is the item-accuracy field, not in this pool
    if par == "confidence_calibration":
        return float(np.mean([c for _, c in half])) if half else None
    if par == "post_decision_wagering":
        return float(np.mean([c for _, c in half])) if half else None
    return None


def split_half(pools_by_model, models, n_splits=N_SPLITS, seed=RNG_SEED):
    """Spearman-Brown split-half reliability of each v2 metric (and the accuracy
    computed on the same units), correlating model vectors across random halves."""
    rng = np.random.default_rng(seed)
    out = {}
    for par in sorted(V2_PARADIGMS):
        rs_v2, rs_acc = [], []
        for _ in range(n_splits):
            h1v, h2v, h1a, h2a = [], [], [], []
            for m in models:
                pool = pools_by_model[m].get(par) or []
                idx = rng.permutation(len(pool))
                a = [pool[i] for i in idx[: len(pool) // 2]]
                bhalf = [pool[i] for i in idx[len(pool) // 2:]]
                h1v.append(metric_from_half(par, a))
                h2v.append(metric_from_half(par, bhalf))
                h1a.append(acc_from_half(par, a))
                h2a.append(acc_from_half(par, bhalf))
            def sb(x, y):
                x, y = np.array(x, float), np.array(y, float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 10 or x[ok].std() == 0 or y[ok].std() == 0:
                    return None
                r = float(np.corrcoef(x[ok], y[ok])[0, 1])
                return 2 * r / (1 + r) if r > -0.5 else None
            r2 = sb(h1v, h2v)
            ra = sb(h1a, h2a)
            if r2 is not None:
                rs_v2.append(r2)
            if ra is not None:
                rs_acc.append(ra)
        # median across splits: robust to the SB transform's blow-up near r=-1
        out[par] = {"v2_splithalf_SB": round(float(np.median(rs_v2)), 3) if rs_v2 else None,
                    "acc_same_units_SB": round(float(np.median(rs_acc)), 3) if rs_acc else None,
                    "n_valid_splits_v2": len(rs_v2),
                    "n_valid_splits_acc": len(rs_acc)}
    return out


# --------------------------------------------------------------------------- #
# stats on a matrix
# --------------------------------------------------------------------------- #
def two_sided_p(corr_matrix, labels, obs, n_perm=50000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    perm = []
    for _ in range(n_perm):
        sh = rng.permutation(labels).tolist()
        w, c = [], []
        for i in range(n):
            for j in range(i + 1, n):
                (w if sh[i] == sh[j] else c).append(corr_matrix[i, j])
        if w and c:
            perm.append(np.mean(w) - np.mean(c))
    perm = np.array(perm)
    return (float(np.mean(perm >= obs)), float(np.mean(np.abs(perm) >= abs(obs))),
            [float(x) for x in np.percentile(perm, [2.5, 97.5])])


def family_boot_ci(rows, models, fams, n_boot=5000, seed=42):
    """Cluster bootstrap over model families for delta."""
    matrix = np.array([[rows[m].get(p, np.nan) for p in b2.PARADIGMS_ORDER] for m in models])
    labels = [b2.DOMAIN_MAP[p] for p in b2.PARADIGMS_ORDER]
    fam_ids = sorted(set(fams[m] for m in models))
    members = {f: [i for i, m in enumerate(models) if fams[m] == f] for f in fam_ids}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = []
        for f in rng.choice(fam_ids, len(fam_ids), replace=True):
            idx.extend(members[f])
        sub = matrix[idx]
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(sub.T)
        w, c = [], []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                r = corr[i, j]
                if np.isfinite(r):
                    (w if labels[i] == labels[j] else c).append(r)
        if w and c:
            deltas.append(np.mean(w) - np.mean(c))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def loo_paradigm(rows, models):
    """Leave-one-paradigm-out delta and one-sided p."""
    out = {}
    full = b2.PARADIGMS_ORDER
    matrix = np.array([[rows[m].get(p, np.nan) for p in full] for m in models])
    for k, drop in enumerate(full):
        keep = [i for i in range(len(full)) if i != k]
        labels = [b2.DOMAIN_MAP[full[i]] for i in keep]
        sub = matrix[:, keep]
        corr = np.corrcoef(sub.T)
        w, c = [], []
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                (w if labels[i] == labels[j] else c).append(corr[i, j])
        obs = float(np.mean(w) - np.mean(c))
        p1, p2, _ = two_sided_p(corr, labels, obs, n_perm=10000, seed=42)
        out[drop] = {"delta": round(obs, 4), "p1": round(p1, 4), "p2": round(p2, 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true",
                    help="3 models, 2 paradigm groups, no stats")
    args = ap.parse_args()

    from cogarena.dimensions.metacognition import ConfidenceCalibrationGenerator as CC
    cc_gold = load_conf_cal_gold()

    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    old_models = list(b2.OLD_MODELS.keys())
    new_models = sorted(new_meta.keys())
    models_all = old_models + new_models
    fams = {m: (b2.OLD_MODELS[m][1] if m in b2.OLD_MODELS
                else new_meta[m].get("family", m)) for m in models_all}
    if args.sanity:
        models_all = old_models[:2] + new_models[:1]

    v1_rows, v2_rows, pools_bm, aux_bm, dropped = {}, {}, {}, {}, []
    for m in models_all:
        got = collect_model(m, m in b2.OLD_MODELS, cc_gold, CC)
        if got[0] is None:
            dropped.append((m, got[3])); continue
        v1_rows[m], v2_rows[m], pools_bm[m], aux_bm[m] = got
    models = [m for m in models_all if m in v1_rows]
    print(f"[collect] {len(models)} models ok, dropped={dropped}", flush=True)

    if args.sanity:
        for m in models:
            print(m, json.dumps({p: (round(v2_rows[m][p], 4)
                                     if v2_rows[m].get(p) is not None else None)
                                 for p in sorted(V2_PARADIGMS)}))
            print("   aux:", json.dumps(aux_bm[m], default=str)[:220])
        missing = {m: [p for p in b2.PARADIGMS_ORDER if v2_rows[m].get(p) is None]
                   for m in models}
        print("[sanity] missing cells:", {m: v for m, v in missing.items() if v})
        return

    # ---- SELF-CHECK: v1 assembly must equal the canonical corrected matrix ----
    canon = {}
    with open(f"{ROOT}/results/recompute_20260703/corrected_matrix.csv") as fh:
        rd = csv.reader(fh)
        header = next(rd)[1:]
        for row in rd:
            canon[row[0]] = {p: float(x) for p, x in zip(header, row[1:])}
    not_in_canon = [m for m in models if m not in canon]
    if not_in_canon:
        sys.exit(f"SELF-CHECK IMPOSSIBLE: models absent from corrected_matrix.csv: "
                 f"{not_in_canon}")
    diffs = []
    for m in models:
        for p in b2.PARADIGMS_ORDER:
            a, bb = canon[m].get(p), v1_rows[m].get(p)
            if a is not None and bb is not None and abs(a - bb) > 5e-4:
                diffs.append((m, p, a, bb))
    if diffs:
        print("SELF-CHECK FAILED (v1 assembly != canonical corrected matrix):")
        for d in diffs[:20]:
            print("  ", d)
        sys.exit(1)
    print(f"[self-check] v1 assembly matches corrected_matrix.csv "
          f"({len(models)} models x 13 paradigms)", flush=True)

    # ---- v2 completeness ----
    for m in models:
        miss = [p for p in b2.PARADIGMS_ORDER if v2_rows[m].get(p) is None]
        if miss:
            print(f"[warn] {m} missing v2 cells: {miss}", flush=True)

    # ---- v1 <-> v2 agreement per swapped paradigm (BEFORE imputation) ----
    agree = {}
    for p in sorted(V2_PARADIGMS):
        a = np.array([v1_rows[m].get(p, np.nan) for m in models], float)
        bb = np.array([v2_rows[m][p] if v2_rows[m].get(p) is not None else np.nan
                       for m in models], float)
        ok = np.isfinite(a) & np.isfinite(bb)
        agree[p] = round(float(np.corrcoef(a[ok], bb[ok])[0, 1]), 3) if ok.sum() > 5 else None

    # ---- mean-impute missing v2 cells (degenerate models), fully reported ----
    imputed = []
    for p in b2.PARADIGMS_ORDER:
        vals = [v2_rows[m][p] for m in models if v2_rows[m].get(p) is not None]
        mu = float(np.mean(vals))
        for m in models:
            if v2_rows[m].get(p) is None:
                v2_rows[m][p] = mu
                imputed.append([m, p])
    print(f"[impute] {len(imputed)} v2 cells mean-imputed: {imputed[:12]}", flush=True)

    # ---- stats ----
    labels = [b2.DOMAIN_MAP[p] for p in b2.PARADIGMS_ORDER]
    res = {}
    for tag, rows in (("v1_accuracy", v1_rows), ("v2_construct", v2_rows)):
        stats, matrix = b2.run_b2(rows, models)
        corr = np.corrcoef(np.array(
            [[rows[m].get(p, np.nan) for p in b2.PARADIGMS_ORDER] for m in models]).T)
        w_obs, c_obs = [], []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                (w_obs if labels[i] == labels[j] else c_obs).append(corr[i, j])
        obs = float(np.mean(w_obs) - np.mean(c_obs))
        p1, p2, perm_ci = two_sided_p(corr, labels, obs, n_perm=50000, seed=42)
        pc1, ev = pc1_share(rows, models)
        res[tag] = {
            "run_b2": stats, "delta_exact": round(obs, 6),
            "p1_50k": round(p1, 4), "p2_50k": round(p2, 4),
            "perm_null_ci": [round(x, 4) for x in perm_ci],
            "family_boot_ci": family_boot_ci(rows, models, fams),
            "pc1_share": round(pc1, 4), "eigen_top3": [round(x, 3) for x in ev],
            "loo": loo_paradigm(rows, models),
        }
        print(f"[{tag}] delta={stats['delta']} p1={p1:.4f} p2={p2:.4f} "
              f"famCI={res[tag]['family_boot_ci']} PC1={pc1:.3f}", flush=True)

    # reliability
    rel = split_half(pools_bm, models)
    print("[reliability]", json.dumps(rel, indent=1), flush=True)

    # ---- outputs ----
    with open(os.path.join(HERE, "construct_matrix.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model"] + b2.PARADIGMS_ORDER)
        for m in models:
            w.writerow([m] + [round(v2_rows[m][p], 5) if v2_rows[m].get(p) is not None
                              else "" for p in b2.PARADIGMS_ORDER])
    with open(os.path.join(HERE, "reliability.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["paradigm", "v2_splithalf_SB", "acc_same_units_SB",
                    "n_valid_splits_v2", "n_valid_splits_acc"])
        for p, v in rel.items():
            w.writerow([p, v["v2_splithalf_SB"], v["acc_same_units_SB"],
                        v["n_valid_splits_v2"], v["n_valid_splits_acc"]])
    out = {
        "spec": {p: ("construct" if p in V2_PARADIGMS else "accuracy (unchanged)")
                 for p in b2.PARADIGMS_ORDER},
        "n_models": len(models), "dropped": dropped, "imputed_cells": imputed,
        "results": res, "v1_v2_agreement_r": agree, "reliability": rel,
        "aux_means": {
            "go_nogo_criterion": round(float(np.nanmean(
                [aux_bm[m]["go_nogo"]["criterion"] for m in models
                 if aux_bm[m]["go_nogo"]])), 3),
            "n_back_load_slope": round(float(np.nanmean(
                [aux_bm[m]["n_back"]["load_slope"] for m in models
                 if aux_bm[m]["n_back"] and aux_bm[m]["n_back"]["load_slope"] is not None])), 4),
            "conf_cal_ece": round(float(np.nanmean(
                [aux_bm[m]["conf_cal_ece"] for m in models
                 if aux_bm[m]["conf_cal_ece"] is not None])), 3),
        },
    }
    json.dump(out, open(os.path.join(HERE, "construct_summary.json"), "w"),
              indent=1, default=str)
    print("[done] wrote construct_matrix.csv / construct_summary.json / reliability.csv",
          flush=True)


if __name__ == "__main__":
    main()
