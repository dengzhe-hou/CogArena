#!/usr/bin/env python3
"""Step-4 artifact generator: every paper-cited quantity from CORRECTED data.

Reads the corrected per-item layer (archived details.json + rescore overlays +
go_nogo rerun + multiturn overlays) and produces step4_artifacts.json with:
  paradigm/group means (20/55), difficulty & condition splits, FB order split,
  n-back load split, scaling correlations + family mixed model, split-half
  reliability (with/without EPITOME), signature tests (stroop/flanker
  congruency, FB order, n-back load) with BH, go_nogo GO/NO-GO rates,
  predictive validity vs published benchmarks, restricted-range PC1 variants,
  and the specific model claims quoted in the paper.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.environ.get("COGARENA_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "recompute_20260703")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ["COGARENA_ROOT"] = ROOT
import compute_b2_expanded as b2

RESCORE = os.path.join(ROOT, "results", "rescore_20260702", "new_scores")
# Final SM overlay adapter: see build_and_recompute.py. When set, every
# source_monitoring item takes the 55x50 corrected+rerun overlay.
SM_OVERLAY = (json.load(open(os.environ["COGARENA_SM_OVERLAY"]))
              if os.environ.get("COGARENA_SM_OVERLAY") else None)
WAGER_OVERLAY = (json.load(open(os.environ["COGARENA_WAGER_OVERLAY"]))
                 if os.environ.get("COGARENA_WAGER_OVERLAY") else None)
GONOGO = os.path.join(ROOT, "results", "gonogo_rerun_20260702")
GROUP = b2.DOMAIN_MAP


def static_set(model, is_old):
    return "full_eval_20260526_2208" if is_old else "full_eval_expansion"


def corrected_items(model, is_old):
    """task_id -> (paradigm, corrected accuracy, archived score dict)."""
    s = static_set(model, is_old)
    det = b2.load_details(f"{ROOT}/results/{s}/openai_{model}/text/details.json")
    ov_p = os.path.join(RESCORE, f"{s}__openai_{model}.json")
    ov = json.load(open(ov_p)) if os.path.exists(ov_p) else {}
    out = {}
    for r in det:
        p = r.get("paradigm")
        if p not in b2.STATIC_PARADIGMS:
            continue
        if p == "source_monitoring" and SM_OVERLAY is not None and model in SM_OVERLAY:
            acc = SM_OVERLAY[model][r["task_id"]]
        elif p == "post_decision_wagering" and WAGER_OVERLAY is not None:
            acc = WAGER_OVERLAY[model][r["task_id"]]
        else:
            acc = ov.get(r["task_id"], b2.item_accuracy(r.get("score")))
        out[r["task_id"]] = (p, float(acc), r.get("score") or {})
    # go_nogo replaced wholesale by the fixed-prompt rerun
    gg = json.load(open(f"{GONOGO}/openai_{model}/text/details.json"))
    for r in gg:
        out[r["task_id"]] = ("go_nogo", b2.item_accuracy(r.get("score")), r.get("score") or {})
    return out


_PRIMARY = os.environ.get("COGARENA_PRIMARY_MATRIX")
_PRIMARY_CONFIG = os.environ.get("COGARENA_PRIMARY_CONFIG", "aplus_strict")
_APLUS = os.path.join(ROOT, "results", "reanalysis", "aplus_20260718")


def _aplus_episode_scores():
    """Under the primary regime, OSpan and CVLT episode scores come from the
    frozen A+ artifacts (selected OSpan recall / fixed CVLT estimator)
    instead of the production turn-average overlays."""
    if not _PRIMARY:
        return None
    if _PRIMARY_CONFIG not in ("aplus_strict", "aplus_adjudicated"):
        raise RuntimeError(f"unsupported COGARENA_PRIMARY_CONFIG={_PRIMARY_CONFIG!r}")
    ospan_field = "strict" if _PRIMARY_CONFIG == "aplus_strict" else "adj_strict"
    osp = json.load(open(os.path.join(_APLUS, "ospan_recall_scores.json")))
    cvl = json.load(open(os.path.join(_APLUS, "cvlt_fixed_scores.json")))
    out = {}
    for m, eps in osp.items():
        for e in eps:
            if ospan_field not in e:
                raise RuntimeError(f"missing {ospan_field} for OSpan {m}/{e['task_id']}")
            out[(m, e["task_id"])] = float(e[ospan_field])
    for m, eps in cvl.items():
        for e in eps:
            out[(m, e["task_id"])] = float(e["fixed_accuracy"])
    return out


_EPISODE_PRIMARY = None


def mt_files(model, is_old):
    """[(paradigm, difficulty, corrected accuracy)] from multiturn files."""
    global _EPISODE_PRIMARY
    if _PRIMARY and _EPISODE_PRIMARY is None:
        _EPISODE_PRIMARY = _aplus_episode_scores()
    if is_old:
        mt_set, base = "multiturn_eval_v3", f"{ROOT}/results/multiturn_eval_v3/openai_{model}"
    else:
        mt_set, base = "multiturn_expansion", f"{ROOT}/results/multiturn_expansion/openai_{model}/text"
    ov_p = os.path.join(RESCORE, f"{mt_set}__openai_{model}__multiturn.json")
    ov = json.load(open(ov_p)) if os.path.exists(ov_p) else {}
    set_root = f"{ROOT}/results/{mt_set}"
    rows = []
    for par in sorted(b2.MT_PARADIGMS):
        for f in sorted(glob.glob(os.path.join(base, "*", par, "*.json"))):
            d = json.load(open(f))
            sc = d.get("score") or {}
            if "accuracy" not in sc:
                continue
            key = (model, d.get("task_id"))
            if _EPISODE_PRIMARY is not None and par in ("operation_span", "cvlt_word_list"):
                acc = _EPISODE_PRIMARY[key]
            else:
                rel = os.path.relpath(f, set_root)
                acc = float(ov.get(rel, sc["accuracy"]))
            rows.append((par, d.get("difficulty", "?"), acc))
    return rows


def bh(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order)):
        k = n - rank
        prev = min(prev, pvals[i] * n / k)
        adj[i] = prev
    return adj.tolist()


def main():
    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    old_models = list(b2.OLD_MODELS.keys())
    new_models = sorted(new_meta.keys())
    all_models = old_models + new_models
    sizes = {m: b2.OLD_MODELS[m][0] for m in old_models}
    sizes.update({m: new_meta[m]["size_b"] for m in new_models})
    fams = {m: b2.OLD_MODELS[m][1] for m in old_models}
    fams.update({m: new_meta[m].get("family", m) for m in new_models})

    # battery parameters (FB order, difficulty) keyed by task_id
    from cogarena.cli import _collect_items
    batt = {it.task_id: it for it in _collect_items(50, 42, None)}

    # ---- corrected matrix (from job 5571 rebuild) ---------------------------
    rows = list(csv.reader(open(os.environ.get("COGARENA_PRIMARY_MATRIX")
                                or os.path.join(OUT, "corrected_matrix.csv"))))
    hdr = rows[0][1:]
    mat = {r[0]: {p: float(v) for p, v in zip(hdr, r[1:])} for r in rows[1:]}

    art = {}

    # ---- 1. paradigm + group means ------------------------------------------
    for tag, pool in (("20", old_models), ("55", all_models)):
        art[f"paradigm_means_{tag}"] = {
            p: round(float(np.mean([mat[m][p] for m in pool])), 4) for p in hdr}
        gm = defaultdict(list)
        for m in pool:
            per_g = defaultdict(list)
            for p in hdr:
                per_g[GROUP[p]].append(mat[m][p])
            for g, v in per_g.items():
                gm[g].append(np.mean(v))
        art[f"group_means_{tag}"] = {g: round(float(np.mean(v)), 4) for g, v in gm.items()}

    # ---- 2. per-item derived splits (20-model set) --------------------------
    per_model_items = {m: corrected_items(m, True) for m in old_models}
    per_model_mt = {m: mt_files(m, True) for m in old_models}

    # FB by order
    fb_o = {1: [], 2: []}
    for m in old_models:
        acc_o = {1: [], 2: []}
        for tid, (p, acc, _) in per_model_items[m].items():
            if p == "false_belief" and tid in batt:
                acc_o[int(batt[tid].metadata.parameters["order"])].append(acc)
        for o in (1, 2):
            if acc_o[o]:
                fb_o[o].append(np.mean(acc_o[o]))
    art["false_belief_by_order_20"] = {f"order{o}": round(float(np.mean(v)), 4)
                                       for o, v in fb_o.items()}

    # n-back by difficulty (easy=1-back .. hard=3-back)
    nb = defaultdict(list)
    for m in old_models:
        acc_d = defaultdict(list)
        for par, diff, acc in per_model_mt[m]:
            if par == "n_back":
                acc_d[diff].append(acc)
        for d, v in acc_d.items():
            nb[d].append(np.mean(v))
    art["n_back_by_difficulty_20"] = {d: round(float(np.mean(v)), 4) for d, v in nb.items()}

    # paradigm x difficulty means (static, 20)
    pd = defaultdict(lambda: defaultdict(list))
    for m in old_models:
        acc_pd = defaultdict(lambda: defaultdict(list))
        for tid, (p, acc, _) in per_model_items[m].items():
            if tid in batt:
                acc_pd[p][batt[tid].metadata.difficulty].append(acc)
        for p, dd in acc_pd.items():
            for d, v in dd.items():
                pd[p][d].append(np.mean(v))
    art["paradigm_by_difficulty_20"] = {
        p: {d: round(float(np.mean(v)), 4) for d, v in dd.items()} for p, dd in pd.items()}

    # go_nogo GO / NO-GO rates (55 models, rerun)
    go, nogo = [], []
    for m in all_models:
        g_acc, n_acc = [], []
        for r in json.load(open(f"{GONOGO}/openai_{m}/text/details.json")):
            exp = str(batt[r["task_id"]].expected_response) if r["task_id"] in batt else "?"
            (g_acc if exp == "GO" else n_acc).append(b2.item_accuracy(r.get("score")))
        go.append(np.mean(g_acc)); nogo.append(np.mean(n_acc))
    art["gonogo_rates_55"] = {"go": round(float(np.mean(go)), 4),
                              "nogo": round(float(np.mean(nogo)), 4)}

    # ---- 3. scaling: r vs log10(size), 55 models + family mixed model -------
    logsize = np.array([np.log10(sizes[m]) for m in all_models])
    scaling = {}
    for p in hdr:
        y = np.array([mat[m][p] for m in all_models])
        r, pv = stats.pearsonr(logsize, y)
        scaling[p] = {"r": round(float(r), 3), "p": round(float(pv), 4)}
    art["scaling_r_55"] = scaling
    y_overall = np.array([np.mean([mat[m][p] for p in hdr]) for m in all_models])
    r, pv = stats.pearsonr(logsize, y_overall)
    art["scaling_overall_55"] = {"r": round(float(r), 3), "p": round(float(pv), 4)}
    try:
        import statsmodels.formula.api as smf
        import pandas as pd_
        mm = []
        for p in hdr:
            df = pd_.DataFrame({"acc": [mat[m][p] for m in all_models],
                                "ls": logsize, "fam": [fams[m] for m in all_models]})
            md = smf.mixedlm("acc ~ ls", df, groups=df["fam"]).fit(reml=False)
            mm.append({"paradigm": p, "slope": round(float(md.params["ls"]), 4),
                       "p": round(float(md.pvalues["ls"]), 4)})
        art["scaling_mixedlm_55"] = mm
        art["scaling_mixedlm_n_sig"] = sum(1 for x in mm if x["p"] < 0.05)
    except Exception as e:
        art["scaling_mixedlm_55"] = f"failed: {e}"

    # ---- 4. split-half reliability (20-model, corrected items) --------------
    def split_half(paradigms):
        rng = np.random.default_rng(42)
        out = {}
        for p in paradigms:
            tids = sorted(t for t, it in batt.items() if it.metadata.paradigm == p)
            if p == "go_nogo":
                pass  # rerun items share the same battery ids
            accs = {m: {t: per_model_items[m][t][1] for t in tids if t in per_model_items[m]}
                    for m in old_models}
            tids = [t for t in tids if all(t in accs[m] for m in old_models)]
            if len(tids) < 4:
                continue
            rs = []
            for _ in range(2000):
                perm = rng.permutation(len(tids))
                h1 = [tids[i] for i in perm[: len(tids) // 2]]
                h2 = [tids[i] for i in perm[len(tids) // 2:]]
                a = [np.mean([accs[m][t] for t in h1]) for m in old_models]
                bvals = [np.mean([accs[m][t] for t in h2]) for m in old_models]
                if np.std(a) > 0 and np.std(bvals) > 0:
                    r = np.corrcoef(a, bvals)[0, 1]
                    rs.append(2 * r / (1 + r))
            out[p] = round(float(np.mean(rs)), 3)
        return out

    static_pars = [p for p in hdr if p in b2.STATIC_PARADIGMS]
    sh = split_half(static_pars)
    art["split_half_corrected_20"] = sh
    vals_no_ep = [v for p, v in sh.items() if p != "epitome_tom"]
    art["split_half_mean_excl_epitome"] = round(float(np.mean(vals_no_ep)), 3)
    art["split_half_mean_incl_epitome_CAVEAT_synthetic"] = round(float(np.mean(list(sh.values()))), 3)

    # ---- 5. signatures with BH (20-model, corrected correctness) ------------
    sig = {}
    # stroop/flanker congruency (condition from archived score dicts)
    for par in ("stroop", "flanker"):
        cong, incong = [], []
        for m in old_models:
            c, i_ = [], []
            for tid, (p, acc, scd) in per_model_items[m].items():
                if p != par:
                    continue
                if scd.get("condition") == "congruent":
                    c.append(acc)
                elif scd.get("condition") == "incongruent":
                    i_.append(acc)
            if c and i_:
                cong.append(np.mean(c)); incong.append(np.mean(i_))
        t, pv = stats.wilcoxon(cong, incong, alternative="greater")
        sig[par] = {"easy": round(float(np.mean(cong)), 4), "hard": round(float(np.mean(incong)), 4),
                    "p": float(pv), "direction": "congruent>incongruent"}
    # FB order1 > order2
    o1, o2 = [], []
    for m in old_models:
        a = {1: [], 2: []}
        for tid, (p, acc, _) in per_model_items[m].items():
            if p == "false_belief" and tid in batt:
                a[int(batt[tid].metadata.parameters["order"])].append(acc)
        if a[1] and a[2]:
            o1.append(np.mean(a[1])); o2.append(np.mean(a[2]))
    t, pv = stats.wilcoxon(o1, o2, alternative="greater")
    sig["false_belief"] = {"easy": round(float(np.mean(o1)), 4), "hard": round(float(np.mean(o2)), 4),
                           "p": float(pv), "direction": "order1>order2"}
    # n-back load (easy > hard)
    e_, h_ = [], []
    for m in old_models:
        a = defaultdict(list)
        for par, diff, acc in per_model_mt[m]:
            if par == "n_back":
                a[diff].append(acc)
        if a.get("easy") and a.get("hard"):
            e_.append(np.mean(a["easy"])); h_.append(np.mean(a["hard"]))
    t, pv = stats.wilcoxon(e_, h_, alternative="greater")
    sig["n_back"] = {"easy": round(float(np.mean(e_)), 4), "hard": round(float(np.mean(h_)), 4),
                     "p": float(pv), "direction": "1back>3back"}
    ps = [sig[k]["p"] for k in sig]
    adj = bh(ps)
    for k, a in zip(sig, adj):
        sig[k]["p_bh"] = round(float(a), 5)
        sig[k]["p"] = round(sig[k]["p"], 5)
    art["signatures_20"] = sig

    # ---- 6. predictive validity (grouping scores vs published benchmarks) ---
    try:
        bench = {}
        with open(f"{ROOT}/data/published_benchmarks.csv") as f:
            rd = csv.DictReader(f)
            cols = [c for c in rd.fieldnames if c != "model"]
            for row in rd:
                bench[row["model"]] = {c: (float(row[c]) if row.get(c, "N/A") not in ("N/A", "", None) else None)
                                       for c in cols}
        pv_out = {}
        for bm in cols:
            ms = [m for m in all_models if m in bench and bench[m].get(bm) is not None]
            if len(ms) < 8:
                continue
            overall = np.array([np.mean([mat[m][p] for p in hdr]) for m in ms])
            yb = np.array([bench[m][bm] for m in ms])
            ls = np.array([np.log10(sizes[m]) for m in ms])
            r, p_ = stats.pearsonr(overall, yb)
            # partial r controlling log size
            rx = overall - np.poly1d(np.polyfit(ls, overall, 1))(ls)
            ry = yb - np.poly1d(np.polyfit(ls, yb, 1))(ls)
            pr, pp = stats.pearsonr(rx, ry)
            pv_out[bm] = {"n": len(ms), "r": round(float(r), 3), "p": round(float(p_), 4),
                          "partial_r_given_size": round(float(pr), 3),
                          "partial_p": round(float(pp), 4)}
        art["predictive_validity_overall"] = pv_out
    except Exception as e:
        art["predictive_validity_overall"] = f"failed: {e}"

    # ---- 7. restricted-range PC1 variants (corrected 55) --------------------
    M = np.array([[mat[m][p] for p in hdr] for m in all_models])
    def pc1(Msub):
        ev = np.linalg.eigvalsh(np.corrcoef(Msub.T))[::-1]
        return round(float(ev[0] / ev.sum()), 4)
    rr = {"full": pc1(M)}
    sds = M.std(axis=0)
    means = M.mean(axis=0)
    hi_i = [i for i in range(len(hdr)) if means[i] < 0.9]
    rr["excl_mean_ge_0.9"] = pc1(M[:, hi_i]) if len(hi_i) >= 3 else None
    lo_var = [i for i in range(len(hdr)) if sds[i] >= np.median(sds)]
    rr["high_variance_half"] = pc1(M[:, lo_var])
    art["restricted_range_pc1_variants"] = rr
    art["paradigm_sd_55"] = {p: round(float(s), 4) for p, s in zip(hdr, sds)}

    # ---- 8. model-specific claims -------------------------------------------
    def cell(m, p):
        return round(mat[m][p], 4) if m in mat else None
    art["model_claims"] = {
        "phi3:14b": {p: cell("phi3:14b", p) for p in ("n_back", "operation_span", "digit_span")},
        "deepseek-r1:7b": {p: cell("deepseek-r1:7b", p) for p in ("false_belief", "source_monitoring", "epitome_tom")},
        "deepseek-r1:14b": {p: cell("deepseek-r1:14b", p) for p in ("false_belief", "source_monitoring")},
        "llama3.2:1b": {p: cell("llama3.2:1b", p) for p in ("false_belief", "n_back")},
        "mixtral:8x7b": {p: cell("mixtral:8x7b", p) for p in ("digit_span", "operation_span")},
    }

    json.dump(art, open(os.path.join(OUT, "step4_artifacts.json"), "w"), indent=1)
    print(json.dumps({k: art[k] for k in ("paradigm_means_20", "group_means_20",
          "false_belief_by_order_20", "n_back_by_difficulty_20", "gonogo_rates_55",
          "split_half_mean_excl_epitome", "signatures_20", "scaling_overall_55")}, indent=1))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
