#!/usr/bin/env python3
"""Step-4b: remaining paper numbers (tables + prose) from corrected data.

Produces step4b_artifacts.json + ready-to-paste LaTeX rows for tab:results and
tab:multiturn. Complements step4_numbers.py.
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
# Final SM overlay adapter: see build_and_recompute.py.
SM_OVERLAY = (json.load(open(os.environ["COGARENA_SM_OVERLAY"]))
              if os.environ.get("COGARENA_SM_OVERLAY") else None)
WAGER_OVERLAY = (json.load(open(os.environ["COGARENA_WAGER_OVERLAY"]))
                 if os.environ.get("COGARENA_WAGER_OVERLAY") else None)
GONOGO = os.path.join(ROOT, "results", "gonogo_rerun_20260702")

STATIC_COLS = ["digit_span", "stroop", "flanker", "go_nogo", "drm_false_memory",
               "source_monitoring", "false_belief", "epitome_tom",
               "confidence_calibration", "post_decision_wagering"]
MT_COLS = ["n_back", "operation_span", "cvlt_word_list"]


def corrected_items(model, is_old, set_name=None):
    s = set_name or ("full_eval_20260526_2208" if is_old else "full_eval_expansion")
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
        elif (p == "post_decision_wagering" and WAGER_OVERLAY is not None
              and s in ("full_eval_20260526_2208", "full_eval_expansion")):
            # The overlay is for the primary quantisation only.  Never leak it
            # into the fp16 historical sensitivity arm, whose model IDs overlap.
            acc = WAGER_OVERLAY[model][r["task_id"]]
        else:
            acc = ov.get(r["task_id"], b2.item_accuracy(r.get("score")))
        out[r["task_id"]] = (p, float(acc), r.get("score") or {})
    gg_p = f"{GONOGO}/openai_{model}/text/details.json"
    if os.path.exists(gg_p):
        for r in json.load(open(gg_p)):
            out[r["task_id"]] = ("go_nogo", b2.item_accuracy(r.get("score")), r.get("score") or {})
    return out


def main():
    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    old_models = list(b2.OLD_MODELS.keys())
    new_models = sorted(new_meta.keys())
    all_models = old_models + new_models
    sizes = {m: b2.OLD_MODELS[m][0] for m in old_models}
    sizes.update({m: new_meta[m]["size_b"] for m in new_models})

    from cogarena.cli import _collect_items
    batt = {it.task_id: it for it in _collect_items(50, 42, None)}

    rows = list(csv.reader(open(os.environ.get("COGARENA_PRIMARY_MATRIX")
                                or os.path.join(OUT, "corrected_matrix.csv"))))
    hdr = rows[0][1:]
    mat = {r[0]: {p: float(v) for p, v in zip(hdr, r[1:])} for r in rows[1:]}

    art = {}

    # ---- 1. table cells (percent, 20 models) --------------------------------
    art["tab_results_cells"] = {
        m: {p: round(mat[m][p] * 100) for p in STATIC_COLS} for m in old_models}
    art["tab_multiturn_cells"] = {
        m: {p: round(mat[m][p] * 100) for p in MT_COLS} for m in old_models}
    # column means / best markers
    art["tab_results_col_means"] = {p: round(np.mean([mat[m][p] for m in old_models]) * 100, 1)
                                    for p in STATIC_COLS}
    art["tab_multiturn_col_means"] = {p: round(np.mean([mat[m][p] for m in old_models]) * 100, 1)
                                      for p in MT_COLS}
    art["tab_results_col_best"] = {p: max(old_models, key=lambda m: mat[m][p]) for p in STATIC_COLS}
    art["paradigm_ranges_20"] = {p: [round(min(mat[m][p] for m in old_models) * 100),
                                     round(max(mat[m][p] for m in old_models) * 100)]
                                 for p in STATIC_COLS + MT_COLS}

    # ---- 2. scaling r on 20 models + mixedlm --------------------------------
    ls20 = np.array([np.log10(sizes[m]) for m in old_models])
    sc20 = {}
    for p in hdr:
        y = np.array([mat[m][p] for m in old_models])
        r, pv = stats.pearsonr(ls20, y)
        sc20[p] = {"r": round(float(r), 2), "p": round(float(pv), 4)}
    art["scaling_r_20"] = sc20
    try:
        import statsmodels.formula.api as smf
        import pandas as pd_
        mm20 = {}
        for p in hdr:
            df = pd_.DataFrame({"acc": [mat[m][p] for m in old_models], "ls": ls20,
                                "fam": [b2.OLD_MODELS[m][1] for m in old_models]})
            md = smf.mixedlm("acc ~ ls", df, groups=df["fam"]).fit(reml=False)
            mm20[p] = {"slope": round(float(md.params["ls"]), 3),
                       "p": round(float(md.pvalues["ls"]), 4)}
        art["scaling_mixedlm_20"] = mm20
    except Exception as e:
        art["scaling_mixedlm_20"] = f"failed: {e}"

    # ---- 3. manifold stats (55) ----------------------------------------------
    M = np.array([[mat[m][p] for p in hdr] for m in all_models])
    corr = np.corrcoef(M.T)
    iu = np.triu_indices(len(hdr), 1)
    art["pct_positive_pairs_55"] = round(float(np.mean(corr[iu] > 0)) * 100, 1)
    pc = np.linalg.eigh(np.corrcoef(M.T))
    v1 = pc[1][:, -1]
    z = (M - M.mean(0)) / M.std(0, ddof=1)
    pc1_scores = z @ v1
    r, _ = stats.pearsonr(pc1_scores, M.mean(1))
    art["pc1_vs_mean_acc_r_55"] = round(abs(float(r)), 2)

    labels = [b2.DOMAIN_MAP[p] for p in hdr]
    def delta_p(Mx, n_perm=5000, seed=42):
        c = np.corrcoef(Mx.T)
        prs = [(i, j) for i in range(len(hdr)) for j in range(i + 1, len(hdr))]
        w = [c[i, j] for i, j in prs if labels[i] == labels[j]]
        cr = [c[i, j] for i, j in prs if labels[i] != labels[j]]
        obs = float(np.mean(w) - np.mean(cr))
        rng = np.random.default_rng(seed)
        perm = []
        for _ in range(n_perm):
            sh = rng.permutation(labels).tolist()
            ww = [c[i, j] for i, j in prs if sh[i] == sh[j]]
            cc = [c[i, j] for i, j in prs if sh[i] != sh[j]]
            if ww and cc:
                perm.append(np.mean(ww) - np.mean(cc))
        return round(obs, 3), round(float(np.mean([d >= obs for d in perm])), 3)

    # PC1-removal residual delta (the known-artifact defense analysis)
    Z_res = z - np.outer(pc1_scores, v1)
    art["pc1_removal_delta_p"] = delta_p(Z_res)

    # ---- 4. EPITOME sub-capacity means (35 expansion, corrected parser) -----
    sub = defaultdict(list)
    ge90 = 0; n_ge7b = 0
    for m in new_models:
        items = corrected_items(m, False)
        per = defaultdict(list)
        for tid, (p, acc, _) in items.items():
            if p == "epitome_tom" and tid in batt:
                per[batt[tid].metadata.parameters["sub_capacity"]].append(acc)
        for k, v in per.items():
            sub[k].append(np.mean(v))
        if sizes[m] >= 7:
            n_ge7b += 1
            if per.get("desire") and np.mean(per["desire"]) >= 0.9:
                ge90 += 1
    art["epitome_subcapacity_35"] = {k: round(float(np.mean(v)) * 100, 1) for k, v in sub.items()}
    art["epitome_desire_ge90_of_ge7b"] = f"{ge90}/{n_ge7b}"

    # ---- 5. srcmon difficulty sweep, qwen2.5:7b ------------------------------
    it7 = corrected_items("qwen2.5:7b", True)
    sweep = defaultdict(list)
    for tid, (p, acc, _) in it7.items():
        if p == "source_monitoring" and tid in batt:
            sweep[batt[tid].metadata.difficulty].append(acc)
    art["srcmon_sweep_qwen7b"] = {d: round(float(np.mean(v)) * 100) for d, v in sweep.items()}

    # ---- 6. per-model signature replication (20 models, corrected) ----------
    per_items = {m: corrected_items(m, True) for m in old_models}

    def repl_static_condition(par, cond_key, easy_val, hard_val):
        n_ok = 0
        for m in old_models:
            e, h = [], []
            for tid, (p, acc, scd) in per_items[m].items():
                if p != par:
                    continue
                c = scd.get(cond_key)
                if c == easy_val:
                    e.append(acc)
                elif c == hard_val:
                    h.append(acc)
            if e and h and np.mean(e) > np.mean(h):
                n_ok += 1
        return n_ok

    sig_counts = {
        "stroop": repl_static_condition("stroop", "condition", "congruent", "incongruent"),
        "flanker": repl_static_condition("flanker", "condition", "congruent", "incongruent"),
    }
    # FB order per model
    n_ok = 0
    for m in old_models:
        a = {1: [], 2: []}
        for tid, (p, acc, _) in per_items[m].items():
            if p == "false_belief" and tid in batt:
                a[int(batt[tid].metadata.parameters["order"])].append(acc)
        if a[1] and a[2] and np.mean(a[1]) > np.mean(a[2]):
            n_ok += 1
    sig_counts["false_belief"] = n_ok
    # n-back load per model (multiturn, corrected via overlay)
    n_ok = 0
    for m in old_models:
        base = f"{ROOT}/results/multiturn_eval_v3/openai_{m}"
        ov_p = os.path.join(RESCORE, f"multiturn_eval_v3__openai_{m}__multiturn.json")
        ov = json.load(open(ov_p)) if os.path.exists(ov_p) else {}
        acc_d = defaultdict(list)
        for f in glob.glob(os.path.join(base, "*", "n_back", "*.json")):
            d = json.load(open(f)); sc = d.get("score") or {}
            if "accuracy" not in sc:
                continue
            rel = os.path.relpath(f, f"{ROOT}/results/multiturn_eval_v3")
            acc_d[d.get("difficulty")].append(float(ov.get(rel, sc["accuracy"])))
        if acc_d.get("easy") and acc_d.get("hard") and np.mean(acc_d["easy"]) > np.mean(acc_d["hard"]):
            n_ok += 1
    sig_counts["n_back"] = n_ok
    # DRM per model (scorer unchanged; from archived score dicts)
    n_ok = 0
    for m in old_models:
        l, u = [], []
        for tid, (p, acc, scd) in per_items[m].items():
            if p == "drm_false_memory":
                if scd.get("false_alarm_to_critical_lures") is not None:
                    l.append(scd["false_alarm_to_critical_lures"])
                    u.append(scd.get("false_alarm_to_unrelated", 0))
        if l and np.mean(l) > np.mean(u):
            n_ok += 1
    sig_counts["drm"] = n_ok
    # EPITOME sub-capacity signature on 35 expansion (desire > belief)
    n_ok = n_tot = 0
    for m in new_models:
        per = defaultdict(list)
        for tid, (p, acc, _) in corrected_items(m, False).items():
            if p == "epitome_tom" and tid in batt:
                per[batt[tid].metadata.parameters["sub_capacity"]].append(acc)
        if per.get("desire") and per.get("belief"):
            n_tot += 1
            if np.mean(per["desire"]) > np.mean(per["belief"]):
                n_ok += 1
    sig_counts["epitome_desire_gt_belief_35"] = f"{n_ok}/{n_tot}"
    # sign-test + BH over the five 20-model signatures
    names = ["stroop", "flanker", "false_belief", "n_back", "drm"]
    pvals = [float(stats.binomtest(sig_counts[k], 20, 0.5, alternative="greater").pvalue)
             for k in names]
    n = len(pvals); order = np.argsort(pvals); adj = np.empty(n); prev = 1.0
    for rank, i in enumerate(reversed(order)):
        prev = min(prev, pvals[i] * n / (n - rank)); adj[i] = prev
    art["signature_replication_20"] = {
        k: {"count": sig_counts[k], "p": round(pvals[i], 5), "p_bh": round(float(adj[i]), 5)}
        for i, k in enumerate(names)}
    art["signature_epitome_35"] = sig_counts["epitome_desire_gt_belief_35"]

    # ---- 7. restricted-range table (corrected 55) ----------------------------
    def pc1_share(Mx):
        ev = np.linalg.eigvalsh(np.corrcoef(Mx.T))[::-1]
        return round(float(ev[0] / ev.sum()), 2)

    def resid(Mx):
        rm = Mx.mean(axis=1, keepdims=True)
        x = np.column_stack([np.ones(Mx.shape[0]), rm.ravel()])
        R = np.empty_like(Mx)
        for j in range(Mx.shape[1]):
            beta, *_ = np.linalg.lstsq(x, Mx[:, j], rcond=None)
            R[:, j] = Mx[:, j] - x @ beta
        return R

    def delta_p_sub(cols, n_perm=5000, seed=42):
        keep = [i for i, h in enumerate(hdr) if h in cols]
        sub_labels = [labels[i] for i in keep]
        def dp(Mx):
            c = np.corrcoef(Mx.T)
            prs = [(i, j) for i in range(len(keep)) for j in range(i + 1, len(keep))]
            w = [c[i, j] for i, j in prs if sub_labels[i] == sub_labels[j]]
            cr = [c[i, j] for i, j in prs if sub_labels[i] != sub_labels[j]]
            if not w or not cr:
                return None, None
            obs = float(np.mean(w) - np.mean(cr))
            rng = np.random.default_rng(seed); perm = []
            for _ in range(n_perm):
                sh = rng.permutation(sub_labels).tolist()
                ww = [c[i, j] for i, j in prs if sh[i] == sh[j]]
                cc = [c[i, j] for i, j in prs if sh[i] != sh[j]]
                if ww and cc:
                    perm.append(np.mean(ww) - np.mean(cc))
            return round(obs, 3), round(float(np.mean([d >= obs for d in perm])), 3)
        Msub = M[:, keep]
        raw = dp(Msub); rs = dp(resid(Msub))
        return {"pc1": pc1_share(Msub), "raw_delta": raw[0], "raw_p": raw[1],
                "resid_delta": rs[0], "resid_p": rs[1]}

    sds = {p: float(M[:, i].std(ddof=1)) for i, p in enumerate(hdr)}
    art["paradigm_sds_55"] = {p: round(v, 2) for p, v in sds.items()}
    lowvar3 = sorted(sds, key=sds.get)[:3]
    art["restricted_range_table"] = {
        "full": delta_p_sub(hdr),
        "drop_nb_cvlt_gn": delta_p_sub([p for p in hdr if p not in ("n_back", "cvlt_word_list", "go_nogo")]),
        "drop_3_lowest_var": delta_p_sub([p for p in hdr if p not in lowvar3]),
        "lowvar3_identity": lowvar3,
        "lo_cvlt": delta_p_sub([p for p in hdr if p != "cvlt_word_list"]),
        "lo_gonogo": delta_p_sub([p for p in hdr if p != "go_nogo"]),
        "lo_nback": delta_p_sub([p for p in hdr if p != "n_back"]),
    }

    # ---- 8. predictive validity: grouping x benchmark Spearman --------------
    bench = {}
    with open(f"{ROOT}/data/published_benchmarks.csv") as f:
        rd = csv.DictReader(f)
        for row in rd:
            m = row.get("model", "").strip()
            if m not in mat:
                continue
            vals = {}
            for c in ("mmlu", "arc_challenge", "gsm8k"):
                try:
                    vals[c] = float(row[c])
                except (TypeError, ValueError):
                    vals[c] = None
            bench[m] = vals
    group_score = {}
    for m in all_models:
        gs = defaultdict(list)
        for p in hdr:
            gs[b2.DOMAIN_MAP[p]].append(mat[m][p])
        group_score[m] = {g: float(np.mean(v)) for g, v in gs.items()}
    pv_cells = {}; pv_ps = []
    for g in ("WM", "Control", "Episodic", "ToM", "Meta"):
        for bm in ("mmlu", "arc_challenge", "gsm8k"):
            ms = [m for m in bench if bench[m].get(bm) is not None]
            x = [group_score[m][g] for m in ms]
            y = [bench[m][bm] for m in ms]
            rho, pv = stats.spearmanr(x, y)
            pv_cells[f"{g}|{bm}"] = {"rho": round(float(rho), 2), "p": round(float(pv), 4), "n": len(ms)}
            pv_ps.append((f"{g}|{bm}", float(pv)))
    ps_only = [p for _, p in pv_ps]
    order = np.argsort(ps_only); adj = np.empty(len(ps_only)); prev = 1.0
    for rank, i in enumerate(reversed(order)):
        prev = min(prev, ps_only[i] * len(ps_only) / (len(ps_only) - rank)); adj[i] = prev
    for (k, _), a in zip(pv_ps, adj):
        pv_cells[k]["p_bh"] = round(float(a), 4)
    art["pv_grouping_cells"] = pv_cells
    art["pv_n_survive_bh"] = int(sum(1 for k in pv_cells if pv_cells[k]["p_bh"] < 0.05))
    # partials for the two cited cells
    def partial(g, bm):
        ms = [m for m in bench if bench[m].get(bm) is not None]
        x = np.array([group_score[m][g] for m in ms])
        y = np.array([bench[m][bm] for m in ms])
        s = np.array([np.log10(sizes[m]) for m in ms])
        rx = x - np.poly1d(np.polyfit(s, x, 1))(s)
        ry = y - np.poly1d(np.polyfit(s, y, 1))(s)
        r, p = stats.pearsonr(rx, ry)
        return round(float(r), 2), round(float(p), 4)
    art["pv_partial_meta_mmlu"] = partial("Meta", "mmlu")
    art["pv_partial_episodic_arc"] = partial("Episodic", "arc_challenge")

    # ---- 9. fp16 robustness ---------------------------------------------------
    fp16_models = [d.split("openai_")[1] for d in
                   sorted(glob.glob(f"{ROOT}/results/full_eval_fp16/openai_*")) if os.path.isdir(d)]
    quant_of = {m: m.replace("-instruct-fp16", "").replace(":latest", "") for m in fp16_models}
    cells_fp, cells_q, grp_fp, grp_q = [], [], [], []
    for fm in fp16_models:
        qm = quant_of[fm]
        if qm not in mat:
            continue
        it_fp = corrected_items(fm, False, set_name="full_eval_fp16")
        by_par = defaultdict(list)
        for tid, (p, acc, _) in it_fp.items():
            by_par[p].append(acc)
        row_fp = {p: float(np.mean(v)) for p, v in by_par.items()}
        for p in STATIC_COLS:
            if p in row_fp:
                cells_fp.append(row_fp[p]); cells_q.append(mat[qm][p])
        gs_fp = defaultdict(list); gs_q = defaultdict(list)
        for p in STATIC_COLS:
            if p in row_fp:
                gs_fp[b2.DOMAIN_MAP[p]].append(row_fp[p]); gs_q[b2.DOMAIN_MAP[p]].append(mat[qm][p])
        for g in gs_fp:
            grp_fp.append(np.mean(gs_fp[g])); grp_q.append(np.mean(gs_q[g]))
    if cells_fp:
        r1, _ = stats.pearsonr(cells_fp, cells_q)
        r2, _ = stats.pearsonr(grp_fp, grp_q)
        art["fp16_cell_r"] = round(float(r1), 2)
        art["fp16_grouping_r"] = round(float(r2), 2)
        art["fp16_n_pairs"] = len(cells_fp)

    # ---- 10. misc model claims -----------------------------------------------
    nb20 = {m: mat[m]["n_back"] for m in old_models}
    mx = max(nb20, key=nb20.get)
    art["nback_max_20"] = {"model": mx, "value": round(nb20[mx] * 100)}
    art["dsr1_cells"] = {
        "7b": {p: round(mat["deepseek-r1:7b"][p] * 100) for p in
               ("flanker", "go_nogo", "false_belief", "epitome_tom", "operation_span")},
        "14b": {p: round(mat["deepseek-r1:14b"][p] * 100) for p in
                ("flanker", "go_nogo", "false_belief", "epitome_tom", "operation_span")},
        "qwen7b": {p: round(mat["qwen2.5:7b"][p] * 100) for p in ("flanker", "false_belief", "go_nogo")},
    }
    art["mistral_vs_qwen7b"] = {
        "mistral_gn": round(mat["mistral:7b"]["go_nogo"] * 100),
        "qwen_gn": round(mat["qwen2.5:7b"]["go_nogo"] * 100),
        "mistral_ds": round(mat["mistral:7b"]["digit_span"] * 100),
        "qwen_ds": round(mat["qwen2.5:7b"]["digit_span"] * 100),
    }
    art["stroop_ceiling_ge7b"] = sum(1 for m in old_models if sizes[m] >= 7 and mat[m]["stroop"] >= 0.95)

    json.dump(art, open(os.path.join(OUT, "step4b_artifacts.json"), "w"), indent=1)

    # LaTeX-ready table rows
    lines = ["% tab:results corrected cells (model | DS ST FL GN DRM SM FB EP CC WG)"]
    for m in old_models:
        c = art["tab_results_cells"][m]
        lines.append(f"{m}: " + " & ".join(str(c[p]) for p in STATIC_COLS))
    lines.append("% tab:multiturn corrected cells (model | NB OS CV)")
    for m in old_models:
        c = art["tab_multiturn_cells"][m]
        lines.append(f"{m}: " + " & ".join(str(c[p]) for p in MT_COLS))
    open(os.path.join(OUT, "step4b_table_rows.txt"), "w").write("\n".join(lines) + "\n")

    print(json.dumps({k: art[k] for k in ("signature_replication_20", "pct_positive_pairs_55",
          "pc1_vs_mean_acc_r_55", "pc1_removal_delta_p", "epitome_subcapacity_35",
          "srcmon_sweep_qwen7b", "pv_n_survive_bh", "pv_partial_meta_mmlu",
          "pv_partial_episodic_arc", "fp16_cell_r", "fp16_grouping_r", "nback_max_20",
          "restricted_range_table")}, indent=1))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
