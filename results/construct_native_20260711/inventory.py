#!/usr/bin/env python3
"""Inventory: which construct-native metrics are computable from existing data.

For each of the 13 paradigms, determine
  (a) what condition labels are recoverable from task_ids / file layout,
  (b) whether raw `response` payloads exist where the construct metric needs
      them (conf-cal Brier/ECE, wagering type-2 AUC, go/no-go d'),
  (c) per-model item counts per condition (median across models),
so we can decide which construct-native scores are buildable WITHOUT any new
model runs. Read-only; writes inventory.json + INVENTORY.md here.
"""
import collections
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(
    os.environ.get(
        "COGARENA_ROOT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
    )
)
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ["COGARENA_ROOT"] = ROOT
import compute_b2_expanded as b2

GONOGO = os.path.join(ROOT, "results", "gonogo_rerun_20260702")

def static_paths(model, is_old):
    set_name = "full_eval_20260526_2208" if is_old else "full_eval_expansion"
    return set_name, f"{ROOT}/results/{set_name}/openai_{model}/text"

def mt_base(model, is_old):
    if is_old:
        return f"{ROOT}/results/multiturn_eval_v3/openai_{model}"
    return f"{ROOT}/results/multiturn_expansion/openai_{model}/text"

def parse_tokens(task_id, paradigm):
    """Tokens between the paradigm name and the trailing index in a task_id."""
    m = re.search(re.escape(paradigm) + r"_(.+)_(\d+)$", task_id)
    if not m:
        return ()
    return tuple(m.group(1).split("_"))

def main():
    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    old_models = list(b2.OLD_MODELS.keys())
    new_models = sorted(new_meta.keys())
    print(f"[pools] old={len(old_models)} new={len(new_models)}")
    print(f"[paradigms] {b2.PARADIGMS_ORDER}")
    print(f"[static] {sorted(b2.STATIC_PARADIGMS)}  [mt] {sorted(b2.MT_PARADIGMS)}")

    inv = {}

    # ---------- static paradigms: token vocab + counts from details.json ------
    per_par_tokens = collections.defaultdict(collections.Counter)   # paradigm -> token tuple -> n items (pooled)
    per_par_counts = collections.defaultdict(list)                  # paradigm -> [n items per model]
    missing = []
    for m in old_models + new_models:
        is_old = m in b2.OLD_MODELS
        set_name, tdir = static_paths(m, is_old)
        det = os.path.join(tdir, "details.json")
        if not os.path.exists(det):
            missing.append(m); continue
        items = b2.load_details(det)
        bypar = collections.Counter()
        for it in items:
            p = it.get("paradigm")
            if p not in b2.STATIC_PARADIGMS:
                continue
            bypar[p] += 1
            per_par_tokens[p][parse_tokens(it.get("task_id", ""), p)] += 1
        for p, n in bypar.items():
            per_par_counts[p].append(n)
    print(f"[static scanned] missing details for: {missing}")

    # details.json item schema (does it carry response?)
    _, tdir0 = static_paths(old_models[0], True)
    it0 = b2.load_details(os.path.join(tdir0, "details.json"))[0]
    inv["details_item_keys"] = sorted(it0.keys())

    # ---------- per-trial subdir files: response payload sampling -------------
    def sample_trial_files(pattern, k=3):
        out = []
        for f in sorted(glob.glob(pattern))[:k]:
            try:
                d = json.load(open(f))
            except Exception as e:
                out.append({"file": f, "error": str(e)}); continue
            r = d.get("response")
            out.append({
                "file": os.path.relpath(f, ROOT),
                "keys": sorted(d.keys()),
                "response_type": type(r).__name__,
                "response_head": (r[:220] if isinstance(r, str) else
                                  (sorted(r.keys()) if isinstance(r, dict) else r)),
                "score": d.get("score"),
                "difficulty": d.get("difficulty"),
            })
        return out

    resp_samples = {}
    probe_models = [old_models[0], new_models[0] if new_models else old_models[0]]
    for p in sorted(b2.STATIC_PARADIGMS):
        for m, is_old in ((probe_models[0], True), (probe_models[1], False)):
            _, tdir = static_paths(m, is_old)
            hits = glob.glob(os.path.join(tdir, "*", p, "*.json"))
            if hits:
                resp_samples[f"{p}__{'old' if is_old else 'new'}"] = sample_trial_files(
                    os.path.join(tdir, "*", p, "*.json"))
                break
    inv["static_trial_samples"] = resp_samples

    # ---------- go_nogo rerun: trial-type tokens + response availability ------
    gg_tokens = collections.Counter(); gg_counts = []
    gg_missing = 0
    for m in old_models + new_models:
        det = os.path.join(GONOGO, f"openai_{m}", "text", "details.json")
        if not os.path.exists(det):
            gg_missing += 1; continue
        items = json.load(open(det))
        gg_counts.append(len(items))
        for it in items:
            gg_tokens[parse_tokens(it.get("task_id", ""), "go_nogo")] += 1
    inv["go_nogo_rerun"] = {
        "models_missing": gg_missing,
        "items_per_model_median": float(np.median(gg_counts)) if gg_counts else None,
        "token_vocab_top": [(list(t), n) for t, n in gg_tokens.most_common(12)],
        "trial_samples": sample_trial_files(os.path.join(
            GONOGO, f"openai_{old_models[0]}", "text", "*", "go_nogo", "*.json")),
    }

    # ---------- multiturn paradigms: file schema (loads/turns) ----------------
    mt = {}
    for par in sorted(b2.MT_PARADIGMS):
        rec = {"files_per_model": [], "sample": None, "score_keys": None,
               "turn_structure": None, "file_name_tokens": collections.Counter()}
        for m, is_old in ((old_models[0], True), (new_models[0], False)):
            base = mt_base(m, is_old)
            files = sorted(glob.glob(os.path.join(base, "*", par, "*.json")))
            if not files:
                continue
            d = json.load(open(files[0]))
            rec["sample"] = os.path.relpath(files[0], ROOT)
            rec["top_keys"] = sorted(d.keys())
            sc = d.get("score") or {}
            rec["score_keys"] = sorted(sc.keys()) if isinstance(sc, dict) else type(sc).__name__
            for key in ("turns", "trials", "turn_scores", "per_turn", "history", "transcript"):
                v = d.get(key) or (sc.get(key) if isinstance(sc, dict) else None)
                if isinstance(v, list) and v:
                    rec["turn_structure"] = {"key": key, "n": len(v),
                        "elem_keys": sorted(v[0].keys()) if isinstance(v[0], dict) else type(v[0]).__name__}
                    break
            for f in files:
                rec["file_name_tokens"][re.sub(r"\d+", "#", os.path.basename(f))] += 1
            break
        # files per model across cohorts
        for m in old_models + new_models:
            base = mt_base(m, m in b2.OLD_MODELS)
            rec["files_per_model"].append(len(glob.glob(os.path.join(base, "*", par, "*.json"))))
        rec["files_per_model"] = {
            "median": float(np.median(rec["files_per_model"])),
            "min": int(np.min(rec["files_per_model"])),
            "n_zero": int(sum(1 for x in rec["files_per_model"] if x == 0)),
        }
        rec["file_name_tokens"] = [(t, n) for t, n in rec["file_name_tokens"].most_common(8)]
        mt[par] = rec
    inv["multiturn"] = mt

    # ---------- condition vocab summary per static paradigm -------------------
    cond = {}
    for p in sorted(per_par_tokens):
        top = per_par_tokens[p].most_common(14)
        # candidate condition token = the token position with smallest vocab > 1
        positions = collections.defaultdict(collections.Counter)
        for toks, n in per_par_tokens[p].items():
            for i, t in enumerate(toks):
                positions[i][t] += n
        pos_vocab = {i: sorted(c.keys()) for i, c in positions.items() if len(c) <= 8}
        cond[p] = {
            "items_per_model_median": float(np.median(per_par_counts[p])),
            "token_tuples_top": [(list(t), n) for t, n in top],
            "small_vocab_positions": pos_vocab,
        }
    inv["static_conditions"] = cond

    json.dump(inv, open(os.path.join(OUT, "inventory.json"), "w"), indent=1, default=str)
    print(json.dumps({"static_conditions": {p: c["small_vocab_positions"] for p, c in cond.items()},
                      "details_item_keys": inv["details_item_keys"],
                      "go_nogo_tokens": inv["go_nogo_rerun"]["token_vocab_top"][:6]},
                     indent=1)[:4000])
    print("[done] wrote inventory.json")

if __name__ == "__main__":
    main()
