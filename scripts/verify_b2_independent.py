#!/usr/bin/env python3
"""FULLY INDEPENDENT adversarial verification of B2 expanded result.
Does NOT import compute_b2_expanded. Rebuilds matrices from raw files.
"""
import json, os, glob
import numpy as np

import os
ROOT = os.environ.get("COGARENA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOMAIN_MAP = {
    'digit_span': 'WM', 'n_back': 'WM', 'operation_span': 'WM',
    'stroop': 'Control', 'flanker': 'Control', 'go_nogo': 'Control',
    'cvlt_word_list': 'Episodic', 'drm_false_memory': 'Episodic', 'source_monitoring': 'Episodic',
    'false_belief': 'ToM', 'epitome_tom': 'ToM',
    'confidence_calibration': 'Meta', 'post_decision_wagering': 'Meta'
}
PARADIGMS = sorted(DOMAIN_MAP.keys())
STATIC = {'digit_span','stroop','flanker','go_nogo','drm_false_memory','source_monitoring',
          'false_belief','epitome_tom','confidence_calibration','post_decision_wagering'}
MT = {'n_back','operation_span','cvlt_word_list'}

OLD_MODELS = ['tinyllama:1.1b','qwen2.5:0.5b','qwen2.5:1.5b','qwen2.5:3b','qwen2.5:7b',
    'qwen2.5:14b','qwen2.5:32b','gemma2:2b','gemma2:9b','gemma2:27b','llama3.2:1b',
    'llama3.2:3b','llama3.1:8b','deepseek-r1:7b','deepseek-r1:14b','mistral:7b',
    'mixtral:8x7b','phi3:14b','yi:34b','command-r:35b']


def item_acc(score):
    if isinstance(score, dict):
        if 'accuracy' in score: return float(score['accuracy'])
        if 'score' in score: return float(score['score'])
        if 'correct' in score: return 1.0 if score['correct'] else 0.0
        return 0.0
    try: return float(score)
    except (TypeError, ValueError): return 0.0


def load_details(path):
    d = json.load(open(path))
    return d if isinstance(d, list) else d.get('results', d.get('details', []))


def reagg_static(path, binarize=False):
    items = load_details(path)
    byp = {}
    for it in items:
        p = it.get('paradigm')
        if p not in STATIC: continue
        a = item_acc(it.get('score'))
        v = (1.0 if a >= 0.5 else 0.0) if binarize else a
        byp.setdefault(p, []).append(v)
    return {p: float(np.mean(v)) for p, v in byp.items() if v}


def mt_old(path):
    d = json.load(open(path)); out = {}
    for k, v in d.get('dimensions', {}).items():
        par = k.split('/')[-1]
        if par in MT: out[par] = float(v['accuracy'])
    return out


def mt_new(path):
    d = json.load(open(path)); out = {}
    for k, v in d.get('paradigms', {}).items():
        par = k.split('/')[-1]
        if par in MT:
            out[par] = float(v['accuracy'] if isinstance(v, dict) else v)
    return out


def build_old_from_aggregate(model):
    """Canonical compute_b2.py path: read aggregate.json directly."""
    agg = json.load(open(f"{ROOT}/results/full_eval_20260526_2208/openai_{model}/text/aggregate.json"))
    row = {p: agg['paradigms'][p]['accuracy'] for p in STATIC if p in agg.get('paradigms', {})}
    mtp = f"{ROOT}/results/multiturn_eval_v3/openai_{model}/aggregate.json"
    if os.path.exists(mtp): row.update(mt_old(mtp))
    return row


def build_old_from_details(model, binarize=False):
    sp = f"{ROOT}/results/full_eval_20260526_2208/openai_{model}/text/details.json"
    row = reagg_static(sp, binarize)
    mtp = f"{ROOT}/results/multiturn_eval_v3/openai_{model}/aggregate.json"
    if os.path.exists(mtp): row.update(mt_old(mtp))
    return row


def build_new(model, binarize=False):
    sp = f"{ROOT}/results/full_eval_expansion/openai_{model}/text/details.json"
    if not os.path.exists(sp): return None
    row = reagg_static(sp, binarize)
    mtp = f"{ROOT}/results/multiturn_expansion/openai_{model}/text/aggregate.json"
    if os.path.exists(mtp): row.update(mt_new(mtp))
    return row


def run_b2(data, models, n_perm=5000, seed=42):
    matrix = np.array([[data[m].get(p, 0) for p in PARADIGMS] for m in models])
    corr = np.corrcoef(matrix.T)
    within, cross = [], []
    for i in range(len(PARADIGMS)):
        for j in range(i+1, len(PARADIGMS)):
            r = corr[i, j]
            (within if DOMAIN_MAP[PARADIGMS[i]] == DOMAIN_MAP[PARADIGMS[j]] else cross).append(r)
    wm, cm = float(np.mean(within)), float(np.mean(cross))
    delta = wm - cm
    rng = np.random.default_rng(seed)
    labels = [DOMAIN_MAP[p] for p in PARADIGMS]
    pd_ = []
    for _ in range(n_perm):
        sh = rng.permutation(labels).tolist()
        w, c = [], []
        for i in range(len(PARADIGMS)):
            for j in range(i+1, len(PARADIGMS)):
                (w if sh[i] == sh[j] else c).append(corr[i, j])
        if w and c: pd_.append(np.mean(w)-np.mean(c))
    p = float(np.mean([x >= delta for x in pd_]))
    lo, hi = [float(x) for x in np.percentile(pd_, [2.5, 97.5])]
    return dict(n=len(models), within=round(wm,4), cross=round(cm,4),
               delta=round(delta,4), ci=[round(lo,4),round(hi,4)], p=round(p,4)), matrix, corr


print("="*70)
print("CHECK 1: Independent 20-model repro from BOTH aggregate.json and details.json")
print("="*70)
old_agg = {m: build_old_from_aggregate(m) for m in OLD_MODELS}
r_agg, mat_agg, _ = run_b2(old_agg, sorted(old_agg.keys()))
print("From aggregate.json (canonical compute_b2.py source):", r_agg)

old_det = {m: build_old_from_details(m, binarize=False) for m in OLD_MODELS}
r_det, _, _ = run_b2(old_det, sorted(old_det.keys()))
print("From details.json CONTINUOUS:", r_det)

old_bin = {m: build_old_from_details(m, binarize=True) for m in OLD_MODELS}
r_bin, _, _ = run_b2(old_bin, sorted(old_bin.keys()))
print("From details.json BINARIZED:", r_bin)

paper = dict(within=0.4137, cross=0.422, delta=-0.0083, p=0.514)
print("\nPAPER target:", paper)
print("aggregate matches paper:", abs(r_agg['within']-0.4137)<0.001 and abs(r_agg['cross']-0.422)<0.001 and abs(r_agg['delta']+0.0083)<0.001 and abs(r_agg['p']-0.514)<0.01)
print("details-continuous matches paper:", abs(r_det['within']-0.4137)<0.001 and abs(r_det['cross']-0.422)<0.001 and abs(r_det['delta']+0.0083)<0.001 and abs(r_det['p']-0.514)<0.01)

# Does aggregate-derived matrix == details-continuous matrix?
print("\naggregate vs details-continuous matrix identical:",
      np.allclose(mat_agg, np.array([[old_det[m].get(p,0) for p in PARADIGMS] for m in sorted(old_det.keys())]), atol=1e-6))

print("\n" + "="*70)
print("CHECK 2: Spot-check (model,paradigm) cells of 52-matrix")
print("="*70)
# new model false_belief from details re-aggregated >=0.5 (binarize) AND continuous
for model, par in [('qwen3:14b','false_belief'), ('phi4:14b','epitome_tom'), ('llama2:7b','stroop')]:
    sp = f"{ROOT}/results/full_eval_expansion/openai_{model}/text/details.json"
    items = load_details(sp)
    accs = [item_acc(it['score']) for it in items if it.get('paradigm')==par]
    cont = np.mean(accs); binm = np.mean([1.0 if a>=0.5 else 0.0 for a in accs])
    print(f"NEW {model} {par}: n={len(accs)} continuous={cont:.4f} binarized>=0.5={binm:.4f}")

# new model n_back from multiturn paradigms
for model in ['qwen3:14b','llama2:13b','phi4:14b']:
    mtp = f"{ROOT}/results/multiturn_expansion/openai_{model}/text/aggregate.json"
    if os.path.exists(mtp):
        d = json.load(open(mtp))
        nb = d.get('paradigms',{}).get('n_back')
        print(f"NEW {model} n_back from MT paradigms: {nb if not isinstance(nb,dict) else nb.get('accuracy')}")

print("\n" + "="*70)
print("CHECK 3: Independent 52-model delta + permutation p (seed=42, n_perm=5000)")
print("="*70)
new_names = [l.strip() for l in open(f"{ROOT}/results/reanalysis/expansion_modellist.txt") if l.strip()]
new_data = {}
for m in new_names:
    row = build_new(m, binarize=False)
    if row is None: print("MISSING NEW:", m); continue
    miss = [p for p in PARADIGMS if p not in row]
    if miss: print("NEW missing paradigms", m, miss)
    new_data[m] = row

# Combined 52 using CONTINUOUS (as artifact claims it used)
all_cont = {**old_det, **new_data}
print("n combined:", len(all_cont), "| overlap old/new:", set(old_det)&set(new_data))
r52, mat52, corr52 = run_b2(all_cont, sorted(all_cont.keys()))
print("EXP52 CONTINUOUS:", r52)

# Also test 52 with binarized static for new (and continuous old) just to see sensitivity
all_bin = {**old_bin, **{m: build_new(m, binarize=True) for m in new_names if build_new(m,True)}}
r52b, _, _ = run_b2(all_bin, sorted(all_bin.keys()))
print("EXP52 BINARIZED:", r52b)

print("\n" + "="*70)
print("CHECK 4: source verification")
print("="*70)
# Confirm new aggregate.json IS buggy (would give different drm than details) -> justifies details
buggy_count = 0
for m in new_names:
    aggp = f"{ROOT}/results/full_eval_expansion/openai_{m}/text/aggregate.json"
    detp = f"{ROOT}/results/full_eval_expansion/openai_{m}/text/details.json"
    if not (os.path.exists(aggp) and os.path.exists(detp)): continue
    agg = json.load(open(aggp))
    items = load_details(detp)
    for par in ['drm_false_memory','source_monitoring']:
        accs=[item_acc(it['score']) for it in items if it.get('paradigm')==par]
        if not accs: continue
        det_cont = np.mean(accs)
        agg_val = agg.get('paradigms',{}).get(par,{}).get('accuracy') if isinstance(agg.get('paradigms',{}).get(par),dict) else None
        if agg_val is not None and abs(agg_val-det_cont)>0.01:
            buggy_count += 1
print("new models where aggregate.json drm/source DIVERGE from details continuous:", buggy_count)

print("\nARTIFACT CLAIMS: exp52 within=0.3379 cross=0.3409 delta=-0.003 p=0.5042 ci=[-0.0932,0.1088]")
print(f"MY EXP52:        within={r52['within']} cross={r52['cross']} delta={r52['delta']} p={r52['p']} ci={r52['ci']}")
