#!/usr/bin/env python3
"""B2 dimensional-separability on the EXPANDED 55-model set.

Methodologically identical to scripts/compute_b2.py, but:
 - STATIC 10 paradigms re-aggregated from per-item details.json (continuous per-item-accuracy mean)
 - MULTI-TURN 3 paradigms (n_back, operation_span, cvlt_word_list) from MT aggregate
 - Includes a 20-model self-check reproducing results/analysis_b2_13paradigms.json
 - Adds family-aware scaling analysis on 55 models.
"""
import json, os, warnings
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd

warnings.filterwarnings("ignore")

import os
ROOT = os.environ.get("COGARENA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOMAIN_MAP = {
    'digit_span': 'WM', 'n_back': 'WM', 'operation_span': 'WM',
    'stroop': 'Control', 'flanker': 'Control', 'go_nogo': 'Control',
    'cvlt_word_list': 'Episodic', 'drm_false_memory': 'Episodic', 'source_monitoring': 'Episodic',
    'false_belief': 'ToM', 'epitome_tom': 'ToM',
    'confidence_calibration': 'Meta', 'post_decision_wagering': 'Meta'
}
PARADIGMS_ORDER = sorted(DOMAIN_MAP.keys())

STATIC_PARADIGMS = {
    'digit_span', 'stroop', 'flanker', 'go_nogo', 'drm_false_memory',
    'source_monitoring', 'false_belief', 'epitome_tom',
    'confidence_calibration', 'post_decision_wagering'
}
MT_PARADIGMS = {'n_back', 'operation_span', 'cvlt_word_list'}

# OLD 20 models: name -> (size_b, family)
OLD_MODELS = {
    'tinyllama:1.1b': (1.1, 'tinyllama'),
    'qwen2.5:0.5b': (0.5, 'qwen2.5'),
    'qwen2.5:1.5b': (1.5, 'qwen2.5'),
    'qwen2.5:3b': (3, 'qwen2.5'),
    'qwen2.5:7b': (7, 'qwen2.5'),
    'qwen2.5:14b': (14, 'qwen2.5'),
    'qwen2.5:32b': (32, 'qwen2.5'),
    'gemma2:2b': (2, 'gemma2'),
    'gemma2:9b': (9, 'gemma2'),
    'gemma2:27b': (27, 'gemma2'),
    'llama3.2:1b': (1, 'llama3.2'),
    'llama3.2:3b': (3, 'llama3.2'),
    'llama3.1:8b': (8, 'llama3.1'),
    'deepseek-r1:7b': (7, 'deepseek-r1'),
    'deepseek-r1:14b': (14, 'deepseek-r1'),
    'mistral:7b': (7, 'mistral'),
    'mixtral:8x7b': (47, 'mixtral'),
    'phi3:14b': (14, 'phi3'),
    'yi:34b': (34, 'yi'),
    'command-r:35b': (35, 'command-r'),
}


def item_accuracy(score):
    """Per-item accuracy from a score field."""
    if isinstance(score, dict):
        if 'accuracy' in score:
            return float(score['accuracy'])
        if 'score' in score:
            return float(score['score'])
        if 'correct' in score:
            return 1.0 if score['correct'] else 0.0
        return 0.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def load_details(path):
    d = json.load(open(path))
    if isinstance(d, list):
        return d
    return d.get('results', d.get('details', []))


def reaggregate_static(details_path, binarize=False):
    """Re-aggregate static paradigms from per-item details.

    NOTE ON BINARIZE: the prompt's literal instruction was to binarize per-item
    accuracy at >=0.5. However, the MANDATORY 20-model self-check
    (results/analysis_b2_13paradigms.json: within=0.4137, cross=0.4220,
    delta=-0.0083, p=0.514) is reproduced EXACTLY only by the CONTINUOUS
    per-item-accuracy mean (paradigm_acc = mean(score['accuracy'])), and NOT by
    the binarized version (which yields within=0.4156, cross=0.4102,
    delta=+0.0054 -- a mismatch the prompt itself flags as "your matrix
    construction is wrong"). The OLD aggregate.json files (which the paper used)
    store the continuous mean for dict-valued paradigms drm_false_memory /
    source_monitoring; binarizing changes exactly those two cells. Per the
    prompt's own arbiter rule, we therefore use the CONTINUOUS mean uniformly
    for all 55 models. Re-aggregating from details.json (rather than trusting
    aggregate.json) is still essential: it repairs NEW models where the inline
    aggregator scores dict-valued paradigms as 0 (the documented bug).
    """
    items = load_details(details_path)
    by_par = {}
    for it in items:
        p = it.get('paradigm')
        if p not in STATIC_PARADIGMS:
            continue
        acc = item_accuracy(it.get('score'))
        val = (1.0 if acc >= 0.5 else 0.0) if binarize else acc
        by_par.setdefault(p, []).append(val)
    return {p: float(np.mean(v)) for p, v in by_par.items() if v}


def load_mt_old(path):
    d = json.load(open(path))
    out = {}
    for k, v in d.get('dimensions', {}).items():
        par = k.split('/')[-1]
        if par in MT_PARADIGMS:
            out[par] = float(v['accuracy'])
    return out


def load_mt_new(path):
    d = json.load(open(path))
    out = {}
    for k, v in d.get('paradigms', {}).items():
        par = k.split('/')[-1]
        if par in MT_PARADIGMS:
            acc = v['accuracy'] if isinstance(v, dict) else v
            out[par] = float(acc)
    return out


_CC_CORR = None


def _cc_corrected():
    """Optional map of confidence_calibration accuracy re-scored with the fixed
    metacognition scorer (results/reanalysis/conf_cal_corrected.json). Empty if
    absent, so behaviour is unchanged until the re-score is produced."""
    global _CC_CORR
    if _CC_CORR is None:
        p = f"{ROOT}/results/reanalysis/conf_cal_corrected.json"
        _CC_CORR = json.load(open(p)) if os.path.exists(p) else {}
    return _CC_CORR


def build_model_row(model, is_old, binarize=False):
    """Return dict paradigm->accuracy for one model, or None if data missing."""
    row = {}
    if is_old:
        static_path = f"{ROOT}/results/full_eval_20260526_2208/openai_{model}/text/details.json"
        mt_path = f"{ROOT}/results/multiturn_eval_v3/openai_{model}/aggregate.json"
        mt = load_mt_old(mt_path) if os.path.exists(mt_path) else {}
    else:
        static_path = f"{ROOT}/results/full_eval_expansion/openai_{model}/text/details.json"
        mt_path = f"{ROOT}/results/multiturn_expansion/openai_{model}/text/aggregate.json"
        mt = load_mt_new(mt_path) if os.path.exists(mt_path) else {}
    if not os.path.exists(static_path):
        return None, f"missing static {static_path}"
    static = reaggregate_static(static_path, binarize=binarize)
    row.update(static)
    row.update(mt)
    _cc = _cc_corrected().get(model)
    if _cc is not None:
        row["confidence_calibration"] = _cc
    return row, None


def run_b2(models_data, models, n_perm=5000, seed=42):
    matrix = np.array([[models_data[m].get(p, 0) for p in PARADIGMS_ORDER] for m in models])
    corr = np.corrcoef(matrix.T)
    within, cross = [], []
    for i in range(len(PARADIGMS_ORDER)):
        for j in range(i + 1, len(PARADIGMS_ORDER)):
            r = corr[i, j]
            if DOMAIN_MAP[PARADIGMS_ORDER[i]] == DOMAIN_MAP[PARADIGMS_ORDER[j]]:
                within.append(r)
            else:
                cross.append(r)
    within_mean = float(np.mean(within))
    cross_mean = float(np.mean(cross))
    delta = within_mean - cross_mean

    rng = np.random.default_rng(seed)
    domain_labels = [DOMAIN_MAP[p] for p in PARADIGMS_ORDER]
    perm_deltas = []
    for _ in range(n_perm):
        shuffled = rng.permutation(domain_labels).tolist()
        w, c = [], []
        for i in range(len(PARADIGMS_ORDER)):
            for j in range(i + 1, len(PARADIGMS_ORDER)):
                if shuffled[i] == shuffled[j]:
                    w.append(corr[i, j])
                else:
                    c.append(corr[i, j])
        if w and c:
            perm_deltas.append(np.mean(w) - np.mean(c))
    p_value = float(np.mean([d >= delta for d in perm_deltas]))
    ci_lo, ci_hi = [float(x) for x in np.percentile(perm_deltas, [2.5, 97.5])]
    return {
        'n_models': len(models),
        'within_mean': round(within_mean, 4),
        'cross_mean': round(cross_mean, 4),
        'delta': round(delta, 4),
        'ci_lo': round(ci_lo, 4),
        'ci_hi': round(ci_hi, 4),
        'p_value': round(p_value, 4),
        'n_perm': n_perm,
        'seed': seed,
        'pass': bool(p_value < 0.05),
    }, matrix


def main():
    # ---- Build OLD 20 ----
    old_data = {}
    old_meta = {}
    for m, (size, fam) in OLD_MODELS.items():
        row, err = build_model_row(m, is_old=True)
        if err:
            print("OLD ERR", m, err)
            continue
        missing = [p for p in PARADIGMS_ORDER if p not in row]
        if missing:
            print("OLD MISSING PARADIGMS", m, missing)
        old_data[m] = row
        old_meta[m] = {'size_b': size, 'family': fam}

    old_models = sorted(old_data.keys())
    repro, _ = run_b2(old_data, old_models)
    print("=== REPRO 20 (continuous) ===")
    print(json.dumps(repro, indent=2))

    # Also compute the binarized variant for transparency on the methodology note.
    old_data_bin = {}
    for m, (size, fam) in OLD_MODELS.items():
        row, err = build_model_row(m, is_old=True, binarize=True)
        if not err:
            old_data_bin[m] = row
    repro_bin, _ = run_b2(old_data_bin, sorted(old_data_bin.keys()))
    print("=== REPRO 20 (binarized, for reference) ===")
    print(json.dumps(repro_bin, indent=2))

    # Current paper 20-model 13-paradigm B2, after the confidence-calibration scorer fix
    # (pre-fix self-check vs analysis_b2_13paradigms.json was delta=-0.0083; see docstring above).
    paper = {'within': 0.4145, 'cross': 0.425, 'delta': -0.0106, 'p': 0.53}
    matches = (abs(repro['within_mean'] - paper['within']) < 0.001 and
               abs(repro['cross_mean'] - paper['cross']) < 0.001 and
               abs(repro['delta'] - paper['delta']) < 0.001 and
               abs(repro['p_value'] - paper['p']) < 0.01)
    print("repro20_matches_paper:", matches)

    # ---- Build NEW 32 ----
    new_meta = json.load(open(f"{ROOT}/results/reanalysis/expansion_models.json"))
    new_names = [l.strip() for l in open(f"{ROOT}/results/reanalysis/expansion_modellist.txt") if l.strip()]
    new_data = {}
    for m in new_names:
        row, err = build_model_row(m, is_old=False)
        if err:
            print("NEW ERR", m, err)
            continue
        missing = [p for p in PARADIGMS_ORDER if p not in row]
        if missing:
            print("NEW MISSING PARADIGMS", m, missing)
        new_data[m] = row

    # ---- Combined 55 ----
    all_data = {**old_data, **new_data}
    all_meta = {}
    for m in old_meta:
        all_meta[m] = old_meta[m]
    for m in new_meta:
        all_meta[m] = {'size_b': float(new_meta[m]['size_b']), 'family': new_meta[m]['family']}

    all_models = sorted(all_data.keys())
    print("n_models_total:", len(all_models))
    print("n_old:", len(old_models), "n_new:", len(new_data))

    exp55, matrix55 = run_b2(all_data, all_models)
    print("=== EXP 55 ===")
    print(json.dumps(exp55, indent=2))

    # ---- Family-aware scaling ----
    rows = []
    for m in all_models:
        meta = all_meta[m]
        for p in PARADIGMS_ORDER:
            rows.append({
                'model': m, 'family': meta['family'],
                'log10_size': np.log10(meta['size_b']),
                'paradigm': p, 'accuracy': all_data[m].get(p, 0.0),
            })
    df = pd.DataFrame(rows)

    scaling = {}
    for p in PARADIGMS_ORDER:
        sub = df[df.paradigm == p]
        x = sub['log10_size'].values
        y = sub['accuracy'].values
        r = float(np.corrcoef(x, y)[0, 1])
        # Mixed effects: accuracy ~ log10_size + (1|family)
        mlm_beta, mlm_p = None, None
        try:
            md = smf.mixedlm("accuracy ~ log10_size", sub, groups=sub['family'])
            mfit = md.fit(method='lbfgs')
            mlm_beta = float(mfit.params['log10_size'])
            mlm_p = float(mfit.pvalues['log10_size'])
        except Exception as e:
            mlm_beta, mlm_p = None, f"err:{e}"
        scaling[p] = {
            'pearson_r': round(r, 4),
            'mlm_beta': round(mlm_beta, 4) if isinstance(mlm_beta, float) else mlm_beta,
            'mlm_p': round(mlm_p, 4) if isinstance(mlm_p, float) else mlm_p,
        }

    # ---- Write out ----
    out = {
        'method': 'B2 dimensional separability, 13 paradigms, 5 domains; '
                  'static re-aggregated from details.json (continuous per-item-accuracy mean), '
                  'multiturn n_back/operation_span/cvlt_word_list from MT aggregate; '
                  'permutation n_perm=5000 seed=42.',
        'n_models_old': len(old_models),
        'n_models_new': len(new_data),
        'n_models_total': len(all_models),
        'repro20': repro,
        'repro20_binarized_reference': repro_bin,
        'repro20_matches_paper': bool(matches),
        'paper_target': paper,
        'aggregation_note': (
            'Static paradigms re-aggregated from per-item details.json. CONTINUOUS '
            'per-item-accuracy mean is used (NOT binarize>=0.5): only the continuous '
            'mean reproduces the published 20-model B2. After the confidence-calibration '
            'scorer fix the current paper reports within 0.4145 / cross 0.4250 / '
            'delta -0.0106 / p 0.53 (the repro20 block here); the pre-fix self-check vs '
            'analysis_b2_13paradigms.json was delta -0.0083. Binarizing alters '
            'drm_false_memory and source_monitoring (dict-valued, continuous in the '
            'paper) and gives a non-matching delta. Re-aggregation from details.json '
            'also fixes the NEW-model inline-aggregator bug (e.g. qwen3:4b had a genuine drm=0).'),
        'exp55': exp55,
        'scaling': scaling,
        'models_old': old_models,
        'models_new': sorted(new_data.keys()),
    }
    out_path = f"{ROOT}/results/reanalysis/b2_expanded.json"
    json.dump(out, open(out_path, 'w'), indent=2)
    print("Saved to", out_path)

    # scaling summary
    strong = [p for p in PARADIGMS_ORDER if abs(scaling[p]['pearson_r']) >= 0.4]
    nearzero = [p for p in PARADIGMS_ORDER if abs(scaling[p]['pearson_r']) < 0.15]
    print("STRONG (|r|>=0.4):", strong)
    print("NEAR-ZERO (|r|<0.15):", nearzero)
    for p in ['stroop', 'epitome_tom', 'drm_false_memory', 'false_belief']:
        print(f"  4-scaler {p}: r={scaling[p]['pearson_r']} mlm_beta={scaling[p]['mlm_beta']} mlm_p={scaling[p]['mlm_p']}")


if __name__ == '__main__':
    main()
