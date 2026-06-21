#!/usr/bin/env python3
"""Compute B2 convergent-discriminant analysis from canonical aggregate.json files.

Usage:
    python scripts/compute_b2.py --eval-dir results/full_eval_20260526_2208

Output: <eval-dir>/analysis_summary_v2.json
"""
import argparse, json, glob
import numpy as np

DOMAIN_MAP = {
    'digit_span': 'WM', 'n_back': 'WM', 'operation_span': 'WM',
    'stroop': 'Control', 'flanker': 'Control', 'go_nogo': 'Control',
    'cvlt_word_list': 'Episodic', 'drm_false_memory': 'Episodic', 'source_monitoring': 'Episodic',
    'false_belief': 'ToM', 'epitome_tom': 'ToM',
    'confidence_calibration': 'Meta', 'post_decision_wagering': 'Meta'
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, help="Static eval dir (10 paradigms)")
    parser.add_argument("--mt-dir", default=None, help="Multi-turn eval dir (n_back, ospan, cvlt)")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paradigms_order = sorted(DOMAIN_MAP.keys())
    models_data = {}

    # Load static paradigms (10)
    for f in sorted(glob.glob(f'{args.eval_dir}/*/text/aggregate.json')):
        d = json.load(open(f))
        model = d['model'].replace('openai/', '')
        models_data[model] = {p: d['paradigms'].get(p, {}).get('accuracy', 0) for p in paradigms_order
                              if p in d.get('paradigms', {})}

    # Merge multi-turn paradigms (n_back, operation_span, cvlt_word_list) if provided
    if args.mt_dir:
        for f in sorted(glob.glob(f'{args.mt_dir}/*/aggregate.json')):
            d = json.load(open(f))
            model = d.get('model', '').replace('openai/', '')
            if model not in models_data:
                models_data[model] = {}
            for k, v in d.get('dimensions', {}).items():
                paradigm = k.split('/')[-1]  # "working_memory/n_back" -> "n_back"
                if paradigm in ('n_back', 'operation_span', 'cvlt_word_list'):
                    models_data[model][paradigm] = v['accuracy']

    # Fill missing paradigms with 0
    for model in models_data:
        for p in paradigms_order:
            if p not in models_data[model]:
                models_data[model][p] = 0

    models = sorted(models_data.keys())
    matrix = np.array([[models_data[m][p] for p in paradigms_order] for m in models])
    corr = np.corrcoef(matrix.T)

    within_corrs, cross_corrs = [], []
    for i in range(len(paradigms_order)):
        for j in range(i+1, len(paradigms_order)):
            r = corr[i, j]
            if DOMAIN_MAP[paradigms_order[i]] == DOMAIN_MAP[paradigms_order[j]]:
                within_corrs.append(r)
            else:
                cross_corrs.append(r)

    within_mean = np.mean(within_corrs)
    cross_mean = np.mean(cross_corrs)
    delta = within_mean - cross_mean

    rng = np.random.default_rng(args.seed)
    domain_labels = [DOMAIN_MAP[p] for p in paradigms_order]
    perm_deltas = []
    for _ in range(args.n_perm):
        shuffled = rng.permutation(domain_labels).tolist()
        w, c = [], []
        for i in range(len(paradigms_order)):
            for j in range(i+1, len(paradigms_order)):
                if shuffled[i] == shuffled[j]: w.append(corr[i,j])
                else: c.append(corr[i,j])
        if w and c: perm_deltas.append(np.mean(w) - np.mean(c))

    p_value = np.mean([d >= delta for d in perm_deltas])
    ci_lo, ci_hi = np.percentile(perm_deltas, [2.5, 97.5])

    result = {
        "eval_dir": args.eval_dir,
        "n_models": len(models),
        "n_paradigms": len(paradigms_order),
        "within_mean": round(float(within_mean), 4),
        "cross_mean": round(float(cross_mean), 4),
        "diff": round(float(delta), 4),
        "ci_lo": round(float(ci_lo), 4),
        "ci_hi": round(float(ci_hi), 4),
        "p_value": round(float(p_value), 4),
        "n_perm": args.n_perm,
        "seed": args.seed,
        "pass": bool(p_value < 0.05),
    }

    out_path = f"{args.eval_dir}/analysis_summary_v2.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
