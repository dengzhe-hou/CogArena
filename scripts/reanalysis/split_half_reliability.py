#!/usr/bin/env python3
"""Split-half reliability of the 10 single-turn paradigms (paper Section 6).

For each paradigm we estimate how consistently it RANKS the 20 text LLMs:
split that paradigm's items into two random halves, score every model on each
half, correlate the two half-scores across the 20 models, and apply the
Spearman-Brown prophecy correction r_sb = 2r / (1 + r). To remove the
arbitrariness of any single split, we AVERAGE r_sb over K random splits
(seed=42), the standard robust split-half estimator.

Items are identical across models (procedural generation with seed=42), so each
random split is the SAME item partition for every model, which is what the
split-half interpretation requires. Per-item accuracy is read from the stored
per-item score (continuous, via item_accuracy, matching the main analysis), so
dict-scored paradigms (DRM, source monitoring) are handled consistently with
the rest of the paper.

Writes results/reanalysis/split_half_reliability.json
(paper Section 6: mean Spearman-Brown ~0.96, all paradigms high).

CPU-only, numpy/scipy. Run on a Slurm node, not the login node.
"""
import json, os
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr

import os
ROOT = os.environ.get("COGARENA_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STATIC_DIR = os.path.join(ROOT, "results/full_eval_20260526_2208")
OUT = os.path.join(ROOT, "results/reanalysis/split_half_reliability.json")
K_SPLITS = 2000
SEED = 42

MODELS = [
    'tinyllama:1.1b', 'qwen2.5:0.5b', 'qwen2.5:1.5b', 'gemma2:2b', 'llama3.2:1b',
    'qwen2.5:3b', 'llama3.2:3b', 'qwen2.5:7b', 'mistral:7b', 'llama3.1:8b',
    'deepseek-r1:7b', 'gemma2:9b', 'qwen2.5:14b', 'phi3:14b', 'deepseek-r1:14b',
    'gemma2:27b', 'qwen2.5:32b', 'mixtral:8x7b', 'yi:34b', 'command-r:35b',
]

# 9 single-turn paradigms (n-back, operation span, CVLT are multi-turn -> excluded).
# epitome_tom is also excluded: its details.json per-item records are synthesized
# placeholders reconstructing the forced-choice rerun's sub-capacity aggregates
# (response text says "per-item response not stored"; first-k-correct patterns),
# so a split-half estimate on them is an artifact of the reconstruction, not data.
PARADIGMS = [
    'stroop', 'flanker', 'go_nogo', 'digit_span',
    'drm_false_memory', 'source_monitoring',
    'false_belief',
    'confidence_calibration', 'post_decision_wagering',
]


def item_accuracy(score):
    """Per-item accuracy from a stored score field (matches compute_b2_expanded)."""
    if isinstance(score, dict):
        if 'accuracy' in score:
            return float(score['accuracy'])
        if 'score' in score:
            return float(score['score'])
        if 'correct' in score:
            return 1.0 if score['correct'] else 0.0
        return 0.0
    return float(score)


def model_items(model):
    """Return {paradigm: [(task_id, accuracy), ...]} for one model, using the
    CORRECTED per-item accuracies (rescore overlays + go_nogo rerun) via
    apply_corrected_results.corrected_static_items, the same item source as
    the corrected matrix, so this artifact matches the paper's scoring canon."""
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import apply_corrected_results as ACR
    try:
        items = ACR.corrected_static_items(model)
    except FileNotFoundError:
        return None
    by_par = defaultdict(list)
    for tid, (par, acc, _score) in items.items():
        if par in PARADIGMS:
            by_par[par].append((tid, float(acc)))
    return by_par or None


def spearman_brown(r):
    return 2.0 * r / (1.0 + r) if (1.0 + r) != 0 else float('nan')


def main():
    per_model = {m: model_items(m) for m in MODELS}
    missing = [m for m in MODELS if per_model[m] is None]
    if missing:
        print("WARNING missing details.json for:", missing)

    rng = np.random.default_rng(SEED)
    result = {}
    for par in PARADIGMS:
        # Build a models x items accuracy matrix (items aligned across models by sorted task_id).
        rows, n_items = [], None
        for m in MODELS:
            bp = per_model.get(m)
            if not bp or par not in bp:
                continue
            accs = [a for _, a in sorted(bp[par], key=lambda t: t[0])]
            if n_items is None:
                n_items = len(accs)
            if len(accs) != n_items:
                continue  # skip a model with a mismatched item count
            rows.append(accs)
        if len(rows) < 3 or not n_items or n_items < 4:
            result[par] = None
            continue
        M = np.array(rows)  # [n_models x n_items]
        half = n_items // 2
        sbs = []
        for _ in range(K_SPLITS):
            perm = rng.permutation(n_items)
            a = M[:, perm[:half]].mean(axis=1)
            b = M[:, perm[half:2 * half]].mean(axis=1)
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            r, _ = pearsonr(a, b)
            sbs.append(spearman_brown(r))
        result[par] = round(float(np.mean(sbs)), 3) if sbs else None

    vals = [v for v in result.values() if v is not None]
    result_out = dict(sorted(result.items()))
    result_out["mean"] = round(float(np.mean(vals)), 3)
    result_out["mean_note"] = ("mean over the nine single-turn paradigms with real per-item "
                               "records; epitome_tom excluded because its details.json per-item "
                               "records are forced-choice-rerun placeholders, not measurements")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result_out, open(OUT, "w"), indent=2)

    print(f"Split-half (Spearman-Brown), 20 models, mean over {K_SPLITS} random splits (seed={SEED}):")
    for k in sorted(result.keys()):
        print(f"  {k:24s} {result[k]}")
    print(f"  {'mean':24s} {result_out['mean']}   (min={min(vals):.3f})")
    print(f"\nWrote {OUT}")
    print(f"Paper-claim check: mean rounds to {round(result_out['mean'],2)}; min paradigm = {min(vals):.3f}")


if __name__ == "__main__":
    main()
