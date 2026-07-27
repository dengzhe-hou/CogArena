#!/usr/bin/env python3
"""Generate all CogArena paper figures from experimental results."""

import json
import glob
import hashlib
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
# Embed TrueType (Type 42) fonts in EVERY figure this module writes; the
# per-figure PUB_RC blocks only cover fig2/fig6, which left Type 3 fonts in
# fig3/4/5. Set globally so no figure can regress.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Match AAAI's newtx body face when a figure is promoted to the main paper.
# Text is converted to vector paths at export, so the PDF remains font-free.
_NEWTX_FONT_DIR = Path.home() / '.TinyTeX/texmf-dist/fonts/opentype/public/newtx'
_NEWTX_FONT_FILES = [
    _NEWTX_FONT_DIR / 'TeXGyreTermesX-Regular.otf',
    _NEWTX_FONT_DIR / 'TeXGyreTermesX-Italic.otf',
    _NEWTX_FONT_DIR / 'TeXGyreTermesX-Bold.otf',
    _NEWTX_FONT_DIR / 'TeXGyreTermesX-BoldItalic.otf',
]
for _font_path in _NEWTX_FONT_FILES:
    if _font_path.exists():
        font_manager.fontManager.addfont(str(_font_path))
_MAIN_FIGURE_FONT = (
    'TeX Gyre TermesX'
    if all(_font_path.exists() for _font_path in _NEWTX_FONT_FILES)
    else 'DejaVu Serif'
)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [_MAIN_FIGURE_FONT, 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
})

# --- figure-compliance hook (2026-07-19, v2): every PDF save (a) uniformly
# scales EVERY matplotlib Text object by the figure's k = exported-width /
# LaTeX-insert-width so fonts land at 9pt EFFECTIVE size regardless of where
# they were authored (suptitles, panel titles, default ticks included), then
# (b) renders through an SVG with text converted to paths (svg.fonttype
# 'path') and cairosvg, so the shipped PDFs embed NO fonts at all, and (c)
# measures the exported width and records the minimum effective font size.
# verify_min_effective() hard-fails the run if any figure lands under 9.0pt.
import io as _io
import cairosvg as _cairosvg
import matplotlib.text as _mtext
plt.rcParams['svg.fonttype'] = 'path'
_CURRENT_K = 1.0
_FIG_K = {'compact': 1.22, 'signatures': 1.12, 'fig3': 2.17,
          'fig4': 1.35, 'fig5': 1.33, 'scaling': 1.24}
_CW, _TW = 3.31, 7.03
_INSERT_W = {'fig2_signatures': .94 * _TW, 'fig2_compact': _CW,
             'fig3_scaling': _TW, 'fig4_cross_system': _TW,
             'fig5_profiles': .85 * _CW, 'fig_scaling_bars': .94 * _CW}
_EFF_REPORT = {}
_orig_fig_savefig = plt.Figure.savefig
def _savefig_pathified(self, fname, **kw):
    is_pdf = isinstance(fname, str) and fname.endswith('.pdf')
    texts = [t for t in self.findobj(_mtext.Text) if t.get_text()]
    saved = [t.get_fontsize() for t in texts]
    try:
        if is_pdf and abs(_CURRENT_K - 1.0) > 1e-9:
            for t in texts:
                t.set_fontsize(t.get_fontsize() * _CURRENT_K)
        if is_pdf:
            kw.pop('format', None)
            buf = _io.BytesIO()
            _orig_fig_savefig(self, buf, format='svg', **kw)
            _cairosvg.svg2pdf(bytestring=buf.getvalue(), write_to=fname)
            base = os.path.basename(fname)[:-4]
            if base in _INSERT_W and texts:
                try:
                    import fitz
                    w_in = fitz.open(fname)[0].rect.width / 72.0
                    scale = _INSERT_W[base] / w_in
                    min_eff = min(t.get_fontsize() for t in texts) * scale
                    _EFF_REPORT[base] = round(min_eff, 2)
                except Exception as e:
                    _EFF_REPORT[base] = f'measure-failed: {e}'
        else:
            _orig_fig_savefig(self, fname, **kw)
    finally:
        for t, fs in zip(texts, saved):
            t.set_fontsize(fs)
plt.Figure.savefig = _savefig_pathified

def verify_min_effective(floor=9.0, overrides=None):
    # overrides: per-figure floors (basename -> pt) for figures whose small
    # text is deliberately authored below the global floor.
    floors = dict(overrides or {})
    bad = {k: v for k, v in _EFF_REPORT.items()
           if not isinstance(v, float) or v < floors.get(k, floor)}
    print('effective font sizes:', _EFF_REPORT)
    if bad:
        raise SystemExit(f'FIGURE SIZE GATE FAILED (floor {floor}pt): {bad}')




import matplotlib.patches as mpatches
from collections import defaultdict
from scipy import stats as scipy_stats

# ── Config ──
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../results/full_eval_20260526_2208')
OUT_DIR = os.path.dirname(__file__)

SIZE_MAP = {
    'tinyllama:1.1b': 1.1, 'qwen2.5:0.5b': 0.5, 'qwen2.5:1.5b': 1.5,
    'gemma2:2b': 2, 'llama3.2:1b': 1, 'qwen2.5:3b': 3, 'llama3.2:3b': 3,
    'qwen2.5:7b': 7, 'mistral:7b': 7, 'llama3.1:8b': 8, 'deepseek-r1:7b': 7,
    'gemma2:9b': 9, 'qwen2.5:14b': 14, 'phi3:14b': 14, 'deepseek-r1:14b': 14,
    'gemma2:27b': 27, 'qwen2.5:32b': 32, 'mixtral:8x7b': 47, 'yi:34b': 34,
    'command-r:35b': 35
}

FAMILY_MAP = {
    'qwen2.5:0.5b': 'Qwen2.5', 'qwen2.5:1.5b': 'Qwen2.5', 'qwen2.5:3b': 'Qwen2.5',
    'qwen2.5:7b': 'Qwen2.5', 'qwen2.5:14b': 'Qwen2.5', 'qwen2.5:32b': 'Qwen2.5',
    'gemma2:2b': 'Gemma2', 'gemma2:9b': 'Gemma2', 'gemma2:27b': 'Gemma2',
    'llama3.2:1b': 'Llama3', 'llama3.2:3b': 'Llama3', 'llama3.1:8b': 'Llama3',
    'deepseek-r1:7b': 'DeepSeek-R1', 'deepseek-r1:14b': 'DeepSeek-R1',
    'mistral:7b': 'Mistral', 'mixtral:8x7b': 'Mistral',
    'phi3:14b': 'Phi3', 'yi:34b': 'Yi', 'command-r:35b': 'Command-R',
    'tinyllama:1.1b': 'TinyLlama'
}

FAMILY_COLORS = {
    'Qwen2.5': '#1f77b4', 'Gemma2': '#ff7f0e', 'Llama3': '#2ca02c',
    'DeepSeek-R1': '#d62728', 'Mistral': '#9467bd', 'Phi3': '#8c564b',
    'Yi': '#e377c2', 'Command-R': '#7f7f7f', 'TinyLlama': '#bcbd22'
}

DOMAIN_MAP = {
    'digit_span': 'Working Memory', 'n_back': 'Working Memory', 'operation_span': 'Working Memory',
    'stroop': 'Cognitive Control', 'flanker': 'Cognitive Control', 'go_nogo': 'Cognitive Control',
    'drm_false_memory': 'Episodic Memory', 'source_monitoring': 'Episodic Memory', 'cvlt_word_list': 'Episodic Memory',
    'false_belief': 'Theory of Mind', 'epitome_tom': 'Theory of Mind',
    'confidence_calibration': 'Metacognition', 'post_decision_wagering': 'Metacognition'
}

PARADIGM_LABELS = {
    'digit_span': 'Digit Span', 'n_back': 'N-Back', 'operation_span': 'Op. Span',
    'stroop': 'Stroop', 'flanker': 'Flanker',
    'go_nogo': 'Go/No-Go', 'drm_false_memory': 'DRM', 'source_monitoring': 'Source Mon.',
    'cvlt_word_list': 'CVLT',
    'false_belief': 'False Belief', 'epitome_tom': 'EPITOME',
    'confidence_calibration': 'Conf. Calib.', 'post_decision_wagering': 'Wagering'
}

# Multi-turn results directory
MULTITURN_DIR = os.path.join(os.path.dirname(__file__), '../../results/multiturn_eval_v3')

DOMAIN_COLORS = {
    'Working Memory': '#1f77b4', 'Cognitive Control': '#ff7f0e',
    'Episodic Memory': '#2ca02c', 'Theory of Mind': '#d62728',
    'Metacognition': '#9467bd'
}

# ── Publication style shared by the compact and scaling-summary figures.
#    TeX Gyre TermesX matches the AAAI template's NewTX body face. ──
PUB_RC = {
    'font.family': 'serif',
    'font.serif': [_MAIN_FIGURE_FONT, 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'pdf.fonttype': 42,
    'svg.fonttype': 'path',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
}
SIG_DARK, SIG_LIGHT = '#31629D', '#A9C4DE'   # expected-higher vs expected-lower condition
SIG_SEQ = ['#C9D9EA', '#7FA6C9', '#31629D']  # easy -> hard difficulty sweep
SIG_UNIF = '#6C93BC'                         # uniform multi-category bars
ACCENT_MUTED = '#8C8C8C'                     # near-zero scalers named in the caption, muted gray


def _cc_corrected():
    """Corrected confidence_calibration accuracy from the fixed metacognition
    scorer; empty until results/reanalysis/conf_cal_corrected.json exists."""
    p = os.path.join(os.path.dirname(__file__), '../../results/reanalysis/conf_cal_corrected.json')
    return json.load(open(p)) if os.path.exists(p) else {}


CORR_MATRIX = (os.environ.get('COGARENA_PRIMARY_MATRIX')
               or os.path.join(os.path.dirname(__file__),
                               '../../results/recompute_20260703/corrected_matrix.csv'))
RESCORE_DIR = os.path.join(os.path.dirname(__file__),
                           '../../results/rescore_20260702/new_scores')
GONOGO_DIR = os.path.join(os.path.dirname(__file__),
                          '../../results/gonogo_rerun_20260702')

_MT_PARADIGMS = {'n_back', 'operation_span', 'cvlt_word_list'}


def _corrected_matrix():
    """model -> paradigm -> corrected accuracy (all scorer fixes applied)."""
    import csv
    rows = list(csv.reader(open(CORR_MATRIX)))
    hdr = rows[0][1:]
    return {r[0]: {p: float(v) for p, v in zip(hdr, r[1:])} for r in rows[1:]}


def load_text_results():
    """Corrected per-paradigm accuracy for the 20 primary text models."""
    mat = _corrected_matrix()
    models = {}
    for agg in sorted(glob.glob(f'{RESULTS_DIR}/*/text/aggregate.json')):
        model = json.load(open(agg))['model'].replace('openai/', '')
        if model in mat:
            models[model] = {p: {'accuracy': v} for p, v in mat[model].items()
                             if p not in _MT_PARADIGMS}
    return models


def load_multiturn_results():
    """Corrected multi-turn accuracies (n_back, operation_span, cvlt_word_list)."""
    mat = _corrected_matrix()
    models = {}
    for agg in sorted(glob.glob(f'{MULTITURN_DIR}/*/aggregate.json')):
        model = json.load(open(agg))['model'].replace('openai/', '')
        if model in mat:
            models[model] = {p: {'accuracy': mat[model][p]} for p in _MT_PARADIGMS}
    return models


def merge_text_multiturn(text_data, multiturn_data):
    """Merge multi-turn paradigm results into text_data."""
    merged = {}
    for model in set(list(text_data.keys()) + list(multiturn_data.keys())):
        merged[model] = {}
        if model in text_data:
            merged[model].update(text_data[model])
        if model in multiturn_data:
            for p in ['n_back', 'operation_span', 'cvlt_word_list']:
                if p in multiturn_data[model]:
                    merged[model][p] = multiturn_data[model][p]
    return merged


def _score_simple_legacy(expected, response):
    """Mirror of the fixed scripts/run_unified.py score_simple (empty guard)."""
    if not expected:
        return False
    exp = str(expected).strip().lower().strip('."\'!?,;:')
    resp = str(response).strip().lower().strip('."\'!?,;:')
    for art in ['the ', 'a ', 'an ']:
        if exp.startswith(art):
            exp = exp[len(art):]
        if resp.startswith(art):
            resp = resp[len(art):]
    if not exp or not resp:
        return False
    return exp == resp or exp in resp or resp in exp


def load_image_results():
    """Load VLM results with an optional fail-closed remediation adapter.

    ``COGARENA_VLM_RERUN`` selects the frozen 2026-07-24 rerun.  Without it,
    the legacy responses are replayed with the empty-safe scorer so historical
    artifacts remain reproducible.
    """
    rerun_root = os.environ.get('COGARENA_VLM_RERUN')
    if rerun_root:
        manifest_path = os.path.join(rerun_root, 'VLM_RERUN_MANIFEST.json')
        manifest = json.load(open(manifest_path))
        if (manifest.get('status') != 'final'
                or manifest.get('record_count') != 1500
                or manifest.get('scorer_id') != 'paradigm-label-parser-v1'
                or manifest.get('stimulus_design_id') != 'balanced_montage_v2'):
            raise RuntimeError(f'VLM rerun is not final: {manifest_path}')
        summary_rel = 'VLM_RERUN_SUMMARY.json'
        expected_sha = manifest.get('outputs', {}).get(summary_rel)
        summary_path = os.path.join(rerun_root, summary_rel)
        actual_sha = hashlib.sha256(open(summary_path, 'rb').read()).hexdigest()
        if not expected_sha or actual_sha != expected_sha:
            raise RuntimeError('VLM summary is not bound to the final manifest')
        summary = json.load(open(summary_path))
        if (summary.get('record_count') != 1500
                or summary.get('scorer_id') != manifest.get('scorer_id')
                or summary.get('scoring_contract_sha256')
                != manifest.get('scoring_contract', {}).get('sha256')):
            raise RuntimeError('VLM summary provenance mismatch')
        if len(summary.get('models', {})) != 6:
            raise RuntimeError('VLM summary does not contain six models')
        models = {
            model_id.replace('openai/', ''): {
                paradigm: {'accuracy': float(stats['accuracy'])}
                for paradigm, stats in model_summary['paradigms'].items()
            }
            for model_id, model_summary in summary['models'].items()
        }
        return models

    # Historical compatibility path.
    models = {}
    for dfile in sorted(glob.glob(f'{RESULTS_DIR}/*/image/details.json')):
        model = dfile.split('/')[-3].replace('openai_', '')
        by = {}
        for r in json.load(open(dfile)):
            ok = _score_simple_legacy(r.get('expected', ''), r.get('response', ''))
            by.setdefault(r['paradigm'], []).append(ok)
        models[model] = {p: {'accuracy': float(np.mean(v))} for p, v in by.items()}
    return models


def load_agent_results():
    """Load agent results."""
    models = {}
    for agg in sorted(glob.glob(f'{RESULTS_DIR}/*/agent/aggregate.json')):
        d = json.load(open(agg))
        model = d['model'].replace('openai/', '')
        models[model] = d['paradigms']
    return models


# ═══════════════════════════════════════════════════════
# Figure 2: Behavioral Signatures
# ═══════════════════════════════════════════════════════
def load_all_details():
    """Per-item details for the 20 text models with corrected scores applied:
    rescore overlays (results/rescore_20260702) update score.accuracy/.correct,
    go_nogo items are replaced by the fixed-prompt rerun, and when
    COGARENA_SM_OVERLAY is set every source_monitoring item takes the final
    55x50 corrected+rerun overlay; COGARENA_WAGER_OVERLAY analogously applies
    the full-pool scorer replay for wagering (see build_and_recompute.py)."""
    sm_overlay = (json.load(open(os.environ["COGARENA_SM_OVERLAY"]))
                  if os.environ.get("COGARENA_SM_OVERLAY") else None)
    wager_overlay = (json.load(open(os.environ["COGARENA_WAGER_OVERLAY"]))
                     if os.environ.get("COGARENA_WAGER_OVERLAY") else None)
    all_items = []
    for dfile in sorted(glob.glob(f'{RESULTS_DIR}/*/text/details.json')):
        model = dfile.split('/')[-3].replace('openai_', '')
        ov_path = os.path.join(RESCORE_DIR, f'full_eval_20260526_2208__openai_{model}.json')
        ov = json.load(open(ov_path)) if os.path.exists(ov_path) else {}
        if sm_overlay is not None:
            ov = dict(ov)
            for tid, v in sm_overlay[model].items():
                ov[tid] = v
        if wager_overlay is not None:
            ov = dict(ov)
            for tid, v in wager_overlay[model].items():
                ov[tid] = v
        items = json.load(open(dfile))
        keep = []
        for item in items:
            if item.get('paradigm') == 'go_nogo':
                continue  # replaced by rerun below
            if item['task_id'] in ov:
                v = float(ov[item['task_id']])
                sc = item.get('score') or {}
                if 'accuracy' in sc:
                    sc['accuracy'] = v
                if 'correct' in sc:
                    sc['correct'] = v >= 0.5
                item['score'] = sc
            item['_model'] = model
            keep.append(item)
        gg = os.path.join(GONOGO_DIR, f'openai_{model}', 'text', 'details.json')
        if os.path.exists(gg):
            for item in json.load(open(gg)):
                item['_model'] = model
                keep.append(item)
        all_items.extend(keep)
    return all_items


def fig2_behavioral_signatures(text_data):
    global _CURRENT_K; _CURRENT_K = _FIG_K['signatures']
    """Panel of behavioral signature effects — ALL values computed from details.json."""
    all_items = load_all_details()
    models = sorted(set(i['_model'] for i in all_items))

    signature_rc = dict(PUB_RC)
    signature_rc.update({
        'font.family': 'serif',
        'font.serif': [_MAIN_FIGURE_FONT, 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'stix',
    })
    _saved_rc = {k: plt.rcParams[k] for k in signature_rc}
    plt.rcParams.update(signature_rc)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.62))

    def _bar(ax, labels, vals, colors, title, ylim_lo=0, errs=None):
        bars = ax.bar(labels, vals, color=colors, width=0.55,
                      yerr=errs, capsize=2, error_kw={'linewidth': 0.7, 'ecolor': '#444444'})
        ax.set_ylim(ylim_lo, 1.05)
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        for k, b in enumerate(bars):
            off = (errs[k] if errs is not None else 0) + 0.02
            ax.text(b.get_x() + b.get_width()/2, min(b.get_height() + off, 1.01),
                    f'{b.get_height():.1%}', ha='center', fontsize=9)

    def _sem(per_model_vals):
        v = np.asarray(per_model_vals, dtype=float)
        return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

    # BH-corrected per-model replication p-values from the committed artifact
    _sig_path = os.path.join(os.path.dirname(__file__),
                             '../../results/reanalysis/signature_significance.json')
    _sig = {b['paradigm']: b for b in json.load(open(_sig_path))['paradigms']}

    def _fmt_p(p):
        if p < 0.001:
            return '$<$.001'
        if p < 0.01:
            return f'={p:.3f}'.replace('0.', '.')
        return f'={p:.2f}'.replace('0.', '.')

    def _cond_panel(ax, par, cond_of, labels, colors, title_stem, ylim_lo=0, acc_of=None):
        """Two-condition panel with per-model means, SEM bars, BH p in title."""
        items = [i for i in all_items if i['paradigm'] == par]
        acc = acc_of or (lambda i: 1.0 if i['score'].get('correct') else 0.0)
        easy_pm, hard_pm = [], []
        for m in models:
            e = [acc(i) for i in items if i['_model'] == m and cond_of(i) == 'easy']
            h = [acc(i) for i in items if i['_model'] == m and cond_of(i) == 'hard']
            if e and h:
                easy_pm.append(np.mean(e)); hard_pm.append(np.mean(h))
        n_eff = sum(1 for e, h in zip(easy_pm, hard_pm) if e > h)
        blk = _sig.get(par, {})
        p_bh = blk.get('p_binom_bh', blk.get('p_binom_onesided', 1.0))
        _bar(ax, labels, [np.mean(easy_pm), np.mean(hard_pm)], colors,
             f'{title_stem}\n({n_eff}/{len(easy_pm)} models, $p_{{BH}}${_fmt_p(p_bh)})',
             ylim_lo, errs=[_sem(easy_pm), _sem(hard_pm)])

    # ── (a) Flanker: congruent vs incongruent (per-model means ± SEM);
    #        panel order follows the caption (Flanker, Stroop, False Belief) ──
    _cond_panel(axes[0, 0], 'flanker',
                lambda i: {'congruent': 'easy', 'incongruent': 'hard'}.get(i['score'].get('condition')),
                ['Congruent', 'Incongruent'], [SIG_DARK, SIG_LIGHT], 'Flanker Effect')

    # ── (b) Stroop (per-model means ± SEM; full 0--1 axis, no truncation) ──
    _cond_panel(axes[0, 1], 'stroop',
                lambda i: {'congruent': 'easy', 'incongruent': 'hard'}.get(i['score'].get('condition')),
                ['Congruent', 'Incongruent'], [SIG_DARK, SIG_LIGHT], 'Stroop Effect')

    # ── (c) False Belief order (per-model means ± SEM) ──
    _cond_panel(axes[0, 2], 'false_belief',
                lambda i: {1.0: 'easy', 2.0: 'hard'}.get(i['score'].get('order')),
                ['1st Order', '2nd Order'], [SIG_DARK, SIG_LIGHT], 'False Belief Order',
                acc_of=lambda i: i['score'].get('accuracy', 0.0))

    # ── (d) EPITOME sub-capacities: 35-model expansion pool (corrected parser;
    #        the 20-model pool's per-item records come from the forced-choice
    #        rerun and do not carry responses). Sub-capacity is in the task_id. ──
    exp_dir = os.path.join(os.path.dirname(__file__), '../../results/full_eval_expansion')
    ep_groups = {'Belief': [], 'Desire': [], 'Intention': [], 'Emotion': []}
    for dfile in sorted(glob.glob(f'{exp_dir}/*/text/details.json')):
        emodel = dfile.split('/')[-3].replace('openai_', '')
        ov_path = os.path.join(RESCORE_DIR, f'full_eval_expansion__openai_{emodel}.json')
        ov = json.load(open(ov_path)) if os.path.exists(ov_path) else {}
        per = {'belief': [], 'desire': [], 'intention': [], 'emotion': []}
        for r in json.load(open(dfile)):
            if r.get('paradigm') != 'epitome_tom':
                continue
            acc = float(ov.get(r['task_id'], (r.get('score') or {}).get('accuracy', 0.0)))
            for sub in per:
                if f'_{sub}_' in r['task_id']:
                    per[sub].append(acc)
                    break
        for sub, v in per.items():
            if v:
                ep_groups[sub.capitalize()].append(np.mean(v))
    ep_vals = [np.mean(v) if v else 0 for v in ep_groups.values()]
    ep_errs = [_sem(v) if v else 0 for v in ep_groups.values()]
    ep_blk = _sig.get('epitome', {})
    _bar(axes[1, 0], ['Belief', 'Desire', 'Intent.', 'Emot.'], ep_vals,
         [SIG_UNIF] * 4,
         f'EPITOME sub-capacities (35 models)\n'
         f'(desire$>$belief {ep_blk.get("fraction", "25/35")}, '
         f'$p${_fmt_p(ep_blk.get("p_binom_onesided", 0.008))})', errs=ep_errs)

    # ── (e) Source Monitoring difficulty sweep (Qwen2.5-7B, corrected items) ──
    sm_items = [i for i in all_items if i['paradigm'] == 'source_monitoring'
                and i['_model'] == 'qwen2.5:7b']
    sm_labels = ['Easy', 'Medium', 'Hard']
    sm_by_level = [[i['score'].get('accuracy', 0.0) for i in sm_items
                    if i.get('difficulty') == lab.lower()] for lab in sm_labels]
    sm_vals = [np.mean(v) for v in sm_by_level]
    sm_errs = [_sem(v) for v in sm_by_level]  # SEM across items (single-model sweep)
    axes[1, 0].tick_params(axis='x', labelrotation=20)
    _bar(axes[1, 1], sm_labels, sm_vals,
         SIG_SEQ,
         'Source Monitoring\n(qwen2.5:7b; item SEM)', errs=sm_errs)

    # (f) DRM false memory: critical-lure vs unrelated false alarms
    drm_items = [i for i in all_items if i['paradigm'] == 'drm_false_memory']
    crit_rates, unrel_rates = [], []
    for m in models:
        md = [i for i in drm_items if i['_model'] == m]
        clt = sum(i['score'].get('critical_lure_total', 0) for i in md)
        clf = sum(i['score'].get('critical_lure_false_alarms', 0) for i in md)
        ut = sum(i['score'].get('unrelated_total', 0) for i in md)
        uf = sum(i['score'].get('unrelated_false_alarms', 0) for i in md)
        if clt and ut:
            crit_rates.append(clf / clt); unrel_rates.append(uf / ut)
    crit_m = float(np.mean(crit_rates)); unrel_m = float(np.mean(unrel_rates))
    n_dr = sum(1 for c, u in zip(crit_rates, unrel_rates) if c > u)
    drm_p = _sig.get('drm_false_memory', {}).get('p_binom_bh', 0.001)
    _bar(axes[1, 2], ['Critical Lure', 'Unrelated'], [crit_m, unrel_m],
         [SIG_DARK, SIG_LIGHT],
         f'DRM False Memory\n({n_dr}/{len(crit_rates)} models, $p_{{BH}}$ {_fmt_p(drm_p)})',
         errs=[_sem(crit_rates), _sem(unrel_rates)])
    axes[1, 2].set_ylabel('False-recognition rate')

    plt.tight_layout(pad=0.4, h_pad=1.6)
    plt.savefig(f'{OUT_DIR}/fig2_signatures.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUT_DIR}/fig2_signatures.svg', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig2_signatures.png', bbox_inches='tight', dpi=150)
    plt.close()
    plt.rcParams.update(_saved_rc)
    print('  ✓ Figure 2: Behavioral signatures (data-driven from result files)')


def fig2_compact(_unused=None):
    global _CURRENT_K; _CURRENT_K = _FIG_K['compact']
    """Two-panel signature figure for the main text: Flanker contrast and DRM
    false recognition. Mirrors panels (a) and (f) of fig2_behavioral_signatures
    (same data, same statistics); the full six-panel figure ships in the
    supplementary material."""
    all_items = load_all_details()
    models = sorted(set(i['_model'] for i in all_items))

    _saved_rc = {k: plt.rcParams[k] for k in PUB_RC}
    plt.rcParams.update(PUB_RC)
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.3))

    def _sem(v):
        v = np.asarray(v, dtype=float)
        return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

    _sig_path = os.path.join(os.path.dirname(__file__),
                             '../../results/reanalysis/signature_significance.json')
    _sig = {b['paradigm']: b for b in json.load(open(_sig_path))['paradigms']}

    def _fmt_p(p):
        if p < 0.001:
            return '$<$.001'
        if p < 0.01:
            return f'={p:.3f}'.replace('0.', '.')
        return f'={p:.2f}'.replace('0.', '.')

    def _bar(ax, labels, vals, title, errs):
        bars = ax.bar(labels, vals, color=[SIG_DARK, SIG_LIGHT], width=0.55,
                      yerr=errs, capsize=2, error_kw={'linewidth': 0.7, 'ecolor': '#444444'})
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        for k, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width()/2, min(b.get_height() + errs[k] + 0.02, 1.01),
                    f'{b.get_height():.1%}', ha='center', fontsize=9)

    fl_items = [i for i in all_items if i['paradigm'] == 'flanker']
    easy_pm, hard_pm = [], []
    for m in models:
        e = [1.0 if i['score'].get('correct') else 0.0 for i in fl_items
             if i['_model'] == m and i['score'].get('condition') == 'congruent']
        h = [1.0 if i['score'].get('correct') else 0.0 for i in fl_items
             if i['_model'] == m and i['score'].get('condition') == 'incongruent']
        if e and h:
            easy_pm.append(np.mean(e)); hard_pm.append(np.mean(h))
    n_fl = sum(1 for e, h in zip(easy_pm, hard_pm) if e > h)
    p_fl = _sig['flanker'].get('p_binom_bh', 1.0)
    _bar(axes[0], ['Congr.', 'Incongr.'], [np.mean(easy_pm), np.mean(hard_pm)],
         f'Flanker Effect\n({n_fl}/{len(easy_pm)}, $p_{{BH}}${_fmt_p(p_fl)})',
         [_sem(easy_pm), _sem(hard_pm)])
    axes[0].set_ylabel('Accuracy')

    drm_items = [i for i in all_items if i['paradigm'] == 'drm_false_memory']
    crit_rates, unrel_rates = [], []
    for m in models:
        md = [i for i in drm_items if i['_model'] == m]
        clt = sum(i['score'].get('critical_lure_total', 0) for i in md)
        clf = sum(i['score'].get('critical_lure_false_alarms', 0) for i in md)
        ut = sum(i['score'].get('unrelated_total', 0) for i in md)
        uf = sum(i['score'].get('unrelated_false_alarms', 0) for i in md)
        if clt and ut:
            crit_rates.append(clf / clt); unrel_rates.append(uf / ut)
    n_dr = sum(1 for c, u in zip(crit_rates, unrel_rates) if c > u)
    p_dr = _sig['drm_false_memory'].get('p_binom_bh', 0.001)
    _bar(axes[1], ['Crit.\nLure', 'Unrel.'], [np.mean(crit_rates), np.mean(unrel_rates)],
         f'DRM False Memory\n({n_dr}/{len(crit_rates)}, $p_{{BH}}$ {_fmt_p(p_dr)})',
         [_sem(crit_rates), _sem(unrel_rates)])
    axes[1].set_ylabel('False-recog. rate')

    plt.tight_layout(pad=0.6)
    fig.subplots_adjust(wspace=0.75)
    plt.savefig(f'{OUT_DIR}/fig2_compact.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUT_DIR}/fig2_compact.svg', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig2_compact.png', bbox_inches='tight', dpi=150)
    plt.close()
    plt.rcParams.update(_saved_rc)
    print('  ✓ Figure 2-compact: Two-panel signatures (main text)')


# ═══════════════════════════════════════════════════════
# Figure 3: Scaling Patterns
# ═══════════════════════════════════════════════════════
def fig3_scaling(text_data):
    global _CURRENT_K; _CURRENT_K = _FIG_K['fig3']
    """Accuracy vs model size per paradigm, colored by family."""
    paradigms_ordered = ['epitome_tom', 'stroop', 'false_belief', 'confidence_calibration',
                         'post_decision_wagering', 'drm_false_memory', 'digit_span',
                         'flanker', 'operation_span', 'go_nogo', 'source_monitoring',
                         'n_back', 'cvlt_word_list']

    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 7.2))
    fig.suptitle('Per-paradigm scaling', fontsize=12, fontweight='bold')

    # Hide unused subplots
    for idx in range(len(paradigms_ordered), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    for idx, paradigm in enumerate(paradigms_ordered):
        ax = axes[idx // ncols, idx % ncols]
        sizes, accs, colors = [], [], []
        for model, pdata in text_data.items():
            if paradigm in pdata and model in SIZE_MAP:
                s = SIZE_MAP[model]
                a = pdata[paradigm]['accuracy']
                sizes.append(s)
                accs.append(a)
                fam = FAMILY_MAP.get(model, 'Other')
                colors.append(FAMILY_COLORS.get(fam, '#333333'))

        if sizes:
            ax.scatter(sizes, accs, c=colors, s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
            # Fit line
            log_sizes = np.log(sizes)
            z = np.polyfit(log_sizes, accs, 1)
            r, p_r = scipy_stats.pearsonr(log_sizes, accs)
            xs = np.linspace(min(sizes), max(sizes), 50)
            ys = np.polyval(z, np.log(xs))
            ax.plot(xs, ys, 'k--', alpha=0.4, linewidth=1)
            star = '*' if p_r < 0.05 else ''
            ax.text(0.95, 0.05, f'r={r:.2f}{star}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        ax.set_xscale('log')
        ax.set_xlim(0.3, 60)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(PARADIGM_LABELS.get(paradigm, paradigm), fontsize=9)
        if idx // ncols == nrows - 1 or idx + ncols >= len(paradigms_ordered):
            ax.set_xlabel('Parameters (B)', fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel('Accuracy', fontsize=9)
        ax.tick_params(labelsize=9)
        # shared-axis behavior: tick labels only on the leftmost column and on
        # the bottom-most panel of each column, so scaled-up labels cannot
        # spill into a neighboring panel
        if idx % ncols != 0:
            ax.tick_params(labelleft=False)
        if not (idx // ncols == nrows - 1 or idx + ncols >= len(paradigms_ordered)):
            ax.tick_params(labelbottom=False)

    # Legend
    legend_handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    # Use the two intentionally empty panels instead of widening the tight
    # bounding box with an external legend.
    fig.legend(handles=legend_handles, loc='center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.80, 0.17), frameon=False,
               columnspacing=0.8, handlelength=1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.94], h_pad=0.9, w_pad=0.6)
    plt.savefig(f'{OUT_DIR}/fig3_scaling.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUT_DIR}/fig3_scaling.svg', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig3_scaling.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('  ✓ Figure 3: Scaling patterns')


# ═══════════════════════════════════════════════════════
# Figure 4: Cross-System Comparison (Text vs VLM vs Agent)
# ═══════════════════════════════════════════════════════
def fig4_cross_system(text_data, image_data, agent_data):
    global _CURRENT_K; _CURRENT_K = _FIG_K['fig4']
    """Unpaired descriptive comparison of text and VLM model pools.

    Individual checkpoints are shown so the figure cannot be mistaken for a
    paired modality effect. Agent is excluded because it only shares false
    belief; see the appendix.
    """
    cross_system_rc = dict(PUB_RC)
    cross_system_rc.update({
        'font.family': 'serif',
        'font.serif': [_MAIN_FIGURE_FONT, 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'stix',
    })
    _saved_rc = {k: plt.rcParams[k] for k in cross_system_rc}
    plt.rcParams.update(cross_system_rc)
    fig, axes = plt.subplots(1, 3, figsize=(8.7, 3.25), sharey=True)
    fig.suptitle('Cross-modal adaptation check', fontsize=12)

    shared = ['stroop', 'flanker', 'false_belief']
    labels = ['Stroop', 'Flanker', 'False Belief']

    for i, (paradigm, label) in enumerate(zip(shared, labels)):
        ax = axes[i]
        text_accs = [p[paradigm]['accuracy'] for m, p in text_data.items() if paradigm in p]
        vlm_accs = [p[paradigm]['accuracy'] for m, p in image_data.items() if paradigm in p]
        pools = [text_accs, vlm_accs]
        mode_labels = [f'Text\n(n={len(text_accs)})', f'VLM\n(n={len(vlm_accs)})']
        mode_colors = ['#356A9A', '#C46A2A']

        rng = np.random.default_rng(100 + i)
        for x, (values, color) in enumerate(zip(pools, mode_colors)):
            values = np.asarray(values, dtype=float)
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            ax.scatter(np.full(len(values), x) + jitter, values, s=23,
                       color=color, alpha=0.58, edgecolors='white',
                       linewidth=0.35, zorder=2)
            mean = float(np.mean(values))
            ax.plot([x - 0.17, x + 0.17], [mean, mean], color='#111111',
                    linewidth=2.0, zorder=3)
            ax.scatter([x], [mean], marker='D', s=30, color='#111111',
                       edgecolors='white', linewidth=0.45, zorder=4)
        ax.set_xticks([0, 1], mode_labels)
        ax.set_xlim(-0.42, 1.42)
        ax.set_ylim(-0.04, 1.04)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('Accuracy' if i == 0 else '')
        ax.grid(axis='y', color='#E1E1E1', linewidth=0.6)
        ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.91], w_pad=1.0)
    plt.savefig(f'{OUT_DIR}/fig4_cross_system.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUT_DIR}/fig4_cross_system.svg', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig4_cross_system.png', bbox_inches='tight', dpi=150)
    plt.close()
    plt.rcParams.update(_saved_rc)
    print('  ✓ Figure 4: Cross-system comparison (text vs VLM)')


# ═══════════════════════════════════════════════════════
# Figure 5: Cognitive Profiles (Radar)
# ═══════════════════════════════════════════════════════
def fig5_profiles(text_data):
    global _CURRENT_K; _CURRENT_K = _FIG_K['fig5']
    """Compact heatmap for representative within-family score profiles."""
    domains = ['Working Memory', 'Cognitive Control', 'Episodic Memory',
               'Theory of Mind', 'Metacognition']
    domain_paradigms = {
        'Working Memory': ['digit_span', 'n_back', 'operation_span'],
        'Cognitive Control': ['stroop', 'flanker', 'go_nogo'],
        'Episodic Memory': ['drm_false_memory', 'source_monitoring', 'cvlt_word_list'],
        'Theory of Mind': ['false_belief', 'epitome_tom'],
        'Metacognition': ['confidence_calibration', 'post_decision_wagering']
    }

    rep_models = ['qwen2.5:0.5b', 'qwen2.5:3b', 'qwen2.5:14b', 'qwen2.5:32b']
    model_labels = ['0.5B', '3B', '14B', '32B']  # family named in the title
    rows, present_labels = [], []
    for model, label in zip(rep_models, model_labels):
        if model not in text_data:
            continue
        pdata = text_data[model]
        rows.append([
            np.mean([pdata[p]['accuracy'] for p in domain_paradigms[domain]
                     if p in pdata])
            for domain in domains
        ])
        present_labels.append(label)

    values = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.25))
    image = ax.imshow(values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    _short = {'Working Memory': 'WM', 'Cognitive Control': 'Ctrl',
              'Episodic Memory': 'Epi', 'Theory of Mind': 'ToM',
              'Metacognition': 'Meta'}
    ax.set_xticks(range(len(domains)),
                  [_short.get(domain, domain) for domain in domains])
    ax.set_yticks(range(len(present_labels)), present_labels)
    ax.tick_params(length=0)
    ax.set_title('Qwen2.5 grouping scores', fontsize=11, fontweight='bold')
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(col, row, f'{100 * value:.0f}',
                    ha='center', va='center',
                    color='white' if value >= 0.58 else '#111111',
                    fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_ticks([0, 0.5, 1.0], labels=['0', '50', '100'])
    cbar.set_label('Accuracy (%)', fontsize=9)
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout(pad=0.4)
    plt.savefig(f'{OUT_DIR}/fig5_profiles.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUT_DIR}/fig5_profiles.svg', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig5_profiles.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('  ✓ Figure 5: Cognitive profiles')


# ═══════════════════════════════════════════════════════
# Figure 6: Scaling Correlation Heatmap
# ═══════════════════════════════════════════════════════
def fig_scaling_bars(text_data):
    global _CURRENT_K; _CURRENT_K = _FIG_K['scaling']
    """Horizontal bar chart of per-paradigm scaling correlations."""
    paradigms_ordered = ['epitome_tom', 'stroop', 'false_belief', 'confidence_calibration',
                         'post_decision_wagering', 'drm_false_memory', 'digit_span',
                         'flanker', 'operation_span', 'go_nogo', 'source_monitoring',
                         'n_back', 'cvlt_word_list']

    correlations = []
    pvalues = []
    labels = []
    for p in paradigms_ordered:
        sizes, accs = [], []
        for model, pdata in text_data.items():
            if p in pdata and model in SIZE_MAP:
                sizes.append(SIZE_MAP[model])
                accs.append(pdata[p]['accuracy'])
        if len(sizes) >= 3:
            r, p_r = scipy_stats.pearsonr(np.log(sizes), accs)
        else:
            r, p_r = 0, 1.0
        correlations.append(r)
        pvalues.append(p_r)
        labels.append(PARADIGM_LABELS.get(p, p))

    # sort paradigms by scaling correlation (descending) so the heatmap reads green->yellow
    _order = sorted(range(len(correlations)), key=lambda i: correlations[i], reverse=True)
    correlations = [correlations[i] for i in _order]
    pvalues = [pvalues[i] for i in _order]
    labels = [labels[i] for i in _order]

    # Horizontal bars sorted by r, with a visual gap between the significant
    # block and the n.s. tail; '*' marks p<.05. Tick labels and bar
    # annotations are authored compactly, then uniformly scaled at export to
    # clear the same 9 pt effective floor as the other paper figures.
    _saved_rc = {k: plt.rcParams[k] for k in PUB_RC}
    plt.rcParams.update(PUB_RC)
    fig, ax = plt.subplots(figsize=(3.35, 2.28))

    n = len(correlations)
    n_sig = sum(1 for p_r in pvalues if p_r < 0.05)  # sorted desc, so sig block is first
    gap = 0.65                                       # visual break before the n.s. tail
    ypos = [(n - 1 - i) + (gap if i < n_sig else 0.0) for i in range(n)]
    # Gray marks every bar that does not reach p<.05, so color, the '*'
    # marks, and the layout gap all encode the same significance boundary
    # (2026-07-23; previously gray = near-zero multi-turn memory paradigms,
    # which left Op. Span blue despite being nonsignificant).
    accent = {lab for lab, p_r in zip(labels, pvalues) if p_r >= 0.05}
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='#E8E8E8', linewidth=0.5)
    for y, r, p_r, lab in zip(ypos, correlations, pvalues, labels):
        color = ACCENT_MUTED if lab in accent else SIG_DARK
        sig = p_r < 0.05
        ax.barh(y, r, height=0.62, color=color, edgecolor='none', zorder=2)
        ax.text(r + 0.02, y, f'{r:.2f}' + ('*' if sig else ''),
                va='center', fontsize=8.4, color='#222222')

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.4)
    for tick, lab in zip(ax.get_yticklabels(), labels):
        if lab in accent:
            tick.set_color(ACCENT_MUTED)
    ax.set_xlim(0, 0.88)
    ax.set_ylim(-0.7, n - 0.3 + gap)
    ax.set_xlabel('Pearson $r$ (acc. vs. log params)')
    ax.text(0.96, 0.03, '* $p<.05$;  $n$=20', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='#555555')

    plt.tight_layout(pad=0.4)
    plt.savefig(f'{OUT_DIR}/fig_scaling_bars.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUT_DIR}/fig_scaling_bars.svg', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig_scaling_bars.png', bbox_inches='tight', dpi=150)
    plt.close()
    plt.rcParams.update(_saved_rc)
    print('  ✓ Figure 1: Scaling bars (fig_scaling_bars)')


if __name__ == '__main__':
    print('Loading results...')
    print(f'  primary matrix: {CORR_MATRIX}')  # chain exports COGARENA_PRIMARY_MATRIX; bare runs fall back to recompute_20260703
    text_data = load_text_results()
    multiturn_data = load_multiturn_results()
    image_data = load_image_results()
    agent_data = load_agent_results()
    print(f'  Loaded: {len(text_data)} text, {len(multiturn_data)} multiturn, {len(image_data)} VLM, {len(agent_data)} agent models')

    # Merge multi-turn results into text data for scaling/profile figures
    merged_data = merge_text_multiturn(text_data, multiturn_data)
    print(f'  Merged: {len(merged_data)} models with up to 13 paradigms')

    print('\nGenerating figures...')
    fig2_behavioral_signatures(text_data)
    fig2_compact()
    fig3_scaling(merged_data)
    fig4_cross_system(text_data, image_data, agent_data)
    fig5_profiles(merged_data)
    fig_scaling_bars(merged_data)
    verify_min_effective()
    print('\nAll figures generated!')
