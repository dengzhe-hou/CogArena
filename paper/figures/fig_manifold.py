#!/usr/bin/env python3
"""Positive-manifold figure: 13x13 paradigm correlation matrix on the 55-model set,
ordered by the 5 theory-motivated groupings. If domains were separable, the within-
grouping blocks (outlined) would be visibly stronger than off-block cells.
Reads the corrected 55x13 matrix (all scorer fixes + fixed-prompt go_nogo rerun +
uniform multiturn metric) produced by results/recompute_20260703/build_and_recompute.py.
"""
import csv
import json
import sys, os
from pathlib import Path
_REPO = os.environ.get("COGARENA_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import compute_b2_expanded as B2
import numpy as np
import matplotlib
matplotlib.use("Agg")
# Embed TrueType (Type 42) fonts; Type 3 is a camera-ready compliance defect.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Match the AAAI template's NewTX body face. The final PDF converts text to
# paths, but loading the matching source font keeps glyph shapes consistent
# with the manuscript before pathification.
_NEWTX_FONT_DIR = Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/newtx"
_NEWTX_FONT_FILES = [
    _NEWTX_FONT_DIR / "TeXGyreTermesX-Regular.otf",
    _NEWTX_FONT_DIR / "TeXGyreTermesX-Italic.otf",
    _NEWTX_FONT_DIR / "TeXGyreTermesX-Bold.otf",
    _NEWTX_FONT_DIR / "TeXGyreTermesX-BoldItalic.otf",
]
for _font_path in _NEWTX_FONT_FILES:
    if _font_path.exists():
        font_manager.fontManager.addfont(str(_font_path))
_FIGURE_FONT = (
    "TeX Gyre TermesX"
    if all(_font_path.exists() for _font_path in _NEWTX_FONT_FILES)
    else "DejaVu Serif"
)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [_FIGURE_FONT, "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
})

# --- figure-compliance hook (2026-07-19): every PDF save renders through an
# SVG with text converted to paths (svg.fonttype='path') and cairosvg, so the
# shipped PDFs embed NO fonts at all (no Type 3, no Identity-H, no CIDFont).
import io as _io
import cairosvg as _cairosvg
plt.rcParams['svg.fonttype'] = 'path'
_orig_fig_savefig = plt.Figure.savefig
_K = 1.0        # uniform text scale applied at save time (like generate_all)
_INSERT_W = 3.31  # \columnwidth in inches; effective size is gated against it.
# CONTRACT: the online main.tex must include this figure at width=\columnwidth
# (NOT 0.9\columnwidth); at 0.9 the true effective size drops below the 9pt gate.
def _savefig_pathified(self, fname, **kw):
    import matplotlib.text as _mtext
    is_pdf = isinstance(fname, str) and fname.endswith('.pdf')
    texts = [t for t in self.findobj(_mtext.Text) if t.get_text()]
    saved = [t.get_fontsize() for t in texts]
    try:
        if is_pdf and abs(_K - 1.0) > 1e-9:
            for t in texts:
                t.set_fontsize(t.get_fontsize() * _K)
        if is_pdf:
            kw.pop('format', None)
            buf = _io.BytesIO()
            _orig_fig_savefig(self, buf, format='svg', **kw)
            _cairosvg.svg2pdf(bytestring=buf.getvalue(), write_to=fname)
            import fitz
            w_in = fitz.open(fname)[0].rect.width / 72.0
            eff = min(t.get_fontsize() for t in texts) * _INSERT_W / w_in
            print(f"fig_manifold effective min font: {eff:.2f}pt "
                  f"(exported {w_in:.2f}in at columnwidth {_INSERT_W}in)")
            if eff < 9.0:
                raise SystemExit(f"FIGURE SIZE GATE FAILED: fig_manifold {eff:.2f}pt < 9.0pt")
        else:
            _orig_fig_savefig(self, fname, **kw)
    finally:
        for t, fs in zip(texts, saved):
            t.set_fontsize(fs)
plt.Figure.savefig = _savefig_pathified


ROOT = B2.ROOT

# --- corrected 55-model paradigm-accuracy matrix (or the frozen primary
# adjudicated matrix when COGARENA_PRIMARY_MATRIX is set) ---
_rows = list(csv.reader(open(
    os.environ.get("COGARENA_PRIMARY_MATRIX")
    or f"{ROOT}/results/recompute_20260703/corrected_matrix.csv")))
_hdr = _rows[0][1:]
all_data = {r[0]: {p: float(v) for p, v in zip(_hdr, r[1:])} for r in _rows[1:]}
models = sorted(all_data.keys())

# --- domain-ordered paradigms so within-grouping pairs are adjacent ---
DOMAIN_ORDER = ['WM', 'Control', 'Episodic', 'ToM', 'Meta']
BY_DOMAIN = {
    'WM': ['digit_span', 'n_back', 'operation_span'],
    'Control': ['stroop', 'flanker', 'go_nogo'],
    'Episodic': ['cvlt_word_list', 'drm_false_memory', 'source_monitoring'],
    'ToM': ['false_belief', 'epitome_tom'],
    'Meta': ['confidence_calibration', 'post_decision_wagering'],
}
LABEL = {'digit_span': 'DS', 'n_back': 'NB', 'operation_span': 'OS', 'stroop': 'ST',
         'flanker': 'FL', 'go_nogo': 'GN', 'cvlt_word_list': 'CV', 'drm_false_memory': 'DRM',
         'source_monitoring': 'SM', 'false_belief': 'FB', 'epitome_tom': 'EP',
         'confidence_calibration': 'CAL', 'post_decision_wagering': 'WG'}
ordered = [p for d in DOMAIN_ORDER for p in BY_DOMAIN[d]]

M = np.array([[all_data[m].get(p, 0.0) for p in ordered] for m in models])
corr = np.corrcoef(M.T)
n = len(ordered)
pct_pos = 100.0 * np.mean([corr[i, j] > 0 for i in range(n) for j in range(i + 1, n)])

disp = corr.copy()
np.fill_diagonal(disp, np.nan)  # omit trivial self-correlations

fig, ax = plt.subplots(figsize=(3.95, 3.06))
TEXT_FS = 9.8
cmap = plt.cm.RdBu_r.copy()
cmap.set_bad('white')
# symmetric range wide enough that no cell saturates (max |r| ~ 0.88)
im = ax.imshow(disp, cmap=cmap, vmin=-0.9, vmax=0.9)

ax.set_xticks(range(n)); ax.set_xticklabels([LABEL[p] for p in ordered], rotation=90, fontsize=TEXT_FS)
ax.set_yticks(range(n)); ax.set_yticklabels([LABEL[p] for p in ordered], fontsize=TEXT_FS)
ax.tick_params(length=0)

# domain block boundaries + labels
bounds, c = [], 0
for d in DOMAIN_ORDER:
    start = c
    c += len(BY_DOMAIN[d])
    bounds.append((start, c))
for di, (dname, (s, e)) in enumerate(zip(DOMAIN_ORDER, bounds)):
    # outline each within-grouping block
    ax.add_patch(plt.Rectangle((s - 0.5, s - 0.5), e - s, e - s,
                               fill=False, edgecolor='black', lw=1.3))
    # keys must match DOMAIN_ORDER entries exactly ('Control'/'Episodic'),
    # otherwise the fallback prints the long name and widens the bbox
    short = {'WM': 'WM', 'Control': 'CC', 'Episodic': 'EM',
             'ToM': 'ToM', 'Meta': 'MC'}[dname]
    off = {'ToM': -0.35, 'MC': 0.35}.get(short, 0.0)
    ax.text((s + e - 1) / 2.0 + off, -1.15, short, ha='center',
            va='bottom', fontsize=TEXT_FS, fontweight='bold')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=TEXT_FS)
cbar.set_label('Pairwise $r$', fontsize=TEXT_FS)

# Use the paper's frozen two-sided headline inference. B2.run_b2 reports a
# one-sided Monte Carlo value, which is useful internally but must not appear
# beside the manuscript's exact two-sided result.
_inference_path = os.path.join(ROOT, "results", "recompute_20260703",
                               "final_inference.json")
with open(_inference_path) as _f:
    _headline = json.load(_f)["corrected_raw"]
ax.set_title(f"within$-$cross $\\delta$={_headline['delta']:+.3f}\n"
             f"$n$={len(models)}; exact $p_2$={_headline['perm_p_two_sided']:.3f}",
             fontsize=TEXT_FS, pad=22)

plt.tight_layout()
plt.savefig(f"{ROOT}/paper/figures/fig_manifold.pdf", bbox_inches='tight', dpi=300)
plt.savefig(f"{ROOT}/paper/figures/fig_manifold.svg", bbox_inches='tight')
plt.savefig(f"{ROOT}/paper/figures/fig_manifold.png", bbox_inches='tight', dpi=150)
plt.close()
res, _ = B2.run_b2(all_data, models)
print(f"saved fig_manifold.pdf | n_models={len(models)} | pct_pairwise_positive={pct_pos:.1f}% | "
      f"within={res['within_mean']} cross={res['cross_mean']} delta={res['delta']} p={res['p_value']}")
