#!/usr/bin/env python3
"""Generate the main-paper intervention-selectivity summary figure.

This figure is deliberately descriptive: the frozen primary estimand and all
confirmatory decisions are read from ``analysis_results.json`` without
recomputing or modifying the formal analysis.  Panel A shows the full crossed
intervention-by-group gain matrix, panel B shows family-level omnibus effects,
and panel C summarizes the nine frozen gates in three decision groups.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import cairosvg
import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")

# AAAI's ``newtxtext`` package uses TeX Gyre TermesX. Load the same local
# OpenType faces when TinyTeX is available, then pathify them at export.
_NEWTX_FONT_DIR = Path(
    os.environ.get(
        "COGARENA_NEWTX_FONT_DIR",
        Path.home()
        / ".TinyTeX/texmf-dist/fonts/opentype/public/newtx",
    )
)
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

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [_FIGURE_FONT, "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        # Authored above 9 pt because the fixed export canvas is downscaled
        # slightly to the AAAI text width at insertion time.
        "font.size": 9.60,
        "axes.titlesize": 10.7,
        "axes.labelsize": 10.2,
        "xtick.labelsize": 9.8,
        "ytick.labelsize": 9.8,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(
    os.environ.get("COGARENA_ROOT")
    or Path(__file__).resolve().parents[2]
).resolve()
RESULTS = ROOT / "results/causal_selectivity_20260720/analysis/analysis_results.json"
OUT = ROOT / "paper/figures/fig_causal_selectivity.pdf"
OUT_SVG = OUT.with_suffix(".svg")
OUT_PNG = OUT.with_suffix(".png")

GROUPS = [
    ("working_memory", "WM"),
    ("cognitive_control", "CC"),
    ("episodic_memory", "EM"),
    ("theory_of_mind", "ToM"),
    ("metacognition", "MC"),
]
INTERVENTION_LABELS = {
    "working_memory_ledger": "WM ledger",
    "control_rule_rehearsal": "Rule rehearsal",
    "episodic_source_binding": "Source binding",
    "belief_state_ledger": "Belief ledger",
    "metacognitive_forecast": "Meta forecast",
}
GATE_GROUPS = [
    (
        "Signal",
        [
            ("gamma_family_item_ci_lower_gt_zero", "CI>0"),
            ("exact_mapping_one_sided_p_le_0_05", "map"),
            ("at_least_four_of_six_family_gamma_positive", "≥4/6 fam."),
        ],
    ),
    (
        "Robust",
        [
            ("each_condition_task_record_invalid_rate_le_0_01", "invalid"),
            (
                "protocol_invalid_exclusion_preserves_at_least_half_gamma_and_three_items_per_cell",
                "exclusion",
            ),
            (
                "response_length_adjustment_preserves_at_least_half_gamma",
                "length",
            ),
        ],
    ),
    (
        "Transport",
        [
            (
                "family_lofo_selective_delta_log_likelihood_gt_zero",
                "LOFO",
            ),
            (
                "empty_exclusion_preserves_at_least_half_gamma_and_three_items_per_cell",
                "nonempty",
            ),
            (
                "ospan_parse_exclusion_preserves_at_least_half_gamma_and_three_items_per_cell",
                "OS parse",
            ),
        ],
    ),
]


def _pathified_save(fig: plt.Figure, pdf: Path, svg: Path, png: Path) -> None:
    """Write a zero-embedded-font PDF and verify effective text is >= 9 pt."""
    pdf.parent.mkdir(parents=True, exist_ok=True)
    svg_buf = io.BytesIO()
    # Keep an exact canvas instead of ``bbox_inches='tight'``. Long gate labels
    # otherwise enlarge the exported bounding box and are silently downscaled
    # below the effective 9 pt floor when LaTeX inserts the figure.
    fig.savefig(svg_buf, format="svg")
    svg_bytes = svg_buf.getvalue()
    svg.write_bytes(svg_bytes)
    cairosvg.svg2pdf(bytestring=svg_bytes, write_to=str(pdf))
    fig.savefig(png, format="png", dpi=180)

    import fitz

    document = fitz.open(pdf)
    exported_width_in = document[0].rect.width / 72.0
    inserted_width_in = 3.35  # main.tex uses \columnwidth
    scale = inserted_width_in / exported_width_in
    text_sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text()]
    effective_min = min(text_sizes) * scale
    print(
        f"fig_causal_selectivity effective min font: {effective_min:.2f}pt "
        f"(exported {exported_width_in:.2f}in at textwidth {inserted_width_in:.2f}in)"
    )
    if effective_min < 9.0:
        raise SystemExit(
            f"FIGURE SIZE GATE FAILED: fig_causal_selectivity "
            f"{effective_min:.2f}pt < 9.0pt"
        )
    if document[0].get_fonts(full=True):
        raise SystemExit("FIGURE FONT GATE FAILED: PDF contains embedded fonts")
    document.close()


def main() -> None:
    data = json.loads(RESULTS.read_text())
    primary = data["primary"]
    gate = data["confirmatory_gate"]

    matrix_rows = primary["intervention_by_group_gain_matrix"]
    matrix = np.asarray(
        [
            [row["group_mean_gain"][group] for group, _ in GROUPS]
            for row in matrix_rows
        ],
        dtype=float,
    )
    row_labels = [INTERVENTION_LABELS[row["intervention"]] for row in matrix_rows]

    family_items = sorted(primary["family_gamma"].items(), key=lambda kv: kv[1])
    family_short = {
        "llama2": "Llama2", "falcon3": "Falcon3", "gemma3": "Gemma3",
        "qwen2.5": "Qwen2.5", "olmo2": "OLMo2", "gemma2": "Gemma2",
    }
    family_labels = [family_short.get(name, name) for name, _ in family_items]
    family_values = [value for _, value in family_items]

    fig = plt.figure(figsize=(3.55, 5.20))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[2.02, 1.28, 1.12],
        hspace=0.58,
    )
    fig.subplots_adjust(left=0.28, right=0.885, bottom=0.035, top=0.955)

    # Panel A: the fully crossed gain matrix.
    ax = fig.add_subplot(grid[0, 0])
    ax_a = ax
    lim = max(0.05, float(np.max(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-lim, vmax=lim, aspect="equal")
    ax.set_xticks(range(5), [label for _, label in GROUPS], rotation=0, ha="center")
    ax.set_yticks(range(5), row_labels)
    ax.tick_params(length=0)
    ax.set_title(
        "A  Crossed gains", x=-0.31, ha="left", fontweight="bold"
    )
    cell_text_rows = []
    for i in range(5):
        cell_texts = []
        ax.add_patch(Rectangle((i - 0.46, i - 0.46), 0.92, 0.92,
                               fill=False, edgecolor="#111111", linewidth=1.0))
        for j in range(5):
            value = matrix[i, j]
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            cell_texts.append(
                ax.text(j, i, f"{100 * value:.1f}", ha="center", va="center",
                        color="white" if luminance < 0.48 else "#111111",
                        fontsize=9.55, fontstretch="condensed")
            )
        cell_text_rows.append(cell_texts)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_ticks(
        [-lim, 0, lim],
        labels=[f"{-100*lim:.0f}", "0", f"{100*lim:.0f}"],
    )
    cbar.ax.tick_params(labelsize=9.60, pad=2)

    # Panel B: family consistency (descriptive, because there are only six families).
    ax = fig.add_subplot(grid[1, 0])
    ax_b = ax
    b_position = ax.get_position()
    ax.set_position(
        [b_position.x0, b_position.y0 + 0.018, b_position.width, b_position.height]
    )
    y = np.arange(len(family_values))
    colors = ["#356A9A" if value > 0 else "#B55A4A" for value in family_values]
    values_pp = 100 * np.asarray(family_values)
    ax.hlines(
        y,
        np.minimum(values_pp, 0),
        np.maximum(values_pp, 0),
        color="#AEB8C2",
        linewidth=1.4,
        zorder=1,
    )
    ax.scatter(
        values_pp,
        y,
        s=33,
        c=colors,
        edgecolor="white",
        linewidth=0.55,
        zorder=2,
    )
    ax.axvline(0, color="#333333", linewidth=0.7)
    ax.set_yticks(y, family_labels)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(-0.72, 5.35)
    ax.set_xticks([0, 2.5, 5], ["0", "2.5", "5"])
    ax.set_xlabel(r"Family $\Gamma$ (pp)")
    ax.set_title(r"B  By family", loc="left", fontweight="bold", pad=4)
    ax.grid(axis="x", color="#E7EAED", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    family_value_texts = []
    for yy, value in zip(y, family_values):
        value_pp = 100 * value
        value_x = value_pp + 0.16 if value_pp >= 0 else 0.16
        family_value_texts.append(
            ax.text(value_x, yy,
                    f"{value_pp:+.1f}", va="center",
                    ha="left")
        )

    # Panel C: nine dots retain the full frozen decision structure without the
    # visual weight of a dashboard. Text labels make status legible without
    # relying on red/green color.
    ax = fig.add_subplot(grid[2, 0])
    ax_c = ax
    ax.axis("off")
    c_position = ax.get_position()
    ax.set_position(
        [
            c_position.x0 - 0.08,
            c_position.y0 + 0.012,
            c_position.width,
            c_position.height,
        ]
    )
    ax.set_title(
        "C  Confirmation", x=0.13, ha="left", fontweight="bold", pad=4
    )
    components = gate["components"]
    pass_count = sum(bool(value) for value in components.values())
    decision_text = ax.text(
        0.98,
        0.84,
        f"{pass_count}/9 FAIL",
        ha="right",
        va="center",
        color="#A33F35",
        fontweight="bold",
        transform=ax.transAxes,
    )
    gamma_summary = ax.text(
        0.02,
        0.84,
        rf"$\Gamma$={100*primary['gamma']:.2f} pp",
        ha="left",
        va="center",
        fontweight="bold",
        transform=ax.transAxes,
    )
    ci_lower, ci_upper = primary["bootstrap"]["gamma_ci95"]
    ci_summary = ax.text(
        0.02,
        0.70,
        rf"95% CI [{100*ci_lower:.2f}, {100*ci_upper:.2f}]",
        ha="left",
        va="center",
        color="#47515B",
        transform=ax.transAxes,
    )
    ax.plot(
        [0.02, 0.98],
        [0.61, 0.61],
        color="#D5DADF",
        linewidth=0.65,
        transform=ax.transAxes,
        clip_on=False,
    )
    row_centers = [0.46, 0.27, 0.08]
    gate_texts = []
    gate_row_texts = []
    for center_y, (group_label, entries) in zip(row_centers, GATE_GROUPS):
        row_passes = [bool(components[key]) for key, _ in entries]
        if not all(value == row_passes[0] for value in row_passes):
            raise SystemExit(
                f"FIGURE DATA GATE FAILED: mixed status in {group_label}"
            )
        edge = "#2E7047" if row_passes[0] else "#A33F35"
        group_text = ax.text(
            0.02,
            center_y,
            group_label,
            ha="left",
            va="center",
            fontweight="bold",
            transform=ax.transAxes,
        )
        dot_x = [0.53, 0.63, 0.73]
        ax.scatter(
            dot_x,
            [center_y] * 3,
            s=43,
            c=[edge] * 3,
            marker="o" if row_passes[0] else "X",
            edgecolor="white",
            linewidth=0.6,
            transform=ax.transAxes,
            clip_on=False,
            zorder=2,
        )
        for key, _ in entries:
            if key not in components:
                raise SystemExit(f"FIGURE DATA GATE FAILED: missing {key}")
        status_text = ax.text(
            0.98,
            center_y,
            "PASS" if row_passes[0] else "FAIL",
            ha="right",
            va="center",
            color=edge,
            fontweight="bold",
            transform=ax.transAxes,
        )
        gate_texts.extend([group_text, status_text])
        gate_row_texts.append((group_text, status_text))

    for axis in fig.axes:
        if axis is not cbar.ax:
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)

    # Fail closed on the exact overlap classes that made the original figure
    # illegible.  This runs on the authored canvas before pathifying the text.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def bbox(text):
        return text.get_window_extent(renderer=renderer)

    # Two restrained rules separate the evidence layers without enclosing the
    # panels in heavy boxes. Compute their positions from rendered text bounds
    # rather than axes boxes so a rule cannot cross a title or axis label.
    divider_style = {
        "transform": fig.transFigure,
        "color": "#C8CDD2",
        "linewidth": 0.65,
        "solid_capstyle": "butt",
        "zorder": 10,
    }
    divider_left, divider_right = 0.085, 0.94
    a_content_bottom = min(label.get_window_extent(renderer).y0
                           for label in ax_a.get_xticklabels())
    b_content_top = ax_b.title.get_window_extent(renderer).y1
    b_content_bottom = ax_b.xaxis.label.get_window_extent(renderer).y0
    c_content_top = ax_c.title.get_window_extent(renderer).y1
    inverse_figure = fig.transFigure.inverted()
    divider_ab_y = inverse_figure.transform(
        (0, 0.5 * (a_content_bottom + b_content_top))
    )[1]
    divider_bc_y = inverse_figure.transform(
        (0, 0.5 * (b_content_bottom + c_content_top))
    )[1]
    for divider_y in (divider_ab_y, divider_bc_y):
        fig.add_artist(
            Line2D(
                [divider_left, divider_right],
                [divider_y, divider_y],
                **divider_style,
            )
        )

    for row in cell_text_rows:
        for left, right in zip(row, row[1:]):
            if bbox(left).overlaps(bbox(right)):
                raise SystemExit("FIGURE COLLISION GATE FAILED: Panel A cell values")
    xlabels = ax_a.get_xticklabels()
    for left, right in zip(xlabels, xlabels[1:]):
        if bbox(left).overlaps(bbox(right)):
            raise SystemExit("FIGURE COLLISION GATE FAILED: Panel A x labels")
    for label, status in gate_row_texts:
        if bbox(label).overlaps(bbox(status)):
            raise SystemExit(
                f"FIGURE COLLISION GATE FAILED: Panel C row heading "
                f"{label.get_text()!r}"
            )
        dot_left = ax.transAxes.transform((0.505, 0))[0]
        dot_right = ax.transAxes.transform((0.755, 0))[0]
        if bbox(label).x1 >= dot_left or bbox(status).x0 <= dot_right:
            raise SystemExit(
                f"FIGURE COLLISION GATE FAILED: Panel C dot gutter "
                f"{label.get_text()!r}; label_right={bbox(label).x1:.1f}, "
                f"dot_left={dot_left:.1f}, dot_right={dot_right:.1f}, "
                f"status_left={bbox(status).x0:.1f}"
            )
    for label, value in zip(ax_b.get_yticklabels(), family_value_texts):
        if bbox(label).overlaps(bbox(value)):
            raise SystemExit("FIGURE COLLISION GATE FAILED: Panel B labels")

    selected_text = [
        *[text for row in cell_text_rows for text in row],
        *ax_a.get_xticklabels(),
        *ax_a.get_yticklabels(),
        *ax_b.get_yticklabels(),
        *family_value_texts,
        *gate_texts,
        gamma_summary,
        ci_summary,
        decision_text,
    ]
    canvas = fig.bbox
    for text in selected_text:
        box = bbox(text)
        if (
            box.x0 < canvas.x0 - 0.5
            or box.y0 < canvas.y0 - 0.5
            or box.x1 > canvas.x1 + 0.5
            or box.y1 > canvas.y1 + 0.5
        ):
            raise SystemExit(
                f"FIGURE CLIPPING GATE FAILED: {text.get_text()!r} leaves the canvas "
                f"{tuple(round(value, 1) for value in box.extents)} versus "
                f"{tuple(round(value, 1) for value in canvas.extents)}"
            )

    _pathified_save(fig, OUT, OUT_SVG, OUT_PNG)
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
