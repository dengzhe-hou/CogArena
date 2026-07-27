"""Clean-install figure-compliance smoke test.

Imports the figure module (which installs the text-to-path savefig hook),
renders a tiny figure with text to a temporary PDF, and asserts the PDF
embeds NO fonts (no Type 3, no Identity-H, no CIDFont). Fails loudly if
cairosvg is missing, which pins the packaging requirement.
"""
import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))


def test_cairosvg_importable():
    import cairosvg  # noqa: F401


def test_pathified_pdf_embeds_no_fonts(tmp_path):
    import fitz
    os.environ.setdefault("COGARENA_PRIMARY_MATRIX", os.path.join(
        ROOT, "results", "reanalysis", "aplus_20260718", "matrix_aplus_strict.csv"))
    spec = importlib.util.spec_from_file_location(
        "ga_smoke", os.path.join(ROOT, "paper", "figures", "generate_all.py"))
    ga = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(ga)
    except FileNotFoundError:
        pytest.skip("results tree not present (clean checkout without data)")
    plt = ga.plt
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.set_title("smoke $\\delta$=0.08")
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x label")
    out = str(tmp_path / "smoke.pdf")
    fig.savefig(out)
    plt.close(fig)
    doc = fitz.open(out)
    fonts = set()
    for page in doc:
        for f in page.get_fonts(full=True):
            fonts.add((f[3], f[2]))
    assert not fonts, f"fonts embedded: {fonts}"
