"""Regression tests for the frozen 2026-07-24 VLM remediation."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageChops

from cogarena.image_gen.flanker_images import (
    FONT_SIZE as FLANKER_FONT_SIZE,
    generate_balanced_flanker_set,
    generate_flanker_image,
)
from cogarena.image_gen.false_belief_images import generate_false_belief_set
from cogarena.image_gen.font_utils import (
    DEJAVU_SANS_BOLD_SHA256,
    resolve_font_path,
)
from cogarena.image_gen.stroop_images import (
    FONT_SIZE as STROOP_FONT_SIZE,
    generate_balanced_stroop_set,
    generate_stroop_image,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiments.vlm_rerun_20260724.scoring import parse_response


def _parse(
    paradigm: str,
    response: str,
    *,
    expected: str = "blue",
    allowed: list[str] | None = None,
    label_space: list[str] | None = None,
    query_subject: str | None = None,
):
    allowed = allowed or ["red", "blue"]
    label_space = label_space or list(
        dict.fromkeys([*allowed, "red", "blue", "green", "orange"])
    )
    return parse_response(
        paradigm=paradigm,
        response=response,
        expected=expected,
        allowed_labels=allowed,
        label_space=label_space,
        query_subject=query_subject,
    )


def _content_bbox(path: Path) -> tuple[int, int, int, int]:
    with Image.open(path) as image:
        white = Image.new(image.mode, image.size, "white")
        bbox = ImageChops.difference(image, white).getbbox()
    assert bbox is not None
    return bbox


def test_frozen_font_has_expected_hash():
    font = resolve_font_path()
    assert hashlib.sha256(font.read_bytes()).hexdigest() == DEJAVU_SANS_BOLD_SHA256


def test_stroop_and_flanker_render_at_nominal_scale(tmp_path):
    stroop = tmp_path / "stroop.png"
    flanker = tmp_path / "flanker.png"
    generate_stroop_image("RED", "blue", str(stroop))
    generate_flanker_image("left", "right", 3, str(flanker))

    sx0, sy0, sx1, sy1 = _content_bbox(stroop)
    fx0, fy0, fx1, fy1 = _content_bbox(flanker)
    assert sx1 - sx0 >= 2 * STROOP_FONT_SIZE
    assert sy1 - sy0 >= int(0.65 * STROOP_FONT_SIZE)
    assert fx1 - fx0 >= 5 * FLANKER_FONT_SIZE
    assert fy1 - fy0 >= int(0.55 * FLANKER_FONT_SIZE)


def test_rendering_is_byte_deterministic(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    generate_stroop_image("BLUE", "orange", str(first))
    generate_stroop_image("BLUE", "orange", str(second))
    assert first.read_bytes() == second.read_bytes()


def test_parser_exact_markdown_and_blank():
    assert not _parse("stroop", "")["correct"]
    assert not _parse("stroop", "   ")["correct"]
    assert _parse("stroop", "The blue.")["correct"]
    assert _parse(
        "stroop",
        "**Answer:** Orange",
        expected="orange",
        allowed=["orange", "purple"],
    )["correct"]


def test_parser_never_uses_substrings_or_unanchored_narrative():
    assert _parse("stroop", "The image is brightly colored.")["status"] == "no_label"
    assert not _parse("stroop", "Red is the word; blue is visible.")["correct"]
    parsed = _parse("stroop", "The word RED is written in blue letters.")
    assert parsed["label"] == "blue" and parsed["correct"]


def test_explicit_answer_uses_immediate_value_only():
    parsed = _parse("stroop", "Answer: blue because the word is red")
    assert parsed["label"] == "blue" and parsed["correct"]
    assert _parse("stroop", "Answer: not blue")["label"] is None
    assert _parse("stroop", "Answer: blue or red")["label"] is None
    assert _parse("stroop", "Answer: blue, or red")["label"] is None
    assert _parse("stroop", "Answer: blue, and red")["label"] is None
    assert _parse("stroop", "Answer: blue (or red)")["label"] is None
    assert _parse("stroop", "Answer: blue\nAnswer: blue")["label"] is None
    wrong = _parse("stroop", "The ink is blue.\nAnswer: red")
    assert wrong["label"] == "red" and not wrong["correct"]


def test_flanker_parser_anchors_center_and_rejects_alternatives():
    parsed = _parse(
        "flanker",
        "The center arrow points right while the flankers point left.",
        expected="right",
        allowed=["left", "right"],
        label_space=["left", "right"],
    )
    assert parsed["label"] == "right" and parsed["correct"]
    ambiguous = _parse(
        "flanker",
        "The center arrow points left/right.",
        expected="left",
        allowed=["left", "right"],
        label_space=["left", "right"],
    )
    assert ambiguous["status"] == "ambiguous_anchor"
    assert _parse("stroop", "The ink is blue, or red")["status"] == "ambiguous_anchor"
    assert _parse(
        "flanker",
        "The center arrow points left, or right.",
        expected="left",
        allowed=["left", "right"],
        label_space=["left", "right"],
    )["status"] == "ambiguous_anchor"


def test_false_belief_parser_binds_subject_and_clause_locality():
    kwargs = {
        "expected": "basket",
        "allowed": ["basket", "box"],
        "label_space": ["basket", "box", "jar"],
        "query_subject": "Anne",
    }
    wrong_subject = _parse(
        "false_belief", "Sally will first look in the basket.", **kwargs
    )
    assert wrong_subject["status"] == "unanchored_labels"
    good = _parse(
        "false_belief",
        "Anne will first look in the basket. The story contains a basket and box.",
        **kwargs,
    )
    assert good["label"] == "basket" and good["correct"]
    ambiguous = _parse(
        "false_belief", "Anne will first look in basket or box.", **kwargs
    )
    assert ambiguous["status"] == "ambiguous_anchor"
    invalid = _parse("false_belief", "jar", **kwargs)
    assert invalid["label"] == "jar"
    assert invalid["status"] == "invalid_label"
    assert not invalid["correct"]


def test_balanced_factorial_generators(tmp_path):
    stroop = generate_balanced_stroop_set(
        seed=42, n_per_condition=50, out_dir=str(tmp_path / "stroop")
    )
    flanker = generate_balanced_flanker_set(
        seed=42, n_per_condition=50, out_dir=str(tmp_path / "flanker")
    )
    for condition in (True, False):
        ink = sorted(
            trial["ink_color"] for trial in stroop if trial["congruent"] is condition
        )
        words = sorted(
            trial["word"].lower()
            for trial in stroop
            if trial["congruent"] is condition
        )
        assert len(ink) == 50
        if condition:
            reference_ink, reference_words = ink, words
        else:
            assert ink == reference_ink
            assert words == reference_words
            assert all(
                trial["word"].lower() != trial["ink_color"]
                for trial in stroop
                if not trial["congruent"]
            )
    for condition in (True, False):
        targets = [
            trial["target_dir"]
            for trial in flanker
            if trial["congruent"] is condition
        ]
        assert targets.count("left") == 25
        assert targets.count("right") == 25


def test_false_belief_montage_profile_preserves_semantics(tmp_path):
    standard = generate_false_belief_set(
        seed=42, n_items=2, out_dir=str(tmp_path / "standard")
    )
    montage = generate_false_belief_set(
        seed=42,
        n_items=2,
        out_dir=str(tmp_path / "montage"),
        montage_readable=True,
    )
    for standard_item, montage_item in zip(standard, montage):
        for key in (
            "characters",
            "object",
            "containers",
            "correct_answer",
            "stimulus_text",
        ):
            assert standard_item[key] == montage_item[key]
        with Image.open(montage_item["image_paths"][0]) as image:
            assert image.size == (500, 360)
