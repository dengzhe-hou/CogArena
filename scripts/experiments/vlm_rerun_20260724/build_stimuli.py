#!/usr/bin/env python3
"""Generate the VLM stimuli once and freeze their content hashes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import PIL
from PIL import Image, ImageChops, ImageDraw

from cogarena.image_gen.false_belief_images import generate_false_belief_set
from cogarena.image_gen.flanker_images import (
    generate_balanced_flanker_set,
    generate_flanker_set,
)
from cogarena.image_gen.font_utils import font_provenance, load_frozen_font
from cogarena.image_gen.stroop_images import (
    generate_balanced_stroop_set,
    generate_stroop_set,
)

from .common import (
    EXPECTED_IMAGE_COUNTS,
    EXPECTED_ITEM_COUNTS,
    EXPECTED_ITEMS_PER_MODEL,
    IMAGE_MANIFEST,
    N_ITEMS,
    ROOT,
    RUN_ROOT,
    SEED,
    STIMULUS_ROOT,
    atomic_write_json,
    load_image_manifest,
    sha256_file,
)


SOURCE_FILES = (
    ROOT / "cogarena" / "image_gen" / "font_utils.py",
    ROOT / "cogarena" / "image_gen" / "stroop_images.py",
    ROOT / "cogarena" / "image_gen" / "flanker_images.py",
    ROOT / "cogarena" / "image_gen" / "false_belief_images.py",
    Path(__file__).resolve(),
)


def _git_head() -> str:
    declared = os.environ.get("COGARENA_GIT_HEAD")
    if declared:
        return declared
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _image_meta(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
        white = Image.new(mode, image.size, "white")
        bbox = ImageChops.difference(image, white).getbbox()
    return {
        "path": path.relative_to(RUN_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "mode": mode,
        "content_bbox": list(bbox) if bbox else None,
    }


def _build_readiness_probe() -> dict[str, Any]:
    """Freeze an out-of-sample API health probe outside the scientific items."""

    path = STIMULUS_ROOT / "readiness" / "black_circle.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (400, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((135, 25, 265, 155), fill="black")
    image.save(path)
    return {
        "probe_id": "vlm_transport_probe_v1",
        "prompt": (
            "This is an out-of-sample serving health check. "
            "Name the centered shape with one word."
        ),
        "images": [_image_meta(path)],
        "expected_transport_status": "ok",
        "output_is_not_scored": True,
    }


def _task(
    paradigm: str,
    index: int,
    trial: dict[str, Any],
    prompt: str,
    image_paths: list[str],
) -> dict[str, Any]:
    expected = str(trial.get("correct_answer", trial.get("expected_response", "")))
    return {
        "task_id": f"img_{paradigm}_{index:04d}",
        "paradigm": paradigm,
        "dimension": (
            "theory_of_mind" if paradigm == "false_belief" else "cognitive_control"
        ),
        "expected": expected,
        "congruent": trial.get("congruent"),
        "prompt": prompt,
        "images": [_image_meta(Path(path)) for path in image_paths],
        "factors": {
            key: trial[key]
            for key in (
                "word",
                "ink_color",
                "target_dir",
                "flanker_dir",
                "characters",
                "object",
                "containers",
            )
            if key in trial
        },
    }


def _assert_geometry(tasks: list[dict[str, Any]]) -> None:
    for task in tasks:
        paradigm = task["paradigm"]
        if paradigm not in {"stroop", "flanker"}:
            continue
        if len(task["images"]) != 1:
            raise RuntimeError(f"{task['task_id']} must have one image")
        image = task["images"][0]
        bbox = image["content_bbox"]
        if bbox is None:
            raise RuntimeError(f"{task['task_id']} rendered a blank image")
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        min_width, min_height = (
            (120, 39) if paradigm == "stroop" else (275, 30)
        )
        if width < min_width or height < min_height:
            raise RuntimeError(
                f"{task['task_id']} content bbox {width}x{height} is below "
                f"{min_width}x{min_height}"
            )


def _assert_false_belief_montages(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    false_belief = [
        task for task in tasks if task["paradigm"] == "false_belief"
    ]
    for task in false_belief:
        if len(task["images"]) != 1:
            raise RuntimeError(f"{task['task_id']} must use one montage")
        meta = task["images"][0]
        if (meta["width"], meta["height"]) != (1000, 720):
            raise RuntimeError(f"{task['task_id']} montage dimensions drifted")
        path = RUN_ROOT / meta["path"]
        with Image.open(path) as image:
            image.load()
            for quadrant, box in enumerate(
                ((0, 0, 500, 360), (500, 0, 1000, 360),
                 (0, 360, 500, 720), (500, 360, 1000, 720)),
                start=1,
            ):
                crop = image.crop(box)
                white = Image.new(crop.mode, crop.size, "white")
                if ImageChops.difference(crop, white).getbbox() is None:
                    raise RuntimeError(
                        f"{task['task_id']} montage quadrant {quadrant} is blank"
                    )
    effective_min_font = 24 * 384 / 1000
    if effective_min_font < 9:
        raise RuntimeError("montage typography falls below the 9px conservative gate")
    return {
        "render_profile": "montage_readable_v2",
        "panel_size": [500, 360],
        "montage_size": [1000, 720],
        "primary_font_px": 24,
        "auxiliary_font_px": 16,
        "conservative_long_side_px": 384,
        "effective_primary_font_px": effective_min_font,
    }


def _false_belief_montage(paths: list[str], output: Path) -> str:
    if len(paths) != 4:
        raise RuntimeError("false-belief montage requires exactly four frames")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    if any(frame.size != frames[0].size for frame in frames):
        raise RuntimeError("false-belief frame-size mismatch")
    width, height = frames[0].size
    montage = Image.new("RGB", (2 * width, 2 * height), "white")
    for index, frame in enumerate(frames):
        montage.paste(frame, ((index % 2) * width, (index // 2) * height))
    from PIL import ImageDraw

    divider = ImageDraw.Draw(montage)
    divider.line([(width, 0), (width, 2 * height)], fill=(100, 100, 100), width=3)
    divider.line([(0, height), (2 * width, height)], fill=(100, 100, 100), width=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output)
    return str(output)


def _assert_balanced(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    stroop = [task for task in tasks if task["paradigm"] == "stroop"]
    flanker = [task for task in tasks if task["paradigm"] == "flanker"]
    stroop_counts = {
        str(condition): Counter(
            task["factors"]["ink_color"]
            for task in stroop
            if task["congruent"] is condition
        )
        for condition in (True, False)
    }
    if stroop_counts["True"] != stroop_counts["False"]:
        raise RuntimeError(f"Stroop ink-label imbalance: {stroop_counts}")
    stroop_word_counts = {
        str(condition): Counter(
            task["factors"]["word"].lower()
            for task in stroop
            if task["congruent"] is condition
        )
        for condition in (True, False)
    }
    if stroop_word_counts["True"] != stroop_word_counts["False"]:
        raise RuntimeError(f"Stroop word-label imbalance: {stroop_word_counts}")
    if any(
        task["factors"]["word"].lower() == task["factors"]["ink_color"]
        for task in stroop
        if task["congruent"] is False
    ):
        raise RuntimeError("Stroop incongruent condition contains a matching label")
    flanker_counts = {
        str(condition): Counter(
            task["factors"]["target_dir"]
            for task in flanker
            if task["congruent"] is condition
        )
        for condition in (True, False)
    }
    expected_flanker = Counter({"left": 25, "right": 25})
    if any(counts != expected_flanker for counts in flanker_counts.values()):
        raise RuntimeError(f"Flanker target-label imbalance: {flanker_counts}")
    return {
        "stroop_ink_counts_by_condition": {
            key: dict(sorted(value.items())) for key, value in stroop_counts.items()
        },
        "stroop_word_counts_by_condition": {
            key: dict(sorted(value.items()))
            for key, value in stroop_word_counts.items()
        },
        "flanker_target_counts_by_condition": {
            key: dict(sorted(value.items())) for key, value in flanker_counts.items()
        },
    }


def build(design_id: str) -> None:
    if IMAGE_MANIFEST.exists():
        load_image_manifest(verify_images=True)
        with IMAGE_MANIFEST.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("design_id", "legacy_readable_v1") != design_id:
            raise RuntimeError(
                f"existing design does not match requested {design_id}: {IMAGE_MANIFEST}"
            )
        print(f"verified existing frozen stimuli: {IMAGE_MANIFEST}")
        return
    if STIMULUS_ROOT.exists() and any(STIMULUS_ROOT.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite nonempty stimulus directory: {STIMULUS_ROOT}"
        )

    image_root = STIMULUS_ROOT / "images"
    if design_id == "balanced_montage_v2":
        stroop = generate_balanced_stroop_set(
            seed=SEED,
            n_per_condition=N_ITEMS,
            out_dir=str(image_root / "stroop"),
        )
        flanker = generate_balanced_flanker_set(
            seed=SEED,
            n_per_condition=N_ITEMS,
            out_dir=str(image_root / "flanker"),
        )
    elif design_id == "legacy_readable_v1":
        stroop = generate_stroop_set(
            seed=SEED,
            n_congruent=N_ITEMS,
            n_incongruent=N_ITEMS,
            out_dir=str(image_root / "stroop"),
        )
        flanker = generate_flanker_set(
            seed=SEED,
            n_congruent=N_ITEMS,
            n_incongruent=N_ITEMS,
            out_dir=str(image_root / "flanker"),
        )
    else:
        raise RuntimeError(f"unsupported design: {design_id}")

    if design_id == "balanced_montage_v2":
        temporary = tempfile.TemporaryDirectory(prefix="cogarena-fb-frames-")
        false_belief_frame_root = Path(temporary.name)
    else:
        temporary = None
        false_belief_frame_root = image_root / "false_belief"
    false_belief = generate_false_belief_set(
        seed=SEED,
        n_items=N_ITEMS,
        out_dir=str(false_belief_frame_root),
        montage_readable=(design_id == "balanced_montage_v2"),
    )

    tasks: list[dict[str, Any]] = []
    for index, trial in enumerate(stroop):
        tasks.append(
            _task(
                "stroop",
                index,
                trial,
                trial["stimulus_text"],
                [trial["image_path"]],
            )
        )
    for index, trial in enumerate(flanker):
        tasks.append(
            _task(
                "flanker",
                index,
                trial,
                trial["stimulus_text"],
                [trial["image_path"]],
            )
        )
    for index, trial in enumerate(false_belief):
        characters = trial.get("characters", ["Character A", "Character B"])
        if design_id == "balanced_montage_v2":
            montage = _false_belief_montage(
                list(trial["image_paths"]),
                image_root / "false_belief" / f"story_{index:03d}.png",
            )
            image_paths = [montage]
            prompt = (
                "This image contains four story panels in reading order: "
                "top-left, top-right, bottom-left, bottom-right. "
                "Study panels 1 through 4 carefully.\n"
                f"Question: Where will {characters[1]} first look for the object? "
                "Answer with the location only."
            )
        else:
            image_paths = list(trial["image_paths"])
            prompt = (
                f"These images show a sequence of events in order. "
                f"Study all {len(image_paths)} scenes carefully.\n"
                f"Question: Where will {characters[1]} look for the object? "
                f"Answer with the location only."
            )
        tasks.append(
            _task(
                "false_belief",
                index,
                trial,
                prompt,
                image_paths,
            )
        )
    if temporary is not None:
        temporary.cleanup()

    counts = {
        paradigm: sum(task["paradigm"] == paradigm for task in tasks)
        for paradigm in EXPECTED_ITEM_COUNTS
    }
    image_count = sum(len(task["images"]) for task in tasks)
    if counts != EXPECTED_ITEM_COUNTS:
        raise RuntimeError(f"unexpected task counts: {counts}")
    expected_image_count = EXPECTED_IMAGE_COUNTS[design_id]
    if len(tasks) != EXPECTED_ITEMS_PER_MODEL or image_count != expected_image_count:
        raise RuntimeError(f"unexpected totals: {len(tasks)} tasks, {image_count} images")
    _assert_geometry(tasks)
    balance_checks = (
        _assert_balanced(tasks) if design_id == "balanced_montage_v2" else None
    )
    montage_checks = (
        _assert_false_belief_montages(tasks)
        if design_id == "balanced_montage_v2"
        else None
    )

    readiness_probe = _build_readiness_probe()
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "design_id": design_id,
        "seed": SEED,
        "n_items": N_ITEMS,
        "item_count": len(tasks),
        "image_count": image_count,
        "scientific_image_count": image_count,
        "readiness_image_count": len(readiness_probe["images"]),
        "total_frozen_image_count": image_count + len(readiness_probe["images"]),
        "task_counts": counts,
        "readiness_probe": readiness_probe,
        "balance_checks": balance_checks,
        "false_belief_render_contract": montage_checks,
        "font": font_provenance(),
        "pillow_version": PIL.__version__,
        "source_revision": _git_head(),
        "source_files": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in SOURCE_FILES
        },
        "tasks": tasks,
    }
    atomic_write_json(IMAGE_MANIFEST, manifest)
    load_image_manifest(verify_images=True)
    print(f"frozen {len(tasks)} tasks and {image_count} images at {IMAGE_MANIFEST}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design",
        choices=sorted(EXPECTED_IMAGE_COUNTS),
        default="legacy_readable_v1",
    )
    args = parser.parse_args()
    build(args.design)


if __name__ == "__main__":
    main()
