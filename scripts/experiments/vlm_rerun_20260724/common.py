"""Shared contracts for the frozen VLM remediation run."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("COGARENA_ROOT", Path(__file__).resolve().parents[3])).resolve()
RUN_ROOT = Path(
    os.environ.get("COGARENA_VLM_RUN_ROOT", ROOT / "results" / "vlm_rerun_20260724")
).resolve()
STIMULUS_ROOT = RUN_ROOT / "stimuli"
IMAGE_MANIFEST = STIMULUS_ROOT / "IMAGE_MANIFEST.json"
SCORING_CONTRACT = STIMULUS_ROOT / "SCORING_LABELS.json"

MODELS = (
    "openai/qwen2.5vl:7b",
    "openai/llava:7b",
    "openai/gemma3:4b",
    "openai/moondream:1.8b",
    "openai/llama3.2-vision:11b",
    "openai/minicpm-v:8b",
)
PARADIGMS = ("stroop", "flanker", "false_belief")
EXPECTED_ITEM_COUNTS = {"stroop": 100, "flanker": 100, "false_belief": 50}
EXPECTED_ITEMS_PER_MODEL = sum(EXPECTED_ITEM_COUNTS.values())
EXPECTED_IMAGE_COUNTS = {
    "legacy_readable_v1": 400,
    "balanced_montage_v2": 250,
}
SEED = 42
N_ITEMS = 50
MAX_TOKENS = 256
TEMPERATURE = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def model_safe(model_id: str) -> str:
    if model_id not in MODELS:
        raise ValueError(f"model is outside the frozen six-model set: {model_id}")
    return model_id.replace("/", "_")


def request_fingerprint(
    model_id: str,
    task: dict[str, Any],
    image_manifest_sha256: str,
    scoring_contract_sha256: str,
) -> str:
    return sha256_json(
        {
            "model_id": model_id,
            "api_model": model_id.split("/", 1)[-1],
            "task_id": task["task_id"],
            "prompt": task["prompt"],
            "image_sha256": [entry["sha256"] for entry in task["images"]],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "image_manifest_sha256": image_manifest_sha256,
            "scoring_contract_sha256": scoring_contract_sha256,
        }
    )


def load_image_manifest(*, verify_images: bool = True) -> tuple[dict[str, Any], str]:
    if not IMAGE_MANIFEST.is_file():
        raise RuntimeError(f"missing frozen stimulus manifest: {IMAGE_MANIFEST}")
    manifest = read_json(IMAGE_MANIFEST)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported IMAGE_MANIFEST schema")
    if manifest.get("status") != "frozen":
        raise RuntimeError("stimulus manifest is not frozen")
    if manifest.get("seed") != SEED or manifest.get("n_items") != N_ITEMS:
        raise RuntimeError("stimulus seed/count drift")
    if manifest.get("item_count") != EXPECTED_ITEMS_PER_MODEL:
        raise RuntimeError("stimulus item-count drift")
    design_id = manifest.get("design_id", "legacy_readable_v1")
    if design_id not in EXPECTED_IMAGE_COUNTS:
        raise RuntimeError(f"unsupported VLM stimulus design: {design_id}")
    if manifest.get("image_count") != EXPECTED_IMAGE_COUNTS[design_id]:
        raise RuntimeError("stimulus image-count drift")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_ITEMS_PER_MODEL:
        raise RuntimeError("invalid task manifest")
    task_ids = [task.get("task_id") for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("duplicate task IDs in image manifest")

    counts = {name: 0 for name in PARADIGMS}
    image_count = 0
    for task in tasks:
        paradigm = task.get("paradigm")
        if paradigm not in counts:
            raise RuntimeError(f"unexpected paradigm: {paradigm!r}")
        counts[paradigm] += 1
        images = task.get("images")
        if not isinstance(images, list) or not images:
            raise RuntimeError(f"task {task.get('task_id')} has no images")
        image_count += len(images)
        for image_meta in images:
            rel = image_meta.get("path")
            if not isinstance(rel, str):
                raise RuntimeError("invalid image path")
            path = RUN_ROOT / rel
            if not path.is_file():
                raise RuntimeError(f"missing frozen image: {path}")
            if verify_images and sha256_file(path) != image_meta.get("sha256"):
                raise RuntimeError(f"frozen image hash mismatch: {path}")
    if (
        counts != EXPECTED_ITEM_COUNTS
        or image_count != EXPECTED_IMAGE_COUNTS[design_id]
    ):
        raise RuntimeError(f"stimulus distribution drift: {counts}, images={image_count}")
    readiness_probe = manifest.get("readiness_probe")
    if design_id == "balanced_montage_v2":
        stroop = [task for task in tasks if task["paradigm"] == "stroop"]
        flanker = [task for task in tasks if task["paradigm"] == "flanker"]
        stroop_ink_counts = {
            str(condition): Counter(
                task["factors"]["ink_color"]
                for task in stroop
                if task["congruent"] is condition
            )
            for condition in (True, False)
        }
        stroop_word_counts = {
            str(condition): Counter(
                task["factors"]["word"].lower()
                for task in stroop
                if task["congruent"] is condition
            )
            for condition in (True, False)
        }
        flanker_counts = {
            str(condition): Counter(
                task["factors"]["target_dir"]
                for task in flanker
                if task["congruent"] is condition
            )
            for condition in (True, False)
        }
        expected_flanker = Counter({"left": 25, "right": 25})
        if (
            stroop_ink_counts["True"] != stroop_ink_counts["False"]
            or stroop_word_counts["True"] != stroop_word_counts["False"]
            or any(
                task["factors"]["word"].lower()
                == task["factors"]["ink_color"]
                for task in stroop
                if task["congruent"] is False
            )
            or any(
                counts != expected_flanker
                for counts in flanker_counts.values()
            )
        ):
            raise RuntimeError("balanced stimulus contract is not factorial")
        expected_balance_checks = {
            "stroop_ink_counts_by_condition": {
                key: dict(sorted(value.items()))
                for key, value in stroop_ink_counts.items()
            },
            "stroop_word_counts_by_condition": {
                key: dict(sorted(value.items()))
                for key, value in stroop_word_counts.items()
            },
            "flanker_target_counts_by_condition": {
                key: dict(sorted(value.items()))
                for key, value in flanker_counts.items()
            },
        }
        if manifest.get("balance_checks") != expected_balance_checks:
            raise RuntimeError("stored balance audit does not match frozen tasks")
        render_contract = manifest.get("false_belief_render_contract")
        if (
            not isinstance(render_contract, dict)
            or render_contract.get("render_profile") != "montage_readable_v2"
            or render_contract.get("montage_size") != [1000, 720]
            or float(render_contract.get("effective_primary_font_px", 0)) < 9
        ):
            raise RuntimeError("invalid false-belief montage contract")
        if (
            not isinstance(readiness_probe, dict)
            or readiness_probe.get("output_is_not_scored") is not True
            or readiness_probe.get("expected_transport_status") != "ok"
            or not isinstance(readiness_probe.get("prompt"), str)
        ):
            raise RuntimeError("balanced design lacks the frozen readiness probe")
        probe_images = readiness_probe.get("images")
        if not isinstance(probe_images, list) or len(probe_images) != 1:
            raise RuntimeError("invalid readiness-probe image contract")
        if (
            manifest.get("scientific_image_count")
            != EXPECTED_IMAGE_COUNTS[design_id]
            or manifest.get("readiness_image_count") != 1
            or manifest.get("total_frozen_image_count")
            != EXPECTED_IMAGE_COUNTS[design_id] + 1
        ):
            raise RuntimeError("frozen-image count contract drift")
        for image_meta in probe_images:
            path = RUN_ROOT / image_meta["path"]
            if not path.is_file():
                raise RuntimeError(f"missing readiness-probe image: {path}")
            if verify_images and sha256_file(path) != image_meta.get("sha256"):
                raise RuntimeError(f"readiness-probe image hash mismatch: {path}")
    return manifest, sha256_file(IMAGE_MANIFEST)


def load_scoring_contract(
    manifest: dict[str, Any], image_manifest_sha256: str
) -> tuple[dict[str, Any], str]:
    from .scoring import SCORER_ID

    if not SCORING_CONTRACT.is_file():
        raise RuntimeError(f"missing scoring contract: {SCORING_CONTRACT}")
    contract = read_json(SCORING_CONTRACT)
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != "frozen"
        or contract.get("scorer_id") != SCORER_ID
        or contract.get("image_manifest_sha256") != image_manifest_sha256
    ):
        raise RuntimeError("scoring-contract provenance mismatch")
    tasks = contract.get("tasks")
    manifest_tasks = {task["task_id"]: task for task in manifest["tasks"]}
    if not isinstance(tasks, dict) or set(tasks) != set(manifest_tasks):
        raise RuntimeError("scoring-contract task coverage mismatch")
    for task_id, scoring in tasks.items():
        stimulus = manifest_tasks[task_id]
        if (
            scoring.get("paradigm") != stimulus["paradigm"]
            or scoring.get("expected") != stimulus["expected"]
        ):
            raise RuntimeError(f"scoring-contract identity mismatch: {task_id}")
        allowed = scoring.get("allowed_labels")
        label_space = scoring.get("label_space")
        if (
            not isinstance(allowed, list)
            or not allowed
            or not isinstance(label_space, list)
            or not label_space
            or any(not isinstance(value, str) or not value.strip() for value in allowed)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in label_space
            )
            or len({value.strip().lower() for value in allowed}) != len(allowed)
            or len({value.strip().lower() for value in label_space})
            != len(label_space)
            or not {
                value.strip().lower() for value in allowed
            }.issubset({value.strip().lower() for value in label_space})
            or stimulus["expected"] not in allowed
        ):
            raise RuntimeError(f"invalid scoring labels: {task_id}")
        normalized_allowed = [value.strip().lower() for value in allowed]
        normalized_space = [value.strip().lower() for value in label_space]
        paradigm = stimulus["paradigm"]
        if paradigm == "stroop":
            from cogarena.image_gen.stroop_images import COLOR_NAMES

            if (
                normalized_allowed != list(COLOR_NAMES)
                or normalized_space != list(COLOR_NAMES)
            ):
                raise RuntimeError(f"Stroop label-space drift: {task_id}")
        elif paradigm == "flanker":
            if (
                normalized_allowed != ["left", "right"]
                or normalized_space != ["left", "right"]
            ):
                raise RuntimeError(f"Flanker label-space drift: {task_id}")
        if stimulus["paradigm"] == "false_belief":
            from cogarena.image_gen.false_belief_images import CONTAINER_OPTIONS

            factors = stimulus.get("factors", {})
            characters = factors.get("characters")
            frozen_containers = [
                str(value).strip().lower()
                for value in factors.get("containers", [])
            ]
            full_container_space = [name for name, _ in CONTAINER_OPTIONS]
            if (
                not isinstance(characters, list)
                or len(characters) != 2
                or scoring.get("query_subject") != characters[1]
                or normalized_allowed != frozen_containers
                or normalized_space != full_container_space
            ):
                raise RuntimeError(f"false-belief scoring-contract mismatch: {task_id}")
    source_files = contract.get("source_files", {})
    scoring_rel = "scripts/experiments/vlm_rerun_20260724/scoring.py"
    scoring_path = ROOT / scoring_rel
    if (
        not isinstance(source_files, dict)
        or source_files.get(scoring_rel) != sha256_file(scoring_path)
    ):
        raise RuntimeError("scoring-contract source hash mismatch")
    return contract, sha256_file(SCORING_CONTRACT)


def raw_tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        rel = path.relative_to(RUN_ROOT).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()
