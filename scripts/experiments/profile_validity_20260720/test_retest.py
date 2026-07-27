#!/usr/bin/env python3
"""Adjacent-administration stability for the frozen CogArena static battery.

This analysis replays two archived administrations through the *current*
paradigm scorers.  It deliberately covers only the static items for which the
same frozen item is present on both occasions:

* 20 models x 421 paired items = 8,420 response pairs;
* eight paradigms (Stroop, Flanker, Digit Span, DRM, Source Monitoring,
  False Belief, Confidence Calibration, and Post-decision Wagering);
* Go/No-Go and EPITOME are excluded because their production versions were
  superseded; the 11 Source Monitoring episodes regenerated after the
  episode-wide de-duplication fix are excluded, leaving the 39 byte-identical
  episodes.

The unit of the primary reliability analysis is a model's mean score in one
paradigm on one occasion (n=20 paired model scores per paradigm).  Item-level
paired summaries are descriptive.  Merged model families are resampled as
clusters for percentile intervals.  The eight-paradigm delta/PC1 analysis is
secondary: two theory groups are represented by only one paradigm in this
eligible subset.

Outputs contain model names, task IDs, scores, hashes, and aggregate metrics,
but never raw model-response text, stimuli, or answer keys.  The formal
occasion is cross-checked against the final scorer overlays and the frozen
strict-primary matrix before any output is committed.

Run through ``test_retest.sbatch`` on c01; do not run on a login node.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import math
import os
import socket
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


ROOT = Path(os.environ.get("COGARENA_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_OCCASION_A = ROOT / "results/_archive/full_eval_20260525_1522"
DEFAULT_OCCASION_B = ROOT / "results/full_eval_20260526_2208"
DEFAULT_EXCLUSIONS = (
    ROOT / "results/reanalysis/sm_20260718/source_monitoring_exclusions.json"
)
DEFAULT_SM_OVERLAY = ROOT / "results/sm_rerun_20260718/sm_scores_overlay.json"
DEFAULT_WAGER_OVERLAY = (
    ROOT
    / "results/reanalysis/profile_validity_20260720/wager_replay/wager_accuracy_overlay.json"
)
DEFAULT_WAGER_MANIFEST = (
    ROOT
    / "results/reanalysis/profile_validity_20260720/wager_replay/WAGER_REPLAY_MANIFEST.json"
)
DEFAULT_RESCORE_DIR = ROOT / "results/rescore_20260702/new_scores"
DEFAULT_PRIMARY_MATRIX = (
    ROOT / "results/reanalysis/aplus_20260718/matrix_aplus_strict.csv"
)
DEFAULT_FAMILY_MAP = ROOT / "results/reanalysis/aplus_20260718/family_map.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/reanalysis/profile_validity_20260720/test_retest"

MODELS = (
    "tinyllama:1.1b",
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen2.5:32b",
    "gemma2:2b",
    "gemma2:9b",
    "gemma2:27b",
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.1:8b",
    "deepseek-r1:7b",
    "deepseek-r1:14b",
    "mistral:7b",
    "mixtral:8x7b",
    "phi3:14b",
    "yi:34b",
    "command-r:35b",
)

PARADIGMS = (
    "digit_span",
    "stroop",
    "flanker",
    "drm_false_memory",
    "source_monitoring",
    "false_belief",
    "confidence_calibration",
    "post_decision_wagering",
)

DOMAIN_MAP = {
    "digit_span": "Working Memory",
    "stroop": "Cognitive Control",
    "flanker": "Cognitive Control",
    "drm_false_memory": "Episodic Memory",
    "source_monitoring": "Episodic Memory",
    "false_belief": "Theory of Mind",
    "confidence_calibration": "Metacognition",
    "post_decision_wagering": "Metacognition",
}

EXPECTED_RAW_COUNTS = {
    "digit_span": 50,
    "stroop": 66,
    "flanker": 66,
    "drm_false_memory": 50,
    "source_monitoring": 50,
    "false_belief": 50,
    "confidence_calibration": 50,
    "post_decision_wagering": 50,
}
EXPECTED_ELIGIBLE_COUNTS = {**EXPECTED_RAW_COUNTS, "source_monitoring": 39}
EXPECTED_PAIRS_PER_MODEL = 421
EXPECTED_TOTAL_PAIRS = 8_420
EXPECTED_N_MODELS = 20
EXPECTED_N_EXCLUDED_SM = 11
OVERLAY_TOL = 5.1e-5  # final per-item overlays are persisted to four decimals
MATRIX_TOL = 6.0e-5   # frozen paper matrix is persisted to five decimals


def req(condition: bool, message: str) -> None:
    """Fail closed with a stable, searchable gate prefix."""
    if not condition:
        raise SystemExit(f"TEST-RETEST GATE FAILED: {message}")


def enforce_c01(
    environment: Mapping[str, str] | None = None,
    hostname: str | None = None,
) -> None:
    env = os.environ if environment is None else environment
    node = socket.gethostname() if hostname is None else hostname
    req(bool(env.get("SLURM_JOB_ID")), "analysis must run inside Slurm")
    req(node.split(".", 1)[0].startswith("c01"), f"analysis must run on c01, got {node}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise SystemExit(f"TEST-RETEST GATE FAILED: path outside COGARENA_ROOT: {path}") from error


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def finite_float(value: Any, context: str) -> float:
    req(not isinstance(value, bool), f"boolean where score expected ({context})")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"TEST-RETEST GATE FAILED: nonnumeric score ({context})") from error
    req(math.isfinite(number), f"non-finite score ({context})")
    req(-1e-12 <= number <= 1.0 + 1e-12, f"score outside [0,1] ({context}): {number}")
    return min(1.0, max(0.0, number))


def metadata_value(item: Any, name: str, default: Any = None) -> Any:
    metadata = getattr(item, "metadata", None)
    value = getattr(metadata, name, default) if metadata is not None else default
    return value.value if hasattr(value, "value") else value


def generate_frozen_items() -> tuple[dict[str, Any], str]:
    """Regenerate seed-42 items and return eligible item map plus gold-free hash."""
    from cogarena.generators.cognitive_control_gen import generate_cc_items
    from cogarena.generators.episodic_memory_gen import generate_em_items
    from cogarena.generators.metacognition_gen import generate_mc_items
    from cogarena.generators.theory_of_mind_gen import generate_tom_items
    from cogarena.generators.working_memory_gen import generate_wm_items

    generators = (
        generate_wm_items,
        generate_cc_items,
        generate_em_items,
        generate_tom_items,
        generate_mc_items,
    )
    generated: dict[str, Any] = {}
    fingerprints: list[dict[str, str]] = []
    for generator in generators:
        items = generator(seed=42, n_per_paradigm=50, include_contamination_probes=False)
        for item in items:
            paradigm = str(metadata_value(item, "paradigm", ""))
            if paradigm not in PARADIGMS:
                continue
            task_id = str(item.task_id)
            req(task_id not in generated, f"duplicate generated task_id: {task_id}")
            generated[task_id] = item
            metadata = getattr(item, "metadata", None)
            parameters = getattr(metadata, "parameters", {}) if metadata is not None else {}
            # The combined digest pins the regenerated stimulus/gold/parameters,
            # while the public output discloses none of those strings.
            fingerprints.append(
                {
                    "task_id": task_id,
                    "paradigm": paradigm,
                    "dimension": str(metadata_value(item, "dimension", "")),
                    "difficulty": str(metadata_value(item, "difficulty", "")),
                    "stimulus_sha256": hashlib.sha256(
                        str(getattr(item, "stimulus", "")).encode("utf-8")
                    ).hexdigest(),
                    "gold_sha256": canonical_hash(getattr(item, "expected_response", None)),
                    "parameters_sha256": canonical_hash(parameters),
                }
            )

    by_paradigm: dict[str, int] = defaultdict(int)
    for item in generated.values():
        by_paradigm[str(metadata_value(item, "paradigm", ""))] += 1
    req(dict(by_paradigm) == EXPECTED_RAW_COUNTS, f"generated counts changed: {dict(by_paradigm)}")
    return generated, canonical_hash(sorted(fingerprints, key=lambda x: x["task_id"]))


def load_excluded_sm(path: Path) -> set[str]:
    req(path.is_file(), f"missing Source Monitoring exclusions: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    req(data.get("n_affected_episodes") == EXPECTED_N_EXCLUDED_SM, "SM exclusion count drift")
    req(
        data.get("unaffected_episodes_verified_byte_identical") == 39,
        "SM unaffected-item verification drift",
    )
    task_ids = {str(entry["task_id"]) for entry in data.get("episodes", [])}
    req(len(task_ids) == EXPECTED_N_EXCLUDED_SM, "SM exclusions are not 11 unique task IDs")
    return task_ids


def resolve_scorer(path: str):
    module_name, attributes = path.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for attribute in attributes.split("."):
        value = getattr(value, attribute)
    return value


def replay_score(item: Any, response: str) -> float:
    """Call the registered paradigm scorer directly; never silently fall back."""
    from cogarena.scoring import PARADIGM_SCORERS, item_accuracy

    paradigm = str(metadata_value(item, "paradigm", ""))
    req(paradigm in PARADIGM_SCORERS, f"no registered scorer for {paradigm}")
    score = resolve_scorer(PARADIGM_SCORERS[paradigm])(item, response)
    req(isinstance(score, dict), f"scorer returned non-dict for {paradigm}")
    return finite_float(item_accuracy(score), f"{paradigm}/{item.task_id}")


def model_dirname(model: str) -> str:
    return "openai_" + model


def raw_paradigm_ids(root: Path, model: str, item_map: Mapping[str, Any]) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {paradigm: set() for paradigm in PARADIGMS}
    for task_id, item in item_map.items():
        expected[str(metadata_value(item, "paradigm", ""))].add(task_id)

    observed: dict[str, set[str]] = {}
    for paradigm in PARADIGMS:
        examples = [item for item in item_map.values() if metadata_value(item, "paradigm") == paradigm]
        req(examples, f"no regenerated item for {paradigm}")
        dimension = str(metadata_value(examples[0], "dimension", ""))
        directory = root / model_dirname(model) / "text" / dimension / paradigm
        req(directory.is_dir(), f"missing paradigm directory: {directory}")
        paths = sorted(directory.glob("*.json"))
        req(
            len(paths) == EXPECTED_RAW_COUNTS[paradigm],
            f"{root.name}/{model}/{paradigm}: expected {EXPECTED_RAW_COUNTS[paradigm]} JSONs, got {len(paths)}",
        )
        ids = {path.stem for path in paths}
        req(len(ids) == len(paths), f"duplicate file stems in {directory}")
        req(ids == expected[paradigm], f"task-id set drift in {root.name}/{model}/{paradigm}")
        observed[paradigm] = ids
    return observed


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"TEST-RETEST GATE FAILED: unreadable JSON {path}") from error
    req(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


def replay_occasion(
    root: Path,
    item_map: Mapping[str, Any],
    excluded_sm: set[str],
    families: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from cogarena.scoring import item_accuracy

    req(root.is_dir(), f"occasion root missing: {root}")
    rows: list[dict[str, Any]] = []
    consumed: list[Path] = []

    actual_model_dirs = {
        path.name for path in root.iterdir() if path.is_dir() and (path / "text").is_dir()
    }
    expected_model_dirs = {model_dirname(model) for model in MODELS}
    # Image-only model directories may coexist; text-capable extras would alter
    # the sampling frame and are therefore rejected.
    req(actual_model_dirs == expected_model_dirs, f"text model set drift in {root}")

    for model in MODELS:
        req(model in families, f"missing family for {model}")
        observed = raw_paradigm_ids(root, model, item_map)
        n_model = 0
        for paradigm in PARADIGMS:
            task_ids = sorted(observed[paradigm] - (excluded_sm if paradigm == "source_monitoring" else set()))
            req(
                len(task_ids) == EXPECTED_ELIGIBLE_COUNTS[paradigm],
                f"eligible count drift for {model}/{paradigm}",
            )
            for task_id in task_ids:
                item = item_map[task_id]
                dimension = str(metadata_value(item, "dimension", ""))
                path = root / model_dirname(model) / "text" / dimension / paradigm / f"{task_id}.json"
                record = load_json(path)
                req(record.get("task_id") == task_id, f"task_id identity mismatch in {path}")
                req(record.get("model_id") == f"openai/{model}", f"model_id identity mismatch in {path}")
                req(record.get("dimension") == dimension, f"dimension identity mismatch in {path}")
                req(record.get("paradigm") == paradigm, f"paradigm identity mismatch in {path}")
                response = record.get("response")
                req(isinstance(response, str), f"response is not a string in {path}")
                score = replay_score(item, response)
                archived_accuracy = finite_float(
                    item_accuracy(record.get("score")),
                    f"archived/{model}/{paradigm}/{task_id}",
                )
                rows.append(
                    {
                        "model": model,
                        "family": families[model],
                        "task_id": task_id,
                        "paradigm": paradigm,
                        "difficulty": str(metadata_value(item, "difficulty", "")),
                        "score": score,
                        "archived_accuracy": archived_accuracy,
                    }
                )
                consumed.append(path)
                n_model += 1
        req(n_model == EXPECTED_PAIRS_PER_MODEL, f"{model}: expected 421 eligible items, got {n_model}")

    req(len(rows) == EXPECTED_TOTAL_PAIRS, f"expected 8420 occasion rows, got {len(rows)}")
    req(len(consumed) == len(set(consumed)), "same input file consumed more than once")
    tree = hashlib.sha256()
    for path in sorted(consumed, key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(bytes.fromhex(sha256(path)))
    return rows, {"n_files": len(consumed), "consumed_tree_sha256": tree.hexdigest()}


def scorer_replay_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Quantify current-scorer differences from the originally stored score."""
    mismatches = []
    by_paradigm: dict[str, int] = defaultdict(int)
    max_error = 0.0
    for row in rows:
        error = abs(float(row["score"]) - float(row["archived_accuracy"]))
        max_error = max(max_error, error)
        if error > 1e-12:
            by_paradigm[str(row["paradigm"])] += 1
            mismatches.append(
                {
                    "model": row["model"],
                    "task_id": row["task_id"],
                    "paradigm": row["paradigm"],
                    "archived_accuracy": float(row["archived_accuracy"]),
                    "replayed_accuracy": float(row["score"]),
                }
            )
    return {
        "n_items": len(rows),
        "n_score_differences": len(mismatches),
        "differences_by_paradigm": dict(sorted(by_paradigm.items())),
        "max_absolute_difference": max_error,
        "differences": mismatches,
    }


def paired_rows(
    occasion_a: Sequence[Mapping[str, Any]], occasion_b: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    def keyed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
        out: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in rows:
            key = (str(row["model"]), str(row["task_id"]))
            req(key not in out, f"duplicate model/task pair: {key}")
            out[key] = row
        return out

    left, right = keyed(occasion_a), keyed(occasion_b)
    req(left.keys() == right.keys(), "occasion pair keys differ")
    out = []
    for key in sorted(left):
        a, b = left[key], right[key]
        for field in ("family", "paradigm", "difficulty"):
            req(a[field] == b[field], f"paired metadata mismatch for {key}/{field}")
        out.append(
            {
                "model": a["model"],
                "family": a["family"],
                "task_id": a["task_id"],
                "paradigm": a["paradigm"],
                "difficulty": a["difficulty"],
                "occasion_a": float(a["score"]),
                "occasion_b": float(b["score"]),
            }
        )
    req(len(out) == EXPECTED_TOTAL_PAIRS, f"paired row count is {len(out)}, not 8420")
    return out


def load_primary_matrix(path: Path) -> dict[str, dict[str, float]]:
    req(path.is_file(), f"primary matrix missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        req(reader.fieldnames is not None and reader.fieldnames[0] == "model", "invalid primary matrix header")
        out = {
            str(row["model"]): {
                paradigm: finite_float(row[paradigm], f"primary/{row['model']}/{paradigm}")
                for paradigm in PARADIGMS
            }
            for row in reader
        }
    req(set(MODELS).issubset(out), "primary matrix lacks one or more 20-model rows")
    return out


def crosscheck_formal_occasion(
    rows: Sequence[Mapping[str, Any]],
    sm_overlay_path: Path,
    wager_overlay_path: Path,
    rescore_dir: Path,
    primary_matrix_path: Path,
) -> dict[str, Any]:
    """Bind replayed formal scores to the final paper's frozen score sources."""
    req(sm_overlay_path.is_file(), f"SM overlay missing: {sm_overlay_path}")
    sm_overlay = json.loads(sm_overlay_path.read_text(encoding="utf-8"))
    req(wager_overlay_path.is_file(), f"wager overlay missing: {wager_overlay_path}")
    wager_overlay = json.loads(wager_overlay_path.read_text(encoding="utf-8"))
    primary = load_primary_matrix(primary_matrix_path)
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)

    overlay_checks = 0
    matrix_checks = 0
    max_overlay_error = 0.0
    max_matrix_error = 0.0
    matrix_mismatches: list[dict[str, Any]] = []
    for model in MODELS:
        req(model in sm_overlay and len(sm_overlay[model]) == 50, f"SM overlay incomplete for {model}")
        req(
            model in wager_overlay and len(wager_overlay[model]) == 50,
            f"wager overlay incomplete for {model}",
        )
        rescore_path = rescore_dir / f"full_eval_20260526_2208__openai_{model}.json"
        req(rescore_path.is_file(), f"rescore overlay missing for {model}")
        rescore = json.loads(rescore_path.read_text(encoding="utf-8"))
        for row in by_model[model]:
            paradigm = str(row["paradigm"])
            task_id = str(row["task_id"])
            actual = float(row["score"])
            expected: Any | None = None
            if paradigm == "source_monitoring":
                req(task_id in sm_overlay[model], f"SM overlay lacks {model}/{task_id}")
                expected = sm_overlay[model][task_id]
            elif paradigm == "post_decision_wagering":
                req(task_id in wager_overlay[model], f"wager overlay lacks {model}/{task_id}")
                expected = wager_overlay[model][task_id]
            elif paradigm in {"digit_span", "stroop", "flanker", "false_belief"}:
                req(task_id in rescore, f"corrected overlay lacks {model}/{task_id}")
                expected = rescore[task_id]
            if expected is not None:
                error = abs(actual - finite_float(expected, f"overlay/{model}/{task_id}"))
                max_overlay_error = max(max_overlay_error, error)
                req(error <= OVERLAY_TOL, f"formal overlay mismatch {model}/{task_id}: {error}")
                overlay_checks += 1

        grouped: dict[str, list[float]] = defaultdict(list)
        paper_source_grouped: dict[str, list[float]] = defaultdict(list)
        for row in by_model[model]:
            paradigm = str(row["paradigm"])
            current = float(row["score"])
            grouped[paradigm].append(current)
            # The final paper matrix uses corrected scorer overlays for the
            # five corrected static paradigms and the current fixed scorer for
            # confidence calibration. DRM was not rescored and therefore
            # consumes its archived per-item scores. Wagering consumes the
            # formal scorer-replay overlay. Bind the cross-check to those exact
            # paper sources while separately reporting scorer drift.
            if paradigm == "source_monitoring":
                paper_value = finite_float(sm_overlay[model][str(row["task_id"])], "paper-source/SM")
            elif paradigm in {"digit_span", "stroop", "flanker", "false_belief"}:
                paper_value = finite_float(rescore[str(row["task_id"])], "paper-source/rescore")
            elif paradigm == "post_decision_wagering":
                paper_value = finite_float(
                    wager_overlay[model][str(row["task_id"])], "paper-source/wager"
                )
            elif paradigm == "confidence_calibration":
                paper_value = current
            else:
                paper_value = float(row["archived_accuracy"])
            paper_source_grouped[paradigm].append(paper_value)
        # The final matrix's SM cell includes the 11 regenerated episodes and
        # is intentionally not comparable to this 39-episode stability subset.
        for paradigm in PARADIGMS:
            if paradigm == "source_monitoring":
                continue
            mean_score = float(np.mean(paper_source_grouped[paradigm]))
            error = abs(mean_score - primary[model][paradigm])
            max_matrix_error = max(max_matrix_error, error)
            if error > MATRIX_TOL:
                matrix_mismatches.append(
                    {
                        "model": model,
                        "paradigm": paradigm,
                        "replayed_mean": mean_score,
                        "matrix_mean": primary[model][paradigm],
                        "absolute_error": error,
                    }
                )
            matrix_checks += 1

    expected_overlay_checks = EXPECTED_N_MODELS * (39 + 50 + 66 + 66 + 50 + 50)
    req(overlay_checks == expected_overlay_checks, f"expected {expected_overlay_checks} overlay checks, got {overlay_checks}")
    req(matrix_checks == EXPECTED_N_MODELS * 7, "formal matrix check count drift")
    req(
        not matrix_mismatches,
        "formal matrix mismatches: " + json.dumps(matrix_mismatches, sort_keys=True),
    )
    return {
        "overlay_item_checks": overlay_checks,
        "matrix_cell_checks": matrix_checks,
        "max_overlay_absolute_error": max_overlay_error,
        "max_matrix_absolute_error": max_matrix_error,
        "status": "all_passed",
        "matrix_source_contract": (
            "Corrected item overlays for digit span, Stroop, Flanker, and false belief; "
            "formal scorer-replay overlay for wagering; current fixed scorer for confidence "
            "calibration; archived production scores for DRM. Source monitoring is excluded "
            "from the 39-item matrix check."
        ),
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x, method="average"), rankdata(y, method="average"))


def icc_a1(values: np.ndarray) -> float:
    """McGraw-Wong/Shrout-Fleiss absolute-agreement single-measure ICC."""
    values = np.asarray(values, dtype=float)
    req(values.ndim == 2 and values.shape[1] == 2, "ICC(A,1) requires n x 2 matrix")
    n, k = values.shape
    if n < 3:
        return math.nan
    grand = float(values.mean())
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ms_rows = k * float(np.sum((row_means - grand) ** 2)) / (n - 1)
    ms_columns = n * float(np.sum((column_means - grand) ** 2)) / (k - 1)
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ms_error = float(np.sum(residual**2)) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return (ms_rows - ms_error) / denominator if abs(denominator) > 1e-15 else math.nan


def pair_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    req(x.shape == y.shape and x.ndim == 1, "pair metric shape mismatch")
    diff = y - x
    shift = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else math.nan
    return {
        "n": int(len(x)),
        "icc_a1": icc_a1(np.column_stack([x, y])),
        "pearson_r": pearson(x, y),
        "spearman_rho": spearman(x, y),
        "mad": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "occasion_shift_b_minus_a": shift,
        "sd_difference": sd_diff,
        "bland_altman_lower": shift - 1.96 * sd_diff,
        "bland_altman_upper": shift + 1.96 * sd_diff,
    }


def structure_metrics(matrix: np.ndarray) -> dict[str, float | int]:
    matrix = np.asarray(matrix, dtype=float)
    req(matrix.ndim == 2 and matrix.shape[1] == len(PARADIGMS), "structure matrix shape mismatch")
    corr = np.corrcoef(matrix, rowvar=False)
    if not np.all(np.isfinite(corr)):
        return {
            "within_mean": math.nan,
            "cross_mean": math.nan,
            "delta": math.nan,
            "pc1_variance_share": math.nan,
            "positive_pair_fraction": math.nan,
        }
    labels = [DOMAIN_MAP[p] for p in PARADIGMS]
    within, cross, all_pairs = [], [], []
    for i in range(len(PARADIGMS)):
        for j in range(i + 1, len(PARADIGMS)):
            value = float(corr[i, j])
            all_pairs.append(value)
            (within if labels[i] == labels[j] else cross).append(value)
    eigenvalues = np.linalg.eigvalsh(corr)
    return {
        "within_mean": float(np.mean(within)),
        "cross_mean": float(np.mean(cross)),
        "delta": float(np.mean(within) - np.mean(cross)),
        "pc1_variance_share": float(eigenvalues[-1] / eigenvalues.sum()),
        "positive_pair_fraction": float(np.mean(np.asarray(all_pairs) > 0)),
        "n_within_pairs": len(within),
        "n_cross_pairs": len(cross),
    }


def exact_grouping_test(matrix: np.ndarray) -> dict[str, float | int]:
    observed = float(structure_metrics(matrix)["delta"])
    labels = tuple(DOMAIN_MAP[p] for p in PARADIGMS)
    assignments = sorted(set(itertools.permutations(labels)))
    corr = np.corrcoef(matrix, rowvar=False)
    values = []
    for assigned in assignments:
        within, cross = [], []
        for i in range(len(assigned)):
            for j in range(i + 1, len(assigned)):
                (within if assigned[i] == assigned[j] else cross).append(corr[i, j])
        values.append(float(np.mean(within) - np.mean(cross)))
    values_array = np.asarray(values)
    return {
        "n_unique_label_assignments": len(values),
        "p_one_sided": float(np.mean(values_array >= observed - 1e-15)),
        "p_two_sided": float(np.mean(np.abs(values_array) >= abs(observed) - 1e-15)),
    }


def aggregate_scores(pairs: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_key[(str(row["model"]), str(row["paradigm"]))].append(row)
    a = np.empty((len(MODELS), len(PARADIGMS)), dtype=float)
    b = np.empty_like(a)
    for i, model in enumerate(MODELS):
        for j, paradigm in enumerate(PARADIGMS):
            rows = by_key[(model, paradigm)]
            req(len(rows) == EXPECTED_ELIGIBLE_COUNTS[paradigm], f"aggregate count drift {model}/{paradigm}")
            a[i, j] = np.mean([float(row["occasion_a"]) for row in rows])
            b[i, j] = np.mean([float(row["occasion_b"]) for row in rows])
    return a, b


def profile_summary(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    correlations = np.asarray([pearson(a[i], b[i]) for i in range(len(a))])
    finite = correlations[np.isfinite(correlations)]
    req(len(finite) == len(MODELS), "undefined model profile correlation")
    return {
        "n_models": len(correlations),
        "mean_model_profile_r": float(np.mean(finite)),
        "median_model_profile_r": float(np.median(finite)),
        "q25_model_profile_r": float(np.quantile(finite, 0.25)),
        "q75_model_profile_r": float(np.quantile(finite, 0.75)),
        "mean_profile_mad": float(np.mean(np.abs(b - a), axis=1).mean()),
        "per_model": {
            model: {
                "profile_r": float(correlations[i]),
                "profile_mad": float(np.mean(np.abs(b[i] - a[i]))),
            }
            for i, model in enumerate(MODELS)
        },
    }


def analysis_points(pairs: Sequence[Mapping[str, Any]], a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    per_paradigm = {
        paradigm: pair_metrics(a[:, j], b[:, j]) for j, paradigm in enumerate(PARADIGMS)
    }
    item_level = {}
    for paradigm in PARADIGMS:
        rows = [row for row in pairs if row["paradigm"] == paradigm]
        x = np.asarray([float(row["occasion_a"]) for row in rows])
        y = np.asarray([float(row["occasion_b"]) for row in rows])
        metrics = pair_metrics(x, y)
        metrics["exact_score_agreement"] = float(np.mean(np.isclose(x, y, atol=1e-12, rtol=0)))
        item_level[paradigm] = metrics

    grand = pair_metrics(a.mean(axis=1), b.mean(axis=1))
    paradigm_centered_a = a - a.mean(axis=0, keepdims=True)
    paradigm_centered_b = b - b.mean(axis=0, keepdims=True)
    pooled_paradigm_centered = pair_metrics(
        paradigm_centered_a.ravel(), paradigm_centered_b.ravel()
    )
    # This is the direct profile-shape estimand: remove each model's grand
    # level separately on each occasion before comparing its deviations across
    # paradigms.  It cannot be driven by stable overall model competence.
    model_centered_a = a - a.mean(axis=1, keepdims=True)
    model_centered_b = b - b.mean(axis=1, keepdims=True)
    pooled_model_centered = pair_metrics(
        model_centered_a.ravel(), model_centered_b.ravel()
    )
    struct_a, struct_b = structure_metrics(a), structure_metrics(b)
    struct_a.update(exact_grouping_test(a))
    struct_b.update(exact_grouping_test(b))
    return {
        "model_level_per_paradigm": per_paradigm,
        "item_level_descriptive": item_level,
        "grand_score_across_models": grand,
        "pooled_paradigm_centered_cells": pooled_paradigm_centered,
        "pooled_model_centered_profile_cells": pooled_model_centered,
        "model_profile_stability": profile_summary(a, b),
        "secondary_eight_paradigm_structure": {
            "scope_warning": (
                "Secondary eligible-subset analysis: Working Memory and Theory of Mind "
                "each contain only one paradigm; do not generalize to the full 13-paradigm battery."
            ),
            "occasion_a": struct_a,
            "occasion_b": struct_b,
            "change_b_minus_a": {
                key: float(struct_b[key]) - float(struct_a[key])
                for key in ("within_mean", "cross_mean", "delta", "pc1_variance_share", "positive_pair_fraction")
            },
        },
    }


BOOT_METRICS = (
    "icc_a1",
    "pearson_r",
    "spearman_rho",
    "mad",
    "rmse",
    "occasion_shift_b_minus_a",
)


def interval(values: Iterable[float], requested: int) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    req(len(array) >= max(100, requested // 2), f"too few valid bootstrap replicates: {len(array)}/{requested}")
    return {
        "ci95": [float(x) for x in np.quantile(array, [0.025, 0.975])],
        "n_valid": int(len(array)),
    }


def family_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    families: Mapping[str, str],
    n_reps: int,
    seed: int,
) -> dict[str, Any]:
    req(n_reps >= 200, "family bootstrap requires at least 200 replicates")
    indices: dict[str, list[int]] = defaultdict(list)
    for index, model in enumerate(MODELS):
        indices[families[model]].append(index)
    family_names = sorted(indices)
    req(len(family_names) >= 8, f"too few merged families: {len(family_names)}")
    rng = np.random.default_rng(seed)

    collected: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_reps):
        sampled_families = rng.choice(family_names, size=len(family_names), replace=True)
        sampled_indices = np.asarray(
            [index for family in sampled_families for index in indices[str(family)]], dtype=int
        )
        aa, bb = a[sampled_indices], b[sampled_indices]
        for j, paradigm in enumerate(PARADIGMS):
            metrics = pair_metrics(aa[:, j], bb[:, j])
            for metric in BOOT_METRICS:
                collected[f"paradigm/{paradigm}/{metric}"].append(float(metrics[metric]))
        for label, x, y in (
            ("grand", aa.mean(axis=1), bb.mean(axis=1)),
            (
                "pooled_paradigm_centered",
                (aa - aa.mean(axis=0, keepdims=True)).ravel(),
                (bb - bb.mean(axis=0, keepdims=True)).ravel(),
            ),
            (
                "pooled_model_centered_profile",
                (aa - aa.mean(axis=1, keepdims=True)).ravel(),
                (bb - bb.mean(axis=1, keepdims=True)).ravel(),
            ),
        ):
            metrics = pair_metrics(x, y)
            for metric in BOOT_METRICS:
                collected[f"aggregate/{label}/{metric}"].append(float(metrics[metric]))

        profile_rs = np.asarray([pearson(aa[i], bb[i]) for i in range(len(aa))])
        collected["profile/mean_r"].append(float(np.nanmean(profile_rs)))
        collected["profile/median_r"].append(float(np.nanmedian(profile_rs)))
        collected["profile/mean_mad"].append(float(np.mean(np.abs(bb - aa), axis=1).mean()))

        sa, sb = structure_metrics(aa), structure_metrics(bb)
        for occasion, structure in (("a", sa), ("b", sb)):
            for metric in ("delta", "pc1_variance_share"):
                collected[f"structure/{occasion}/{metric}"].append(float(structure[metric]))
        for metric in ("delta", "pc1_variance_share"):
            collected[f"structure/change/{metric}"].append(float(sb[metric]) - float(sa[metric]))

    nested: dict[str, Any] = {
        "method": (
            "Merged families sampled with replacement; every checkpoint in each sampled "
            "family retained as a cluster. Percentile 95% intervals."
        ),
        "seed": seed,
        "n_reps": n_reps,
        "n_merged_families": len(family_names),
        "family_sizes": {family: len(indices[family]) for family in family_names},
        "model_level_per_paradigm": {},
        "aggregates": {},
        "model_profile_stability": {},
        "secondary_eight_paradigm_structure": {},
    }
    for paradigm in PARADIGMS:
        nested["model_level_per_paradigm"][paradigm] = {
            metric: interval(collected[f"paradigm/{paradigm}/{metric}"], n_reps)
            for metric in BOOT_METRICS
        }
    for label in (
        "grand",
        "pooled_paradigm_centered",
        "pooled_model_centered_profile",
    ):
        nested["aggregates"][label] = {
            metric: interval(collected[f"aggregate/{label}/{metric}"], n_reps)
            for metric in BOOT_METRICS
        }
    nested["model_profile_stability"] = {
        metric: interval(collected[f"profile/{metric}"], n_reps)
        for metric in ("mean_r", "median_r", "mean_mad")
    }
    for occasion in ("a", "b", "change"):
        nested["secondary_eight_paradigm_structure"][occasion] = {
            metric: interval(collected[f"structure/{occasion}/{metric}"], n_reps)
            for metric in ("delta", "pc1_variance_share")
        }
    return nested


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def assert_no_raw_text_fields(value: Any) -> None:
    """Prevent accidental persistence of raw text-bearing record fields."""
    forbidden = {"response", "responses", "stimulus", "expected_response", "turns"}
    if isinstance(value, dict):
        for key, item in value.items():
            req(str(key).lower() not in forbidden, f"forbidden raw-text field in output: {key}")
            assert_no_raw_text_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_raw_text_fields(item)


def write_pairs_csv(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "model",
        "family",
        "task_id",
        "paradigm",
        "difficulty",
        "occasion_a",
        "occasion_b",
    )
    req(not any("response" in field for field in fields), "raw response column requested")
    lines: list[str] = []
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in pairs:
        writer.writerow({field: row[field] for field in fields})
    lines.append(buffer.getvalue())
    atomic_write(path, "".join(lines))


def git_head() -> str:
    revision = os.environ.get("COGARENA_GIT_HEAD", "").strip()
    req(bool(revision), "COGARENA_GIT_HEAD was not injected at Slurm submission")
    req(
        len(revision) == 40 and all(char in "0123456789abcdef" for char in revision.lower()),
        "COGARENA_GIT_HEAD is not a full 40-character commit SHA",
    )
    return revision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occasion-a", type=Path, default=DEFAULT_OCCASION_A)
    parser.add_argument("--occasion-b", type=Path, default=DEFAULT_OCCASION_B)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--sm-overlay", type=Path, default=DEFAULT_SM_OVERLAY)
    parser.add_argument("--wager-overlay", type=Path, default=DEFAULT_WAGER_OVERLAY)
    parser.add_argument("--wager-manifest", type=Path, default=DEFAULT_WAGER_MANIFEST)
    parser.add_argument("--rescore-dir", type=Path, default=DEFAULT_RESCORE_DIR)
    parser.add_argument("--primary-matrix", type=Path, default=DEFAULT_PRIMARY_MATRIX)
    parser.add_argument("--family-map", type=Path, default=DEFAULT_FAMILY_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    enforce_c01()
    args = parse_args()
    output_dir = args.output_dir.resolve()
    relative_to_root(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "test_retest_results.json"
    pairs_path = output_dir / "paired_scores.csv"
    manifest_path = output_dir / "TEST_RETEST_MANIFEST.json"
    allowed_output_names = {results_path.name, pairs_path.name, manifest_path.name}
    # Invalidate any previous PASS before touching inputs. A failed rerun must
    # leave a non-consumable state rather than an old success manifest.
    atomic_write(
        manifest_path,
        json.dumps(
            {
                "schema_version": "cogarena-test-retest-manifest-v1",
                "status": "running",
                "all_gates_passed": False,
                "git_head": os.environ.get("COGARENA_GIT_HEAD", ""),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    req(not list(output_dir.glob("*.tmp")), "temporary output files pre-exist")
    req(
        all(path.is_file() and not path.is_symlink() and path.name in allowed_output_names
            for path in output_dir.iterdir()),
        "unexpected, non-file, symlink, or stale entry in test-retest output directory",
    )
    for path in (
        args.occasion_a,
        args.occasion_b,
        args.exclusions,
        args.sm_overlay,
        args.wager_overlay,
        args.wager_manifest,
        args.rescore_dir,
        args.primary_matrix,
        args.family_map,
    ):
        req(path.exists(), f"required input missing: {path}")

    family_data = json.loads(args.family_map.read_text(encoding="utf-8"))
    families = {model: str(family_data["merged"][model]) for model in MODELS}
    req(len(families) == EXPECTED_N_MODELS, "family map does not cover exactly 20 models")

    revision = git_head()
    wager_manifest = json.loads(args.wager_manifest.read_text(encoding="utf-8"))
    req(
        wager_manifest.get("schema_version") == "cogarena-wager-replay-manifest-v1"
        and wager_manifest.get("status") == "final"
        and wager_manifest.get("all_gates_passed") is True,
        "formal wagering replay manifest is not final",
    )
    req(
        wager_manifest.get("execution", {}).get("git_head") == revision,
        "formal wagering replay manifest source revision mismatch",
    )
    req(
        wager_manifest.get("checks", {}).get("wager_construct_overlay_representable") is True,
        "formal wagering overlay lacks construct representability gate",
    )
    wager_relative = relative_to_root(args.wager_overlay)
    req(
        wager_manifest.get("outputs", {}).get(wager_relative) == sha256(args.wager_overlay),
        "formal wagering overlay hash mismatch",
    )

    item_map, generated_items_sha = generate_frozen_items()
    excluded_sm = load_excluded_sm(args.exclusions)
    generated_sm_ids = {
        task_id
        for task_id, item in item_map.items()
        if metadata_value(item, "paradigm") == "source_monitoring"
    }
    req(excluded_sm.issubset(generated_sm_ids), "SM exclusion manifest does not match generated battery")

    occasion_a, hash_a = replay_occasion(args.occasion_a, item_map, excluded_sm, families)
    occasion_b, hash_b = replay_occasion(args.occasion_b, item_map, excluded_sm, families)
    replay_audit = {
        "occasion_a": scorer_replay_audit(occasion_a),
        "occasion_b": scorer_replay_audit(occasion_b),
    }
    pairs = paired_rows(occasion_a, occasion_b)
    formal_crosscheck = crosscheck_formal_occasion(
        occasion_b,
        args.sm_overlay,
        args.wager_overlay,
        args.rescore_dir,
        args.primary_matrix,
    )
    a, b = aggregate_scores(pairs)
    points = analysis_points(pairs, a, b)
    bootstrap = family_bootstrap(a, b, families, args.bootstrap_reps, args.seed)

    results = sanitize_json(
        {
            "schema_version": "cogarena-test-retest-v1",
            "estimand": (
                "Adjacent-administration absolute agreement for model-level paradigm means "
                "on the eligible same-item static subset."
            ),
            "scope": {
                "n_models": EXPECTED_N_MODELS,
                "n_paradigms": len(PARADIGMS),
                "paradigms": list(PARADIGMS),
                "n_pairs_per_model": EXPECTED_PAIRS_PER_MODEL,
                "n_response_pairs": EXPECTED_TOTAL_PAIRS,
                "eligible_items_per_paradigm_per_model": EXPECTED_ELIGIBLE_COUNTS,
                "excluded": {
                    "go_nogo": "superseded production form",
                    "epitome_tom": "superseded production form",
                    "source_monitoring": "11 regenerated episodes; only 39 verified-identical episodes retained",
                },
            },
            "formal_crosscheck": formal_crosscheck,
            "current_scorer_vs_archived_audit": replay_audit,
            "point_estimates": points,
            "family_bootstrap": bootstrap,
        }
    )
    assert_no_raw_text_fields(results)

    script_path = Path(__file__).resolve()
    rescore_paths = sorted(args.rescore_dir.glob("full_eval_20260526_2208__openai_*.json"))
    expected_rescore_names = {
        f"full_eval_20260526_2208__openai_{model}.json" for model in MODELS
    }
    req(
        {path.name for path in rescore_paths} == expected_rescore_names,
        "formal rescore overlay file set is not exactly the frozen 20-model panel",
    )
    dependency_paths = [
        ROOT / "cogarena/scoring/__init__.py",
        ROOT / "cogarena/generators/working_memory_gen.py",
        ROOT / "cogarena/generators/cognitive_control_gen.py",
        ROOT / "cogarena/generators/episodic_memory_gen.py",
        ROOT / "cogarena/generators/theory_of_mind_gen.py",
        ROOT / "cogarena/generators/metacognition_gen.py",
        ROOT / "cogarena/dimensions/working_memory.py",
        ROOT / "cogarena/dimensions/cognitive_control.py",
        ROOT / "cogarena/dimensions/episodic_memory.py",
        ROOT / "cogarena/dimensions/theory_of_mind.py",
        ROOT / "cogarena/dimensions/metacognition.py",
    ]
    req(all(path.is_file() for path in dependency_paths), "scorer/generator dependency missing")
    job_path = script_path.with_name("test_retest.sbatch")
    spec_path = script_path.with_name("TEST_RETEST_SPEC.md")
    req(job_path.is_file() and spec_path.is_file(), "job or frozen specification missing")

    inputs = {
        "occasion_a": {
            "path": args.occasion_a.relative_to(ROOT).as_posix(),
            **hash_a,
        },
        "occasion_b": {
            "path": args.occasion_b.relative_to(ROOT).as_posix(),
            **hash_b,
        },
        "source_monitoring_exclusions_sha256": sha256(args.exclusions),
        "sm_overlay_sha256": sha256(args.sm_overlay),
        "wager_overlay_sha256": sha256(args.wager_overlay),
        "wager_manifest_sha256": sha256(args.wager_manifest),
        "primary_matrix_sha256": sha256(args.primary_matrix),
        "family_map_sha256": sha256(args.family_map),
        "generated_item_bundle_sha256": generated_items_sha,
        "rescore_overlay_tree_sha256": canonical_hash(
            [(path.name, sha256(path)) for path in rescore_paths]
        ),
        "scorer_generator_code_tree_sha256": canonical_hash(
            [
                (path.relative_to(ROOT).as_posix(), sha256(path))
                for path in dependency_paths
            ]
        ),
    }

    # No result payload is written until scorer replay, frozen-source
    # cross-checks, statistics, bootstrap, privacy audit, and every input-hash
    # gate pass. The only earlier write is the non-consumable running marker.
    write_pairs_csv(pairs_path, pairs)
    atomic_write(
        results_path,
        json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )

    req(not list(output_dir.glob("*.tmp")), "temporary output files remain")
    req(
        {path.name for path in output_dir.iterdir() if path.is_file()}
        == allowed_output_names,
        "unexpected or stale file in formal test-retest output directory",
    )
    manifest = {
        "schema_version": "cogarena-test-retest-manifest-v1",
        "status": "final",
        "git_head": revision,
        "analysis_script_sha256": sha256(script_path),
        "slurm_job_script_sha256": sha256(job_path),
        "frozen_specification_sha256": sha256(spec_path),
        "bootstrap": {"method": "merged-family cluster percentile", "n_reps": args.bootstrap_reps, "seed": args.seed},
        "inputs": inputs,
        "outputs": {
            "test_retest_results.json": sha256(results_path),
            "paired_scores.csv": sha256(pairs_path),
        },
        "privacy_gate": {
            "raw_response_text_written": False,
            "stimulus_text_written": False,
            "answer_keys_written": False,
        },
        "all_gates_passed": True,
    }
    assert_no_raw_text_fields(manifest)
    atomic_write(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )
    print(
        f"ALL GATES PASSED: {EXPECTED_TOTAL_PAIRS} paired items, "
        f"{EXPECTED_N_MODELS} models, {len(PARADIGMS)} paradigms -> {output_dir}"
    )


if __name__ == "__main__":
    main()
