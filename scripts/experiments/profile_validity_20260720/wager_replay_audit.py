#!/usr/bin/env python3
"""Fail-closed replay audit for the frozen Post-Decision Wagering battery.

This script regenerates the 50 seed-42 wagering items, replays every stored
response from the frozen 55-model panel through the current
``PostDecisionWageringGenerator.score``, and compares the result with the
archived production score.  It is deliberately restricted to a c01 Slurm
allocation: importing its helpers is safe, but ``main`` refuses to execute on
a login node or any other compute node.

The output is disclosure-minimal.  It contains model/task identifiers,
difficulty, numeric archived/replayed scores, score differences, aggregate
means, and hashes.  Raw responses, stimuli, questions, and answer keys are
never persisted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import socket
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(os.environ.get("COGARENA_ROOT", Path(__file__).resolve().parents[3])).resolve()
DEFAULT_PRIMARY = ROOT / "results/full_eval_20260526_2208"
DEFAULT_EXPANSION = ROOT / "results/full_eval_expansion"
DEFAULT_OUTPUT = ROOT / "results/reanalysis/profile_validity_20260720/wager_replay"

PARADIGM = "post_decision_wagering"
DIMENSION = "metacognition"
N_ITEMS = 50
N_MODELS = 55
N_RECORDS = N_ITEMS * N_MODELS
SEED = 42
SCORE_FIELDS = ("accuracy", "did_bet", "is_correct", "points")
KNOWN_MODEL = "phi3:14b"
KNOWN_TASK = "mc_pdw_gen_easy_math_s1167672954"

PRIMARY_MODELS = (
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

EXPANSION_MODELS = (
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:14b",
    "gemma3:1b",
    "gemma3:12b",
    "gemma3:27b",
    "phi3:3.8b",
    "phi4:14b",
    "llama2:7b",
    "llama2:13b",
    "mistral-nemo:12b",
    "yi:6b",
    "yi:9b",
    "falcon3:7b",
    "falcon3:10b",
    "olmo2:7b",
    "olmo2:13b",
    "smollm2:1.7b",
    "smollm2:360m",
    "internlm2:7b",
    "glm4:9b",
    "aya:8b",
    "solar:10.7b",
    "exaone3.5:7.8b",
    "stablelm2:1.6b",
    "deepseek-llm:7b",
    "openchat:7b",
    "zephyr:7b",
    "starling-lm:7b",
    "nemotron-mini:4b",
    "llama3.1:70b",
    "qwen2.5:72b",
    "mixtral:8x22b",
)


def req(condition: bool, message: str) -> None:
    """Fail closed with a stable, searchable gate prefix."""
    if not condition:
        raise SystemExit(f"WAGER-REPLAY GATE FAILED: {message}")


def enforce_c01(
    environment: Mapping[str, str] | None = None,
    hostname: str | None = None,
) -> None:
    """Refuse substantive execution outside a c01 Slurm allocation."""
    env = os.environ if environment is None else environment
    node = socket.gethostname() if hostname is None else hostname
    req(bool(env.get("SLURM_JOB_ID")), "analysis must run inside Slurm")
    req(node.split(".", 1)[0].startswith("c01"), f"analysis must run on c01, got {node}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise SystemExit(f"WAGER-REPLAY GATE FAILED: path outside COGARENA_ROOT: {path}") from error


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def finite_number(value: Any, context: str) -> float:
    req(not isinstance(value, bool), f"boolean where numeric score expected ({context})")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"WAGER-REPLAY GATE FAILED: nonnumeric score ({context})") from error
    req(math.isfinite(number), f"non-finite score ({context})")
    return number


def validate_score(score: Any, context: str) -> dict[str, float]:
    req(isinstance(score, dict), f"score is not an object ({context})")
    req(set(SCORE_FIELDS).issubset(score), f"score fields missing ({context})")
    out = {field: finite_number(score[field], f"{context}/{field}") for field in SCORE_FIELDS}
    for field in ("accuracy", "did_bet", "is_correct"):
        req(out[field] in (0.0, 1.0), f"{field} is not binary ({context})")
    req(out["accuracy"] == out["is_correct"], f"accuracy/is_correct disagree ({context})")
    req(out["points"] in (-10.0, 2.0, 10.0), f"unexpected payoff ({context})")
    return out


def metadata_value(item: Any, field: str) -> str:
    metadata = getattr(item, "metadata", None)
    value = getattr(metadata, field, "") if metadata is not None else ""
    return str(value.value if hasattr(value, "value") else value)


def generated_wagering_items() -> tuple[dict[str, Any], str]:
    """Regenerate the frozen items and return an aggregate, non-disclosing hash."""
    from cogarena.generators.metacognition_gen import generate_mc_items

    all_items = generate_mc_items(
        seed=SEED,
        n_per_paradigm=N_ITEMS,
        include_contamination_probes=False,
    )
    items = {
        str(item.task_id): item
        for item in all_items
        if metadata_value(item, "paradigm") == PARADIGM
    }
    req(len(items) == N_ITEMS, f"current generator produced {len(items)} wagering items, not 50")
    req(len(set(items)) == N_ITEMS, "current generator produced duplicate wagering task IDs")
    req(KNOWN_TASK in items, f"known task absent from regenerated battery: {KNOWN_TASK}")

    fingerprints: list[dict[str, str]] = []
    for task_id, item in sorted(items.items()):
        req(metadata_value(item, "dimension") == DIMENSION, f"generated dimension drift: {task_id}")
        req(metadata_value(item, "paradigm") == PARADIGM, f"generated paradigm drift: {task_id}")
        req(getattr(item, "expected_response", None) not in (None, ""), f"missing generated gold: {task_id}")
        metadata = getattr(item, "metadata", None)
        parameters = getattr(metadata, "parameters", {}) if metadata is not None else {}
        fingerprints.append(
            {
                "task_id": task_id,
                "dimension": metadata_value(item, "dimension"),
                "paradigm": metadata_value(item, "paradigm"),
                "difficulty": metadata_value(item, "difficulty"),
                "stimulus_sha256": hashlib.sha256(
                    str(getattr(item, "stimulus", "")).encode("utf-8")
                ).hexdigest(),
                "expected_sha256": canonical_hash(getattr(item, "expected_response")),
                "parameters_sha256": canonical_hash(parameters),
            }
        )
    return items, canonical_hash(fingerprints)


def audit_record(
    record: Any,
    item: Any,
    expected_model: str,
    dataset: str,
) -> dict[str, Any]:
    """Validate one archived record and return a disclosure-minimal score row."""
    from cogarena.dimensions.metacognition import PostDecisionWageringGenerator

    task_id = str(getattr(item, "task_id", ""))
    context = f"{dataset}/{expected_model}/{task_id}"
    req(isinstance(record, dict), f"record is not an object ({context})")
    req(record.get("task_id") == task_id, f"task identity mismatch ({context})")
    req(record.get("model_id") == f"openai/{expected_model}", f"model identity mismatch ({context})")
    req(record.get("dimension") == DIMENSION, f"dimension identity mismatch ({context})")
    req(record.get("paradigm") == PARADIGM, f"paradigm identity mismatch ({context})")
    req(record.get("difficulty") == metadata_value(item, "difficulty"), f"difficulty mismatch ({context})")
    response = record.get("response")
    req(isinstance(response, str), f"stored response is not a string ({context})")

    archived = validate_score(record.get("score"), f"archived/{context}")
    replayed = validate_score(
        PostDecisionWageringGenerator.score(item, response),
        f"replayed/{context}",
    )
    differences = {field: replayed[field] - archived[field] for field in SCORE_FIELDS}
    return {
        "dataset": dataset,
        "model": expected_model,
        "task_id": task_id,
        "difficulty": metadata_value(item, "difficulty"),
        **{f"archived_{field}": archived[field] for field in SCORE_FIELDS},
        **{f"replayed_{field}": replayed[field] for field in SCORE_FIELDS},
        **{f"difference_{field}": differences[field] for field in SCORE_FIELDS},
        "score_changed": any(abs(value) > 1e-12 for value in differences.values()),
    }


def tree_hash(root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    ordered = sorted(paths, key=lambda path: path.relative_to(root).as_posix())
    req(len(ordered) == len(set(ordered)), f"duplicate consumed path under {root}")
    digest = hashlib.sha256()
    for path in ordered:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return {"n_files": len(ordered), "tree_sha256": digest.hexdigest()}


def audit_source(
    root: Path,
    models: Sequence[str],
    dataset: str,
    items: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    req(root.is_dir(), f"missing input root: {root}")
    expected_ids = set(items)
    rows: list[dict[str, Any]] = []
    consumed: list[Path] = []
    for model in models:
        directory = root / f"openai_{model}/text/{DIMENSION}/{PARADIGM}"
        req(directory.is_dir(), f"missing wagering directory: {directory}")
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
        req(
            all(path.is_file() and not path.is_symlink() and path.suffix == ".json" for path in entries),
            f"unexpected directory entry in {directory}",
        )
        req(len(entries) == N_ITEMS, f"{model}: expected 50 item files, got {len(entries)}")
        file_ids = {path.stem for path in entries}
        req(file_ids == expected_ids, f"{model}: archived task set differs from regenerated task set")
        for path in entries:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise SystemExit(f"WAGER-REPLAY GATE FAILED: unreadable JSON: {path}") from error
            rows.append(audit_record(record, items[path.stem], model, dataset))
            consumed.append(path)
    req(len(rows) == len(models) * N_ITEMS, f"{dataset}: row count drift")
    return rows, {
        "path": relative_to_root(root),
        **tree_hash(root, consumed),
    }


def summarize_models(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["model"]))].append(row)
    req(len(grouped) == N_MODELS, f"expected 55 model cells, got {len(grouped)}")
    output = []
    for (dataset, model), model_rows in sorted(grouped.items()):
        req(len(model_rows) == N_ITEMS, f"{model}: corrected mean is not based on 50 items")
        archived = sum(float(row["archived_accuracy"]) for row in model_rows) / N_ITEMS
        replayed = sum(float(row["replayed_accuracy"]) for row in model_rows) / N_ITEMS
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "n_items": N_ITEMS,
                "archived_mean_accuracy": archived,
                "corrected_mean_accuracy": replayed,
                "mean_accuracy_difference": replayed - archived,
                "n_score_changed": sum(bool(row["score_changed"]) for row in model_rows),
                "n_accuracy_changed": sum(
                    abs(float(row["difference_accuracy"])) > 1e-12 for row in model_rows
                ),
            }
        )
    return output


def assert_known_correction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matches = [
        row for row in rows
        if row["model"] == KNOWN_MODEL and row["task_id"] == KNOWN_TASK
    ]
    req(len(matches) == 1, "known phi3:14b audit record is not unique")
    row = matches[0]
    req(row["dataset"] == "full_eval_20260526_2208", "known record is in wrong source")
    req(row["archived_accuracy"] == 0.0, "known record archived accuracy is not 0")
    req(row["replayed_accuracy"] == 1.0, "known record replayed accuracy is not 1")
    req(row["difference_accuracy"] == 1.0, "known record accuracy difference is not +1")
    return {
        "model": KNOWN_MODEL,
        "task_id": KNOWN_TASK,
        "archived_accuracy": 0.0,
        "replayed_accuracy": 1.0,
        "status": "verified",
    }


def assert_disclosure_minimal(value: Any, location: str = "root") -> None:
    """Ensure no private text-bearing payload fields enter persisted outputs."""
    banned = {"response", "stimulus", "question", "answer", "gold", "expected_response"}
    if isinstance(value, dict):
        for key, child in value.items():
            req(str(key).lower() not in banned, f"private field in output ({location}/{key})")
            assert_disclosure_minimal(child, f"{location}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_disclosure_minimal(child, f"{location}/{index}")


def rows_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "dataset",
        "model",
        "task_id",
        "difficulty",
        "archived_accuracy",
        "replayed_accuracy",
        "difference_accuracy",
        "archived_did_bet",
        "replayed_did_bet",
        "difference_did_bet",
        "archived_is_correct",
        "replayed_is_correct",
        "difference_is_correct",
        "archived_points",
        "replayed_points",
        "difference_points",
        "score_changed",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return buffer.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--expansion-root", type=Path, default=DEFAULT_EXPANSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    enforce_c01()
    args = parse_args()
    req(args.primary_root.resolve() == DEFAULT_PRIMARY.resolve(),
        "formal replay refuses a noncanonical primary input root")
    req(args.expansion_root.resolve() == DEFAULT_EXPANSION.resolve(),
        "formal replay refuses a noncanonical expansion input root")
    req(args.output_dir.resolve() == DEFAULT_OUTPUT.resolve(),
        "formal replay refuses a noncanonical output root")
    git_head = os.environ.get("COGARENA_GIT_HEAD", "")
    req(bool(re.fullmatch(r"[0-9a-f]{40}", git_head)), "COGARENA_GIT_HEAD must be a full commit SHA")
    req(len(PRIMARY_MODELS) == 20, "primary model panel drift")
    req(len(EXPANSION_MODELS) == 35, "expansion model panel drift")
    req(not set(PRIMARY_MODELS) & set(EXPANSION_MODELS), "model panels overlap")

    output_dir = args.output_dir.resolve()
    relative_to_root(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "wager_replay_items.csv"
    results_path = output_dir / "wager_replay_results.json"
    overlay_path = output_dir / "wager_accuracy_overlay.json"
    manifest_path = output_dir / "WAGER_REPLAY_MANIFEST.json"
    allowed_names = {
        rows_path.name,
        results_path.name,
        overlay_path.name,
        manifest_path.name,
    }
    # Invalidate any earlier PASS before performing a new replay. A failed
    # rerun must never leave a stale, consumable success manifest behind.
    atomic_write(
        manifest_path,
        json.dumps(
            {
                "schema_version": "cogarena-wager-replay-manifest-v1",
                "all_gates_passed": False,
                "status": "running",
                "execution": {"slurm_job_id": os.environ["SLURM_JOB_ID"], "git_head": git_head},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    req(not list(output_dir.glob("*.tmp")), "temporary output files pre-exist")
    req(
        all(path.is_file() and not path.is_symlink() and path.name in allowed_names
            for path in output_dir.iterdir()),
        "unexpected, non-file, symlink, or stale entry in wagering replay output directory",
    )

    items, generated_bundle_sha = generated_wagering_items()
    primary_rows, primary_tree = audit_source(
        args.primary_root.resolve(), PRIMARY_MODELS, "full_eval_20260526_2208", items
    )
    expansion_rows, expansion_tree = audit_source(
        args.expansion_root.resolve(), EXPANSION_MODELS, "full_eval_expansion", items
    )
    rows = sorted(primary_rows + expansion_rows, key=lambda row: (row["model"], row["task_id"]))
    req(len(rows) == N_RECORDS, f"expected 2,750 replayed records, got {len(rows)}")
    req(
        len({(row["model"], row["task_id"]) for row in rows}) == N_RECORDS,
        "duplicate model/task replay key",
    )
    known = assert_known_correction(rows)
    did_bet_changes = sum(
        abs(float(row["difference_did_bet"])) > 1e-12 for row in rows
    )
    req(
        did_bet_changes == 0,
        "accuracy-only overlay cannot represent a did_bet scorer change",
    )
    means = summarize_models(rows)
    changed = [row for row in rows if row["score_changed"]]
    accuracy_overlay: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        model = str(row["model"])
        task_id = str(row["task_id"])
        req(task_id not in accuracy_overlay[model], f"duplicate overlay key {model}/{task_id}")
        accuracy_overlay[model][task_id] = float(row["replayed_accuracy"])
    req(len(accuracy_overlay) == N_MODELS, "accuracy overlay model count drift")
    req(all(len(values) == N_ITEMS for values in accuracy_overlay.values()),
        "accuracy overlay item count drift")

    results = {
        "schema_version": "cogarena-wager-replay-v1",
        "estimand": (
            "Per-model mean accuracy across the frozen 50-item Post-Decision Wagering "
            "battery after replay through the current generator scorer."
        ),
        "scope": {
            "seed": SEED,
            "n_models": N_MODELS,
            "n_items_per_model": N_ITEMS,
            "n_replayed_records": N_RECORDS,
            "n_score_differences": len(changed),
            "n_accuracy_differences": sum(
                abs(float(row["difference_accuracy"])) > 1e-12 for row in rows
            ),
            "n_did_bet_differences": did_bet_changes,
        },
        "known_correction_gate": known,
        "corrected_model_means": means,
        "grand_means": {
            "archived_accuracy": sum(float(row["archived_accuracy"]) for row in rows) / N_RECORDS,
            "corrected_accuracy": sum(float(row["replayed_accuracy"]) for row in rows) / N_RECORDS,
        },
        "privacy": {
            "raw_text_persisted": False,
            "item_payload_persisted": False,
            "persisted_fields": "identifiers, difficulty, numeric scores, differences, means, and hashes only",
        },
    }
    assert_disclosure_minimal(results)
    assert_disclosure_minimal(rows)
    assert_disclosure_minimal(accuracy_overlay)

    script_path = Path(__file__).resolve()
    job_path = script_path.with_suffix(".sbatch")
    test_path = ROOT / "tests/test_wager_replay_audit.py"
    chain_test_path = ROOT / "tests/test_wager_overlay_chain.py"
    dependency_paths = (
        script_path,
        job_path,
        test_path,
        chain_test_path,
        ROOT / "cogarena/core.py",
        ROOT / "cogarena/generators/metacognition_gen.py",
        ROOT / "cogarena/dimensions/metacognition.py",
    )
    req(all(path.is_file() for path in dependency_paths), "code dependency missing")
    code_hashes = {
        relative_to_root(path): sha256(path)
        for path in dependency_paths
    }

    atomic_write(rows_path, rows_csv(rows))
    atomic_write(
        results_path,
        json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )
    atomic_write(
        overlay_path,
        json.dumps(accuracy_overlay, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    # All payload files must be the only non-manifest outputs before a PASS is
    # constructed. No validation is allowed after the final atomic manifest
    # replacement.
    req(not list(output_dir.glob("*.tmp")), "temporary output files remain")
    req(
        {path.name for path in output_dir.iterdir() if path.is_file()}
        == allowed_names,
        "unexpected or stale file in wagering replay output directory",
    )
    manifest = {
        "schema_version": "cogarena-wager-replay-manifest-v1",
        "status": "final",
        "all_gates_passed": True,
        "execution": {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "node": socket.gethostname(),
            "required_node": "c01",
            "git_head": git_head,
        },
        "generator_contract": {
            "entrypoint": "cogarena.generators.metacognition_gen.generate_mc_items",
            "score_entrypoint": (
                "cogarena.dimensions.metacognition.PostDecisionWageringGenerator.score"
            ),
            "seed": SEED,
            "n_per_paradigm": N_ITEMS,
            "include_contamination_probes": False,
            "generated_item_bundle_sha256": generated_bundle_sha,
            "task_id_alignment": "50/50 for every one of 55 models",
        },
        "inputs": {
            "full_eval_20260526_2208": primary_tree,
            "full_eval_expansion": expansion_tree,
            "combined_consumed_tree_sha256": canonical_hash(
                {
                    "full_eval_20260526_2208": primary_tree,
                    "full_eval_expansion": expansion_tree,
                }
            ),
            "frozen_model_panel_sha256": canonical_hash(
                {"primary": PRIMARY_MODELS, "expansion": EXPANSION_MODELS}
            ),
        },
        "code": {
            "files": code_hashes,
            "tree_sha256": canonical_hash(sorted(code_hashes.items())),
        },
        "checks": {
            "model_identity_checks": N_RECORDS,
            "task_identity_checks": N_RECORDS,
            "paradigm_dimension_checks": N_RECORDS,
            "stored_text_type_checks": N_RECORDS,
            "score_schema_checks": 2 * N_RECORDS,
            "known_phi3_numeric_normalization_case": known,
            "wager_construct_overlay_representable": did_bet_changes == 0,
            "did_bet_difference_count": did_bet_changes,
            "disclosure_minimal_output": True,
        },
        "outputs": {
            relative_to_root(rows_path): sha256(rows_path),
            relative_to_root(results_path): sha256(results_path),
            relative_to_root(overlay_path): sha256(overlay_path),
        },
    }
    assert_disclosure_minimal(manifest)
    atomic_write(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )
    print(
        "ALL GATES PASSED: "
        f"{N_RECORDS} wagering records replayed; "
        f"{len(changed)} score records changed; known phi3 correction verified"
    )


if __name__ == "__main__":
    main()
