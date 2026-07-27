#!/usr/bin/env python3
"""Prospective family-structure audit for the frozen CogArena matrices.

This analysis separates the observed grouping contrast into:

* between-family structure (one centroid per merged model family), and
* within-family structure (paradigm scores centered within families that have
  at least two checkpoints).

It was specified before the confirmatory extension was run.  The script is
CPU-only and is intended to be executed through Slurm, not on a login node.
It writes no model responses and consumes only frozen matrices/family labels.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import socket
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("COGARENA_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_STRICT = ROOT / "results/reanalysis/aplus_20260718/matrix_aplus_strict.csv"
DEFAULT_CONSTRUCT = (
    ROOT / "results/reanalysis/aplus_20260718/matrix_construct_aplus_strict.csv"
)
DEFAULT_FAMILIES = ROOT / "results/reanalysis/aplus_20260718/family_map.json"
DEFAULT_OUT = ROOT / "results/reanalysis/profile_validity_20260720/family_structure.json"

GROUPS = {
    "Working Memory": {"digit_span", "n_back", "operation_span"},
    "Cognitive Control": {"stroop", "flanker", "go_nogo"},
    "Episodic Memory": {"drm_false_memory", "source_monitoring", "cvlt_word_list"},
    "Theory of Mind": {"false_belief", "epitome_tom"},
    "Metacognition": {"confidence_calibration", "post_decision_wagering"},
}


def _req(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAMILY-STRUCTURE GATE FAILED: {message}")


def enforce_c01() -> str:
    revision = os.environ.get("COGARENA_GIT_HEAD", "").strip()
    _req(bool(os.environ.get("SLURM_JOB_ID")), "analysis must run inside Slurm")
    _req(socket.gethostname().split(".", 1)[0].startswith("c01"),
         f"analysis must run on c01, got {socket.gethostname()}")
    _req(bool(re.fullmatch(r"[0-9a-f]{40}", revision)),
         "COGARENA_GIT_HEAD must be a full commit SHA")
    return revision


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def labels_for(columns: Sequence[str]) -> list[str]:
    inverse = {paradigm: group for group, paradigms in GROUPS.items() for paradigm in paradigms}
    _req(set(columns) == set(inverse), f"unexpected paradigm set: {sorted(set(columns) ^ set(inverse))}")
    return [inverse[column] for column in columns]


def pair_indices(labels: Sequence[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    within, cross = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (within if labels[i] == labels[j] else cross).append((i, j))
    return within, cross


def delta_from_matrix(matrix: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    _req(matrix.ndim == 2 and matrix.shape[1] == len(labels), "matrix shape mismatch")
    _req(matrix.shape[0] >= 3, "fewer than three rows")
    corr = np.corrcoef(matrix, rowvar=False)
    within, cross = pair_indices(labels)
    w = np.asarray([corr[i, j] for i, j in within], dtype=float)
    c = np.asarray([corr[i, j] for i, j in cross], dtype=float)
    _req(np.isfinite(w).all() and np.isfinite(c).all(), "non-finite correlation")
    return {
        "within": float(w.mean()),
        "cross": float(c.mean()),
        "delta": float(w.mean() - c.mean()),
    }


def _partitions_equal_size(
    remaining: tuple[int, ...],
    size: int,
    count: int,
    previous: tuple[int, ...] | None = None,
) -> Iterator[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]]:
    """Yield unordered groups of equal size once, plus unused indices."""
    if count == 0:
        yield (), remaining
        return
    for group in itertools.combinations(remaining, size):
        if previous is not None and group <= previous:
            continue
        left = tuple(i for i in remaining if i not in group)
        for tail, unused in _partitions_equal_size(left, size, count - 1, group):
            yield (group,) + tail, unused


def theory_partitions(n: int = 13) -> Iterator[tuple[tuple[int, ...], ...]]:
    """Yield all 600,600 unlabeled partitions with sizes 3,3,3,2,2."""
    _req(n == 13, "the frozen theory partition requires 13 paradigms")
    all_indices = tuple(range(n))
    for triples, left in _partitions_equal_size(all_indices, 3, 3):
        for pairs, unused in _partitions_equal_size(left, 2, 2):
            _req(not unused, "partition left unused indices")
            yield triples + pairs


def exact_partition_p(corr: np.ndarray, observed: float) -> dict[str, float | int]:
    """Exact one/two-sided p over unique 3,3,3,2,2 partitions."""
    n = corr.shape[0]
    _req(corr.shape == (13, 13), "exact test requires a 13x13 correlation matrix")
    upper = np.asarray([corr[i, j] for i in range(n) for j in range(i + 1, n)])
    _req(np.isfinite(upper).all(), "non-finite exact-test input")
    total_sum = float(upper.sum())
    n_within = 3 * math.comb(3, 2) + 2 * math.comb(2, 2)
    n_cross = math.comb(13, 2) - n_within
    ge = abs_ge = count = 0
    tolerance = 1e-12
    for partition in theory_partitions():
        within_sum = 0.0
        for group in partition:
            within_sum += sum(corr[i, j] for i, j in itertools.combinations(group, 2))
        delta = within_sum / n_within - (total_sum - within_sum) / n_cross
        ge += delta >= observed - tolerance
        abs_ge += abs(delta) >= abs(observed) - tolerance
        count += 1
    _req(count == 600_600, f"expected 600600 partitions, got {count}")
    return {"n_partitions": count, "p_one_sided": ge / count, "p_two_sided": abs_ge / count}


def family_bootstrap(
    frame: pd.DataFrame,
    family_by_model: dict[str, str],
    labels: Sequence[str],
    *,
    n_boot: int,
    seed: int,
    center_within_family: bool,
) -> dict[str, object]:
    families = sorted({family_by_model[m] for m in frame.index})
    members = {family: [m for m in frame.index if family_by_model[m] == family] for family in families}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(families, size=len(families), replace=True)
        blocks = []
        for draw, family in enumerate(sampled):
            block = frame.loc[members[family]].copy()
            if center_within_family:
                block = block - block.mean(axis=0)
            block.index = [f"{family}__draw{draw}__{i}" for i in range(len(block))]
            blocks.append(block)
        matrix = pd.concat(blocks).to_numpy(dtype=float)
        if np.any(np.std(matrix, axis=0) == 0):
            continue
        values.append(delta_from_matrix(matrix, labels)["delta"])
    _req(len(values) >= int(n_boot * 0.95), "too many invalid bootstrap replicates")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {
        "n_requested": n_boot,
        "n_effective": len(values),
        "seed": seed,
        "mean": float(np.mean(values)),
        "ci95": [float(lo), float(hi)],
        "fraction_le_zero": float(np.mean(np.asarray(values) <= 0)),
    }


def paired_decomposition_bootstrap(
    frame: pd.DataFrame,
    family_by_model: dict[str, str],
    labels: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap between- and within-family deltas on the same families.

    Each draw samples one of the eligible multi-checkpoint families. The
    between component contributes that family's centroid once, while the
    within component contributes all of its family-centered checkpoints. This
    makes the between-minus-within contrast comparable rather than mixing the
    24-family centroid panel with the 11-family within-lineage panel.
    """
    families = sorted({family_by_model[m] for m in frame.index})
    members = {
        family: [m for m in frame.index if family_by_model[m] == family]
        for family in families
    }
    _req(all(len(value) >= 2 for value in members.values()), "paired decomposition includes singleton")
    rng = np.random.default_rng(seed)
    between_values: list[float] = []
    within_values: list[float] = []
    contrasts: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(families, size=len(families), replace=True)
        centroid_rows = []
        centered_blocks = []
        for draw, family in enumerate(sampled):
            block = frame.loc[members[str(family)]].copy()
            centroid_rows.append(block.mean(axis=0).to_numpy(dtype=float))
            centered = block - block.mean(axis=0)
            centered.index = [f"{family}__draw{draw}__{i}" for i in range(len(centered))]
            centered_blocks.append(centered)
        centroid_matrix = np.asarray(centroid_rows, dtype=float)
        centered_matrix = pd.concat(centered_blocks).to_numpy(dtype=float)
        if np.any(np.std(centroid_matrix, axis=0) == 0) or np.any(
            np.std(centered_matrix, axis=0) == 0
        ):
            continue
        between_delta = delta_from_matrix(centroid_matrix, labels)["delta"]
        within_delta = delta_from_matrix(centered_matrix, labels)["delta"]
        between_values.append(between_delta)
        within_values.append(within_delta)
        contrasts.append(between_delta - within_delta)
    _req(len(contrasts) >= int(n_boot * 0.95), "too many invalid paired bootstrap replicates")

    def summarize(values: list[float]) -> dict[str, object]:
        lo, hi = np.percentile(values, [2.5, 97.5])
        return {
            "mean": float(np.mean(values)),
            "ci95": [float(lo), float(hi)],
            "fraction_le_zero": float(np.mean(np.asarray(values) <= 0)),
        }

    return {
        "n_requested": n_boot,
        "n_effective": len(contrasts),
        "seed": seed,
        "between_family_delta": summarize(between_values),
        "within_family_delta": summarize(within_values),
        "between_minus_within_delta": summarize(contrasts),
    }


def analyze_matrix(
    path: Path,
    family_by_model: dict[str, str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    frame = pd.read_csv(path, index_col=0)
    _req(frame.shape == (55, 13), f"{path} is {frame.shape}, expected 55x13")
    _req(frame.index.is_unique and frame.columns.is_unique, "duplicate model/paradigm labels")
    _req(np.isfinite(frame.to_numpy(dtype=float)).all(), "non-finite matrix cell")
    _req(set(frame.index) == set(family_by_model), "family map/model set mismatch")
    labels = labels_for(list(frame.columns))

    base = delta_from_matrix(frame.to_numpy(dtype=float), labels)
    base_corr = np.corrcoef(frame.to_numpy(dtype=float), rowvar=False)
    base["exact_partition"] = exact_partition_p(base_corr, base["delta"])

    family_series = pd.Series(family_by_model).reindex(frame.index)
    _req(family_series.notna().all(), "missing family label during centroid construction")
    centroids = frame.groupby(family_series).mean()
    _req(len(centroids) == 24, f"expected 24 merged families, got {len(centroids)}")
    between = delta_from_matrix(centroids.to_numpy(dtype=float), labels)
    between["exact_partition"] = exact_partition_p(
        np.corrcoef(centroids.to_numpy(dtype=float), rowvar=False), between["delta"]
    )

    counts = pd.Series(family_by_model).value_counts()
    eligible_families = sorted(counts[counts >= 2].index)
    eligible_models = [m for m in frame.index if family_by_model[m] in eligible_families]
    within_frame = frame.loc[eligible_models].copy()
    within_family_series = pd.Series(family_by_model).reindex(within_frame.index)
    centered = within_frame - within_frame.groupby(within_family_series).transform("mean")
    _req(len(eligible_families) == 11, f"expected 11 multi-checkpoint families, got {len(eligible_families)}")
    _req(len(centered) == 42, f"expected 42 eligible models, got {len(centered)}")
    within = delta_from_matrix(centered.to_numpy(dtype=float), labels)
    within["exact_partition"] = exact_partition_p(
        np.corrcoef(centered.to_numpy(dtype=float), rowvar=False), within["delta"]
    )
    within["family_bootstrap"] = family_bootstrap(
        within_frame,
        family_by_model,
        labels,
        n_boot=n_boot,
        seed=seed,
        center_within_family=True,
    )

    eligible_centroids = within_frame.groupby(within_family_series).mean()
    between_eligible = delta_from_matrix(eligible_centroids.to_numpy(dtype=float), labels)
    between_eligible["exact_partition"] = exact_partition_p(
        np.corrcoef(eligible_centroids.to_numpy(dtype=float), rowvar=False),
        between_eligible["delta"],
    )
    paired_decomposition = paired_decomposition_bootstrap(
        within_frame,
        family_by_model,
        labels,
        n_boot=n_boot,
        seed=seed,
    )

    leave_one_out = {}
    for family in eligible_families:
        kept = within_frame.loc[[m for m in within_frame.index if family_by_model[m] != family]]
        kept_family_series = pd.Series(family_by_model).reindex(kept.index)
        kept_centered = kept - kept.groupby(kept_family_series).transform("mean")
        leave_one_out[family] = delta_from_matrix(kept_centered.to_numpy(dtype=float), labels)["delta"]

    return {
        "input": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "base_55_models": base,
        "between_24_family_centroids": between,
        "same_11_families_decomposition": {
            "between_family_centroids": between_eligible,
            "within_family_centered": {
                "within": within["within"],
                "cross": within["cross"],
                "delta": within["delta"],
                "exact_partition": within["exact_partition"],
            },
            "paired_family_bootstrap": paired_decomposition,
        },
        "within_11_multicheckpoint_families": {
            **within,
            "n_models": len(centered),
            "n_families": len(eligible_families),
            "families": eligible_families,
            "leave_one_family_out_delta": leave_one_out,
            "leave_one_family_out_range": [min(leave_one_out.values()), max(leave_one_out.values())],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--construct", type=Path, default=DEFAULT_CONSTRUCT)
    parser.add_argument("--families", type=Path, default=DEFAULT_FAMILIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    revision = enforce_c01()
    args = parse_args()
    args.strict = args.strict.resolve()
    args.construct = args.construct.resolve()
    args.families = args.families.resolve()
    args.output = args.output.resolve()
    for path in (args.strict, args.construct, args.families):
        _req(path.is_file(), f"missing input {path}")
    family_payload = json.loads(args.families.read_text())
    _req(set(family_payload) >= {"merged", "raw"}, "family map lacks raw/merged labels")
    merged = family_payload["merged"]
    code_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_suffix(".sbatch").resolve(),
        (ROOT / "tests/test_family_structure.py").resolve(),
    )
    _req(all(path.is_file() for path in code_paths), "analysis code dependency missing")

    output = {
        "spec": "family-structure-v1-prospective-20260720",
        "status": "complete",
        "execution": {
            "git_head": revision,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "node": socket.gethostname(),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in code_paths
        },
        "family_map": {
            "path": str(args.families.relative_to(ROOT)),
            "sha256": sha256(args.families),
            "n_merged": len(set(merged.values())),
        },
        "inference": {
            "exact_partition_space": "all 600600 unlabeled 3,3,3,2,2 partitions",
            "bootstrap_unit": "merged model family",
            "bootstrap_replicates": args.n_boot,
            "seed": args.seed,
            "post_hoc": True,
        },
        "strict_accuracy": analyze_matrix(
            args.strict, merged, n_boot=args.n_boot, seed=args.seed
        ),
        "construct_native": analyze_matrix(
            args.construct, merged, n_boot=args.n_boot, seed=args.seed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({
        "output": str(args.output),
        "strict_within_family_delta": output["strict_accuracy"]
            ["within_11_multicheckpoint_families"]["delta"],
        "strict_within_family_p2": output["strict_accuracy"]
            ["within_11_multicheckpoint_families"]["exact_partition"]["p_two_sided"],
    }, indent=2))


if __name__ == "__main__":
    main()
