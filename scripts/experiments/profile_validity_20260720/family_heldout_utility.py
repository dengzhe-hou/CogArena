#!/usr/bin/env python3
"""Family-held-out incremental utility of CogArena grouping information.

For each target paradigm and held-out merged family, compare:

  g-only  : target ~ PC1(other 12 paradigms)
  g+group : target ~ PC1 + residualized mean of same-group peer paradigms

All standardization, PCA, residualization, and regression fits use training
families only.  The target paradigm is never used as a predictor.  A tiny,
pre-frozen ridge penalty is used solely for numerical stability; there is no
data-dependent hyperparameter search.

CPU-only.  Intended for Slurm execution, never a login node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


ROOT = Path(os.environ.get("COGARENA_ROOT", Path(__file__).resolve().parents[3]))
STRICT = ROOT / "results/reanalysis/aplus_20260718/matrix_aplus_strict.csv"
CONSTRUCT = ROOT / "results/reanalysis/aplus_20260718/matrix_construct_aplus_strict.csv"
FAMILIES = ROOT / "results/reanalysis/aplus_20260718/family_map.json"
OUT = ROOT / "results/reanalysis/profile_validity_20260720/family_heldout_utility.json"

GROUPS = {
    "Working Memory": {"digit_span", "n_back", "operation_span"},
    "Cognitive Control": {"stroop", "flanker", "go_nogo"},
    "Episodic Memory": {"drm_false_memory", "source_monitoring", "cvlt_word_list"},
    "Theory of Mind": {"false_belief", "epitome_tom"},
    "Metacognition": {"confidence_calibration", "post_decision_wagering"},
}
RIDGE_ALPHA = 1e-6


def req(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAMILY-HELDOUT GATE FAILED: {message}")


def enforce_c01() -> str:
    revision = os.environ.get("COGARENA_GIT_HEAD", "").strip()
    req(bool(os.environ.get("SLURM_JOB_ID")), "analysis must run inside Slurm")
    req(socket.gethostname().split(".", 1)[0].startswith("c01"),
        f"analysis must run on c01, got {socket.gethostname()}")
    req(bool(re.fullmatch(r"[0-9a-f]{40}", revision)),
        "COGARENA_GIT_HEAD must be a full commit SHA")
    return revision


def sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()


def group_map(columns: Sequence[str]) -> dict[str, str]:
    out = {p: g for g, paradigms in GROUPS.items() for p in paradigms}
    req(set(columns) == set(out), "theory grouping does not match matrix columns")
    return out


def build_features_from_frames(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target: str,
    labels: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build fold features with every transform fitted on training families."""
    req(target in train_frame and target in test_frame, f"missing target={target}")
    req(
        list(train_frame.columns) == list(test_frame.columns),
        "train/test column order differs",
    )
    predictors = [p for p in train_frame.columns if p != target]
    train_raw = train_frame.loc[:, predictors].to_numpy(dtype=float)
    test_raw = test_frame.loc[:, predictors].to_numpy(dtype=float)
    mean = train_raw.mean(axis=0)
    sd = train_raw.std(axis=0, ddof=0)
    req(np.all(sd > 1e-10), f"zero-variance predictor while holding out target={target}")
    train_z = (train_raw - mean) / sd
    test_z = (test_raw - mean) / sd

    pca = PCA(n_components=1, svd_solver="full")
    g_train = pca.fit_transform(train_z).ravel()
    g_test = pca.transform(test_z).ravel()
    # Deterministic orientation: higher PC1 means higher average performance.
    if np.corrcoef(g_train, train_z.mean(axis=1))[0, 1] < 0:
        g_train *= -1
        g_test *= -1

    peer_columns = [
        p for p in predictors if labels[p] == labels[target]
    ]
    req(bool(peer_columns), f"target {target} has no same-group peer")
    peer_idx = [predictors.index(p) for p in peer_columns]
    peer_train = train_z[:, peer_idx].mean(axis=1)
    peer_test = test_z[:, peer_idx].mean(axis=1)
    design = np.column_stack([np.ones_like(g_train), g_train])
    beta, *_ = np.linalg.lstsq(design, peer_train, rcond=None)
    peer_resid_train = peer_train - design @ beta
    peer_resid_test = peer_test - np.column_stack([np.ones_like(g_test), g_test]) @ beta

    return (
        g_train[:, None],
        g_test[:, None],
        np.column_stack([g_train, peer_resid_train]),
        np.column_stack([g_test, peer_resid_test]),
    )


def build_features(
    frame: pd.DataFrame,
    train_models: Sequence[str],
    test_models: Sequence[str],
    target: str,
    labels: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper used by unit tests and small diagnostic calls."""
    return build_features_from_frames(
        frame.loc[list(train_models)], frame.loc[list(test_models)], target, labels
    )


def heldout_predictions(
    frame: pd.DataFrame,
    family_by_model: dict[str, str],
    labels: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    families = sorted(set(family_by_model.values()))
    for target in frame.columns:
        for heldout in families:
            test_models = [m for m in frame.index if family_by_model[m] == heldout]
            train_models = [m for m in frame.index if family_by_model[m] != heldout]
            req(test_models and train_models, f"empty fold {heldout}/{target}")
            # One row per training family prevents families with many released
            # checkpoints from dominating PCA or regression.  Held-out models
            # remain separate predictions; downstream metrics average them to
            # one family-target cell.
            train_rows = frame.loc[train_models]
            train_family = pd.Series(family_by_model).reindex(train_rows.index)
            train_centroids = train_rows.groupby(train_family).mean()
            test_frame = frame.loc[test_models]
            req(
                len(train_centroids) == len(families) - 1,
                f"expected {len(families) - 1} training family centroids",
            )
            gtr, gte, xtr, xte = build_features_from_frames(
                train_centroids, test_frame, target, labels
            )
            y_train_raw = train_centroids[target].to_numpy(dtype=float)
            y_test_raw = test_frame[target].to_numpy(dtype=float)
            y_mean = float(y_train_raw.mean())
            y_sd = float(y_train_raw.std(ddof=0))
            req(y_sd > 1e-10, f"zero-variance outcome while holding out {heldout}/{target}")
            y_train = (y_train_raw - y_mean) / y_sd
            y_test = (y_test_raw - y_mean) / y_sd
            baseline = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(gtr, y_train)
            extended = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(xtr, y_train)
            pred_g = baseline.predict(gte)
            pred_group = extended.predict(xte)
            for model, truth, p0, p1 in zip(test_models, y_test, pred_g, pred_group):
                rows.append({
                    "model": model,
                    "family": heldout,
                    "target": target,
                    "truth_z": float(truth),
                    "pred_g_z": float(p0),
                    "pred_g_group_z": float(p1),
                    "train_outcome_mean": y_mean,
                    "train_outcome_sd": y_sd,
                })
    result = pd.DataFrame(rows)
    req(len(result) == len(frame) * len(frame.columns), "prediction count is not 55x13")
    req(not result.isna().any().any(), "NaN in held-out predictions")
    req(result.groupby(["model", "target"]).size().eq(1).all(), "duplicate held-out prediction")
    return result


def metrics(predictions: pd.DataFrame) -> dict[str, object]:
    data = predictions.copy()
    data["se_g"] = (data.truth_z - data.pred_g_z) ** 2
    data["se_group"] = (data.truth_z - data.pred_g_group_z) ** 2
    data["ae_g"] = np.abs(data.truth_z - data.pred_g_z)
    data["ae_group"] = np.abs(data.truth_z - data.pred_g_group_z)

    # Each family-target cell receives equal weight, regardless of checkpoints.
    cells = data.groupby(["family", "target"], as_index=False)[
        ["se_g", "se_group", "ae_g", "ae_group"]
    ].mean()
    rmse_g = float(np.sqrt(cells.se_g.mean()))
    rmse_group = float(np.sqrt(cells.se_group.mean()))
    per_target = {}
    for target, sub in cells.groupby("target"):
        r0 = float(np.sqrt(sub.se_g.mean()))
        r1 = float(np.sqrt(sub.se_group.mean()))
        per_target[target] = {
            "rmse_g": r0,
            "rmse_g_group": r1,
            "relative_rmse_gain": 1 - r1 / r0,
        }
    return {
        "rmse_g": rmse_g,
        "rmse_g_group": rmse_group,
        "relative_rmse_gain": 1 - rmse_group / rmse_g,
        "mae_g": float(cells.ae_g.mean()),
        "mae_g_group": float(cells.ae_group.mean()),
        "n_positive_targets": int(sum(v["relative_rmse_gain"] > 0 for v in per_target.values())),
        "per_target": per_target,
    }


def family_bootstrap(
    predictions: pd.DataFrame, *, n_boot: int, seed: int
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    families = sorted(predictions.family.unique())
    gains = []
    for _ in range(n_boot):
        sampled = rng.choice(families, size=len(families), replace=True)
        blocks = []
        for draw, family in enumerate(sampled):
            block = predictions[predictions.family == family].copy()
            block["family"] = f"{family}__draw{draw}"
            blocks.append(block)
        gains.append(metrics(pd.concat(blocks, ignore_index=True))["relative_rmse_gain"])
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return {
        "n_boot": n_boot,
        "seed": seed,
        "ci95": [float(lo), float(hi)],
        "mean": float(np.mean(gains)),
        "fraction_le_zero": float(np.mean(np.asarray(gains) <= 0)),
    }


def shuffled_group_control(
    frame: pd.DataFrame,
    families: dict[str, str],
    true_labels: dict[str, str],
    *,
    n_perm: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    columns = list(frame.columns)
    label_values = np.asarray([true_labels[p] for p in columns], dtype=object)
    gains = []
    for _ in range(n_perm):
        shuffled = dict(zip(columns, rng.permutation(label_values).tolist()))
        pred = heldout_predictions(frame, families, shuffled)
        gains.append(metrics(pred)["relative_rmse_gain"])
    return {
        "n_permutations": n_perm,
        "seed": seed,
        "gain_mean": float(np.mean(gains)),
        "gain_ci95": [float(x) for x in np.percentile(gains, [2.5, 97.5])],
        "gains_ge_observed_computed_downstream": False,
        "raw_gains": gains,
    }


def analyze(
    path: Path,
    families: dict[str, str],
    *,
    n_boot: int,
    n_label_perm: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    frame = pd.read_csv(path, index_col=0)
    req(frame.shape == (55, 13), f"unexpected matrix shape {frame.shape}")
    req(set(frame.index) == set(families), "family map/model mismatch")
    req(np.isfinite(frame.to_numpy(dtype=float)).all(), "matrix contains non-finite values")
    labels = group_map(list(frame.columns))
    predictions = heldout_predictions(frame, families, labels)
    summary = metrics(predictions)
    summary["family_bootstrap"] = family_bootstrap(predictions, n_boot=n_boot, seed=seed)
    null = shuffled_group_control(
        frame, families, labels, n_perm=n_label_perm, seed=seed + 10_000
    )
    null["gains_ge_observed"] = int(
        sum(g >= summary["relative_rmse_gain"] for g in null.pop("raw_gains"))
    )
    null["p_one_sided_plus_one"] = (null["gains_ge_observed"] + 1) / (n_label_perm + 1)
    null.pop("gains_ge_observed_computed_downstream")
    summary["shuffled_group_control"] = null
    summary["input"] = str(path.relative_to(ROOT))
    summary["sha256"] = sha256(path)
    return summary, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", type=Path, default=STRICT)
    parser.add_argument("--construct", type=Path, default=CONSTRUCT)
    parser.add_argument("--families", type=Path, default=FAMILIES)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=20_000)
    parser.add_argument("--n-label-perm", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    revision = enforce_c01()
    args = parse_args()
    args.strict = args.strict.resolve()
    args.construct = args.construct.resolve()
    args.families = args.families.resolve()
    args.output = args.output.resolve()
    family_payload = json.loads(args.families.read_text())
    merged = family_payload["merged"]
    code_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_suffix(".sbatch").resolve(),
        (ROOT / "tests/test_family_heldout_utility.py").resolve(),
    )
    req(all(path.is_file() for path in code_paths), "analysis code dependency missing")
    req(len(merged) == 55 and len(set(merged.values())) == 24, "unexpected merged family map")
    strict, pred_strict = analyze(
        args.strict,
        merged,
        n_boot=args.n_boot,
        n_label_perm=args.n_label_perm,
        seed=args.seed,
    )
    construct, pred_construct = analyze(
        args.construct,
        merged,
        n_boot=args.n_boot,
        n_label_perm=args.n_label_perm,
        seed=args.seed,
    )
    for summary in (strict, construct):
        summary["decision_gate"] = {
            "point_gain_ge_005": summary["relative_rmse_gain"] >= 0.05,
            "bootstrap_ci_lower_gt_0": summary["family_bootstrap"]["ci95"][0] > 0,
            "positive_targets_ge_7": summary["n_positive_targets"] >= 7,
        }
    payload = {
        "spec": "family-heldout-profile-utility-v1-prospective-20260720",
        "execution": {
            "git_head": revision,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "node": socket.gethostname(),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in code_paths
        },
        "ridge_alpha_fixed": RIDGE_ALPHA,
        "target_leakage_guard": "target excluded before train-family-only scaling and PCA",
        "training_unit": "one centroid per non-held-out merged model family",
        "outcome_scale": "target z score using non-held-out family centroids only",
        "family_weighting": "equal held-out family x target cells",
        "n_models": 55,
        "n_families": 24,
        "n_targets": 13,
        "strict_accuracy": strict,
        "construct_native": construct,
        "cross_configuration_gate": {
            "construct_does_not_reverse_strict": not (
                strict["relative_rmse_gain"] > 0 and construct["relative_rmse_gain"] < 0
            )
        },
        "family_map": {"path": str(args.families.relative_to(ROOT)), "sha256": sha256(args.families)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A custom development output must never overwrite the formal prediction
    # artifact.  Bind the companion table name to the requested JSON stem.
    pred_path = args.output.with_name(f"{args.output.stem}_predictions.csv")
    prediction_frame = pd.concat(
        [pred_strict.assign(configuration="strict_accuracy"),
         pred_construct.assign(configuration="construct_native")],
        ignore_index=True,
    )
    pred_tmp = pred_path.with_suffix(pred_path.suffix + ".tmp")
    prediction_frame.to_csv(pred_tmp, index=False)
    os.replace(pred_tmp, pred_path)
    payload["predictions"] = {
        "path": str(pred_path.relative_to(ROOT)),
        "sha256": sha256(pred_path),
        "n_rows": len(pred_strict) + len(pred_construct),
    }
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({
        "output": str(args.output),
        "strict_gain": strict["relative_rmse_gain"],
        "construct_gain": construct["relative_rmse_gain"],
    }, indent=2))


if __name__ == "__main__":
    main()
