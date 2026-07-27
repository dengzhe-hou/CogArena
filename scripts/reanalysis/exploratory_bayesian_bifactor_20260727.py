#!/usr/bin/env python3
"""Exploratory multilevel Bayesian bifactor analysis for CogArena.

This analysis is deliberately isolated from the frozen manuscript chain.  It
compares a multilevel general-factor model with a multilevel bifactor model
using family-held-out predictive log density.  Both models include a latent
family-level general factor; the bifactor model additionally includes five
orthogonal grouping factors at checkpoint and family levels.

The implementation uses Pyro mean-field variational inference.  It is an
exploratory regularized analysis, not a replacement for the paper's exact
permutation and family-bootstrap inference.  In particular, two proposed
groupings have only two indicators, and mean-field intervals can be too narrow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.initialization import init_to_median
from pyro.optim import ClippedAdam


PARADIGM_GROUPS = {
    "digit_span": "working_memory",
    "n_back": "working_memory",
    "operation_span": "working_memory",
    "stroop": "cognitive_control",
    "flanker": "cognitive_control",
    "go_nogo": "cognitive_control",
    "cvlt_word_list": "episodic_memory",
    "drm_false_memory": "episodic_memory",
    "source_monitoring": "episodic_memory",
    "false_belief": "theory_of_mind",
    "epitome_tom": "theory_of_mind",
    "confidence_calibration": "metacognition",
    "post_decision_wagering": "metacognition",
}
GROUP_ORDER = [
    "working_memory",
    "cognitive_control",
    "episodic_memory",
    "theory_of_mind",
    "metacognition",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def model(
    data: torch.Tensor,
    family_index: torch.Tensor,
    n_families: int,
    group_index: torch.Tensor,
    use_specific: bool,
    specific_prior_scale: float,
) -> None:
    """Hierarchical Gaussian factor model on standardized paradigm scores."""
    n_rows, n_items = data.shape
    n_groups = len(GROUP_ORDER)
    zero = data.new_tensor(0.0)

    mu = pyro.sample(
        "mu",
        dist.Normal(zero, data.new_tensor(0.35))
        .expand([n_items])
        .to_event(1),
    )
    loading_general = pyro.sample(
        "loading_general",
        dist.HalfNormal(data.new_tensor(0.8))
        .expand([n_items])
        .to_event(1),
    )
    sd_family_general = pyro.sample(
        "sd_family_general", dist.HalfNormal(data.new_tensor(0.5))
    )
    family_general = pyro.sample(
        "family_general",
        dist.Normal(zero, sd_family_general)
        .expand([n_families])
        .to_event(1),
    )
    row_general = pyro.sample(
        "row_general",
        dist.Normal(zero, data.new_tensor(1.0))
        .expand([n_rows])
        .to_event(1),
    )

    eta_general = row_general + family_general[family_index]
    expectation = mu.unsqueeze(0) + (
        eta_general.unsqueeze(1) * loading_general.unsqueeze(0)
    )

    if use_specific:
        loading_specific = pyro.sample(
            "loading_specific",
            dist.HalfNormal(data.new_tensor(specific_prior_scale))
            .expand([n_items])
            .to_event(1),
        )
        sd_family_specific = pyro.sample(
            "sd_family_specific",
            dist.HalfNormal(data.new_tensor(0.35))
            .expand([n_groups])
            .to_event(1),
        )
        family_specific = pyro.sample(
            "family_specific",
            dist.Normal(
                torch.zeros(n_families, n_groups, device=data.device),
                sd_family_specific.unsqueeze(0).expand(n_families, n_groups),
            ).to_event(2),
        )
        row_specific = pyro.sample(
            "row_specific",
            dist.Normal(zero, data.new_tensor(1.0))
            .expand([n_rows, n_groups])
            .to_event(2),
        )
        eta_specific = row_specific + family_specific[family_index]
        expectation = expectation + (
            eta_specific[:, group_index] * loading_specific.unsqueeze(0)
        )

    residual_sd = pyro.sample(
        "residual_sd",
        dist.HalfNormal(data.new_tensor(0.75))
        .expand([n_items])
        .to_event(1),
    )
    pyro.sample(
        "obs",
        dist.Normal(expectation, residual_sd.unsqueeze(0)).to_event(2),
        obs=data,
    )


def fit_model(
    x: np.ndarray,
    family_labels: list[str],
    group_index: np.ndarray,
    use_specific: bool,
    specific_prior_scale: float,
    seed: int,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[AutoNormal, dict, dict]:
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    family_names = sorted(set(family_labels))
    family_to_index = {f: i for i, f in enumerate(family_names)}
    family_index_np = np.array([family_to_index[f] for f in family_labels])

    data = torch.as_tensor(x, dtype=torch.float32, device=device)
    family_index = torch.as_tensor(
        family_index_np, dtype=torch.long, device=device
    )
    group_t = torch.as_tensor(group_index, dtype=torch.long, device=device)

    def bound_model(observed: torch.Tensor = data) -> None:
        model(
            observed,
            family_index,
            len(family_names),
            group_t,
            use_specific,
            specific_prior_scale,
        )

    guide = AutoNormal(
        bound_model,
        init_loc_fn=init_to_median(num_samples=20),
    )
    svi = SVI(
        bound_model,
        guide,
        ClippedAdam({"lr": lr, "clip_norm": 10.0}),
        loss=Trace_ELBO(),
    )

    losses: list[float] = []
    best_tail = math.inf
    stale = 0
    minimum_steps = min(2500, steps)
    for step in range(steps):
        loss = float(svi.step()) / x.size
        require(math.isfinite(loss), f"non-finite ELBO at step {step}")
        losses.append(loss)
        if step >= minimum_steps and (step + 1) % 250 == 0:
            tail = float(np.mean(losses[-250:]))
            if tail < best_tail - 1e-4:
                best_tail = tail
                stale = 0
            else:
                stale += 1
            if stale >= 8:
                break

    diagnostics = {
        "seed": seed,
        "steps_requested": steps,
        "steps_run": len(losses),
        "loss_per_cell_final": losses[-1],
        "loss_per_cell_tail_mean": float(np.mean(losses[-250:])),
        "loss_per_cell_tail_sd": float(np.std(losses[-250:])),
        "family_count": len(family_names),
    }
    state = pyro.get_param_store().get_state()
    return guide, diagnostics, state


def posterior_parameter_samples(
    guide: AutoNormal,
    n_samples: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    keep = {
        "mu",
        "loading_general",
        "loading_specific",
        "sd_family_general",
        "sd_family_specific",
        "residual_sd",
    }
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for _ in range(n_samples):
            draw = guide()
            for key, value in draw.items():
                if key in keep:
                    collected[key].append(value.detach().cpu().numpy())
    return {key: np.stack(values) for key, values in collected.items()}


def covariance_components(
    parameters: dict[str, np.ndarray],
    group_index: np.ndarray,
    use_specific: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loading_general = np.asarray(parameters["loading_general"])
    residual_sd = np.asarray(parameters["residual_sd"])
    sd_family_general = float(np.asarray(parameters["sd_family_general"]))

    row_cov = np.outer(loading_general, loading_general)
    family_cov = (
        sd_family_general**2
        * np.outer(loading_general, loading_general)
    )
    if use_specific:
        loading_specific = np.asarray(parameters["loading_specific"])
        sd_family_specific = np.asarray(parameters["sd_family_specific"])
        for k in range(len(GROUP_ORDER)):
            v = np.where(group_index == k, loading_specific, 0.0)
            row_cov += np.outer(v, v)
            family_cov += sd_family_specific[k] ** 2 * np.outer(v, v)
    residual_cov = np.diag(np.square(residual_sd))
    return row_cov, family_cov, residual_cov


def family_log_score(
    x_family: np.ndarray,
    parameters: dict[str, np.ndarray],
    group_index: np.ndarray,
    use_specific: bool,
) -> float:
    """Joint score for all held-out checkpoints from one unseen family."""
    m, n_items = x_family.shape
    row_cov, family_cov, residual_cov = covariance_components(
        parameters, group_index, use_specific
    )
    covariance = (
        np.kron(np.eye(m), row_cov + residual_cov)
        + np.kron(np.ones((m, m)), family_cov)
    )
    covariance += np.eye(m * n_items) * 1e-5
    mean = np.tile(np.asarray(parameters["mu"]), m)
    observed = x_family.reshape(-1)

    cov_t = torch.as_tensor(covariance, dtype=torch.float64)
    mean_t = torch.as_tensor(mean, dtype=torch.float64)
    obs_t = torch.as_tensor(observed, dtype=torch.float64)
    return float(
        torch.distributions.MultivariateNormal(
            mean_t, covariance_matrix=cov_t
        )
        .log_prob(obs_t)
        .item()
    )


def median_parameters(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.median(value, axis=0) for key, value in samples.items()}


def balanced_family_folds(
    family_labels: list[str], n_folds: int
) -> list[list[str]]:
    counts = Counter(family_labels)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    loads = [0] * n_folds
    for family, count in sorted(
        counts.items(), key=lambda pair: (-pair[1], pair[0])
    ):
        target = min(range(n_folds), key=lambda idx: (loads[idx], idx))
        folds[target].append(family)
        loads[target] += count
    return folds


def empirical_correlation_rmse(
    x: np.ndarray,
    parameters: dict[str, np.ndarray],
    group_index: np.ndarray,
    use_specific: bool,
) -> float:
    row_cov, family_cov, residual_cov = covariance_components(
        parameters, group_index, use_specific
    )
    covariance = row_cov + family_cov + residual_cov
    scale = np.sqrt(np.diag(covariance))
    predicted = covariance / np.outer(scale, scale)
    observed = np.corrcoef(x, rowvar=False)
    tri = np.triu_indices_from(observed, k=1)
    return float(np.sqrt(np.mean(np.square(predicted[tri] - observed[tri]))))


def bootstrap_ci(
    values: np.ndarray, seed: int, replicates: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(replicates)
    for b in range(replicates):
        draws[b] = rng.choice(values, size=n, replace=True).mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def full_fit_summary(
    x: np.ndarray,
    family_labels: list[str],
    group_index: np.ndarray,
    use_specific: bool,
    prior_scale: float,
    seeds: list[int],
    steps: int,
    samples: int,
    device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    fits: list[tuple[float, AutoNormal, dict, dict]] = []
    for seed in seeds:
        guide, diag, state = fit_model(
            x,
            family_labels,
            group_index,
            use_specific,
            prior_scale,
            seed,
            steps,
            0.015,
            device,
        )
        fits.append((diag["loss_per_cell_tail_mean"], guide, diag, state))
        print(
            json.dumps(
                {
                    "stage": "full_fit",
                    "model": "bifactor" if use_specific else "general",
                    "specific_prior_scale": (
                        prior_scale if use_specific else None
                    ),
                    **diag,
                }
            ),
            flush=True,
        )
    fits.sort(key=lambda item: item[0])
    _, best_guide, _, best_state = fits[0]
    pyro.clear_param_store()
    pyro.get_param_store().set_state(best_state)
    posterior = posterior_parameter_samples(best_guide, samples, device)
    median = median_parameters(posterior)

    summary: dict = {
        "model": "multilevel_bifactor" if use_specific else "multilevel_general",
        "specific_prior_scale": prior_scale if use_specific else None,
        "best_seed": fits[0][2]["seed"],
            "restarts": [item[2] for item in fits],
        "correlation_rmse": empirical_correlation_rmse(
            x, median, group_index, use_specific
        ),
        "parameters": {},
    }
    for key, value in posterior.items():
        summary["parameters"][key] = {
            "q025": np.quantile(value, 0.025, axis=0).tolist(),
            "median": np.quantile(value, 0.5, axis=0).tolist(),
            "q975": np.quantile(value, 0.975, axis=0).tolist(),
        }

    if use_specific:
        general_loading = posterior["loading_general"]
        specific_loading = posterior["loading_specific"]
        general_family_sd = posterior["sd_family_general"][:, None]
        specific_family_sd = posterior["sd_family_specific"]
        residual_sd = posterior["residual_sd"]
        group_specific_sd = specific_family_sd[:, group_index]
        general_var = (
            (1.0 + np.square(general_family_sd))
            * np.square(general_loading)
        )
        specific_var = (
            (1.0 + np.square(group_specific_sd))
            * np.square(specific_loading)
        )
        total_var = general_var + specific_var + np.square(residual_sd)
        share = specific_var / total_var
        summary["specific_variance_share"] = {
            "per_paradigm": {
                "q025": np.quantile(share, 0.025, axis=0).tolist(),
                "median": np.quantile(share, 0.5, axis=0).tolist(),
                "q975": np.quantile(share, 0.975, axis=0).tolist(),
            },
            "mean_across_paradigms": {
                "q025": float(np.quantile(share.mean(axis=1), 0.025)),
                "median": float(np.quantile(share.mean(axis=1), 0.5)),
                "q975": float(np.quantile(share.mean(axis=1), 0.975)),
            },
        }
    return summary, median


def run_cross_validation(
    x_raw: np.ndarray,
    family_labels: list[str],
    group_index: np.ndarray,
    folds: list[list[str]],
    seeds: list[int],
    steps: int,
    posterior_samples: int,
    prior_scale: float,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    records: list[dict] = []
    for fold_index, held_families in enumerate(folds):
        held_set = set(held_families)
        train_mask = np.array([f not in held_set for f in family_labels])
        mean = x_raw[train_mask].mean(axis=0)
        sd = x_raw[train_mask].std(axis=0, ddof=1)
        require(np.all(sd > 0), f"zero training variance in fold {fold_index}")
        x_train = (x_raw[train_mask] - mean) / sd
        train_families = [
            family_labels[i] for i in np.flatnonzero(train_mask)
        ]

        fitted: dict[bool, dict[str, np.ndarray]] = {}
        fit_diagnostics: dict[bool, list[dict]] = {}
        for use_specific in (False, True):
            restarts: list[tuple[float, AutoNormal, dict, dict]] = []
            for seed in seeds:
                guide, diag, state = fit_model(
                    x_train,
                    train_families,
                    group_index,
                    use_specific,
                    prior_scale,
                    seed + fold_index * 1000,
                    steps,
                    0.018,
                    device,
                )
                restarts.append(
                    (diag["loss_per_cell_tail_mean"], guide, diag, state)
                )
                print(
                    json.dumps(
                        {
                            "stage": "cross_validation_fit",
                            "fold": fold_index,
                            "model": (
                                "bifactor" if use_specific else "general"
                            ),
                            **diag,
                        }
                    ),
                    flush=True,
                )
            restarts.sort(key=lambda item: item[0])
            pyro.clear_param_store()
            pyro.get_param_store().set_state(restarts[0][3])
            posterior = posterior_parameter_samples(
                restarts[0][1], posterior_samples, device
            )
            fitted[use_specific] = median_parameters(posterior)
            fit_diagnostics[use_specific] = [
                item[2] for item in restarts
            ]

        for family in held_families:
            indexes = np.array(
                [i for i, value in enumerate(family_labels) if value == family]
            )
            x_family = (x_raw[indexes] - mean) / sd
            score_general = family_log_score(
                x_family, fitted[False], group_index, False
            )
            score_bifactor = family_log_score(
                x_family, fitted[True], group_index, True
            )
            cells = int(x_family.size)
            records.append(
                {
                    "fold": fold_index,
                    "family": family,
                    "models": len(indexes),
                    "cells": cells,
                    "elpd_general": score_general,
                    "elpd_bifactor": score_bifactor,
                    "delta_elpd": score_bifactor - score_general,
                    "delta_elpd_per_cell": (
                        score_bifactor - score_general
                    )
                    / cells,
                }
            )

        print(
            json.dumps(
                {
                    "fold": fold_index,
                    "held_families": held_families,
                    "general_best_loss": min(
                        d["loss_per_cell_tail_mean"]
                        for d in fit_diagnostics[False]
                    ),
                    "bifactor_best_loss": min(
                        d["loss_per_cell_tail_mean"]
                        for d in fit_diagnostics[True]
                    ),
                }
            ),
            flush=True,
        )

    deltas = np.array([row["delta_elpd_per_cell"] for row in records])
    weighted_delta = sum(row["delta_elpd"] for row in records) / sum(
        row["cells"] for row in records
    )
    ci = bootstrap_ci(deltas, 42, 20000)
    result = {
        "folds": folds,
        "family_count": len(records),
        "family_equal_mean_delta_elpd_per_cell": float(deltas.mean()),
        "family_bootstrap_95_ci": list(ci),
        "cell_weighted_delta_elpd_per_cell": float(weighted_delta),
        "families_favoring_bifactor": int(np.sum(deltas > 0)),
        "families_favoring_general": int(np.sum(deltas < 0)),
    }
    return result, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        default="results/reanalysis/aplus_20260718/matrix_aplus_strict.csv",
    )
    parser.add_argument(
        "--family-map",
        default="results/reanalysis/aplus_20260718/family_map.json",
    )
    parser.add_argument(
        "--out",
        default="results/reanalysis/exploratory_bifactor_20260727",
    )
    parser.add_argument("--full-steps", type=int, default=12000)
    parser.add_argument("--cv-steps", type=int, default=6500)
    parser.add_argument("--posterior-samples", type=int, default=600)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    matrix_path = (root / args.matrix).resolve()
    family_path = (root / args.family_map).resolve()
    output = (root / args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
        device = torch.device("cuda")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    frame = pd.read_csv(matrix_path)
    paradigms = list(frame.columns[1:])
    require(set(paradigms) == set(PARADIGM_GROUPS), "paradigm set mismatch")
    x_raw = frame[paradigms].to_numpy(dtype=float)
    require(np.isfinite(x_raw).all(), "matrix contains non-finite values")
    require(((x_raw >= 0) & (x_raw <= 1)).all(), "matrix outside [0,1]")

    family_document = json.loads(family_path.read_text())
    merged = family_document["merged"]
    models = frame.iloc[:, 0].astype(str).tolist()
    require(set(models) <= set(merged), "family map is incomplete")
    family_labels = [merged[m] for m in models]
    group_index = np.array(
        [GROUP_ORDER.index(PARADIGM_GROUPS[p]) for p in paradigms],
        dtype=int,
    )

    mean = x_raw.mean(axis=0)
    sd = x_raw.std(axis=0, ddof=1)
    x = (x_raw - mean) / sd
    seeds = [42, 43, 44]

    print(
        json.dumps(
            {
                "device": str(device),
                "torch": torch.__version__,
                "pyro": pyro.__version__,
                "rows": len(frame),
                "families": len(set(family_labels)),
                "paradigms": len(paradigms),
            }
        ),
        flush=True,
    )

    general, _ = full_fit_summary(
        x,
        family_labels,
        group_index,
        False,
        0.0,
        seeds,
        args.full_steps,
        args.posterior_samples,
        device,
    )

    prior_sensitivity: list[dict] = []
    primary_bifactor: dict | None = None
    for prior_scale in (0.15, 0.30, 0.60):
        result, _ = full_fit_summary(
            x,
            family_labels,
            group_index,
            True,
            prior_scale,
            seeds,
            args.full_steps,
            args.posterior_samples,
            device,
        )
        prior_sensitivity.append(result)
        if math.isclose(prior_scale, 0.30):
            primary_bifactor = result
    require(primary_bifactor is not None, "primary bifactor fit missing")

    folds = balanced_family_folds(family_labels, args.folds)
    cv_result, cv_records = run_cross_validation(
        x_raw,
        family_labels,
        group_index,
        folds,
        [42, 43],
        args.cv_steps,
        max(250, args.posterior_samples // 2),
        0.30,
        device,
    )

    result = {
        "status": "exploratory_not_in_frozen_chain",
        "analysis": "multilevel_bayesian_bifactor",
        "method": {
            "inference": "Pyro AutoNormal mean-field variational inference",
            "comparison": "family-held-out joint Gaussian log predictive density",
            "general_factor": "checkpoint and family latent effects",
            "specific_factors": (
                "five orthogonal grouping factors at checkpoint and family levels"
            ),
            "loading_constraints": "half-normal positive loadings",
            "warning": (
                "Two groupings have only two indicators. Results can be "
                "prior-sensitive, and mean-field intervals may be too narrow."
            ),
        },
        "data": {
            "matrix": str(matrix_path.relative_to(root)),
            "matrix_sha256": sha256(matrix_path),
            "family_map": str(family_path.relative_to(root)),
            "family_map_sha256": sha256(family_path),
            "rows": len(frame),
            "families": len(set(family_labels)),
            "paradigms": paradigms,
            "group_index": group_index.tolist(),
            "group_order": GROUP_ORDER,
            "standardization_mean": mean.tolist(),
            "standardization_sd": sd.tolist(),
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "pyro": pyro.__version__,
            "cuda_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "seeds": seeds,
        },
        "full_fit": {
            "general": general,
            "bifactor_primary": primary_bifactor,
            "bifactor_prior_sensitivity": prior_sensitivity,
        },
        "family_heldout_cross_validation": cv_result,
    }

    result_path = output / "exploratory_bifactor_results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    csv_path = output / "family_heldout_log_scores.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cv_records[0]))
        writer.writeheader()
        writer.writerows(cv_records)

    manifest = {
        "status": "exploratory_not_in_frozen_chain",
        "script": str(Path(__file__).resolve().relative_to(root)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": {
            str(matrix_path.relative_to(root)): sha256(matrix_path),
            str(family_path.relative_to(root)): sha256(family_path),
        },
        "outputs": {
            result_path.name: sha256(result_path),
            csv_path.name: sha256(csv_path),
        },
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(cv_result, indent=2), flush=True)
    print(f"WROTE {result_path}", flush=True)


if __name__ == "__main__":
    main()
