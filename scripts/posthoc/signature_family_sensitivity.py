#!/usr/bin/env python3
"""Post-hoc family-level sensitivity for directional behavioral signatures.

The main signature summary counts checkpoints. This script treats the family
as the sampling unit by averaging each contrast within family, then reports:

1. a one-sided exact binomial test on the number of family means above zero;
2. an exact one- and two-sided sign-flip test on continuous family means;
3. Benjamini-Hochberg adjustments across the five 20-checkpoint paradigms.

Raw and merged family labels are both reported. EPITOME uses the separate
35-model expansion pool and is not included in the five-paradigm BH family.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


ROOT = Path(
    os.environ.get(
        "COGARENA_ROOT",
        Path(__file__).resolve().parents[2],
    )
).resolve()
SIGNATURE_PATH = ROOT / "results" / "reanalysis" / "signature_significance.json"
FAMILY_PATH = (
    ROOT / "results" / "reanalysis" / "aplus_20260718" / "family_map.json"
)
EXPANSION_PATH = ROOT / "results" / "reanalysis" / "expansion_models.json"
OUTPUT_PATH = (
    ROOT / "results" / "reanalysis" / "signature_family_sensitivity.json"
)
BH_PARADIGMS = (
    "stroop",
    "flanker",
    "n_back_load",
    "false_belief",
    "drm_false_memory",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def exact_sign_flip(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    require(
        array.ndim == 1
        and 1 <= len(array) <= 20
        and bool(np.isfinite(array).all()),
        "exact sign-flip requires 1 to 20 finite family means",
    )
    observed = float(array.mean())
    null = np.fromiter(
        (
            float(np.mean(array * np.asarray(signs, dtype=float)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(array))
        ),
        dtype=float,
        count=2 ** len(array),
    )
    tolerance = 1e-15
    return {
        "n_permutations": int(len(null)),
        "observed_mean_delta": observed,
        "p_one_sided": float(np.mean(null >= observed - tolerance)),
        "p_two_sided": float(
            np.mean(np.abs(null) >= abs(observed) - tolerance)
        ),
    }


def summarize_family_means(
    per_model_delta: dict[str, float],
    family_map: dict[str, str],
    *,
    include_sign_flip: bool = True,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    require(
        per_model_delta and set(per_model_delta).issubset(family_map),
        "family map does not cover every model",
    )
    for model, delta in per_model_delta.items():
        require(
            isinstance(delta, (int, float))
            and not isinstance(delta, bool)
            and math.isfinite(float(delta)),
            f"invalid model delta for {model}",
        )
        grouped[family_map[model]].append(float(delta))
    family_means = {
        family: float(np.mean(values)) for family, values in sorted(grouped.items())
    }
    positive = sum(value > 0 for value in family_means.values())
    n_families = len(family_means)
    row: dict[str, Any] = {
        "n_families": n_families,
        "k_family_means_expected_direction": positive,
        "fraction": f"{positive}/{n_families}",
        "p_direction_binomial_one_sided": float(
            binomtest(positive, n_families, 0.5, alternative="greater").pvalue
        ),
        "mean_of_family_mean_deltas": float(np.mean(list(family_means.values()))),
        "family_mean_deltas": family_means,
    }
    if include_sign_flip:
        row["continuous_exact_sign_flip"] = exact_sign_flip(
            list(family_means.values())
        )
    else:
        row["continuous_exact_sign_flip"] = {
            "status": "not_run_for_separate_35_model_epitome_pool"
        }
    return row


def epitome_expansion_deltas() -> dict[str, float]:
    from cogarena.cli import _collect_items
    from results.recompute_20260703.step4b_numbers import corrected_items

    expansion_models = json.loads(EXPANSION_PATH.read_text(encoding="utf-8"))
    battery = {item.task_id: item for item in _collect_items(50, 42, None)}
    result: dict[str, float] = {}
    for model in sorted(expansion_models):
        by_capacity: dict[str, list[float]] = defaultdict(list)
        for task_id, (paradigm, accuracy, _) in corrected_items(
            model, False
        ).items():
            if paradigm != "epitome_tom" or task_id not in battery:
                continue
            capacity = battery[task_id].metadata.parameters["sub_capacity"]
            by_capacity[str(capacity)].append(float(accuracy))
        if by_capacity["desire"] and by_capacity["belief"]:
            result[model] = float(
                np.mean(by_capacity["desire"]) - np.mean(by_capacity["belief"])
            )
    require(len(result) == 35, f"expected 35 EPITOME expansion models, got {len(result)}")
    return result


def main() -> None:
    signatures = json.loads(SIGNATURE_PATH.read_text(encoding="utf-8"))
    family_maps = json.loads(FAMILY_PATH.read_text(encoding="utf-8"))
    require(
        set(family_maps) == {"raw", "merged"},
        "unexpected family-map schema",
    )
    rows = {row["paradigm"]: row for row in signatures["paradigms"]}
    require(set(BH_PARADIGMS).issubset(rows), "signature artifact is incomplete")

    output: dict[str, Any] = {
        "schema_version": "cogarena.signature_family_sensitivity.v1",
        "status": "post_hoc_sensitivity",
        "checkpoint_analysis_sha256": __import__("hashlib").sha256(
            SIGNATURE_PATH.read_bytes()
        ).hexdigest(),
        "family_map_sha256": __import__("hashlib").sha256(
            FAMILY_PATH.read_bytes()
        ).hexdigest(),
        "sampling_unit_note": (
            "Family means, not checkpoints, are the inferential units. Ties and "
            "zero family means do not count in the expected direction."
        ),
        "paradigms": {},
    }
    for paradigm in BH_PARADIGMS:
        per_model = rows[paradigm].get("per_model")
        require(
            isinstance(per_model, dict) and len(per_model) == 20,
            f"{paradigm} lacks 20 per-model contrasts",
        )
        deltas = {
            model: float(record["delta"]) for model, record in per_model.items()
        }
        output["paradigms"][paradigm] = {
            label: summarize_family_means(deltas, family_maps[label])
            for label in ("raw", "merged")
        }

    for label in ("raw", "merged"):
        direction_p = [
            output["paradigms"][paradigm][label][
                "p_direction_binomial_one_sided"
            ]
            for paradigm in BH_PARADIGMS
        ]
        sign_flip_p = [
            output["paradigms"][paradigm][label][
                "continuous_exact_sign_flip"
            ]["p_one_sided"]
            for paradigm in BH_PARADIGMS
        ]
        direction_bh = multipletests(direction_p, alpha=0.05, method="fdr_bh")[1]
        sign_flip_bh = multipletests(sign_flip_p, alpha=0.05, method="fdr_bh")[1]
        for index, paradigm in enumerate(BH_PARADIGMS):
            output["paradigms"][paradigm][label][
                "p_direction_binomial_bh"
            ] = float(direction_bh[index])
            output["paradigms"][paradigm][label][
                "p_continuous_sign_flip_one_sided_bh"
            ] = float(sign_flip_bh[index])

    epitome_deltas = epitome_expansion_deltas()
    output["paradigms"]["epitome"] = {
        "pool": "35-model expansion",
        "contrast": "desire minus belief accuracy",
        "excluded_from_five_paradigm_bh": True,
        **{
            label: summarize_family_means(
                epitome_deltas,
                family_maps[label],
                include_sign_flip=False,
            )
            for label in ("raw", "merged")
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_PATH}")
    for paradigm, result in output["paradigms"].items():
        merged = result["merged"]
        print(
            f"{paradigm}: merged families {merged['fraction']}, "
            f"direction p={merged['p_direction_binomial_one_sided']:.4f}"
        )


if __name__ == "__main__":
    main()
