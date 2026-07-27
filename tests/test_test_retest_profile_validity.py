from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/experiments/profile_validity_20260720/test_retest.py"
SPEC = importlib.util.spec_from_file_location("cogarena_test_retest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
tr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tr)


def test_frozen_pair_count_contract() -> None:
    assert len(tr.MODELS) == tr.EXPECTED_N_MODELS == 20
    assert sum(tr.EXPECTED_ELIGIBLE_COUNTS.values()) == tr.EXPECTED_PAIRS_PER_MODEL == 421
    assert tr.EXPECTED_PAIRS_PER_MODEL * tr.EXPECTED_N_MODELS == tr.EXPECTED_TOTAL_PAIRS == 8420
    assert tr.EXPECTED_RAW_COUNTS["source_monitoring"] - tr.EXPECTED_ELIGIBLE_COUNTS["source_monitoring"] == 11


def test_icc_a1_identical_and_absolute_shift() -> None:
    x = np.asarray([0.05, 0.2, 0.45, 0.7, 0.95])
    assert tr.icc_a1(np.column_stack([x, x])) == pytest.approx(1.0)
    shifted = tr.icc_a1(np.column_stack([x, x + 0.2]))
    assert shifted < 1.0
    assert shifted < tr.pearson(x, x + 0.2)


def test_pair_metrics_and_structure_are_finite() -> None:
    x = np.linspace(0.05, 0.95, 20)
    y = x * 0.9 + 0.03
    metrics = tr.pair_metrics(x, y)
    assert metrics["n"] == 20
    assert metrics["pearson_r"] == pytest.approx(1.0)
    assert metrics["mad"] > 0

    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(20, len(tr.PARADIGMS)))
    structure = tr.structure_metrics(matrix)
    assert structure["n_within_pairs"] == 3
    assert structure["n_cross_pairs"] == 25
    assert np.isfinite(structure["delta"])
    assert 0 <= structure["pc1_variance_share"] <= 1


def test_exact_grouping_assignment_count() -> None:
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(20, len(tr.PARADIGMS)))
    result = tr.exact_grouping_test(matrix)
    # 8! / (2! 2! 2!) for the three doubleton and two singleton labels.
    assert result["n_unique_label_assignments"] == 5040
    assert 0 <= result["p_one_sided"] <= 1
    assert 0 <= result["p_two_sided"] <= 1


def test_output_privacy_gate_rejects_raw_text_fields() -> None:
    tr.assert_no_raw_text_fields({"scores": [{"task_id": "x", "accuracy": 1.0}]})
    with pytest.raises(SystemExit, match="forbidden raw-text field"):
        tr.assert_no_raw_text_fields({"response": "must never be persisted"})
    with pytest.raises(SystemExit, match="forbidden raw-text field"):
        tr.assert_no_raw_text_fields({"nested": [{"stimulus": "also forbidden"}]})


def test_paired_csv_schema_cannot_persist_raw_text(tmp_path: Path) -> None:
    destination = tmp_path / "pairs.csv"
    tr.write_pairs_csv(
        destination,
        [
            {
                "model": "m",
                "family": "f",
                "task_id": "t",
                "paradigm": "stroop",
                "difficulty": "easy",
                "occasion_a": 0.0,
                "occasion_b": 1.0,
            }
        ],
    )
    header = destination.read_text().splitlines()[0].split(",")
    assert header == [
        "model",
        "family",
        "task_id",
        "paradigm",
        "difficulty",
        "occasion_a",
        "occasion_b",
    ]
    assert not any("response" in column or "stimulus" in column for column in header)


def test_sbatch_is_cpu_c01_and_never_requests_a_gpu() -> None:
    sbatch = (MODULE_PATH.parent / "test_retest.sbatch").read_text()
    assert "#SBATCH --partition=batch" in sbatch
    assert "#SBATCH --nodelist=c01" in sbatch
    assert "SLURM_JOB_ID" in sbatch
    assert "srun python3" in sbatch
    assert "COGARENA_GIT_HEAD" in sbatch
    assert "--gres=gpu" not in sbatch


def test_runtime_guard_requires_slurm_and_c01() -> None:
    tr.enforce_c01({"SLURM_JOB_ID": "123"}, "c01")
    with pytest.raises(SystemExit, match="inside Slurm"):
        tr.enforce_c01({}, "c01")
    with pytest.raises(SystemExit, match="must run on c01"):
        tr.enforce_c01({"SLURM_JOB_ID": "123"}, "login01")


def test_formal_wagering_overlay_is_a_frozen_input() -> None:
    source = MODULE_PATH.read_text()
    assert tr.DEFAULT_WAGER_OVERLAY.name == "wager_accuracy_overlay.json"
    assert tr.DEFAULT_WAGER_MANIFEST.name == "WAGER_REPLAY_MANIFEST.json"
    assert 'parser.add_argument("--wager-overlay"' in source
    assert 'parser.add_argument("--wager-manifest"' in source
    assert '"wager_overlay_sha256": sha256(args.wager_overlay)' in source
    assert '"wager_manifest_sha256": sha256(args.wager_manifest)' in source
    assert 'wager_overlay[model][str(row["task_id"])]' in source
    assert 'wager_manifest.get("execution", {}).get("git_head") == revision' in source


def test_test_retest_invalidates_stale_pass_and_writes_final_manifest_last() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    invalidation = source.index('"all_gates_passed": False')
    replay = source.index("item_map, generated_items_sha = generate_frozen_items()")
    final_pass = source.index('"all_gates_passed": True')
    final_write = source.rindex("atomic_write(\n        manifest_path")
    assert invalidation < replay < final_pass < final_write
    assert source.rfind("req(") < final_write


def test_profile_shape_centering_removes_model_level() -> None:
    rng = np.random.default_rng(33)
    a = rng.normal(size=(20, len(tr.PARADIGMS)))
    offsets = np.linspace(-5, 5, 20)[:, None]
    b = a + offsets
    pairs = [
        {
            "model": tr.MODELS[i],
            "paradigm": paradigm,
            "occasion_a": float(a[i, j]),
            "occasion_b": float(b[i, j]),
        }
        for i in range(20)
        for j, paradigm in enumerate(tr.PARADIGMS)
    ]
    summary = tr.analysis_points(pairs, a, b)
    shape = summary["pooled_model_centered_profile_cells"]
    assert shape["pearson_r"] == pytest.approx(1.0)
    assert shape["rmse"] == pytest.approx(0.0, abs=1e-12)
