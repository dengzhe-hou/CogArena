from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cogarena.core import DifficultyLevel, EvalMode, ScoringConfig, TaskInstance, TaskMetadata


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "scripts/experiments/profile_validity_20260720/wager_replay_audit.py"
)
SPEC = importlib.util.spec_from_file_location("wager_replay_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def synthetic_item() -> TaskInstance:
    return TaskInstance(
        task_id="fixture_pdw_numeric_normalization",
        metadata=TaskMetadata(
            dimension=AUDIT.DIMENSION,
            paradigm=AUDIT.PARADIGM,
            mode=EvalMode.LLM_STATIC,
            parameters={
                "bet_correct": 10,
                "bet_wrong": -10,
                "pass_any": 2,
            },
            scoring=ScoringConfig(method="custom"),
            difficulty=DifficultyLevel.EASY,
        ),
        stimulus="synthetic fixture",
        expected_response="1200",
    )


def archived_record() -> dict:
    return {
        "task_id": "fixture_pdw_numeric_normalization",
        "model_id": "openai/fixture:1b",
        "dimension": AUDIT.DIMENSION,
        "paradigm": AUDIT.PARADIGM,
        "difficulty": "easy",
        "response": "Answer: 1,200\nBet: YES",
        "score": {
            "accuracy": 0.0,
            "did_bet": 1.0,
            "is_correct": 0.0,
            "points": -10.0,
        },
    }


def test_numeric_normalization_replay_and_output_privacy() -> None:
    row = AUDIT.audit_record(
        archived_record(), synthetic_item(), "fixture:1b", "synthetic_source"
    )
    assert row["archived_accuracy"] == 0.0
    assert row["replayed_accuracy"] == 1.0
    assert row["difference_accuracy"] == 1.0
    assert row["difference_did_bet"] == 0.0
    assert row["replayed_points"] == 10.0
    assert row["score_changed"] is True
    assert "response" not in row
    assert "stimulus" not in row
    assert "expected_response" not in row
    AUDIT.assert_disclosure_minimal(row)


def test_identity_mismatch_fails_closed() -> None:
    record = archived_record()
    record["model_id"] = "openai/not-the-model"
    with pytest.raises(SystemExit, match="model identity mismatch"):
        AUDIT.audit_record(
            record, synthetic_item(), "fixture:1b", "synthetic_source"
        )


def test_named_phi3_correction_is_a_hard_gate() -> None:
    row = {
        "dataset": "full_eval_20260526_2208",
        "model": AUDIT.KNOWN_MODEL,
        "task_id": AUDIT.KNOWN_TASK,
        "archived_accuracy": 0.0,
        "replayed_accuracy": 1.0,
        "difference_accuracy": 1.0,
    }
    assert AUDIT.assert_known_correction([row])["status"] == "verified"
    row["replayed_accuracy"] = 0.0
    with pytest.raises(SystemExit, match="replayed accuracy is not 1"):
        AUDIT.assert_known_correction([row])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("accuracy", True, "boolean"),
        ("did_bet", 0.5, "not binary"),
        ("points", float("nan"), "non-finite"),
        ("points", 11, "unexpected payoff"),
    ],
)
def test_score_schema_fail_closed(field: str, value: object, message: str) -> None:
    score = archived_record()["score"]
    score[field] = value
    with pytest.raises(SystemExit, match=message):
        AUDIT.validate_score(score, "fixture")


def test_runtime_guard_requires_slurm_and_c01() -> None:
    AUDIT.enforce_c01({"SLURM_JOB_ID": "123"}, "c01")
    with pytest.raises(SystemExit, match="inside Slurm"):
        AUDIT.enforce_c01({}, "c01")
    with pytest.raises(SystemExit, match="must run on c01"):
        AUDIT.enforce_c01({"SLURM_JOB_ID": "123"}, "login01")


def test_tree_hash_binds_relative_paths_and_contents(tmp_path: Path) -> None:
    left = tmp_path / "a.json"
    right = tmp_path / "b.json"
    left.write_text(json.dumps({"x": 1}), encoding="utf-8")
    right.write_text(json.dumps({"x": 1}), encoding="utf-8")
    both = AUDIT.tree_hash(tmp_path, [left, right])
    one = AUDIT.tree_hash(tmp_path, [left])
    assert both["n_files"] == 2
    assert both["tree_sha256"] != one["tree_sha256"]


def test_private_payload_field_is_rejected() -> None:
    with pytest.raises(SystemExit, match="private field"):
        AUDIT.assert_disclosure_minimal({"response": "must not persist"})


def test_frozen_panels_are_exact_and_disjoint() -> None:
    assert len(AUDIT.PRIMARY_MODELS) == 20
    assert len(AUDIT.EXPANSION_MODELS) == 35
    assert not set(AUDIT.PRIMARY_MODELS).intersection(AUDIT.EXPANSION_MODELS)


def test_manifest_is_invalidated_before_replay_and_pass_is_written_last() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    invalidation = source.index('"all_gates_passed": False')
    replay = source.index("items, generated_bundle_sha = generated_wagering_items()")
    final_pass = source.index('"all_gates_passed": True')
    final_write = source.rindex("atomic_write(\n        manifest_path")
    assert invalidation < replay < final_pass < final_write
    assert source.rfind("req(") < final_write


def test_accuracy_overlay_requires_did_bet_invariance() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'did_bet_changes == 0' in source
    assert '"wager_construct_overlay_representable": did_bet_changes == 0' in source


def test_formal_replay_refuses_noncanonical_roots_and_hashes_chain_test() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "formal replay refuses a noncanonical primary input root" in source
    assert "formal replay refuses a noncanonical expansion input root" in source
    assert 'ROOT / "tests/test_wager_overlay_chain.py"' in source
