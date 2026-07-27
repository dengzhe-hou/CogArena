from scripts.experiments.scaffold_wording_replication_20260725.build_spec import (
    ALTERNATIVE_SCAFFOLDS,
    PARENT_SPEC,
    STUDY_ID,
    build,
)
from scripts.experiments.scaffold_wording_replication_20260725 import analyze

import json
import pytest


def test_replication_changes_only_targeted_scaffold_wordings():
    spec = build()
    parent = json.loads(PARENT_SPEC.read_text(encoding="utf-8"))
    current = {row["id"]: row for row in spec["conditions"]}
    original = {row["id"]: row for row in parent["conditions"]}

    assert spec["study_id"] == STUDY_ID
    assert spec["analysis_role"].startswith("post_hoc_exploratory")
    assert spec["result_interpretation"]["confirmatory_status"] == "none"
    assert current["baseline"] == original["baseline"]
    assert current["neutral_placebo"] == original["neutral_placebo"]
    assert set(ALTERNATIVE_SCAFFOLDS) == {
        row["id"] for row in spec["conditions"] if row["kind"] == "targeted"
    }
    for condition_id, alternative in ALTERNATIVE_SCAFFOLDS.items():
        assert current[condition_id]["scaffold"] == alternative
        assert alternative != original[condition_id]["scaffold"]
        assert current[condition_id]["target_group"] == original[condition_id]["target_group"]


def test_replication_prompts_remain_length_matched_and_answer_free():
    spec = build()
    conditions = {row["id"]: row for row in spec["conditions"]}
    placebo_tokens = len(conditions["neutral_placebo"]["scaffold"].split())
    for condition_id in ALTERNATIVE_SCAFFOLDS:
        scaffold = conditions[condition_id]["scaffold"]
        assert abs(len(scaffold.split()) - placebo_tokens) <= 2
        assert "requested output format" in scaffold
        assert "never reveal" in scaffold
        assert "<END_COGARENA_RESPONSE>" not in scaffold


def test_analysis_adapter_rejects_confirmatory_relabeling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = build()
    monkeypatch.setattr(analyze, "validate_static", lambda: spec)
    monkeypatch.setenv("COGARENA_ANALYSIS_ADAPTER_HEAD", "b" * 40)
    analyze.validate_replication_source_revision(spec, "formal", "a" * 40)
    changed = dict(spec)
    changed["analysis_role"] = "confirmatory"
    monkeypatch.setattr(analyze, "validate_static", lambda: changed)
    with pytest.raises(RuntimeError, match="cannot update"):
        analyze.validate_replication_source_revision(changed, "formal", "a" * 40)
