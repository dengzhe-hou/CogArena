#!/usr/bin/env python3
"""Run the parent analyzer with replication-specific provenance gates.

The exploratory wording replication intentionally reuses the parent study's
items, model panel, scorers, and analysis. It does not inherit the parent's
confirmatory pilot and capacity decision. The parent analyzer's frozen-spec
gate is therefore replaced only at the source-revision boundary. All raw-data,
manifest, replay, scoring, and aggregate-output checks remain active.
"""

from __future__ import annotations

import os

from scripts.experiments.causal_selectivity_20260720 import analyze as parent
from scripts.experiments.causal_selectivity_20260720 import analyze_amended
from scripts.experiments.causal_selectivity_20260720.common import require

from .preflight import STUDY_ID, validate_static


def validate_replication_source_revision(
    spec: dict, profile: str, revision: str
) -> None:
    """Validate the exploratory run without claiming the parent freeze."""
    require(profile == "formal", "wording replication analyzes the formal panel only")
    require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "COGARENA_GIT_HEAD must identify the frozen inference revision",
    )
    validate_static()
    require(spec.get("study_id") == STUDY_ID, "wrong replication study")
    require(
        spec.get("analysis_role")
        == "post_hoc_exploratory_wording_replication_not_part_of_the_parent_confirmation_decision",
        "replication cannot update the parent confirmation decision",
    )
    adapter_revision = os.environ.get("COGARENA_ANALYSIS_ADAPTER_HEAD", "").strip()
    require(
        len(adapter_revision) == 40
        and all(character in "0123456789abcdef" for character in adapter_revision),
        "COGARENA_ANALYSIS_ADAPTER_HEAD must identify the analysis adapter revision",
    )


def main() -> None:
    parent.validate_source_revision = validate_replication_source_revision
    parent.analyze_arrays = analyze_amended.analyze_arrays
    parent.main()


if __name__ == "__main__":
    main()
