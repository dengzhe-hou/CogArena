"""Regression tests for the Source Monitoring generator dedup fix.

The original generator deduplicated statements only within each source, so
two different sources could be attributed the identical statement.  A test
probe built from such a statement is ambiguous (multiple correct answers).
Statements must now be unique across the whole episode.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cogarena.dimensions.episodic_memory import SourceMonitoringGenerator


def _episode_statements(item):
    return [s["statement"] for s in item.metadata.parameters["all_statements"]]


def test_statements_unique_across_sources_many_seeds():
    for seed in range(200):
        for difficulty in ("easy", "medium", "hard"):
            items = SourceMonitoringGenerator.generate(
                seed=seed, n_items=2, difficulty=difficulty
            )
            for item in items:
                stmts = _episode_statements(item)
                assert len(set(stmts)) == len(stmts), (
                    f"duplicate statement in seed={seed} {difficulty}: {stmts}"
                )


def test_probe_gold_is_unambiguous():
    for seed in range(50):
        items = SourceMonitoringGenerator.generate(
            seed=seed, n_items=2, difficulty="hard"
        )
        for item in items:
            by_stmt = {}
            for s in item.metadata.parameters["all_statements"]:
                by_stmt.setdefault(s["statement"], set()).add(s["source"])
            for probe in item.metadata.parameters["test_items"]:
                owners = by_stmt[probe["statement"]]
                assert owners == {probe["correct_source"]}, (
                    f"ambiguous probe seed={seed}: {probe['statement']!r} "
                    f"owned by {owners}"
                )


def test_unaffected_episode_content_stable():
    # Episodes that never drew a cross-source duplicate keep an identical
    # RNG path under the fix, so regeneration must reproduce the frozen
    # battery for them.  Spot-check determinism of the fixed generator.
    a = SourceMonitoringGenerator.generate(seed=1234, n_items=3, difficulty="medium")
    b = SourceMonitoringGenerator.generate(seed=1234, n_items=3, difficulty="medium")
    for x, y in zip(a, b):
        assert x.stimulus == y.stimulus
        assert x.expected_response == y.expected_response
        assert x.metadata.parameters["all_statements"] == y.metadata.parameters["all_statements"]
