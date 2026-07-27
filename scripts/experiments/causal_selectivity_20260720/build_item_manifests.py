#!/usr/bin/env python3
"""Build deterministic, balanced, mutually exclusive formal/pilot item manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from cogarena.generators.cognitive_control_gen import generate_cc_items
from cogarena.generators.episodic_memory_gen import generate_em_items
from cogarena.generators.metacognition_gen import generate_mc_items
from cogarena.generators.theory_of_mind_gen import generate_tom_items
from cogarena.generators.working_memory_gen import generate_wm_items
from cogarena.dimensions.cognitive_control import FlankerParadigm, StroopParadigm

from .common import (
    HERE,
    RESULTS_ROOT,
    ROOT,
    SPEC_PATH,
    atomic_write_json,
    canonical_bytes,
    format_user_prompt,
    item_fingerprint,
    item_payload,
    load_json,
    load_spec,
    manifest_path,
    MULTITURN_HISTORY_LINES,
    presentation_fingerprint,
    require_prompt_budget,
    request_stop_policy,
    request_stop_sequences,
    response_terminator,
    scoring_gold_fingerprint,
    require,
    sha256_bytes,
    sha256_file,
    sha256_text,
    system_prompt,
    turn_shown_text,
)


GENERATORS: dict[str, Callable[..., list[Any]]] = {
    "working_memory": generate_wm_items,
    "cognitive_control": generate_cc_items,
    "episodic_memory": generate_em_items,
    "theory_of_mind": generate_tom_items,
    "metacognition": generate_mc_items,
}

PARADIGM_DIMENSION = {
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

DIFFICULTIES = ("easy", "medium", "hard")
FILLER_STRESS_RESPONSE_CHARACTERS = 1024


def _difficulty(item: Any) -> str:
    value = item.metadata.difficulty
    return value.value if hasattr(value, "value") else str(value)


def _stable(items: list[Any]) -> list[Any]:
    return sorted(
        items,
        key=lambda item: sha256_text(
            f"{item.metadata.paradigm}\0{item.task_id}\0{item_fingerprint(item)}"
        ),
    )


def _take(
    items: list[Any],
    count: int,
    forbidden_presentations: set[str],
    label: str,
) -> list[Any]:
    eligible = [x for x in _stable(items) if presentation_fingerprint(x) not in forbidden_presentations]
    require(len(eligible) >= count, f"insufficient eligible items for {label}: {len(eligible)} < {count}")
    return eligible[:count]


def _quota_select(
    items: list[Any],
    quotas: dict[Any, int],
    key: Callable[[Any], Any],
    forbidden: set[str],
    label: str,
) -> tuple[list[Any], str]:
    chosen: list[Any] = []
    for value, count in quotas.items():
        chosen.extend(_take([x for x in items if key(x) == value], count, forbidden, f"{label}/{value}"))
    require(len(chosen) == sum(quotas.values()), f"quota selection failed for {label}")
    return chosen, ",".join(f"{value}={count}" for value, count in quotas.items())


def _formal_select(
    paradigm: str, difficulty: str, items: list[Any], forbidden: set[str]
) -> tuple[list[Any], str]:
    params = lambda x: x.metadata.parameters
    if paradigm == "digit_span":
        return _quota_select(
            items, {"forward": 2, "backward": 2, "sequencing": 2},
            lambda x: params(x).get("sub_mode"), forbidden, f"{paradigm}/{difficulty}",
        )
    if paradigm in {"stroop", "flanker"}:
        return _quota_select(
            items, {True: 3, False: 3}, lambda x: bool(params(x).get("congruent")),
            forbidden, f"{paradigm}/{difficulty}",
        )
    if paradigm == "go_nogo":
        return _quota_select(
            items, {"go": 5, "nogo": 1},
            lambda x: params(x).get("_scoring_config", {}).get("condition"),
            forbidden, f"{paradigm}/{difficulty}",
        )
    if paradigm == "false_belief":
        return _quota_select(
            items, {1: 3, 2: 3}, lambda x: params(x).get("order"),
            forbidden, f"{paradigm}/{difficulty}",
        )
    if paradigm == "epitome_tom":
        quotas_by_difficulty = {
            "easy": {"belief": 2, "desire": 2, "intention": 1, "emotion": 1},
            "medium": {"belief": 1, "desire": 1, "intention": 2, "emotion": 2},
            "hard": {"belief": 2, "desire": 1, "intention": 2, "emotion": 1},
        }
        return _quota_select(
            items, quotas_by_difficulty[difficulty], lambda x: params(x).get("sub_capacity"),
            forbidden, f"{paradigm}/{difficulty}",
        )
    return _take(items, 6, forbidden, f"{paradigm}/{difficulty}"), "unstratified=6"


def generate_pool(spec: dict[str, Any], profile: str) -> list[Any]:
    cfg = spec["item_generation"]
    seed = cfg[f"{profile}_seed"]
    sizes = cfg[f"{profile}_pool_n_per_paradigm"]
    pool: list[Any] = []
    for dimension, generator in GENERATORS.items():
        pool.extend(
            generator(
                seed=seed,
                n_per_paradigm=int(sizes[dimension]),
                include_contamination_probes=False,
            )
        )
    if profile == "pilot":
        # The formal wrapper intentionally uses small, finite symbolic sets;
        # changing only its RNG seed can therefore reproduce every easy
        # flanker surface form. Add sacrificial-only variants so the pilot's
        # *presented content*, not merely its task ID, is disjoint.
        for offset, difficulty in enumerate(DIFFICULTIES):
            pool.extend(
                StroopParadigm.generate(
                    seed=seed + 70000 + offset,
                    n_congruent=12,
                    n_incongruent=12,
                    conflict_type="color_word",
                    difficulty=difficulty,
                    contamination_probe=False,
                )
            )
            pool.extend(
                FlankerParadigm.generate(
                    seed=seed + 80000 + offset,
                    n_congruent=12,
                    n_incongruent=12,
                    symbol_set="numbers_magnitude",
                    difficulty=difficulty,
                    contamination_probe=False,
                )
            )
    return pool


def _source_hashes() -> dict[str, str]:
    relative = [
        "cogarena/core.py",
        "cogarena/generators/working_memory_gen.py",
        "cogarena/generators/cognitive_control_gen.py",
        "cogarena/generators/episodic_memory_gen.py",
        "cogarena/generators/theory_of_mind_gen.py",
        "cogarena/generators/metacognition_gen.py",
        "cogarena/dimensions/working_memory.py",
        "cogarena/dimensions/cognitive_control.py",
        "cogarena/dimensions/episodic_memory.py",
        "cogarena/dimensions/theory_of_mind.py",
        "cogarena/dimensions/metacognition.py",
        "scripts/reanalysis/aplus_rescore_20260718.py",
        "scripts/experiments/causal_selectivity_20260720/common.py",
        "scripts/experiments/causal_selectivity_20260720/scorer_adapter.py",
        "scripts/experiments/causal_selectivity_20260720/build_item_manifests.py",
        "scripts/experiments/causal_selectivity_20260720/preflight.py",
        "scripts/experiments/causal_selectivity_20260720/analyze.py",
        "scripts/experiments/causal_selectivity_20260720/capacity_probe.py",
        "scripts/experiments/causal_selectivity_20260720/verify_capacity.py",
        "scripts/experiments/causal_selectivity_20260720/run_model.py",
        "scripts/experiments/causal_selectivity_20260720/verify_model.py",
        "scripts/experiments/causal_selectivity_20260720/verify_run.py",
        "scripts/experiments/causal_selectivity_20260720/run_one_model.sh",
        "scripts/experiments/causal_selectivity_20260720/run_capacity_probe.sh",
        "scripts/experiments/causal_selectivity_20260720/pilot.sbatch",
        "scripts/experiments/causal_selectivity_20260720/full.sbatch",
        "scripts/experiments/causal_selectivity_20260720/finalize.sbatch",
        "scripts/experiments/causal_selectivity_20260720/capacity_probe.sbatch",
        "scripts/experiments/causal_selectivity_20260720/finalize_capacity.sbatch",
        "scripts/experiments/causal_selectivity_20260720/analyze.sbatch",
        "scripts/experiments/causal_selectivity_20260720/prepare.sbatch",
    ]
    return {name: sha256_file(ROOT / name) for name in relative}


def _prompt_audit(spec: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for condition in spec["conditions"]:
        scaffold = condition["scaffold"]
        complete = system_prompt(spec, condition["id"])
        rows.append(
            {
                "condition_id": condition["id"],
                "kind": condition["kind"],
                "target_group": condition["target_group"],
                "scaffold_characters": len(scaffold),
                "scaffold_utf8_bytes": len(scaffold.encode("utf-8")),
                "scaffold_whitespace_tokens": len(scaffold.split()),
                "complete_system_characters": len(complete),
                "complete_system_whitespace_tokens": len(complete.split()),
                "complete_system_sha256": sha256_text(complete),
            }
        )
    placebo = next(x for x in rows if x["condition_id"] == "neutral_placebo")
    targeted = [x for x in rows if x["kind"] == "targeted"]
    max_token_delta = max(
        abs(x["scaffold_whitespace_tokens"] - placebo["scaffold_whitespace_tokens"])
        for x in targeted
    )
    max_char_fraction = max(
        abs(x["scaffold_characters"] - placebo["scaffold_characters"])
        / placebo["scaffold_characters"]
        for x in targeted
    )
    require(max_token_delta <= 2, f"target/placebo whitespace-token mismatch: {max_token_delta}")
    return {
        "count_definition": "Whitespace-delimited tokens plus Unicode character and UTF-8 byte counts; no model-specific tokenizer is assumed across six model families.",
        "conditions": rows,
        "targeted_vs_placebo_max_absolute_whitespace_token_delta": max_token_delta,
        "targeted_vs_placebo_max_character_fraction_delta": max_char_fraction,
    }


def _declared_text(value: Any) -> str:
    """Render a gold field only for aggregate answer-shape bounds.

    The rendered content is never persisted. Lists use one item per line so
    the audit conservatively preserves legal CVLT-style multiline responses.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_declared_text(part) for part in value)
    return str(value)


def _declared_response_candidates(item: Any) -> list[str]:
    candidates = [_declared_text(item.expected_response)]
    turns = item.metadata.parameters.get("turns", [])
    for turn in turns:
        for key in (
            "expected_words",
            "expected",
            "math_expected",
            "correct_answer",
            "recall_letter",
        ):
            if turn.get(key) is not None:
                candidates.append(_declared_text(turn[key]))
    paradigm = item.metadata.paradigm
    if paradigm == "confidence_calibration":
        candidates.append(
            f"Answer: {_declared_text(item.expected_response)}\nConfidence: 100%"
        )
    elif paradigm == "post_decision_wagering":
        candidates.append(f"Answer: {_declared_text(item.expected_response)}\nBet: YES")
    return [text for text in candidates if text]


def _declared_call_candidates(item: Any) -> list[str]:
    """Declared response shapes at the API-call unit used by stop routing."""
    turns = item.metadata.parameters.get("turns", [])
    if not turns:
        return _declared_response_candidates(item)
    candidates = []
    for turn in turns:
        for key in (
            "expected_words",
            "expected",
            "math_expected",
            "correct_answer",
            "recall_letter",
        ):
            if turn.get(key) is not None:
                value = turn[key]
                if item.metadata.paradigm == "cvlt_word_list" and isinstance(
                    value, (list, tuple)
                ):
                    candidates.append(", ".join(_declared_text(part) for part in value))
                else:
                    candidates.append(_declared_text(value))
    return [text for text in candidates if text]


def _turn_response_for_context_stress(turn: dict[str, Any]) -> str:
    turn_type = str(turn.get("type", "")).lower()
    if "filler" in turn_type:
        return "X" * FILLER_STRESS_RESPONSE_CHARACTERS
    for key in (
        "expected_words",
        "expected",
        "math_expected",
        "correct_answer",
        "recall_letter",
    ):
        if turn.get(key) is not None:
            return _declared_text(turn[key])
    return "X"


def _response_format_and_context_audit(
    spec: dict[str, Any], chosen: list[tuple[Any, str]]
) -> dict[str, Any]:
    """Outcome-blind proof that legal response shapes survive transport/context.

    Only aggregate maxima and booleans are persisted; no answer, stimulus, or
    task identifier leaves this function.
    """
    terminator = response_terminator(spec)
    stop_policy = request_stop_policy(spec)
    require(terminator not in spec["system_base"], "terminator collides with system base")
    require(
        all(terminator not in condition["scaffold"] for condition in spec["conditions"]),
        "terminator collides with an intervention scaffold",
    )
    prompts = [system_prompt(spec, condition["id"]) for condition in spec["conditions"]]
    require(all(prompt.count(terminator) == 1 for prompt in prompts),
            "transport terminator must occur exactly once in each complete system prompt")

    by_paradigm: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "item_count": 0,
            "max_declared_lines": 0,
            "max_declared_characters": 0,
            "max_declared_whitespace_tokens": 0,
            "max_context_prompt_estimate_tokens": 0,
            "max_reserved_prompt_plus_completion_tokens": 0,
        }
    )
    collisions = 0
    declared_stop_collisions = 0
    worst_paradigm = ""
    worst_reserved = -1
    completion = int(spec["scope"]["max_completion_tokens"])
    context = int(spec["scope"]["served_context_tokens"])

    for item, _ in chosen:
        paradigm = item.metadata.paradigm
        row = by_paradigm[paradigm]
        row["item_count"] += 1
        collisions += canonical_bytes(item_payload(item)).count(terminator.encode("utf-8"))
        candidates = _declared_response_candidates(item)
        require(candidates, f"no declared response shape for {paradigm}")
        for candidate in candidates:
            row["max_declared_lines"] = max(
                row["max_declared_lines"], len(candidate.splitlines()) or 1
            )
            row["max_declared_characters"] = max(
                row["max_declared_characters"], len(candidate)
            )
            row["max_declared_whitespace_tokens"] = max(
                row["max_declared_whitespace_tokens"], len(candidate.split())
            )
        call_candidates = _declared_call_candidates(item)
        require(call_candidates, f"no declared call-level response shape for {paradigm}")
        content_stops = [
            stop for stop in request_stop_sequences(spec, paradigm) if stop != terminator
        ]
        declared_stop_collisions += sum(
            stop in candidate
            for candidate in call_candidates
            for stop in content_stops
        )

        turns = item.metadata.parameters.get("turns", [])
        for sys_prompt in prompts:
            if not turns:
                user_prompt = format_user_prompt(
                    spec, paradigm, item.stimulus
                )
                estimates = [require_prompt_budget(spec, user_prompt, sys_prompt)]
            else:
                estimates = []
                history_lines = [item.stimulus]
                for index, turn in enumerate(turns):
                    kept = (
                        [history_lines[0]] + history_lines[-MULTITURN_HISTORY_LINES:]
                        if len(history_lines) > MULTITURN_HISTORY_LINES + 1
                        else history_lines
                    )
                    shown = turn_shown_text(turn)
                    user_prompt = (
                        "\n".join(kept)
                        + f"\nTrial {index + 1}: {shown}\nYour response:"
                    )
                    user_prompt = format_user_prompt(spec, paradigm, user_prompt)
                    estimates.append(require_prompt_budget(spec, user_prompt, sys_prompt))
                    response = _turn_response_for_context_stress(turn)
                    history_lines.append(f"Trial {index + 1}: {shown}")
                    history_lines.append(f"Your response: {response}")
                    feedback = turn.get("feedback", turn.get("correct_answer", ""))
                    if feedback:
                        history_lines.append(f"Feedback: {feedback}")
            item_max = max(estimates)
            row["max_context_prompt_estimate_tokens"] = max(
                row["max_context_prompt_estimate_tokens"], item_max
            )
            reserved = item_max + completion
            row["max_reserved_prompt_plus_completion_tokens"] = max(
                row["max_reserved_prompt_plus_completion_tokens"], reserved
            )
            if reserved > worst_reserved:
                worst_reserved = reserved
                worst_paradigm = paradigm

    require(collisions == 0, "transport terminator collides with frozen item content")
    require(
        declared_stop_collisions == 0,
        "response-format stop collides with a declared call-level answer",
    )
    global_lines = max(row["max_declared_lines"] for row in by_paradigm.values())
    global_chars = max(row["max_declared_characters"] for row in by_paradigm.values())
    global_tokens = max(
        row["max_declared_whitespace_tokens"] for row in by_paradigm.values()
    )
    require(global_tokens * 4 <= completion,
            "completion budget is under four times the largest declared answer")
    require(worst_reserved <= context, "context stress audit exceeded served context")
    return {
        "contract": (
            "Aggregate-only static audit over all selected items. Multiline lists are "
            "kept one element per line; CVLT filler history is stressed with a 1,024-"
            "character response; runtime's conservative char/3+64 estimator reserves "
            "the complete completion budget for every simulated call."
        ),
        "transport_terminator": terminator,
        "response_format_policy": stop_policy,
        "user_prompt_format_overrides": {
            paradigm: {
                "instruction_sha256": sha256_text(instruction),
                "instruction_characters": len(instruction),
                "instruction_whitespace_tokens": len(instruction.split()),
            }
            for paradigm, instruction in sorted(
                spec["response_format_overrides"].items()
            )
        },
        "terminator_absent_from_all_item_payloads_and_condition_scaffolds": True,
        "item_payload_terminator_collision_count": collisions,
        "stop_sequence_collisions_with_declared_call_responses": declared_stop_collisions,
        "declared_response_global_max": {
            "lines": global_lines,
            "characters": global_chars,
            "whitespace_tokens": global_tokens,
        },
        "completion_budget_tokens": completion,
        "declared_answer_completion_budget_multiple": completion / global_tokens,
        "context_stress": {
            "served_context_tokens": context,
            "filler_response_characters": FILLER_STRESS_RESPONSE_CHARACTERS,
            "max_reserved_prompt_plus_completion_tokens": worst_reserved,
            "minimum_reserved_margin_tokens": context - worst_reserved,
            "worst_paradigm": worst_paradigm,
        },
        "by_paradigm": dict(sorted(by_paradigm.items())),
    }


def build_manifest(
    spec: dict[str, Any], profile: str, forbidden_presentations: set[str] | None = None
) -> dict[str, Any]:
    forbidden_presentations = set(forbidden_presentations or ())
    pool = generate_pool(spec, profile)
    by_cell: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for item in pool:
        paradigm = item.metadata.paradigm
        if paradigm not in PARADIGM_DIMENSION:
            continue
        require(not item.metadata.parameters.get("contamination_probe", False), "probe leaked into pool")
        by_cell[(paradigm, _difficulty(item))].append(item)

    chosen: list[tuple[Any, str]] = []
    for paradigm in PARADIGM_DIMENSION:
        for difficulty in DIFFICULTIES:
            candidates = by_cell[(paradigm, difficulty)]
            if profile == "formal":
                selected, stratum = _formal_select(paradigm, difficulty, candidates, forbidden_presentations)
            else:
                selected = _take(candidates, 1, forbidden_presentations, f"pilot/{paradigm}/{difficulty}")
                stratum = "engineering_pilot=1"
            chosen.extend((item, stratum) for item in selected)

    expected_per_paradigm = spec["scope"][f"{profile}_items_per_paradigm"]
    counts = Counter(item.metadata.paradigm for item, _ in chosen)
    require(set(counts) == set(PARADIGM_DIMENSION), "manifest paradigm set mismatch")
    require(all(n == expected_per_paradigm for n in counts.values()), f"bad paradigm counts: {counts}")

    task_ids = [item.task_id for item, _ in chosen]
    fingerprints = [item_fingerprint(item) for item, _ in chosen]
    presentations = [presentation_fingerprint(item) for item, _ in chosen]
    require(len(task_ids) == len(set(task_ids)), "duplicate task IDs")
    require(len(fingerprints) == len(set(fingerprints)), "duplicate complete item fingerprints")
    require(not (set(presentations) & forbidden_presentations), "pilot/formal presentation overlap")

    group_of = {
        paradigm: group for group, paradigms in spec["grouping"].items() for paradigm in paradigms
    }
    entries = []
    for item, stratum in chosen:
        paradigm = item.metadata.paradigm
        entries.append(
            {
                "task_id": item.task_id,
                "dimension": PARADIGM_DIMENSION[paradigm],
                "group": group_of[paradigm],
                "paradigm": paradigm,
                "difficulty": _difficulty(item),
                "selection_stratum": stratum,
                "is_multiturn": bool(item.metadata.parameters.get("turns")),
                "n_turns": len(item.metadata.parameters.get("turns", [])),
                "item_fingerprint_sha256": item_fingerprint(item),
                "presentation_sha256": presentation_fingerprint(item),
                "scoring_gold_sha256": scoring_gold_fingerprint(item),
                "top_level_stimulus_sha256": sha256_text(item.stimulus),
            }
        )
    entries.sort(key=lambda x: (x["paradigm"], DIFFICULTIES.index(x["difficulty"]), x["task_id"]))
    require(len({entry["task_id"] for entry in entries}) == len(entries),
            f"selected {profile} manifest contains duplicate task IDs")

    return {
        "schema_version": "cogarena.causal_selectivity.item_manifest.v1",
        "study_id": spec["study_id"],
        "profile": profile,
        "spec_sha256": sha256_file(SPEC_PATH),
        "source_sha256": _source_hashes(),
        "prompt_length_audit": _prompt_audit(spec),
        "response_format_and_context_audit": _response_format_and_context_audit(
            spec, chosen
        ),
        "generation_seed": spec["item_generation"][f"{profile}_seed"],
        "item_count": len(entries),
        "condition_count": len(spec["conditions"]),
        "task_record_count_per_model": len(entries) * len(spec["conditions"]),
        "paradigm_counts": dict(sorted(counts.items())),
        "difficulty_counts": dict(sorted(Counter(x["difficulty"] for x in entries).items())),
        "items": entries,
    }


def validate_cross_profile(spec: dict[str, Any], formal: dict, pilot: dict) -> None:
    formal_models = {m["model"] for m in spec["formal_model_panel"]}
    pilot_models = {m["model"] for m in spec["pilot_model_panel"]}
    require(not formal_models & pilot_models, "pilot/formal model overlap")
    for key in ("task_id", "item_fingerprint_sha256", "presentation_sha256"):
        a = {x[key] for x in formal["items"]}
        b = {x[key] for x in pilot["items"]}
        require(not a & b, f"pilot/formal {key} overlap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("formal", "pilot", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    require(args.output_dir.resolve() == RESULTS_ROOT.resolve(), "noncanonical output directory refused")

    spec = load_spec()
    formal: dict | None = None
    pilot: dict | None = None
    if args.profile in {"formal", "all"}:
        formal = build_manifest(spec, "formal")
        atomic_write_json(manifest_path("formal"), formal)
    else:
        formal = load_json(manifest_path("formal"))
    if args.profile in {"pilot", "all"}:
        forbidden = {x["presentation_sha256"] for x in formal["items"]}
        pilot = build_manifest(spec, "pilot", forbidden)
        atomic_write_json(manifest_path("pilot"), pilot)
    else:
        pilot = load_json(manifest_path("pilot"))

    validate_cross_profile(spec, formal, pilot)
    provenance = {
        "schema_version": "cogarena.causal_selectivity.item_provenance.v1",
        "study_id": spec["study_id"],
        "repository_base_revision": spec["repository_base_revision"],
        "spec_sha256": sha256_file(SPEC_PATH),
        "formal_manifest_sha256": sha256_file(manifest_path("formal")),
        "pilot_manifest_sha256": sha256_file(manifest_path("pilot")),
        "source_sha256": _source_hashes(),
        "prompt_length_audit": _prompt_audit(spec),
        "formal_response_format_and_context_audit": formal[
            "response_format_and_context_audit"
        ],
        "pilot_response_format_and_context_audit": pilot[
            "response_format_and_context_audit"
        ],
        "cross_profile_disjoint": True,
    }
    atomic_write_json(RESULTS_ROOT / "ITEM_PROVENANCE.json", provenance)
    print(
        f"formal={formal['item_count']} pilot={pilot['item_count']} "
        f"records/model={formal['task_record_count_per_model']} cross-profile=PASS"
    )


if __name__ == "__main__":
    main()
