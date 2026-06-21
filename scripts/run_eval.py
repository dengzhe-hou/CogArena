#!/usr/bin/env python3
"""Run CogArena evaluation on one or more LLMs.

Usage:
    # Sanity check with a small subset
    python scripts/run_eval.py --model openai/qwen2.5:7b --n-per-paradigm 5 --dimensions cognitive_control theory_of_mind

    # Full pilot
    python scripts/run_eval.py --model openai/qwen2.5:7b --n-per-paradigm 50 --output-dir results/pilot
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

from cogarena.generators.working_memory_gen import generate_wm_items
from cogarena.generators.cognitive_control_gen import generate_cc_items
from cogarena.generators.set_shifting_gen import generate_ss_items
from cogarena.generators.episodic_memory_gen import generate_em_items
from cogarena.generators.theory_of_mind_gen import generate_tom_items
from cogarena.generators.metacognition_gen import generate_mc_items

GENERATORS = {
    "working_memory": generate_wm_items,
    "cognitive_control": generate_cc_items,
    "set_shifting": generate_ss_items,
    "episodic_memory": generate_em_items,
    "theory_of_mind": generate_tom_items,
    "metacognition": generate_mc_items,
}

# Map paradigm names to their module-level score functions
PARADIGM_SCORERS = {
    # cognitive_control
    "stroop": "cogarena.dimensions.cognitive_control:StroopParadigm.score",
    "flanker": "cogarena.dimensions.cognitive_control:FlankerParadigm.score",
    "go_nogo": "cogarena.dimensions.cognitive_control:GoNoGoParadigm.score",
    # working_memory
    "n_back": "cogarena.dimensions.working_memory:NBackGenerator.score",
    "digit_span": "cogarena.dimensions.working_memory:DigitSpanGenerator.score",
    "operation_span": "cogarena.dimensions.working_memory:OperationSpanGenerator.score",
    # set_shifting
    "wcst": "cogarena.dimensions.set_shifting:WCSTGenerator.score",
    "reversal_learning": "cogarena.dimensions.set_shifting:ReversalLearningGenerator.score",
    # episodic_memory
    "cvlt_word_list": "cogarena.dimensions.episodic_memory:CVLTGenerator.score",
    "drm_false_memory": "cogarena.dimensions.episodic_memory:DRMGenerator.score",
    "source_monitoring": "cogarena.dimensions.episodic_memory:SourceMonitoringGenerator.score",
    # theory_of_mind
    "false_belief": "cogarena.dimensions.theory_of_mind:FalseBeliefGenerator.score",
    "epitome_tom": "cogarena.dimensions.theory_of_mind:EpitomeToMGenerator.score",
    # metacognition
    "confidence_calibration": "cogarena.dimensions.metacognition:ConfidenceCalibrationGenerator.score",
    "post_decision_wagering": "cogarena.dimensions.metacognition:PostDecisionWageringGenerator.score",
}


def _resolve_scorer(dotted_path: str):
    """Import 'module.path:Class.method' and return the callable."""
    mod_path, attr_path = dotted_path.split(":", 1)
    mod = importlib.import_module(mod_path)
    obj = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def _get_meta(item):
    """Get metadata object (handles nested schema)."""
    return item.metadata if hasattr(item, "metadata") and item.metadata else None


def _get_paradigm(item) -> str:
    meta = _get_meta(item)
    return meta.paradigm if meta else getattr(item, "paradigm", "unknown")


def _get_dimension(item) -> str:
    meta = _get_meta(item)
    return meta.dimension if meta else getattr(item, "dimension", "unknown")


def _get_difficulty(item) -> str:
    meta = _get_meta(item)
    d = meta.difficulty if meta else getattr(item, "difficulty", "medium")
    return d.value if hasattr(d, "value") else str(d)


def _get_params(item) -> dict:
    meta = _get_meta(item)
    return meta.parameters if meta else getattr(item, "parameters", {})


def _has_turns(item) -> bool:
    params = _get_params(item)
    turns = params.get("turns", [])
    return isinstance(turns, list) and len(turns) > 0


# ---------------------------------------------------------------------------
# LLM calling
# ---------------------------------------------------------------------------

def call_llm(model_id: str, prompt: str, system_prompt: str = None) -> str:
    """Call an LLM via API. Supports openai-compatible APIs."""
    provider, model_name = model_id.split("/", 1) if "/" in model_id else ("openai", model_id)

    if provider in ("openai", "local"):
        import openai
        base_url = os.environ.get("OPENAI_BASE_URL")  # always honor env var
        client = openai.OpenAI(base_url=base_url, timeout=300.0)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=1024,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return f"ERROR: {e}"

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=1024,
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return f"ERROR: {e}"

    return f"ERROR: Unknown provider {provider}"


SYSTEM_PROMPT = (
    "You are taking a cognitive evaluation. Follow the instructions precisely. "
    "Give only the requested answer, nothing else."
)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_static_item(item, response: str) -> dict:
    """Score a static item using the paradigm-specific scorer or TaskInstance.score()."""
    paradigm = _get_paradigm(item)

    # Try paradigm-specific scorer first (handles rich metrics)
    scorer_path = PARADIGM_SCORERS.get(paradigm)
    if scorer_path:
        try:
            scorer_fn = _resolve_scorer(scorer_path)
            return scorer_fn(item, response)
        except Exception as e:
            import sys
            print(f"  [WARN] Paradigm scorer failed for {paradigm}: {e}", file=sys.stderr)
            # Fall through to generic

    # Try TaskInstance.score() (uses ScoringConfig from metadata)
    if hasattr(item, "score") and callable(item.score):
        try:
            result = item.score(response)
            if isinstance(result, dict):
                return result
        except Exception as e:
            import sys
            print(f"  [WARN] TaskInstance.score() failed for {paradigm}: {e}", file=sys.stderr)

    # Fallback: exact match (safe, no substring bugs)
    expected = str(item.expected_response).strip().lower() if item.expected_response is not None else ""
    actual = response.strip().lower()
    if not expected:
        return {"response": response, "scored": False}
    return {
        "correct": expected == actual,
        "expected": str(item.expected_response),
        "response": response,
    }


def score_multiturn_item(item, responses: list) -> dict:
    """Score a multi-turn item by scoring each turn and aggregating."""
    paradigm = _get_paradigm(item)
    params = _get_params(item)
    turns = params.get("turns", [])

    if not turns or not responses:
        return {"scored": False, "n_turns": len(responses)}

    # Score each turn individually
    turn_scores = []
    for i, (turn, resp_entry) in enumerate(zip(turns, responses)):
        resp_text = resp_entry["response"] if isinstance(resp_entry, dict) else str(resp_entry)

        # Try multiple expected-answer keys
        expected = turn.get("expected", turn.get("correct_answer", turn.get("target", None)))
        expected_words = turn.get("expected_words")  # list recall (CVLT)

        if expected_words is not None and isinstance(expected_words, list):
            # List recall scoring: count how many target words appear in response
            resp_words = [w.strip().lower() for w in resp_text.replace(",", "\n").split("\n") if w.strip()]
            target_set = set(w.lower() for w in expected_words)
            hits = sum(1 for w in resp_words if w in target_set)
            recall = hits / len(target_set) if target_set else 0
            turn_scores.append({
                "trial": i + 1, "correct": recall >= 0.5,
                "recall": recall, "hits": hits, "total": len(target_set),
                "response": resp_text[:100],
            })
        elif expected is not None:
            exp_str = str(expected).strip().lower()
            act_str = resp_text.strip().lower()
            # Strict matching for multi-turn responses
            # For n_back: "match" vs "no match" must be exact (not substring)
            if exp_str in ("match", "no match"):
                correct = exp_str == act_str or act_str.startswith(exp_str + " ") or act_str == exp_str
                # Double-check: "no match" should NOT match expected "match"
                if exp_str == "match" and "no" in act_str:
                    correct = False
            else:
                correct = exp_str == act_str or exp_str in act_str
            turn_scores.append({"trial": i + 1, "correct": correct, "expected": exp_str, "response": act_str[:100]})
        else:
            turn_scores.append({"trial": i + 1, "scored": False, "response": resp_text[:100]})

    scored_turns = [t for t in turn_scores if "correct" in t]
    if not scored_turns:
        return {"scored": False, "n_turns": len(responses), "turn_scores": turn_scores}

    accuracy = sum(1 for t in scored_turns if t["correct"]) / len(scored_turns)
    return {
        "accuracy": accuracy,
        "n_scored": len(scored_turns),
        "n_correct": sum(1 for t in scored_turns if t["correct"]),
        "n_turns": len(responses),
        "turn_scores": turn_scores,
    }


# ---------------------------------------------------------------------------
# Eval runners
# ---------------------------------------------------------------------------

def run_static_item(model_id, item, results_dir):
    """Evaluate and score a single static item."""
    task_id = item.task_id
    paradigm = _get_paradigm(item)
    dimension = _get_dimension(item)

    result_path = results_dir / dimension / paradigm / f"{task_id}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    response = call_llm(model_id, item.stimulus, SYSTEM_PROMPT)
    score = score_static_item(item, response)

    result = {
        "task_id": task_id,
        "model_id": model_id,
        "dimension": dimension,
        "paradigm": paradigm,
        "difficulty": _get_difficulty(item),
        "response": response,
        "score": score,
        "timestamp": datetime.now().isoformat(),
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def run_multiturn_item(model_id, item, results_dir, max_context_turns=30):
    """Evaluate a multi-turn item with sliding context window."""
    task_id = item.task_id
    paradigm = _get_paradigm(item)
    dimension = _get_dimension(item)

    result_path = results_dir / dimension / paradigm / f"{task_id}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    params = _get_params(item)
    turns = params.get("turns", [])

    # Build conversation with sliding window to avoid context overflow
    history_lines = [item.stimulus]
    responses = []

    for i, turn in enumerate(turns):
        stimulus_text = turn.get("stimulus", turn.get("presented", turn.get("item", str(turn))))
        if isinstance(stimulus_text, dict):
            stimulus_text = json.dumps(stimulus_text)

        # Sliding window: keep only last max_context_turns of history
        if len(history_lines) > max_context_turns + 1:
            kept = [history_lines[0]] + history_lines[-(max_context_turns):]
            prompt = "\n".join(kept) + f"\nTrial {i+1}: {stimulus_text}\nYour response:"
        else:
            prompt = "\n".join(history_lines) + f"\nTrial {i+1}: {stimulus_text}\nYour response:"

        response = call_llm(model_id, prompt, SYSTEM_PROMPT)
        responses.append({"trial": i + 1, "stimulus": str(stimulus_text)[:200], "response": response})

        # Update history
        feedback = turn.get("feedback", turn.get("correct_answer", ""))
        history_lines.append(f"Trial {i+1}: {stimulus_text}")
        history_lines.append(f"Your response: {response}")
        if feedback:
            history_lines.append(f"Feedback: {feedback}")

    # Score the multi-turn episode
    score = score_multiturn_item(item, responses)

    result = {
        "task_id": task_id,
        "model_id": model_id,
        "dimension": dimension,
        "paradigm": paradigm,
        "difficulty": _get_difficulty(item),
        "n_turns": len(turns),
        "responses": responses,
        "score": score,
        "timestamp": datetime.now().isoformat(),
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_aggregate(all_results: list) -> dict:
    """Compute aggregate statistics from results."""
    summary = {"dimensions": {}}
    for r in all_results:
        dim = r.get("dimension", "unknown")
        par = r.get("paradigm", "unknown")
        key = f"{dim}/{par}"
        if key not in summary["dimensions"]:
            summary["dimensions"][key] = {"count": 0, "correct": 0, "scored": 0}
        entry = summary["dimensions"][key]
        entry["count"] += 1

        score = r.get("score", {})
        if score.get("scored") is False:
            continue
        entry["scored"] += 1

        # Handle multiple score formats
        # Use continuous accuracy (mean) instead of binary threshold
        if "accuracy" in score:
            acc_val = score["accuracy"]
            if "acc_sum" not in entry:
                entry["acc_sum"] = 0.0
            entry["acc_sum"] += acc_val
            entry["correct"] += round(acc_val)  # approximate int for display
        elif "correct" in score:
            val = score["correct"]
            if val is True:
                entry["correct"] += 1
                if "acc_sum" not in entry:
                    entry["acc_sum"] = 0.0
                entry["acc_sum"] += 1.0
            elif val is False:
                if "acc_sum" not in entry:
                    entry["acc_sum"] = 0.0
                entry["acc_sum"] += 0.0

    # Compute accuracy per paradigm using mean accuracy (not pass-rate)
    for key, entry in summary["dimensions"].items():
        n = entry["scored"] if entry["scored"] > 0 else entry["count"]
        if "acc_sum" in entry and n > 0:
            entry["accuracy"] = entry["acc_sum"] / n
        else:
            entry["accuracy"] = entry["correct"] / n if n > 0 else 0.0
        entry.pop("acc_sum", None)  # clean up temp field

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run CogArena evaluation")
    parser.add_argument("--model", type=str, required=True,
                        help="Model ID: provider/model_name (e.g., openai/gpt-4o-mini)")
    parser.add_argument("--n-per-paradigm", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dimensions", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="results/sanity")
    parser.add_argument("--static-only", action="store_true",
                        help="Skip multi-turn items (faster for sanity)")
    parser.add_argument("--max-context-turns", type=int, default=30,
                        help="Sliding window size for multi-turn context")
    args = parser.parse_args()

    model_safe = args.model.replace("/", "_")
    results_dir = Path(args.output_dir) / model_safe

    dims = args.dimensions or list(GENERATORS.keys())
    print(f"Model: {args.model}")
    print(f"Dimensions: {dims}")
    print(f"Items per paradigm: {args.n_per_paradigm}")
    print(f"Output: {results_dir}")
    print()

    all_results = []
    for dim_name in dims:
        gen_fn = GENERATORS.get(dim_name)
        if not gen_fn:
            print(f"Unknown dimension: {dim_name}")
            continue

        print(f"--- {dim_name} ---")
        items = gen_fn(seed=args.seed, n_per_paradigm=args.n_per_paradigm,
                       include_contamination_probes=False)
        print(f"  Generated {len(items)} items")

        for i, item in enumerate(items):
            paradigm = _get_paradigm(item)
            is_multiturn = _has_turns(item)

            if is_multiturn and args.static_only:
                continue

            try:
                if is_multiturn:
                    result = run_multiturn_item(args.model, item, results_dir, args.max_context_turns)
                else:
                    result = run_static_item(args.model, item, results_dir)
                all_results.append(result)

                score = result.get("score", {})
                if score.get("correct") is True:
                    status = "CORRECT"
                elif score.get("correct") is False:
                    status = "WRONG"
                elif "accuracy" in score:
                    status = f"acc={score['accuracy']:.2f}"
                else:
                    status = "OK"
                print(f"  [{i+1}/{len(items)}] {paradigm} — {status}")
            except Exception as e:
                print(f"  [{i+1}/{len(items)}] {paradigm} — EXCEPTION: {e}")

    # Save aggregate
    agg = compute_aggregate(all_results)
    agg["model"] = args.model
    agg["timestamp"] = datetime.now().isoformat()
    agg["n_items"] = len(all_results)

    agg_path = results_dir / "aggregate.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(json.dumps(agg, indent=2))

    # Print summary table
    print(f"\n{'='*65}")
    print(f"{'Paradigm':<40} {'Correct':>8} {'Total':>6} {'Acc':>7}")
    print(f"{'-'*65}")
    for key in sorted(agg["dimensions"]):
        e = agg["dimensions"][key]
        print(f"{key:<40} {e['correct']:>8}/{e['count']:<6} {e['accuracy']:>6.1%}")
    print(f"{'='*65}")
    print(f"Aggregate saved to {agg_path}")


if __name__ == "__main__":
    main()
