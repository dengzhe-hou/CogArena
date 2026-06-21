#!/usr/bin/env python3
"""Unified CogArena evaluation runner supporting 3 modes: text, image, agent.

Usage:
    # Text mode (default) — LLM answers directly
    python scripts/run_unified.py --model openai/qwen2.5:7b --mode text

    # Image mode — VLM sees image stimuli
    python scripts/run_unified.py --model openai/qwen2.5vl:7b --mode image

    # Agent mode — LLM + ReAct scaffold + tools
    python scripts/run_unified.py --model openai/qwen2.5:7b --mode agent
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path
from datetime import datetime

# ── Text mode imports ──
import sys, importlib
sys.path.insert(0, str(Path(__file__).resolve().parent))
_re = importlib.import_module("run_eval")
call_llm = _re.call_llm
score_static_item = _re.score_static_item
score_multiturn_item = _re.score_multiturn_item
_get_paradigm = _re._get_paradigm
_get_dimension = _re._get_dimension
_get_difficulty = _re._get_difficulty
_get_params = _re._get_params
_has_turns = _re._has_turns
GENERATORS = _re.GENERATORS
SYSTEM_PROMPT = _re.SYSTEM_PROMPT

# ── Image mode imports ──
from cogarena.image_gen.stroop_images import generate_stroop_set
from cogarena.image_gen.flanker_images import generate_flanker_set
from cogarena.image_gen.false_belief_images import generate_false_belief_set

# ── Agent mode imports ──
from cogarena.agent import CogArenaAgent
from cogarena.llm_client import LLMClient
from cogarena.core import CogArenaEnv, TaskMetadata, EvalMode, ScoringConfig


def call_vlm(model_id: str, prompt: str, image_paths) -> str:
    """Call VLM with text + one or more images."""
    import openai
    client = openai.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    content = [{"type": "text", "text": prompt}]
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    messages = [{"role": "user", "content": content}]
    model_name = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name, messages=messages, temperature=0, max_tokens=256)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"
    return "ERROR"


def score_simple(expected: str, response: str) -> bool:
    """Flexible scoring for VLM verbose responses."""
    if not expected:
        return False
    exp = expected.strip().lower().strip('."\'!?,;:')
    resp = response.strip().lower().strip('."\'!?,;:')
    # Remove common articles
    for art in ['the ', 'a ', 'an ']:
        if exp.startswith(art):
            exp = exp[len(art):]
        if resp.startswith(art):
            resp = resp[len(art):]
    if not exp:
        return False
    return exp == resp or exp in resp or resp in exp


# ── Image mode paradigms ──────────────────────────────────────────────

# WCST removed from image mode — it requires multi-turn feedback loop (use agent mode instead)
IMAGE_PARADIGMS = {
    "stroop": {"gen": generate_stroop_set, "dimension": "cognitive_control"},
    "flanker": {"gen": generate_flanker_set, "dimension": "cognitive_control"},
    "false_belief": {"gen": generate_false_belief_set, "dimension": "theory_of_mind"},
}


def run_image_mode(model_id: str, paradigms: list, n_items: int, seed: int,
                   results_dir: Path) -> list:
    """Run image-mode evaluation on VLM."""
    all_results = []

    for para_name in paradigms:
        info = IMAGE_PARADIGMS.get(para_name)
        if not info:
            print(f"  {para_name}: no image generator, skipping")
            continue

        gen_fn = info["gen"]
        dimension = info["dimension"]
        print(f"  --- {para_name} (image) ---")

        if para_name in ["stroop", "flanker"]:
            trials = gen_fn(seed=seed, n_congruent=n_items, n_incongruent=n_items,
                           out_dir=f"data/images/{para_name}")
        elif para_name == "wcst":
            trials = gen_fn(seed=seed, n_trials=n_items * 3,
                           out_dir=f"data/images/{para_name}")
        elif para_name == "false_belief":
            trials = gen_fn(seed=seed, n_items=n_items,
                           out_dir=f"data/images/{para_name}")
        else:
            trials = gen_fn(seed=seed, n_items=n_items,
                           out_dir=f"data/images/{para_name}")

        print(f"  Generated {len(trials)} image trials")

        for i, trial in enumerate(trials):
            # Get image path(s)
            if "image_path" in trial:
                img_paths = [trial["image_path"]]
            elif "image_paths" in trial:
                img_paths = trial["image_paths"]  # all frames for false belief
            else:
                continue

            # For false_belief: send ALL scene frames + image-only prompt
            if para_name == "false_belief" and len(img_paths) > 1:
                chars = trial.get("characters", ["Character A", "Character B"])
                prompt = (
                    f"These images show a sequence of events in order. "
                    f"Study all {len(img_paths)} scenes carefully.\n"
                    f"Question: Where will {chars[1]} look for the object? "
                    f"Answer with the location only."
                )
            else:
                prompt = trial.get("stimulus_text", "What do you see?")

            expected = str(trial.get("correct_answer", trial.get("expected_response", "")))

            resp = call_vlm(model_id, prompt, img_paths)
            correct = score_simple(expected, resp)

            result = {
                "task_id": f"img_{para_name}_{i:04d}",
                "model_id": model_id,
                "mode": "image",
                "dimension": dimension,
                "paradigm": para_name,
                "response": resp,
                "expected": expected,
                "correct": correct,
                "congruent": trial.get("congruent"),
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result)

            cond = "cong" if trial.get("congruent") else "incong" if trial.get("congruent") is False else ""
            status = "✓" if correct else "✗"
            print(f"    [{i+1}/{len(trials)}] {cond:5s} exp={expected[:10]:10s} resp={resp[:15]:15s} {status}")

    return all_results


# ── Agent mode ─────────────────────────────────────────────────────────

# reversal_learning removed: needs probabilistic rewards, CogArenaEnv gives deterministic 0/1
AGENT_PARADIGMS = ["wcst", "n_back", "false_belief"]


def _format_wcst_stimulus(turn: dict) -> str:
    """Format a WCST turn into readable text for the agent."""
    target = turn.get("target_card", {})
    refs = turn.get("reference_cards", [])
    trial_num = turn.get("trial", 0) + 1

    lines = [f"Trial {trial_num}:"]
    dims = list(target.keys())
    lines.append(f"Target card: {', '.join(f'{k}={v}' for k, v in target.items())}")
    lines.append("Reference cards:")
    for i, rc in enumerate(refs):
        lines.append(f"  {i+1}: {', '.join(f'{k}={v}' for k, v in rc.items())}")
    lines.append("Which reference card (1-4) matches the target on the current hidden rule?")
    lines.append("Answer with just the number (1, 2, 3, or 4).")
    return "\n".join(lines)


def _format_nback_stimulus(turn: dict) -> str:
    """Format an n-back turn."""
    token = turn.get("stimulus", turn.get("token", "?"))
    return f"Token: {token}\nRespond MATCH or NO MATCH."


def make_trial_generator(paradigm_name: str, seed: int, difficulty: str):
    """Create a trial generator function for CogArenaEnv from a paradigm."""
    def generator(config):
        if paradigm_name == "wcst":
            from cogarena.dimensions.set_shifting import WCSTGenerator
            items = WCSTGenerator().generate(seed=seed, n_items=1, difficulty=difficulty)
        elif paradigm_name == "reversal_learning":
            from cogarena.dimensions.set_shifting import ReversalLearningGenerator
            items = ReversalLearningGenerator().generate(seed=seed, n_items=1, difficulty=difficulty)
        elif paradigm_name == "n_back":
            from cogarena.dimensions.working_memory import NBackGenerator
            items = NBackGenerator().generate(seed=seed, n_items=1, difficulty=difficulty)
        elif paradigm_name == "false_belief":
            from cogarena.dimensions.theory_of_mind import FalseBeliefGenerator
            items = FalseBeliefGenerator().generate(seed=seed, n_items=1, difficulty=difficulty)
        else:
            return []

        if not items:
            return []

        item = items[0]
        params = item.metadata.parameters if hasattr(item, "metadata") else {}
        turns = params.get("turns", [])

        # Static paradigms (no turns) — wrap as single-turn episode
        if not turns and item.stimulus:
            return [{
                "stimulus": item.stimulus,
                "expected": str(item.expected_response or ""),
            }]

        env_trials = []
        for i, t in enumerate(turns):
            # Build proper stimulus text
            if paradigm_name == "wcst":
                stimulus = _format_wcst_stimulus(t)
            elif paradigm_name == "n_back":
                stimulus = _format_nback_stimulus(t)
            else:
                stimulus = t.get("stimulus", str(t))

            expected = str(t.get("expected", t.get("correct_answer", "")))

            # Build dynamic feedback template
            # {correct} will be filled by CogArenaEnv.step() with the expected answer
            # {response} with the agent's response
            # {score} with 0.0 or 1.0
            # CogArenaEnv fills: feedback_template.format(correct=expected, response=action, score=reward)
            # score is 1.0 (correct) or 0.0 (wrong)
            if paradigm_name == "wcst":
                feedback_tpl = "Feedback: score={score}. The correct card was {correct}. Your choice was {response}."
            elif paradigm_name == "reversal_learning":
                feedback_tpl = "Reward: {score}. You chose {response}."
            elif paradigm_name == "n_back":
                feedback_tpl = None
            else:
                feedback_tpl = None

            trial = {
                "stimulus": stimulus,
                "expected": expected,
            }
            if feedback_tpl:
                trial["feedback_template"] = feedback_tpl

            env_trials.append(trial)

        return env_trials

    return generator


def run_agent_mode(model_id: str, paradigms: list, n_items: int, seed: int,
                   results_dir: Path) -> list:
    """Run agent-mode evaluation with ReAct scaffold + tools."""
    client = LLMClient(config={
        "provider": "openai",
        "model": model_id.split("/", 1)[-1] if "/" in model_id else model_id,
        "api_key": os.environ.get("OPENAI_API_KEY", "ollama"),
        "base_url": os.environ.get("OPENAI_BASE_URL"),
    })
    agent = CogArenaAgent(client, max_think_steps=2)

    all_results = []
    for para_name in paradigms:
        if para_name not in AGENT_PARADIGMS:
            print(f"  {para_name}: no agent env, skipping")
            continue

        print(f"  --- {para_name} (agent) ---")

        for ep in range(n_items):
            ep_seed = seed + ep * 100
            trial_gen = make_trial_generator(para_name, ep_seed, "easy")
            dim_map = {"wcst": "set_shifting", "reversal_learning": "set_shifting",
                       "n_back": "working_memory", "false_belief": "theory_of_mind"}
            dim = dim_map.get(para_name, "unknown")
            # Use partial_match scoring so "3" matches response "3" or "Card 3"
            meta = TaskMetadata(
                dimension=dim,
                paradigm=para_name,
                mode=EvalMode.AGENT_INTERACTIVE,
                scoring=ScoringConfig(method="partial_match", params={"direction": "expected_in_response"}),
            )
            env = CogArenaEnv(trial_gen, meta)

            agent.reset()
            obs = env.reset(seed=ep_seed)

            # Add task-level instructions to first observation
            if para_name == "wcst":
                obs["instructions"] = (
                    "Card Sorting Task. You will see cards with properties. "
                    "Find the hidden sorting rule by trial and error. "
                    "After each choice you get feedback (score=1.0 means CORRECT, score=0.0 means WRONG). "
                    "The rule may change without warning. "
                    "Use your memory_store tool to track what you've learned about the rule. "
                    "Answer with just the card number (1-4)."
                )
            elif para_name == "n_back":
                obs["instructions"] = (
                    "N-Back Task. You will see tokens one at a time. "
                    "Respond MATCH if the current token is the same as N positions back, otherwise NO MATCH. "
                    "Use your memory_store tool to remember recent tokens."
                )
            elif para_name == "reversal_learning":
                obs["instructions"] = (
                    "Two-Choice Learning Task. Choose option A or B on each trial. "
                    "One option is usually rewarded (score=1.0). "
                    "The reward contingencies may reverse. Track which option is currently better."
                )
            elif para_name == "false_belief":
                obs["instructions"] = (
                    "Theory of Mind Task. Read the story carefully. "
                    "Think about what each character knows and doesn't know. "
                    "Use memory_store to track each character's beliefs. "
                    "Answer with the location only."
                )

            done = False
            step = 0
            max_steps = 60

            while not done and step < max_steps:
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                agent.record_reward(reward)
                step += 1

            scores = env.score()
            trace = env.trace

            result = {
                "task_id": f"agent_{para_name}_{ep:03d}",
                "model_id": model_id,
                "mode": "agent",
                "dimension": meta.dimension,
                "paradigm": para_name,
                "n_steps": step,
                "scores": scores,
                "accuracy": scores.get("accuracy", 0),
                "tools_used": agent.tool_call_count,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result)
            print(f"    [ep {ep+1}/{n_items}] steps={step} acc={scores.get('accuracy',0):.0%} tools={result['tools_used']}")

    return all_results


# ── Text mode (reuse run_eval.py) ──────────────────────────────────────

def run_text_mode(model_id: str, dimensions: list, n_per_paradigm: int,
                  seed: int, results_dir: Path, static_only: bool) -> list:
    """Run text-mode evaluation (delegates to run_eval.py logic)."""
    run_static_item = _re.run_static_item
    run_multiturn_item = _re.run_multiturn_item

    all_results = []
    for dim_name in dimensions:
        gen_fn = GENERATORS.get(dim_name)
        if not gen_fn:
            continue
        print(f"  --- {dim_name} (text) ---")
        items = gen_fn(seed=seed, n_per_paradigm=n_per_paradigm,
                       include_contamination_probes=False)
        print(f"  Generated {len(items)} items")

        for i, item in enumerate(items):
            paradigm = _get_paradigm(item)
            is_mt = _has_turns(item)
            if is_mt and static_only:
                continue
            try:
                if is_mt:
                    result = run_multiturn_item(model_id, item, results_dir)
                else:
                    result = run_static_item(model_id, item, results_dir)
                result["mode"] = "text"
                all_results.append(result)
            except Exception as e:
                print(f"    [{i+1}] {paradigm} ERROR: {e}")

    return all_results


# ── Aggregation helpers ────────────────────────────────────────────────

def _item_accuracy(result: dict) -> float:
    """Scalar accuracy in [0, 1] for a single result record.

    Several paradigms (digit_span, false_belief, epitome_tom,
    confidence_calibration, post_decision_wagering, ...) store ``score`` as a
    dict like ``{"accuracy": 1.0, ...}`` rather than a scalar/bool. Without
    this, the inline aggregator counted those items as 0 even when their
    per-item ``score.accuracy`` was 1.0. Pull the accuracy out of dict scores
    (preferring ``accuracy``, then ``score``, then ``correct``); fall back to a
    scalar ``score`` and, for image/agent modes, the top-level ``correct`` /
    ``accuracy`` fields. Callers binarize at >= 0.5 where a 0/1 outcome is
    needed.
    """
    score = result.get("score")
    if isinstance(score, dict):
        if "accuracy" in score:
            return float(score["accuracy"])
        if "score" in score:
            return float(score["score"])
        if "correct" in score:
            return 1.0 if score["correct"] else 0.0
    elif isinstance(score, bool):
        return 1.0 if score else 0.0
    elif score is not None:
        try:
            return float(score)
        except (TypeError, ValueError):
            pass
    # Top-level fallbacks: image mode stores `correct` (bool), agent mode `accuracy`.
    if result.get("correct") is True:
        return 1.0
    if result.get("correct") is False:
        return 0.0
    try:
        return float(result.get("accuracy", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


# Partial-credit paradigms report mean(accuracy) instead of a binary
# correct/incorrect count.
PARTIAL_CREDIT_PARADIGMS = {"source_monitoring", "drm_false_memory"}


def aggregate_by_paradigm(results: list) -> dict:
    """Aggregate a list of result records into per-paradigm correct/count/accuracy.

    Binary paradigms (stroop, flanker, go_nogo, digit_span, false_belief, ...)
    are counted via :func:`_item_accuracy` binarized at >= 0.5 — this is what
    keeps dict-score paradigms (``{"accuracy": 1.0, ...}``) from being miscounted
    as 0. Partial-credit paradigms use mean(accuracy).
    """
    by_paradigm = {}
    paradigm_accs = {}  # collect per-item accuracy for partial-credit
    for r in results:
        p = r.get("paradigm", "unknown")
        if p not in by_paradigm:
            by_paradigm[p] = {"count": 0, "correct": 0}
            paradigm_accs[p] = []
        by_paradigm[p]["count"] += 1

        if p in PARTIAL_CREDIT_PARADIGMS and isinstance(r.get("score"), dict):
            # Use continuous accuracy for partial-credit paradigms
            acc = r["score"].get("accuracy", 0)
            paradigm_accs[p].append(acc)
            by_paradigm[p]["correct"] += acc  # sum of accuracies
        else:
            # Binary correct/incorrect for other paradigms. Use the shared
            # helper so dict scores (e.g. digit_span's {"accuracy": 1.0, ...})
            # are extracted instead of silently counted as 0, then binarize.
            if _item_accuracy(r) >= 0.5:
                by_paradigm[p]["correct"] += 1

    for p, v in by_paradigm.items():
        if p in PARTIAL_CREDIT_PARADIGMS and paradigm_accs.get(p):
            v["accuracy"] = sum(paradigm_accs[p]) / len(paradigm_accs[p])
            v["correct"] = round(v["accuracy"] * v["count"])  # approx for display
        else:
            v["accuracy"] = v["correct"] / v["count"] if v["count"] > 0 else 0
    return by_paradigm


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CogArena Unified Evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["text", "image", "agent"], default="text")
    parser.add_argument("--n-items", type=int, default=5, help="Items per paradigm/trial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/unified")
    parser.add_argument("--dimensions", nargs="+", default=None,
                        help="Text mode: specific dimensions")
    parser.add_argument("--paradigms", nargs="+", default=None,
                        help="Image/Agent mode: specific paradigms")
    parser.add_argument("--static-only", action="store_true",
                        help="Text mode: skip multi-turn")
    args = parser.parse_args()

    model_safe = args.model.replace("/", "_")
    results_dir = Path(args.output_dir) / model_safe / args.mode

    print(f"CogArena Unified Evaluation")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Items: {args.n_items}")
    print()

    if args.mode == "text":
        dims = args.dimensions or list(GENERATORS.keys())
        results = run_text_mode(args.model, dims, args.n_items, args.seed,
                               results_dir, args.static_only)

    elif args.mode == "image":
        paradigms = args.paradigms or list(IMAGE_PARADIGMS.keys())
        results = run_image_mode(args.model, paradigms, args.n_items, args.seed,
                                results_dir)

    elif args.mode == "agent":
        paradigms = args.paradigms or AGENT_PARADIGMS
        results = run_agent_mode(args.model, paradigms, args.n_items, args.seed,
                                results_dir)

    # Save aggregate
    results_dir.mkdir(parents=True, exist_ok=True)
    agg_path = results_dir / "aggregate.json"
    summary = {
        "model": args.model,
        "mode": args.mode,
        "timestamp": datetime.now().isoformat(),
        "n_results": len(results),
    }

    # Aggregate by paradigm (partial-credit -> mean accuracy; others ->
    # binarized correct count, handling dict scores via _item_accuracy).
    by_paradigm = aggregate_by_paradigm(results)
    summary["paradigms"] = by_paradigm

    agg_path.write_text(json.dumps(summary, indent=2))

    # Save detailed results
    detail_path = results_dir / "details.json"
    detail_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    # Print summary
    print(f"\n{'='*60}")
    print(f"{'Paradigm':<25} {'Correct':>8} {'Total':>6} {'Acc':>7}")
    print(f"{'-'*60}")
    for p, v in sorted(by_paradigm.items()):
        print(f"{p:<25} {v['correct']:>8}/{v['count']:<6} {v['accuracy']:>6.0%}")
    print(f"{'='*60}")
    print(f"Results: {agg_path}")


if __name__ == "__main__":
    main()
