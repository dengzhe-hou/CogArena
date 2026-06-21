#!/usr/bin/env python3
"""B1.5: Intervention-based construct checks.

For each paradigm, sweep ONE difficulty parameter and measure
the dose-response curve. Compare against published human patterns.

This is CogArena's key methodological contribution — showing that
paradigms respond to construct-relevant interventions, not just
prompt-specific artifacts.

Usage:
    OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 \
    python scripts/run_intervention_sweep.py --model openai/qwen2.5:7b

    # Specific paradigms only
    python scripts/run_intervention_sweep.py --model openai/qwen2.5:7b \
        --paradigms n_back stroop go_nogo
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime

from cogarena.dimensions.working_memory import NBackGenerator, DigitSpanGenerator
from cogarena.dimensions.cognitive_control import StroopParadigm, FlankerParadigm, GoNoGoParadigm
from cogarena.dimensions.set_shifting import WCSTGenerator
from cogarena.dimensions.theory_of_mind import FalseBeliefGenerator
from cogarena.dimensions.episodic_memory import DRMGenerator, SourceMonitoringGenerator
from cogarena.dimensions.metacognition import ConfidenceCalibrationGenerator

# Import paradigm-specific scoring from run_eval
import sys, importlib
sys.path.insert(0, str(Path(__file__).resolve().parent))
_run_eval = importlib.import_module("run_eval")
_score_static_item = _run_eval.score_static_item


def call_llm(model_id: str, prompt: str, system_prompt: str = None) -> str:
    import openai
    client = openai.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_id.split("/", 1)[-1] if "/" in model_id else model_id,
                messages=messages, temperature=0, max_tokens=512)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"
    return "ERROR"


SYS = "You are taking a cognitive evaluation. Follow the instructions precisely. Give only the requested answer."


def score_simple(expected, response):
    """Score with flexible matching."""
    if expected is None:
        return None
    exp = str(expected).strip().lower()
    resp = response.strip().lower()
    if not exp:
        return None
    return exp == resp or exp in resp.split()


def run_paradigm_sweep(model_id, paradigm_name, gen_fn, levels, n_items_per_level=10, seed=42):
    """Run one paradigm across multiple difficulty levels.

    Args:
        gen_fn: callable(level) -> list of TaskInstance
        levels: list of (level_label, level_value) tuples

    Returns:
        dict with per-level accuracy and details
    """
    results = {"paradigm": paradigm_name, "levels": []}

    for label, items in levels:
        correct = 0
        total = 0
        details = []
        for item in items:
            resp = call_llm(model_id, item.stimulus, SYS)
            # Use paradigm-specific scorer
            score = _score_static_item(item, resp)
            # Determine correctness from score dict
            if "accuracy" in score:
                is_correct = score["accuracy"] >= 0.5
            elif "correct" in score:
                val = score["correct"]
                is_correct = val is True or (isinstance(val, (int, float)) and val > 0)
            else:
                is_correct = None
            if is_correct is not None:
                total += 1
                if is_correct:
                    correct += 1
                details.append({
                    "expected": str(item.expected_response)[:50],
                    "response": resp[:80],
                    "correct": is_correct,
                    "score": {k: v for k, v in score.items() if k != "response"},
                })
        acc = correct / total if total > 0 else 0
        results["levels"].append({
            "label": str(label),
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "details": details,
        })
        print(f"    {label}: {acc:.0%} ({correct}/{total})")

    return results


# ── Paradigm-specific sweep definitions ──────────────────────────────────

def sweep_nback(model_id, n_items=8, seed=42):
    """n-back: N = 1, 2, 3 (load sweep)."""
    print("  n_back: N = 1, 2, 3")
    gen = NBackGenerator()
    levels = []
    for n in [1, 2, 3]:
        diff = {1: "easy", 2: "medium", 3: "hard"}[n]
        items = gen.generate(seed=seed, n_items=n_items, difficulty=diff)
        levels.append((f"N={n}", items))
    return run_paradigm_sweep(model_id, "n_back", None, levels)


def sweep_digit_span(model_id, n_items=8, seed=42):
    """digit_span: sequence length via difficulty levels."""
    print("  digit_span: easy (short) → hard (long)")
    gen = DigitSpanGenerator()
    levels = []
    for diff in ["easy", "medium", "hard"]:
        items = gen.generate(seed=seed, n_items=n_items, difficulty=diff)
        levels.append((diff, items))
    return run_paradigm_sweep(model_id, "digit_span", None, levels)


def sweep_stroop(model_id, n_items=10, seed=42):
    """stroop: congruent vs incongruent (congruency effect)."""
    print("  stroop: congruent vs incongruent")
    cong = StroopParadigm.generate(seed=seed, n_congruent=n_items, n_incongruent=0, conflict_type="mixed")
    incong = StroopParadigm.generate(seed=seed+1, n_congruent=0, n_incongruent=n_items, conflict_type="mixed")
    return run_paradigm_sweep(model_id, "stroop", None, [("congruent", cong), ("incongruent", incong)])


def sweep_flanker(model_id, n_items=10, seed=42):
    """flanker: congruent vs incongruent."""
    print("  flanker: congruent vs incongruent")
    cong = FlankerParadigm.generate(seed=seed, n_congruent=n_items, n_incongruent=0)
    incong = FlankerParadigm.generate(seed=seed+1, n_congruent=0, n_incongruent=n_items)
    return run_paradigm_sweep(model_id, "flanker", None, [("congruent", cong), ("incongruent", incong)])


def sweep_gonogo(model_id, n_items=8, seed=42):
    """go_nogo: Go ratio 0.6, 0.75, 0.9 (prepotency sweep)."""
    print("  go_nogo: Go ratio 0.6 → 0.75 → 0.9")
    levels = []
    n_trials = max(n_items * 3, 20)  # need enough trials for signal
    for diff, ratio in [("easy", 0.6), ("medium", 0.75), ("hard", 0.9)]:
        items = GoNoGoParadigm.generate(seed=seed, n_trials=n_trials, difficulty=diff)
        levels.append((f"ratio={ratio}", items))
    return run_paradigm_sweep(model_id, "go_nogo", None, levels)


def sweep_false_belief(model_id, n_items=10, seed=42):
    """sally_anne: 1st order vs 2nd order."""
    print("  sally_anne: 1st order vs 2nd order")
    gen = FalseBeliefGenerator()
    order1 = gen.generate(seed=seed, n_items=n_items, difficulty="medium", order=1)
    order2 = gen.generate(seed=seed, n_items=n_items, difficulty="medium", order=2)
    return run_paradigm_sweep(model_id, "sally_anne", None, [("order_1", order1), ("order_2", order2)])


def sweep_drm(model_id, n_items=5, seed=42):
    """drm: easy (3 lists) → medium (5) → hard (8)."""
    print("  drm: easy → medium → hard")
    gen = DRMGenerator()
    levels = []
    for diff in ["easy", "medium", "hard"]:
        items = gen.generate(seed=seed, n_items=n_items, difficulty=diff)
        levels.append((diff, items))
    return run_paradigm_sweep(model_id, "drm", None, levels)


def sweep_source_monitoring(model_id, n_items=5, seed=42):
    """source_monitoring: easy (3 sources) → medium (4) → hard (5)."""
    print("  source_monitoring: easy → medium → hard")
    gen = SourceMonitoringGenerator()
    levels = []
    for diff in ["easy", "medium", "hard"]:
        items = gen.generate(seed=seed, n_items=n_items, difficulty=diff)
        levels.append((diff, items))
    return run_paradigm_sweep(model_id, "source_monitoring", None, levels)


def sweep_confidence(model_id, n_items=10, seed=42):
    """confidence_calibration: easy → medium → hard questions."""
    print("  confidence_calibration: easy → medium → hard")
    gen = ConfidenceCalibrationGenerator()
    levels = []
    for diff in ["easy", "medium", "hard"]:
        items = gen.generate(seed=seed, n_items=n_items, difficulty=diff)
        levels.append((diff, items))
    return run_paradigm_sweep(model_id, "confidence_calibration", None, levels)


# Registry of all sweeps
SWEEPS = {
    "n_back": sweep_nback,
    "digit_span": sweep_digit_span,
    "stroop": sweep_stroop,
    "flanker": sweep_flanker,
    "go_nogo": sweep_gonogo,
    "sally_anne": sweep_false_belief,
    "drm": sweep_drm,
    "source_monitoring": sweep_source_monitoring,
    "confidence_calibration": sweep_confidence,
}

# Expected human patterns (for comparison)
HUMAN_PATTERNS = {
    "n_back": "Monotonic decrease: N=1 ~95%, N=2 ~85%, N=3 ~70%",
    "digit_span": "Accuracy drops sharply at span limit (~7 forward, ~5 backward)",
    "stroop": "Congruent > incongruent (Stroop effect ~5-20% accuracy gap)",
    "flanker": "Congruent > incongruent (flanker effect, human gap ~2-5%)",
    "go_nogo": "Higher Go ratio → more false alarms (10-15% at 0.75, ~25% at 0.9)",
    "sally_anne": "1st order easier than 2nd order (adults: 1st ~95%, 2nd ~85%)",
    "drm": "Longer lists → more critical lure false alarms (effect size ~0.3-0.5)",
    "source_monitoring": "More sources → lower source accuracy",
    "confidence_calibration": "Harder questions → more overconfidence (ECE increases)",
}


def main():
    parser = argparse.ArgumentParser(description="B1.5: Intervention-based construct checks")
    parser.add_argument("--model", type=str, default="openai/qwen2.5:7b")
    parser.add_argument("--output-dir", type=str, default="results/intervention_sweeps")
    parser.add_argument("--paradigms", nargs="+", default=None,
                        help="Specific paradigms to sweep (default: all)")
    parser.add_argument("--n-items", type=int, default=8,
                        help="Items per difficulty level")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paradigms = args.paradigms or list(SWEEPS.keys())

    print(f"B1.5 Intervention Sweep")
    print(f"Model: {args.model}")
    print(f"Paradigms: {len(paradigms)}")
    print(f"Items per level: {args.n_items}")
    print()

    all_results = []
    for name in paradigms:
        sweep_fn = SWEEPS.get(name)
        if not sweep_fn:
            print(f"  {name}: no sweep function, skipping")
            continue
        print(f"  Running {name}...")
        try:
            result = sweep_fn(args.model, n_items=args.n_items, seed=args.seed)
            result["human_pattern"] = HUMAN_PATTERNS.get(name, "Unknown")

            # Check direction
            accs = [l["accuracy"] for l in result["levels"]]
            if len(accs) >= 2:
                if name in ["stroop", "flanker"]:
                    # Congruent should be higher than incongruent
                    result["direction_match"] = accs[0] > accs[1]
                elif name in ["sally_anne"]:
                    # 1st order should be higher than 2nd
                    result["direction_match"] = accs[0] >= accs[1]
                else:
                    # Generally: easier → harder should show decrease
                    result["direction_match"] = accs[0] >= accs[-1]

            all_results.append(result)
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results.append({"paradigm": name, "error": str(e)})
        print()

    # Summary
    tested = [r for r in all_results if "error" not in r]
    direction_pass = [r for r in tested if r.get("direction_match", False)]

    print(f"{'='*65}")
    print(f"INTERVENTION SWEEP SUMMARY")
    print(f"{'='*65}")
    print(f"{'Paradigm':<25} {'Levels':>6} {'Direction':>10} {'Human Pattern'}")
    print(f"{'-'*65}")
    for r in all_results:
        if "error" in r:
            print(f"{r['paradigm']:<25} ERROR")
            continue
        accs = " → ".join(f"{l['accuracy']:.0%}" for l in r["levels"])
        dm = "✓ PASS" if r.get("direction_match") else "✗ FAIL"
        print(f"{r['paradigm']:<25} {accs:>20} {dm:>10}")
    print(f"{'='*65}")
    print(f"Direction match: {len(direction_pass)}/{len(tested)} paradigms")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_safe = args.model.replace("/", "_")
    out_path = out_dir / f"intervention_{model_safe}.json"
    report = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "n_items_per_level": args.n_items,
        "seed": args.seed,
        "paradigms_tested": len(tested),
        "direction_matches": len(direction_pass),
        "results": all_results,
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
