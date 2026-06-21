#!/usr/bin/env python3
"""Compare text-only LLM vs VLM on Stroop and Flanker tasks.

Tests whether image-based Stroop breaks the text-version ceiling effect.

Usage:
    OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 \
    python scripts/run_text_vs_image.py
"""

import base64
import json
import os
import time
from pathlib import Path
from datetime import datetime

from cogarena.dimensions.cognitive_control import StroopParadigm, FlankerParadigm
from cogarena.image_gen.stroop_images import generate_stroop_set
from cogarena.image_gen.flanker_images import generate_flanker_set


def call_llm_text(model_id: str, prompt: str, system_prompt: str = None) -> str:
    import openai
    client = openai.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_id, messages=messages, temperature=0, max_tokens=50)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2: time.sleep(2 ** attempt)
            else: return f"ERROR: {e}"
    return "ERROR"


def call_vlm_image(model_id: str, prompt: str, image_path: str) -> str:
    import openai
    client = openai.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_id, messages=messages, temperature=0, max_tokens=50)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2: time.sleep(2 ** attempt)
            else: return f"ERROR: {e}"
    return "ERROR"


def score_simple(expected: str, response: str) -> bool:
    exp = expected.strip().lower()
    resp = response.strip().lower()
    return exp == resp or exp in resp.split()


SYS = "You are taking a cognitive evaluation. Give only the requested answer."


def run_stroop_comparison(n: int = 10):
    print(f"\n{'='*60}")
    print(f"STROOP: text (qwen2.5:7b) vs image (qwen2.5vl:7b)")
    print(f"{'='*60}")

    # Generate image stimuli
    image_trials = generate_stroop_set(seed=42, n_congruent=n, n_incongruent=n,
                                        out_dir="data/images/stroop")
    print(f"Generated {len(image_trials)} Stroop images")

    # Generate matched text stimuli (classic color-word Stroop)
    text_items = StroopParadigm.generate(seed=42, n_congruent=n, n_incongruent=n,
                                          conflict_type="color_word")

    # --- Text track ---
    print(f"\n[TEXT] Running qwen2.5:7b on {len(text_items)} text items...")
    text_results = {"congruent": [], "incongruent": []}
    for item in text_items:
        sc = item.metadata.parameters.get("_scoring_config", {})
        cond = "congruent" if sc.get("congruent", True) else "incongruent"
        resp = call_llm_text("qwen2.5:7b", item.stimulus, SYS)
        correct = score_simple(item.expected_response, resp)
        text_results[cond].append({"expected": item.expected_response, "response": resp, "correct": correct})
        print(f"  {cond[:5]:5s} exp={item.expected_response:8s} resp={resp:10s} {'✓' if correct else '✗'}")

    # --- Image track ---
    print(f"\n[IMAGE] Running qwen2.5vl:7b on {len(image_trials)} image items...")
    image_results = {"congruent": [], "incongruent": []}
    for trial in image_trials:
        cond = "congruent" if trial["congruent"] else "incongruent"
        resp = call_vlm_image("qwen2.5vl:7b", trial["stimulus_text"], trial["image_path"])
        correct = score_simple(trial["expected_response"], resp)
        image_results[cond].append({
            "expected": trial["expected_response"], "response": resp,
            "correct": correct, "word": trial["word"],
        })
        print(f"  {cond[:5]:5s} word={trial['word']:8s} ink={trial['ink_color']:8s} resp={resp:10s} {'✓' if correct else '✗'}")

    # --- Summary ---
    print(f"\n{'─'*60}")
    print(f"STROOP RESULTS:")
    for track, results in [("TEXT", text_results), ("IMAGE", image_results)]:
        for cond in ["congruent", "incongruent"]:
            items = results[cond]
            if items:
                acc = sum(1 for r in items if r["correct"]) / len(items)
                print(f"  [{track:5s}] {cond:12s}: {acc:5.0%} ({sum(1 for r in items if r['correct'])}/{len(items)})")
        all_items = results["congruent"] + results["incongruent"]
        total_acc = sum(1 for r in all_items if r["correct"]) / len(all_items) if all_items else 0
        cong_acc = sum(1 for r in results["congruent"] if r["correct"]) / len(results["congruent"]) if results["congruent"] else 0
        incong_acc = sum(1 for r in results["incongruent"] if r["correct"]) / len(results["incongruent"]) if results["incongruent"] else 0
        effect = cong_acc - incong_acc
        print(f"  [{track:5s}] TOTAL: {total_acc:.0%}  congruency_effect: {effect:+.0%}")
    print(f"{'─'*60}")

    return text_results, image_results


def run_flanker_comparison(n: int = 10):
    print(f"\n{'='*60}")
    print(f"FLANKER: text (qwen2.5:7b) vs image (qwen2.5vl:7b)")
    print(f"{'='*60}")

    # Generate image stimuli
    image_trials = generate_flanker_set(seed=42, n_congruent=n, n_incongruent=n,
                                         out_dir="data/images/flanker")
    print(f"Generated {len(image_trials)} Flanker images")

    # Generate text stimuli
    text_items = FlankerParadigm.generate(seed=42, n_congruent=n, n_incongruent=n,
                                           contamination_probe=True)  # arrows

    # --- Text track ---
    print(f"\n[TEXT] Running qwen2.5:7b on {len(text_items)} text items...")
    text_results = {"congruent": [], "incongruent": []}
    for item in text_items:
        sc = item.metadata.parameters.get("_scoring_config", {})
        cond = "congruent" if sc.get("condition") == "congruent" else "incongruent"
        resp = call_llm_text("qwen2.5:7b", item.stimulus, SYS)
        correct = score_simple(item.expected_response, resp)
        text_results[cond].append({"expected": item.expected_response, "response": resp, "correct": correct})
        print(f"  {cond[:5]:5s} exp={item.expected_response:6s} resp={resp:10s} {'✓' if correct else '✗'}")

    # --- Image track ---
    print(f"\n[IMAGE] Running qwen2.5vl:7b on {len(image_trials)} image items...")
    image_results = {"congruent": [], "incongruent": []}
    for trial in image_trials:
        cond = "congruent" if trial["congruent"] else "incongruent"
        resp = call_vlm_image("qwen2.5vl:7b", trial["stimulus_text"], trial["image_path"])
        correct = score_simple(trial["expected_response"], resp)
        image_results[cond].append({
            "expected": trial["expected_response"], "response": resp, "correct": correct,
        })
        print(f"  {cond[:5]:5s} target={trial['target_dir']:5s} flanker={trial['flanker_dir']:5s} resp={resp:10s} {'✓' if correct else '✗'}")

    # --- Summary ---
    print(f"\n{'─'*60}")
    print(f"FLANKER RESULTS:")
    for track, results in [("TEXT", text_results), ("IMAGE", image_results)]:
        for cond in ["congruent", "incongruent"]:
            items = results[cond]
            if items:
                acc = sum(1 for r in items if r["correct"]) / len(items)
                print(f"  [{track:5s}] {cond:12s}: {acc:5.0%} ({sum(1 for r in items if r['correct'])}/{len(items)})")
        all_items = results["congruent"] + results["incongruent"]
        total_acc = sum(1 for r in all_items if r["correct"]) / len(all_items) if all_items else 0
        cong_acc = sum(1 for r in results["congruent"] if r["correct"]) / len(results["congruent"]) if results["congruent"] else 0
        incong_acc = sum(1 for r in results["incongruent"] if r["correct"]) / len(results["incongruent"]) if results["incongruent"] else 0
        effect = cong_acc - incong_acc
        print(f"  [{track:5s}] TOTAL: {total_acc:.0%}  flanker_effect: {effect:+.0%}")
    print(f"{'─'*60}")

    return text_results, image_results


def main():
    print("Text vs Image Cognitive Task Comparison")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Text model: qwen2.5:7b | VLM: qwen2.5vl:7b")
    print()

    stroop_text, stroop_image = run_stroop_comparison(n=10)
    flanker_text, flanker_image = run_flanker_comparison(n=10)

    # Save results
    out_dir = Path("results/text_vs_image")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "text_model": "qwen2.5:7b",
        "vlm_model": "qwen2.5vl:7b",
        "stroop": {"text": stroop_text, "image": stroop_image},
        "flanker": {"text": flanker_text, "image": flanker_image},
    }
    (out_dir / "text_vs_image_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_dir / 'text_vs_image_results.json'}")


if __name__ == "__main__":
    main()
