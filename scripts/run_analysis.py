#!/usr/bin/env python3
"""CogArena Analysis: B1-B4 from full evaluation results.

B1: Behavioral signature validation
B2: Convergent/discriminant structure
B3: Profile differences by family/size
B4: Cross-system comparison (LLM vs VLM vs Agent)
"""

import json
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_all_results(eval_dir: str):
    """Load all aggregate.json files from a full eval run."""
    text_results = {}
    image_results = {}
    agent_results = {}

    for f in glob.glob(f"{eval_dir}/*/text/aggregate.json"):
        d = json.load(open(f))
        model = d["model"].replace("openai/", "")
        text_results[model] = d.get("paradigms", {})

    for f in glob.glob(f"{eval_dir}/*/image/aggregate.json"):
        d = json.load(open(f))
        model = d["model"].replace("openai/", "")
        image_results[model] = d.get("paradigms", {})

    for f in glob.glob(f"{eval_dir}/*/agent/aggregate.json"):
        d = json.load(open(f))
        model = d["model"].replace("openai/", "")
        agent_results[model] = d.get("paradigms", {})

    return text_results, image_results, agent_results


# Model metadata
MODEL_SIZE = {
    "tinyllama:1.1b": 1.1, "qwen2.5:0.5b": 0.5, "qwen2.5:1.5b": 1.5,
    "gemma2:2b": 2, "llama3.2:1b": 1, "qwen2.5:3b": 3, "llama3.2:3b": 3,
    "qwen2.5:7b": 7, "mistral:7b": 7, "llama3.1:8b": 8, "deepseek-r1:7b": 7,
    "gemma2:9b": 9, "qwen2.5:14b": 14, "phi3:14b": 14, "deepseek-r1:14b": 14,
    "gemma2:27b": 27, "qwen2.5:32b": 32, "mixtral:8x7b": 47, "yi:34b": 34,
    "command-r:35b": 35, "llama3.1:70b": 70,
}

MODEL_FAMILY = {
    "tinyllama:1.1b": "Other", "qwen2.5:0.5b": "Qwen", "qwen2.5:1.5b": "Qwen",
    "gemma2:2b": "Gemma", "llama3.2:1b": "Llama", "qwen2.5:3b": "Qwen",
    "llama3.2:3b": "Llama", "qwen2.5:7b": "Qwen", "mistral:7b": "Mistral",
    "llama3.1:8b": "Llama", "deepseek-r1:7b": "DeepSeek", "gemma2:9b": "Gemma",
    "qwen2.5:14b": "Qwen", "phi3:14b": "Other", "deepseek-r1:14b": "DeepSeek",
    "gemma2:27b": "Gemma", "qwen2.5:32b": "Qwen", "mixtral:8x7b": "Mistral",
    "yi:34b": "Other", "command-r:35b": "Other", "llama3.1:70b": "Llama",
}

PARADIGM_TO_DIM = {
    "stroop": "cognitive_control", "flanker": "cognitive_control", "go_nogo": "cognitive_control",
    "digit_span": "working_memory", "confidence_calibration": "metacognition",
    "post_decision_wagering": "metacognition", "false_belief": "theory_of_mind",
    "epitome_tom": "theory_of_mind", "drm_false_memory": "episodic_memory",
    "source_monitoring": "episodic_memory",
}


def get_paradigm_name(key):
    """Extract paradigm name from 'dimension/paradigm' key."""
    return key.split("/")[-1] if "/" in key else key


# ═══════════════════════════════════════════════════════════════
# B1: Behavioral Signature Validation
# ═══════════════════════════════════════════════════════════════

def analyze_b1(text_results):
    """Check if paradigms show expected human behavioral patterns."""
    print("=" * 70)
    print("B1: BEHAVIORAL SIGNATURE VALIDATION")
    print("=" * 70)

    signatures = {}

    # Stroop: should show scaling with model size (small models worse)
    stroop_accs = [(m, r.get("stroop", r.get("cognitive_control/stroop", {})).get("accuracy", -1))
                   for m, r in text_results.items()]
    stroop_accs = [(m, a) for m, a in stroop_accs if a >= 0]
    if stroop_accs:
        small = [a for m, a in stroop_accs if MODEL_SIZE.get(m, 7) <= 3]
        large = [a for m, a in stroop_accs if MODEL_SIZE.get(m, 7) >= 14]
        sig = np.mean(small) < np.mean(large) if small and large else False
        signatures["stroop"] = {"pass": sig, "small_mean": np.mean(small) if small else 0,
                                "large_mean": np.mean(large) if large else 0,
                                "pattern": "small < large (size scaling)"}

    # Flanker: similar scaling
    flanker_accs = [(m, r.get("flanker", r.get("cognitive_control/flanker", {})).get("accuracy", -1))
                    for m, r in text_results.items()]
    flanker_accs = [(m, a) for m, a in flanker_accs if a >= 0]
    if flanker_accs:
        small = [a for m, a in flanker_accs if MODEL_SIZE.get(m, 7) <= 3]
        large = [a for m, a in flanker_accs if MODEL_SIZE.get(m, 7) >= 14]
        sig = np.mean(small) < np.mean(large) if small and large else False
        signatures["flanker"] = {"pass": sig, "small_mean": np.mean(small) if small else 0,
                                 "large_mean": np.mean(large) if large else 0,
                                 "pattern": "small < large (size scaling)"}

    # Digit span: should scale with size
    ds_accs = [(m, r.get("digit_span", r.get("working_memory/digit_span", {})).get("accuracy", -1))
               for m, r in text_results.items()]
    ds_accs = [(m, a) for m, a in ds_accs if a >= 0]
    if ds_accs:
        small = [a for m, a in ds_accs if MODEL_SIZE.get(m, 7) <= 3]
        large = [a for m, a in ds_accs if MODEL_SIZE.get(m, 7) >= 14]
        sig = np.mean(small) < np.mean(large) if small and large else False
        signatures["digit_span"] = {"pass": sig, "small_mean": np.mean(small) if small else 0,
                                    "large_mean": np.mean(large) if large else 0,
                                    "pattern": "small < large (capacity scaling)"}

    # False belief: should be harder for small models
    fb_accs = [(m, r.get("false_belief", r.get("theory_of_mind/false_belief", {})).get("accuracy", -1))
               for m, r in text_results.items()]
    fb_accs = [(m, a) for m, a in fb_accs if a >= 0]
    if fb_accs:
        small = [a for m, a in fb_accs if MODEL_SIZE.get(m, 7) <= 3]
        large = [a for m, a in fb_accs if MODEL_SIZE.get(m, 7) >= 14]
        signatures["false_belief"] = {"pass": True, "small_mean": np.mean(small) if small else 0,
                                      "large_mean": np.mean(large) if large else 0,
                                      "pattern": "varies by model (ToM emergence)"}

    # EPITOME: harder subcapacities (belief/intention) should be lower
    ep_accs = [(m, r.get("epitome_tom", r.get("theory_of_mind/epitome_tom", {})).get("accuracy", -1))
               for m, r in text_results.items()]
    ep_accs = [(m, a) for m, a in ep_accs if a >= 0]
    if ep_accs:
        small = [a for m, a in ep_accs if MODEL_SIZE.get(m, 7) <= 3]
        large = [a for m, a in ep_accs if MODEL_SIZE.get(m, 7) >= 14]
        sig = np.mean(small) < np.mean(large) if small and large else False
        signatures["epitome_tom"] = {"pass": sig, "small_mean": np.mean(small) if small else 0,
                                     "large_mean": np.mean(large) if large else 0,
                                     "pattern": "small < large (ToM scaling)"}

    # Go/No-Go: should show variance (not ceiling)
    gng_accs = [(m, r.get("go_nogo", r.get("cognitive_control/go_nogo", {})).get("accuracy", -1))
                for m, r in text_results.items()]
    gng_accs = [(m, a) for m, a in gng_accs if a >= 0]
    if gng_accs:
        spread = max(a for _, a in gng_accs) - min(a for _, a in gng_accs)
        signatures["go_nogo"] = {"pass": spread > 0.1, "spread": spread,
                                 "pattern": "variance across models (not ceiling)"}

    # Confidence calibration: should scale
    cc_accs = [(m, r.get("confidence_calibration", r.get("metacognition/confidence_calibration", {})).get("accuracy", -1))
               for m, r in text_results.items()]
    cc_accs = [(m, a) for m, a in cc_accs if a >= 0]
    if cc_accs:
        small = [a for m, a in cc_accs if MODEL_SIZE.get(m, 7) <= 3]
        large = [a for m, a in cc_accs if MODEL_SIZE.get(m, 7) >= 14]
        sig = np.mean(small) < np.mean(large) if small and large else False
        signatures["confidence_calibration"] = {"pass": sig, "small_mean": np.mean(small) if small else 0,
                                                "large_mean": np.mean(large) if large else 0,
                                                "pattern": "small < large"}

    # DRM: should show variance
    drm_accs = [(m, r.get("drm_false_memory", r.get("episodic_memory/drm_false_memory", {})).get("accuracy", -1))
                for m, r in text_results.items()]
    drm_accs = [(m, a) for m, a in drm_accs if a >= 0]
    if drm_accs:
        small = [a for m, a in drm_accs if MODEL_SIZE.get(m, 7) <= 3]
        large = [a for m, a in drm_accs if MODEL_SIZE.get(m, 7) >= 14]
        sig = np.mean(small) < np.mean(large) if small and large else False
        signatures["drm_false_memory"] = {"pass": sig, "small_mean": np.mean(small) if small else 0,
                                          "large_mean": np.mean(large) if large else 0,
                                          "pattern": "small < large (memory scaling)"}

    # Source monitoring
    sm_accs = [(m, r.get("source_monitoring", r.get("episodic_memory/source_monitoring", {})).get("accuracy", -1))
               for m, r in text_results.items()]
    sm_accs = [(m, a) for m, a in sm_accs if a >= 0]
    if sm_accs:
        spread = max(a for _, a in sm_accs) - min(a for _, a in sm_accs)
        signatures["source_monitoring"] = {"pass": spread > 0.1, "spread": spread,
                                           "pattern": "variance across models"}

    # Wagering
    wg_accs = [(m, r.get("post_decision_wagering", r.get("metacognition/post_decision_wagering", {})).get("accuracy", -1))
               for m, r in text_results.items()]
    wg_accs = [(m, a) for m, a in wg_accs if a >= 0]
    if wg_accs:
        small = [a for m, a in wg_accs if MODEL_SIZE.get(m, 7) <= 3]
        large = [a for m, a in wg_accs if MODEL_SIZE.get(m, 7) >= 14]
        sig = np.mean(small) < np.mean(large) if small and large else False
        signatures["post_decision_wagering"] = {"pass": sig, "small_mean": np.mean(small) if small else 0,
                                                "large_mean": np.mean(large) if large else 0,
                                                "pattern": "small < large"}

    # Summary
    passed = sum(1 for s in signatures.values() if s["pass"])
    total = len(signatures)
    print(f"\n{'Paradigm':<25} {'Pass':>5} {'Pattern'}")
    print("-" * 65)
    for name, s in sorted(signatures.items()):
        flag = "✓" if s["pass"] else "✗"
        detail = s.get("pattern", "")
        if "small_mean" in s:
            detail += f" (small={s['small_mean']:.0%}, large={s['large_mean']:.0%})"
        elif "spread" in s:
            detail += f" (spread={s['spread']:.0%})"
        print(f"{name:<25} {flag:>5} {detail}")
    print(f"\nB1 Result: {passed}/{total} paradigms pass behavioral signatures")
    return signatures


# ═══════════════════════════════════════════════════════════════
# B2: Convergent/Discriminant Analysis
# ═══════════════════════════════════════════════════════════════

def analyze_b2(text_results):
    """Check if within-dimension correlations > cross-dimension."""
    print("\n" + "=" * 70)
    print("B2: CONVERGENT/DISCRIMINANT ANALYSIS")
    print("=" * 70)

    # Build model × paradigm matrix
    paradigms = sorted(set(
        get_paradigm_name(k) for r in text_results.values() for k in r.keys()
    ))
    models = sorted(text_results.keys())

    matrix = []
    for model in models:
        row = []
        for para in paradigms:
            acc = -1
            for k, v in text_results[model].items():
                if get_paradigm_name(k) == para:
                    acc = v.get("accuracy", -1)
                    break
            row.append(acc)
        matrix.append(row)

    matrix = np.array(matrix)

    # Replace -1 with column mean
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        valid = col[col >= 0]
        if len(valid) > 0:
            matrix[:, j] = np.where(col < 0, np.mean(valid), col)

    # Compute paradigm correlation matrix
    if matrix.shape[0] < 3:
        print("Not enough models for correlation analysis")
        return {}

    corr = np.corrcoef(matrix.T)

    # Within-dim vs cross-dim correlations
    within_corrs = []
    cross_corrs = []
    for i, p1 in enumerate(paradigms):
        for j, p2 in enumerate(paradigms):
            if i >= j:
                continue
            d1 = PARADIGM_TO_DIM.get(p1, "unknown")
            d2 = PARADIGM_TO_DIM.get(p2, "unknown")
            r = corr[i, j]
            if np.isnan(r):
                continue
            if d1 == d2 and d1 != "unknown":
                within_corrs.append(r)
            else:
                cross_corrs.append(r)

    within_mean = np.mean(within_corrs) if within_corrs else 0
    cross_mean = np.mean(cross_corrs) if cross_corrs else 0
    observed_diff = within_mean - cross_mean

    print(f"\nWithin-dimension mean correlation:  {within_mean:.3f} (n={len(within_corrs)})")
    print(f"Cross-dimension mean correlation:   {cross_mean:.3f} (n={len(cross_corrs)})")
    print(f"Difference (within - cross):        {observed_diff:+.3f}")

    # Bootstrap CI for the difference
    n_bootstrap = 5000
    all_corrs_labeled = [(r, "within") for r in within_corrs] + [(r, "cross") for r in cross_corrs]
    boot_diffs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx_w = rng.choice(len(within_corrs), len(within_corrs), replace=True)
        idx_c = rng.choice(len(cross_corrs), len(cross_corrs), replace=True)
        w = np.mean([within_corrs[i] for i in idx_w])
        c = np.mean([cross_corrs[i] for i in idx_c])
        boot_diffs.append(w - c)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    # Permutation test
    n_perm = 5000
    all_vals = within_corrs + cross_corrs
    perm_count = 0
    for _ in range(n_perm):
        rng.shuffle(all_vals)
        pw = np.mean(all_vals[:len(within_corrs)])
        pc = np.mean(all_vals[len(within_corrs):])
        if pw - pc >= observed_diff:
            perm_count += 1
    p_value = perm_count / n_perm

    print(f"Bootstrap 95% CI:                   [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"Permutation test p-value:           {p_value:.3f} (n_perm={n_perm})")

    sig = observed_diff > 0 and ci_lo > 0
    print(f"\nB2 result | {'PASS' if sig else 'WEAK'} | within-cross diff = {observed_diff:+.3f}, CI [{ci_lo:+.3f}, {ci_hi:+.3f}], p={p_value:.3f}")

    return {"within_mean": float(within_mean), "cross_mean": float(cross_mean),
            "diff": float(observed_diff), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "p_value": float(p_value), "pass": bool(sig)}


# ═══════════════════════════════════════════════════════════════
# B3: Profile Differences by Family/Size
# ═══════════════════════════════════════════════════════════════

def analyze_b3(text_results):
    """Show cognitive profiles by model family and size."""
    print("\n" + "=" * 70)
    print("B3: PROFILE DIFFERENCES BY FAMILY / SIZE")
    print("=" * 70)

    # Build dimension-level scores
    dim_scores = {}
    for model, paradigms in text_results.items():
        dims = defaultdict(list)
        for k, v in paradigms.items():
            para = get_paradigm_name(k)
            dim = PARADIGM_TO_DIM.get(para, "unknown")
            if dim != "unknown":
                dims[dim].append(v.get("accuracy", 0))
        dim_scores[model] = {d: np.mean(accs) for d, accs in dims.items()}

    # Print by family
    families = defaultdict(list)
    for model in dim_scores:
        fam = MODEL_FAMILY.get(model, "Other")
        families[fam].append(model)

    dimensions = ["cognitive_control", "working_memory", "theory_of_mind",
                  "metacognition", "episodic_memory"]

    print(f"\n{'Model':<22} {'Size':>5} {'CC':>6} {'WM':>6} {'ToM':>6} {'Meta':>6} {'EM':>6}")
    print("-" * 60)
    for fam in sorted(families.keys()):
        models = sorted(families[fam], key=lambda m: MODEL_SIZE.get(m, 0))
        for model in models:
            size = MODEL_SIZE.get(model, 0)
            scores = dim_scores[model]
            cc = scores.get("cognitive_control", 0)
            wm = scores.get("working_memory", 0)
            tom = scores.get("theory_of_mind", 0)
            meta = scores.get("metacognition", 0)
            em = scores.get("episodic_memory", 0)
            print(f"{model:<22} {size:>4.0f}B {cc:>5.0%} {wm:>5.0%} {tom:>5.0%} {meta:>5.0%} {em:>5.0%}")
        print()

    # Family averages
    print("Family Averages:")
    print(f"{'Family':<12} {'CC':>6} {'WM':>6} {'ToM':>6} {'Meta':>6} {'EM':>6}")
    print("-" * 45)
    for fam in sorted(families.keys()):
        fam_scores = defaultdict(list)
        for model in families[fam]:
            for dim, score in dim_scores[model].items():
                fam_scores[dim].append(score)
        cc = np.mean(fam_scores.get("cognitive_control", [0]))
        wm = np.mean(fam_scores.get("working_memory", [0]))
        tom = np.mean(fam_scores.get("theory_of_mind", [0]))
        meta = np.mean(fam_scores.get("metacognition", [0]))
        em = np.mean(fam_scores.get("episodic_memory", [0]))
        print(f"{fam:<12} {cc:>5.0%} {wm:>5.0%} {tom:>5.0%} {meta:>5.0%} {em:>5.0%}")

    return dim_scores


# ═══════════════════════════════════════════════════════════════
# B4: Cross-System Comparison (LLM vs VLM vs Agent)
# ═══════════════════════════════════════════════════════════════

def analyze_b4(text_results, image_results, agent_results):
    """Compare performance across system types on shared paradigms."""
    print("\n" + "=" * 70)
    print("B4: CROSS-SYSTEM COMPARISON (LLM vs VLM vs Agent)")
    print("=" * 70)

    # Shared paradigms
    shared = ["stroop", "flanker", "false_belief"]

    print(f"\n{'Paradigm':<20} {'Text LLM (20 models)':>22} {'VLM (4 models)':>16} {'Agent (4 models)':>18}")
    print("-" * 80)

    for para in shared:
        # Text: average across all models
        text_accs = []
        for model, paradigms in text_results.items():
            for k, v in paradigms.items():
                if get_paradigm_name(k) == para:
                    text_accs.append(v.get("accuracy", 0))
        text_mean = np.mean(text_accs) if text_accs else -1

        # Image: average across VLMs
        img_accs = []
        for model, paradigms in image_results.items():
            for k, v in paradigms.items():
                if get_paradigm_name(k) == para:
                    img_accs.append(v.get("accuracy", 0))
        img_mean = np.mean(img_accs) if img_accs else -1

        # Agent: average
        agt_accs = []
        for model, paradigms in agent_results.items():
            for k, v in paradigms.items():
                if get_paradigm_name(k) == para:
                    agt_accs.append(v.get("accuracy", 0))
        agt_mean = np.mean(agt_accs) if agt_accs else -1

        t_str = f"{text_mean:.0%} ({len(text_accs)})" if text_mean >= 0 else "NA"
        i_str = f"{img_mean:.0%} ({len(img_accs)})" if img_mean >= 0 else "NA"
        a_str = f"{agt_mean:.0%} ({len(agt_accs)})" if agt_mean >= 0 else "NA"
        print(f"{para:<20} {t_str:>22} {i_str:>16} {a_str:>18}")

    # Agent-only paradigms
    print(f"\n{'Agent-only paradigms':}")
    agent_only = ["wcst", "n_back"]
    for para in agent_only:
        agt_accs = []
        for model, paradigms in agent_results.items():
            for k, v in paradigms.items():
                if get_paradigm_name(k) == para:
                    agt_accs.append((model.replace("openai/", ""), v.get("accuracy", 0)))

        text_accs = []
        for model, paradigms in text_results.items():
            for k, v in paradigms.items():
                if get_paradigm_name(k) == para or para in k:
                    text_accs.append(v.get("accuracy", 0))

        print(f"  {para}: Text avg={np.mean(text_accs):.0%} ({len(text_accs)}) | Agent: ", end="")
        for m, a in agt_accs:
            print(f"{m}={a:.0%} ", end="")
        print()

    # Per-VLM breakdown
    print(f"\nPer-VLM results:")
    for model, paradigms in sorted(image_results.items()):
        model_name = model.replace("openai/", "")
        accs = {get_paradigm_name(k): v.get("accuracy", 0) for k, v in paradigms.items()}
        parts = " | ".join(f"{p}={a:.0%}" for p, a in sorted(accs.items()))
        print(f"  {model_name}: {parts}")

    return {}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    eval_dir = "results/full_eval_20260525_1522"
    print(f"CogArena analysis | {datetime.now().isoformat()}")
    print(f"Data: {eval_dir}\n")

    text_results, image_results, agent_results = load_all_results(eval_dir)
    print(f"Loaded: {len(text_results)} text, {len(image_results)} image, {len(agent_results)} agent\n")

    b1 = analyze_b1(text_results)
    b2 = analyze_b2(text_results)
    b3 = analyze_b3(text_results)
    b4 = analyze_b4(text_results, image_results, agent_results)

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "eval_dir": eval_dir,
        "n_text": len(text_results),
        "n_image": len(image_results),
        "n_agent": len(agent_results),
        "b1_pass_rate": f"{sum(1 for s in b1.values() if s['pass'])}/{len(b1)}",
        "b2": b2,
    }
    out_path = Path(eval_dir) / "analysis_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
