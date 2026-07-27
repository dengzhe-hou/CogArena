#!/usr/bin/env python3
"""CogArena command-line interface for evaluating LLMs on the cognitive-paradigm battery.

The single-turn battery (10 paradigms across 5 theory-motivated groupings) runs out
of the box against any provider. Bring your own model in one of three ways:

  # 1. Any OpenAI-compatible endpoint (Ollama, vLLM, TGI, LM Studio, OpenRouter, Together, ...)
  cogarena eval --provider local --base-url http://localhost:11434/v1 --model qwen2.5:7b

  # 2. Hosted APIs (set the matching API key in your environment)
  cogarena eval --provider openai    --model gpt-4o-mini                 # OPENAI_API_KEY
  cogarena eval --provider anthropic --model claude-3-5-sonnet-20241022  # ANTHROPIC_API_KEY
  cogarena eval --provider google    --model gemini-1.5-pro              # GOOGLE_API_KEY

  # 3. A custom model: subclass cogarena.llm_client.LLMClient and override _dispatch()

  cogarena eval --dry-run --model test   # validate your install with no API key
  cogarena list                          # show the paradigms

Results are written to <output>/<model>/{aggregate.json,details.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from cogarena.scoring import item_accuracy, score_static

# Generators for the 5 paper groupings (the single-turn battery).
from cogarena.generators.working_memory_gen import generate_wm_items
from cogarena.generators.cognitive_control_gen import generate_cc_items
from cogarena.generators.episodic_memory_gen import generate_em_items
from cogarena.generators.theory_of_mind_gen import generate_tom_items
from cogarena.generators.metacognition_gen import generate_mc_items

GENERATORS = {
    "working_memory": generate_wm_items,
    "cognitive_control": generate_cc_items,
    "episodic_memory": generate_em_items,
    "theory_of_mind": generate_tom_items,
    "metacognition": generate_mc_items,
}

# Paradigm -> 5 theory-motivated groupings (paper Table 1). NB: the paper's central
# finding is that these groupings are NOT empirically separable; they are an
# organizing taxonomy, not validated cognitive dimensions.
PARADIGM_GROUPING = {
    "digit_span": "Working Memory",
    "stroop": "Cognitive Control",
    "flanker": "Cognitive Control",
    "go_nogo": "Cognitive Control",
    "drm_false_memory": "Episodic Memory",
    "source_monitoring": "Episodic Memory",
    "false_belief": "Theory of Mind",
    "epitome_tom": "Theory of Mind",
    "confidence_calibration": "Metacognition",
    "post_decision_wagering": "Metacognition",
}

# Same instruction used to produce the paper's results.
SYSTEM_PROMPT = (
    "You are taking a cognitive evaluation. Follow the instructions precisely. "
    "Give only the requested answer, nothing else."
)


class _StubClient:
    """No-op client for --dry-run: validates the pipeline without any model call."""

    def generate(self, prompt, system_prompt=None, temperature=None,
                 max_tokens=None, images=None):
        return "(dry-run: no model called)"


def _build_client(args):
    if args.dry_run:
        return _StubClient()
    from cogarena.llm_client import LLMClient
    cfg = {"provider": args.provider, "model": args.model}
    if args.base_url:
        cfg["base_url"] = args.base_url
    if args.api_key:
        cfg["api_key"] = args.api_key
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.max_tokens is not None:
        cfg["max_tokens"] = args.max_tokens
    return LLMClient(config=cfg)


def _is_multiturn(item):
    turns = item.metadata.parameters.get("turns", [])
    return isinstance(turns, list) and len(turns) > 0


def _collect_items(n_per_paradigm, seed, only_paradigms):
    """Generate the single-turn battery items (one prompt -> one response)."""
    items = []
    for gen in GENERATORS.values():
        for it in gen(seed=seed, n_per_paradigm=n_per_paradigm,
                      include_contamination_probes=False):
            if _is_multiturn(it):
                continue  # n-back, operation span, CVLT need the multi-turn/gym path
            if only_paradigms and it.metadata.paradigm not in only_paradigms:
                continue
            items.append(it)
    return items


def cmd_eval(args):
    only = set(args.paradigms) if args.paradigms else None
    items = _collect_items(args.n, args.seed, only)
    if not items:
        print("No items generated (check --paradigms).", file=sys.stderr)
        return 1
    n_par = len({it.metadata.paradigm for it in items})
    label = "DRY RUN" if args.dry_run else args.model
    print(f"CogArena: {len(items)} items across {n_par} single-turn paradigms "
          f"-> {label} (provider={args.provider}, seed={args.seed})")

    client = _build_client(args)
    t0 = time.time()
    by_par_scores = defaultdict(list)
    details = []
    for i, it in enumerate(items):
        try:
            resp = client.generate(prompt=it.stimulus, system_prompt=SYSTEM_PROMPT)
        except Exception as exc:  # a network/auth error shouldn't abort the whole run
            resp = f"ERROR: {exc}"
        sc = score_static(it, resp)
        by_par_scores[it.metadata.paradigm].append(item_accuracy(sc))
        details.append({
            "task_id": it.task_id,
            "paradigm": it.metadata.paradigm,
            "grouping": PARADIGM_GROUPING.get(it.metadata.paradigm, "Other"),
            "response": resp,
            "score": sc,
        })
        if not args.quiet and (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(items)} items")
    elapsed = time.time() - t0

    par_acc = {p: sum(v) / len(v) for p, v in by_par_scores.items()}
    grp_scores = defaultdict(list)
    for p, a in par_acc.items():
        grp_scores[PARADIGM_GROUPING.get(p, "Other")].append(a)
    grp_acc = {g: sum(v) / len(v) for g, v in grp_scores.items()}
    overall = sum(par_acc.values()) / len(par_acc) if par_acc else 0.0

    _print_table(par_acc, grp_acc, overall, elapsed)

    out_dir = Path(args.output) / args.model.replace("/", "_").replace(":", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "model": args.model,
        "provider": args.provider,
        "seed": args.seed,
        "n_items": len(items),
        "n_per_paradigm": args.n,
        "elapsed_sec": round(elapsed, 1),
        "overall_accuracy": round(overall, 4),
        "paradigm_accuracy": {p: round(a, 4) for p, a in sorted(par_acc.items())},
        "grouping_accuracy": {g: round(a, 4) for g, a in sorted(grp_acc.items())},
    }
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    (out_dir / "details.json").write_text(json.dumps(details, indent=2))
    print(f"\nSaved: {out_dir/'aggregate.json'}  and  details.json")
    return 0


def _print_table(par_acc, grp_acc, overall, elapsed):
    print("\n  Grouping              Paradigm                  Accuracy")
    print("  " + "-" * 56)
    for grp in ["Working Memory", "Cognitive Control", "Episodic Memory",
                "Theory of Mind", "Metacognition"]:
        pars = [p for p, g in PARADIGM_GROUPING.items() if g == grp and p in par_acc]
        for i, p in enumerate(sorted(pars)):
            gcol = grp if i == 0 else ""
            print(f"  {gcol:21s} {p:24s}  {par_acc[p]*100:6.1f}%")
        if grp in grp_acc:
            print(f"  {'':21s} {'  -> grouping mean':24s}  {grp_acc[grp]*100:6.1f}%")
    print("  " + "-" * 56)
    print(f"  {'OVERALL':21s} {'':24s}  {overall*100:6.1f}%   ({elapsed:.0f}s)")


def cmd_list(args):
    print("CogArena single-turn battery (10 paradigms, 5 groupings):\n")
    for grp in ["Working Memory", "Cognitive Control", "Episodic Memory",
                "Theory of Mind", "Metacognition"]:
        pars = [p for p, g in PARADIGM_GROUPING.items() if g == grp]
        print(f"  {grp}: {', '.join(pars)}")
    print("\nMulti-turn paradigms (n-back, operation span, CVLT) and agent/VLM modes use the "
          "Gymnasium API: import cogarena.gym_env; gym.make('CogArena/NBack-v0').")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="cogarena", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    e = sub.add_parser("eval", help="evaluate a model on the battery")
    e.add_argument("--model", required=True, help="model name/tag (e.g. gpt-4o-mini, qwen2.5:7b)")
    e.add_argument("--provider", default="openai",
                   choices=["openai", "local", "anthropic", "google"],
                   help="'local' = any OpenAI-compatible endpoint (Ollama/vLLM/TGI/...)")
    e.add_argument("--base-url", default=None, help="endpoint URL for --provider local")
    e.add_argument("--api-key", default=None, help="API key (else read from env)")
    e.add_argument("--n", type=int, default=50, help="items per paradigm (default 50)")
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--paradigms", nargs="+", default=None,
                   help="restrict to these paradigm names (default: all)")
    e.add_argument("--temperature", type=float, default=None)
    e.add_argument("--max-tokens", type=int, default=None)
    e.add_argument("--output", default="cogarena_results", help="output directory")
    e.add_argument("--dry-run", action="store_true",
                   help="run the pipeline with a stub model (no API key needed)")
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_eval)

    l = sub.add_parser("list", help="list paradigms")
    l.set_defaults(func=cmd_list)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
