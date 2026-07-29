"""CogArena command-line interface for evaluating models on the full battery.

The text mode runs all 13 paradigms, including n-back, operation span, and
CVLT. VLM mode runs the three image adaptations. Agent mode wraps the selected
paradigms in a ReAct-style tool-using loop.

  # 1. Any OpenAI-compatible endpoint (Ollama, vLLM, TGI, LM Studio, OpenRouter, Together, ...)
  cogarena eval --provider local --base-url http://localhost:11434/v1 --model qwen2.5:7b

  # 2. Hosted APIs (set the matching API key in your environment)
  cogarena eval --provider openai    --model gpt-4o-mini                 # OPENAI_API_KEY
  cogarena eval --provider anthropic --model claude-3-5-sonnet-20241022  # ANTHROPIC_API_KEY
  cogarena eval --provider google    --model gemini-1.5-pro              # GOOGLE_API_KEY

  # 3. Load a Hugging Face text model directly
  cogarena eval --provider huggingface --model Qwen/Qwen2.5-7B-Instruct

  # 4. VLM and agent modes
  cogarena eval --mode vlm --provider local --base-url http://localhost:8000/v1 --model my-vlm
  cogarena eval --mode agent --provider local --base-url http://localhost:8000/v1 --model my-llm

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
from datetime import datetime, timezone
from pathlib import Path

# Generators for the 5 paper groupings and complete 13-paradigm text battery.
from cogarena.generators.cognitive_control_gen import generate_cc_items
from cogarena.generators.episodic_memory_gen import generate_em_items
from cogarena.generators.metacognition_gen import generate_mc_items
from cogarena.generators.theory_of_mind_gen import generate_tom_items
from cogarena.generators.working_memory_gen import generate_wm_items
from cogarena.scoring import item_accuracy, score_episode, score_static

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
    "n_back": "Working Memory",
    "operation_span": "Working Memory",
    "stroop": "Cognitive Control",
    "flanker": "Cognitive Control",
    "go_nogo": "Cognitive Control",
    "drm_false_memory": "Episodic Memory",
    "source_monitoring": "Episodic Memory",
    "cvlt_word_list": "Episodic Memory",
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

    is_stub = True

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
    if getattr(args, "device", None):
        cfg["device"] = args.device
    if getattr(args, "dtype", None):
        cfg["dtype"] = args.dtype
    if getattr(args, "trust_remote_code", False):
        cfg["trust_remote_code"] = True
    return LLMClient(config=cfg)


def _is_multiturn(item):
    turns = item.metadata.parameters.get("turns", [])
    return isinstance(turns, list) and len(turns) > 0


def _collect_items(n_per_paradigm, seed, only_paradigms, include_multiturn=True):
    """Generate deterministic text-battery items for the requested paradigms."""
    if n_per_paradigm < 1:
        raise ValueError("--n must be at least 1")
    pools = defaultdict(list)
    for gen in GENERATORS.values():
        # Several research generators enforce minimum condition counts for
        # tiny requests. Generate a standard pool, then select exactly --n
        # items so the public CLI has predictable cost.
        pool_size = max(n_per_paradigm, 50)
        for it in gen(seed=seed, n_per_paradigm=pool_size,
                      include_contamination_probes=False):
            if _is_multiturn(it) and not include_multiturn:
                continue
            if only_paradigms and it.metadata.paradigm not in only_paradigms:
                continue
            pools[it.metadata.paradigm].append(it)
    items = []
    for paradigm in PARADIGM_GROUPING:
        pool = pools.get(paradigm, [])
        if not pool:
            continue
        if len(pool) < n_per_paradigm:
            raise RuntimeError(
                f"Generator returned only {len(pool)} {paradigm} items; "
                f"{n_per_paradigm} requested"
            )
        if n_per_paradigm == 1:
            chosen = [pool[len(pool) // 2]]
        elif n_per_paradigm == len(pool):
            chosen = pool
        else:
            chosen = [
                pool[round(i * (len(pool) - 1) / (n_per_paradigm - 1))]
                for i in range(n_per_paradigm)
            ]
        items.extend(chosen)
    return items


def _safe_generate(client, prompt, *, images=None):
    try:
        return client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            images=images,
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {type(exc).__name__}: {exc}"


def _turn_stimulus(turn):
    for key in ("stimulus", "presented", "item", "prompt"):
        if key in turn:
            value = turn[key]
            return json.dumps(value) if isinstance(value, dict) else str(value)
    return str(turn)


def _run_text_item(client, item, max_context_turns=30):
    if not _is_multiturn(item):
        response = _safe_generate(client, item.stimulus)
        return {
            "task_id": item.task_id,
            "mode": "text",
            "paradigm": item.metadata.paradigm,
            "grouping": PARADIGM_GROUPING[item.metadata.paradigm],
            "response": response,
            "score": (
                {"scored": False, "accuracy": 0.0}
                if getattr(client, "is_stub", False)
                else score_static(item, response)
            ),
        }

    turns = item.metadata.parameters["turns"]
    history = [item.stimulus]
    responses = []
    response_records = []
    for index, turn in enumerate(turns):
        stimulus = _turn_stimulus(turn)
        recent = history[-(max_context_turns * 2):]
        prompt = "\n".join(recent)
        prompt += f"\nTrial {index + 1}: {stimulus}\nYour response:"
        response = _safe_generate(client, prompt)
        responses.append(response)
        response_records.append(
            {
                "trial": index + 1,
                "stimulus": stimulus,
                "response": response,
            }
        )
        history.extend(
            [
                f"Trial {index + 1}: {stimulus}",
                f"Your response: {response}",
            ]
        )
        # Only explicit task feedback is shown. Ground-truth answers are never
        # inserted into the conversation by the evaluator.
        if turn.get("feedback"):
            history.append(f"Feedback: {turn['feedback']}")

    return {
        "task_id": item.task_id,
        "mode": "text",
        "paradigm": item.metadata.paradigm,
        "grouping": PARADIGM_GROUPING[item.metadata.paradigm],
        "n_turns": len(turns),
        "responses": response_records,
        "score": (
            {"scored": False, "accuracy": 0.0}
            if getattr(client, "is_stub", False)
            else score_episode(item, responses)
        ),
    }


def _answer_matches(expected, response):
    import re

    exp = str(expected).strip().lower()
    act = str(response).strip().lower()
    if not exp or not act or act.startswith("error:"):
        return False
    return exp == act or bool(re.search(rf"\b{re.escape(exp)}\b", act))


def _run_vlm(client, n, seed, paradigms, asset_dir):
    try:
        from cogarena.image_gen.false_belief_images import generate_false_belief_set
        from cogarena.image_gen.flanker_images import generate_flanker_set
        from cogarena.image_gen.stroop_images import generate_stroop_set
    except ImportError as exc:
        raise RuntimeError(
            "VLM mode requires the image extra: pip install -e \".[image]\""
        ) from exc

    available = {"stroop", "flanker", "false_belief"}
    selected = paradigms or sorted(available)
    unknown = set(selected) - available
    if unknown:
        raise ValueError(
            "VLM mode supports stroop, flanker, and false_belief; "
            f"unsupported: {', '.join(sorted(unknown))}"
        )

    details = []
    asset_dir.mkdir(parents=True, exist_ok=True)
    for paradigm in selected:
        out_dir = asset_dir / paradigm
        if paradigm == "stroop":
            trials = generate_stroop_set(
                seed=seed,
                n_congruent=(n + 1) // 2,
                n_incongruent=n // 2,
                out_dir=str(out_dir),
            )
        elif paradigm == "flanker":
            trials = generate_flanker_set(
                seed=seed,
                n_congruent=(n + 1) // 2,
                n_incongruent=n // 2,
                out_dir=str(out_dir),
            )
        else:
            trials = generate_false_belief_set(
                seed=seed,
                n_items=n,
                out_dir=str(out_dir),
            )

        for index, trial in enumerate(trials):
            images = trial.get("image_paths") or [trial["image_path"]]
            response = _safe_generate(
                client,
                trial["stimulus_text"],
                images=list(images),
            )
            expected = trial.get(
                "correct_answer",
                trial.get("expected_response", ""),
            )
            correct = (
                False
                if getattr(client, "is_stub", False)
                else _answer_matches(expected, response)
            )
            details.append(
                {
                    "task_id": f"vlm_{paradigm}_{index:04d}",
                    "mode": "vlm",
                    "paradigm": paradigm,
                    "grouping": PARADIGM_GROUPING[paradigm],
                    "response": response,
                    "score": {
                        "accuracy": 1.0 if correct else 0.0,
                        "correct": correct,
                    },
                    "condition": (
                        "congruent"
                        if trial.get("congruent") is True
                        else "incongruent"
                        if trial.get("congruent") is False
                        else None
                    ),
                }
            )
    return details


def _run_agent(client, n, seed, paradigms, max_think_steps):
    from cogarena.agent import CogArenaAgent
    from cogarena.core import CogArenaEnv, EvalMode, ScoringConfig, TaskMetadata
    from cogarena.gym_env import _SPEC, make_trial_generator

    selected = paradigms or list(PARADIGM_GROUPING)
    unknown = set(selected) - set(PARADIGM_GROUPING)
    if unknown:
        raise ValueError(
            f"Unknown agent paradigms: {', '.join(sorted(unknown))}"
        )

    details = []
    for paradigm in selected:
        dimension = _SPEC[paradigm][2]
        for episode in range(n):
            episode_seed = seed + episode
            generator = make_trial_generator(paradigm, episode_seed, "easy")
            metadata = TaskMetadata(
                dimension=dimension,
                paradigm=paradigm,
                mode=EvalMode.AGENT_INTERACTIVE,
                scoring=ScoringConfig(
                    method="partial_match",
                    params={"direction": "expected_in_response"},
                ),
            )
            env = CogArenaEnv(generator, metadata)
            agent = CogArenaAgent(
                client,
                max_think_steps=max_think_steps,
            )
            observation = env.reset(seed=episode_seed)
            done = False
            steps = 0
            while not done:
                action = agent.act(observation)
                observation, reward, done, _ = env.step(action)
                agent.record_reward(reward)
                steps += 1
            score = (
                {"scored": False, "accuracy": 0.0}
                if getattr(client, "is_stub", False)
                else env.score()
            )
            details.append(
                {
                    "task_id": f"agent_{paradigm}_{episode:04d}",
                    "mode": "agent",
                    "paradigm": paradigm,
                    "grouping": PARADIGM_GROUPING[paradigm],
                    "n_turns": steps,
                    "tools_used": agent.tool_call_count,
                    "score": score,
                }
            )
    return details


def _summarize(details):
    by_par_scores = defaultdict(list)
    for detail in details:
        by_par_scores[detail["paradigm"]].append(item_accuracy(detail["score"]))
    par_acc = {p: sum(v) / len(v) for p, v in by_par_scores.items()}
    grp_scores = defaultdict(list)
    for paradigm, accuracy in par_acc.items():
        grp_scores[PARADIGM_GROUPING[paradigm]].append(accuracy)
    grp_acc = {g: sum(v) / len(v) for g, v in grp_scores.items()}
    overall = sum(par_acc.values()) / len(par_acc) if par_acc else 0.0
    return par_acc, grp_acc, overall


def cmd_eval(args):
    only = set(args.paradigms) if args.paradigms else None
    label = "DRY RUN" if args.dry_run else args.model
    client = _build_client(args)
    t0 = time.time()

    if args.mode == "text":
        items = _collect_items(
            args.n,
            args.seed,
            only,
            include_multiturn=not args.single_turn_only,
        )
        if not items:
            print("No items generated (check --paradigms).", file=sys.stderr)
            return 1
        n_par = len({it.metadata.paradigm for it in items})
        print(
            f"CogArena text: {len(items)} episodes across {n_par} paradigms "
            f"-> {label} (provider={args.provider}, seed={args.seed})"
        )
        details = []
        for index, item in enumerate(items):
            details.append(
                _run_text_item(
                    client,
                    item,
                    max_context_turns=args.max_context_turns,
                )
            )
            if not args.quiet and (index + 1) % 25 == 0:
                print(f"  ... {index + 1}/{len(items)} episodes")
    elif args.mode == "vlm":
        print(
            f"CogArena VLM: {label} "
            f"(provider={args.provider}, seed={args.seed})"
        )
        safe_model = args.model.replace("/", "_").replace(":", "_")
        asset_dir = Path(args.output) / safe_model / "vlm_assets"
        details = _run_vlm(
            client,
            args.n,
            args.seed,
            list(only) if only else None,
            asset_dir,
        )
    else:
        print(
            f"CogArena agent: {label} "
            f"(provider={args.provider}, seed={args.seed})"
        )
        details = _run_agent(
            client,
            args.n,
            args.seed,
            list(only) if only else None,
            args.agent_max_think_steps,
        )

    elapsed = time.time() - t0
    par_acc, grp_acc, overall = _summarize(details)
    _print_table(par_acc, grp_acc, overall, elapsed)

    out_dir = (
        Path(args.output)
        / args.model.replace("/", "_").replace(":", "_")
        / args.mode
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "model": args.model,
        "provider": args.provider,
        "mode": args.mode,
        "seed": args.seed,
        "n_records": len(details),
        "n_per_paradigm": args.n,
        "elapsed_sec": round(elapsed, 1),
        "overall_accuracy": round(overall, 4),
        "paradigm_accuracy": {p: round(a, 4) for p, a in sorted(par_acc.items())},
        "grouping_accuracy": {g: round(a, 4) for g, a in sorted(grp_acc.items())},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    (out_dir / "details.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False)
    )
    print(f"\nSaved: {out_dir/'aggregate.json'} and {out_dir/'details.json'}")
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
    print("CogArena text battery (13 paradigms, 5 groupings):\n")
    for grp in ["Working Memory", "Cognitive Control", "Episodic Memory",
                "Theory of Mind", "Metacognition"]:
        pars = [p for p, g in PARADIGM_GROUPING.items() if g == grp]
        print(f"  {grp}: {', '.join(pars)}")
    print("\nModes:")
    print("  text   all 13 paradigms; n-back, operation span, and CVLT are multi-turn")
    print("  vlm    Stroop, Flanker, and false-belief image adaptations")
    print("  agent  ReAct-style tool-using evaluation on any of the 13 paradigms")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="cogarena", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    e = sub.add_parser("eval", help="evaluate a model in text, VLM, or agent mode")
    e.add_argument("--model", required=True, help="model name/tag (e.g. gpt-4o-mini, qwen2.5:7b)")
    e.add_argument("--mode", default="text", choices=["text", "vlm", "agent"],
                   help="evaluation mode (default: text)")
    e.add_argument("--provider", default="openai",
                   choices=["openai", "local", "anthropic", "google", "huggingface"],
                   help="'local' uses an OpenAI-compatible endpoint; 'huggingface' loads a text model directly")
    e.add_argument("--base-url", default=None, help="endpoint URL for --provider local")
    e.add_argument("--api-key", default=None, help="API key (else read from env)")
    e.add_argument("--n", type=int, default=50, help="items per paradigm (default 50)")
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--paradigms", nargs="+", default=None,
                   help="restrict to these paradigm names (default: all)")
    e.add_argument("--temperature", type=float, default=None)
    e.add_argument("--max-tokens", type=int, default=None)
    e.add_argument("--max-context-turns", type=int, default=30,
                   help="prior turns retained in multi-turn text prompts")
    e.add_argument("--single-turn-only", action="store_true",
                   help="text mode: skip n-back, operation span, and CVLT")
    e.add_argument("--agent-max-think-steps", type=int, default=2,
                   help="agent mode: maximum ReAct iterations per turn")
    e.add_argument("--device", default="auto",
                   help="Hugging Face device: auto, cpu, cuda, or cuda:<index>")
    e.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"],
                   help="Hugging Face model dtype")
    e.add_argument("--trust-remote-code", action="store_true",
                   help="allow Hugging Face custom model code")
    e.add_argument("--output", default="cogarena_results", help="output directory")
    e.add_argument("--dry-run", action="store_true",
                   help="run the pipeline with a stub model (no API key needed)")
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_eval)

    list_parser = sub.add_parser("list", help="list paradigms")
    list_parser.set_defaults(func=cmd_list)

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
