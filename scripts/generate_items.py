#!/usr/bin/env python3
"""Generate the full CogArena v1 item pool across all 6 dimensions."""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cogarena.generators.working_memory_gen import generate_wm_items
from cogarena.generators.cognitive_control_gen import generate_cc_items
from cogarena.generators.set_shifting_gen import generate_ss_items
from cogarena.generators.episodic_memory_gen import generate_em_items
from cogarena.generators.theory_of_mind_gen import generate_tom_items
from cogarena.generators.metacognition_gen import generate_mc_items


def _get_attr(item, name):
    """Get attribute from nested (item.metadata.X) or flat (item.X) schema."""
    if hasattr(item, "metadata") and item.metadata is not None:
        return getattr(item.metadata, name, None)
    return getattr(item, name, None)


def _get_params(item):
    if hasattr(item, "metadata") and item.metadata is not None:
        return getattr(item.metadata, "parameters", {})
    return getattr(item, "parameters", {})


def _enum_val(v):
    return v.value if hasattr(v, "value") else str(v)


def serialize_item(item):
    """Convert a TaskInstance to a JSON-serializable dict (handles both schemas)."""
    params = _get_params(item)
    d = {
        "task_id": item.task_id,
        "dimension": _get_attr(item, "dimension"),
        "paradigm": _get_attr(item, "paradigm"),
        "mode": _enum_val(_get_attr(item, "mode") or "llm_static"),
        "difficulty": _enum_val(_get_attr(item, "difficulty") or "medium"),
        "adaptation_distance": _enum_val(_get_attr(item, "adaptation_distance") or "low"),
        "parameters": {k: v for k, v in params.items() if k != "turns"},
        "stimulus": item.stimulus[:200] + ("..." if len(item.stimulus) > 200 else ""),
        "has_turns": "turns" in params,
        "n_turns": len(params.get("turns", [])),
    }
    if item.expected_response is not None:
        er = str(item.expected_response)
        d["expected_response"] = er[:100] + ("..." if len(er) > 100 else "")
    return d


def main():
    parser = argparse.ArgumentParser(description="Generate CogArena item pool")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-paradigm", type=int, default=20,
                        help="Items per paradigm (use 20 for sanity, 100+ for full)")
    parser.add_argument("--output-dir", type=str, default="data/items")
    parser.add_argument("--contamination-probes", action="store_true", default=True)
    parser.add_argument("--dimensions", nargs="+", default=None,
                        help="Specific dimensions to generate (default: all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generators = {
        "working_memory": ("WM", generate_wm_items),
        "cognitive_control": ("CC", generate_cc_items),
        "set_shifting": ("SS", generate_ss_items),
        "episodic_memory": ("EM", generate_em_items),
        "theory_of_mind": ("ToM", generate_tom_items),
        "metacognition": ("MC", generate_mc_items),
    }

    dims = args.dimensions or list(generators.keys())
    all_items = []
    summary = {}

    for dim_name in dims:
        if dim_name not in generators:
            print(f"WARNING: Unknown dimension '{dim_name}', skipping")
            continue

        abbrev, gen_fn = generators[dim_name]
        print(f"Generating {dim_name} ({abbrev})...")
        try:
            items = gen_fn(
                seed=args.seed,
                n_per_paradigm=args.n_per_paradigm,
                include_contamination_probes=args.contamination_probes,
            )
            all_items.extend(items)

            # Count by paradigm
            paradigm_counts = {}
            for item in items:
                p = _get_attr(item, "paradigm") or "unknown"
                paradigm_counts[p] = paradigm_counts.get(p, 0) + 1

            summary[dim_name] = {
                "total": len(items),
                "paradigms": paradigm_counts,
            }
            print(f"  Generated {len(items)} items: {paradigm_counts}")
        except Exception as e:
            print(f"  ERROR generating {dim_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    summary["total_items"] = len(all_items)
    summary["seed"] = args.seed
    summary["n_per_paradigm"] = args.n_per_paradigm

    summary_path = out_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Save serialized items (metadata only, not full turns data)
    items_meta_path = out_dir / "items_metadata.jsonl"
    with open(items_meta_path, "w") as f:
        for item in all_items:
            f.write(json.dumps(serialize_item(item)) + "\n")
    print(f"Item metadata saved to {items_meta_path}")

    # Print overall summary
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_items)} items across {len(summary) - 2} dimensions")
    for dim, info in summary.items():
        if dim in ("total_items", "seed", "n_per_paradigm"):
            continue
        print(f"  {dim}: {info['total']} items ({info['paradigms']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
