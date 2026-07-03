#!/usr/bin/env python3
"""Re-score the contamination probe test with the current (strict) scorers.

The original contamination run (results/contamination_v3/) scored responses at
runtime with the then-deployed scorers. After the scorer corrections, the rest
of the paper is reported under the corrected scorers, so this script brings the
contamination analysis onto the same footing:

1. Regenerates the exact classic/novel item sets (fixed seeds inside
   scripts/run_contamination_test.py) so scorers that need item context
   (e.g. Stroop's option disambiguation) can run.
2. Re-scores every stored response with cogarena.scoring.score_static.
   Stored responses were persisted truncated to 200 chars; for the ~2% of
   items at that cap the stored verdict (scored on the full text at runtime)
   is kept instead, and the count of such items is reported per cell.
3. Runs Fisher's exact test (classic vs novel correct/incorrect counts) for
   all 50 model x paradigm cells.
4. Self-check: re-deriving the table from the stored verdicts must reproduce
   the originally published result (only Qwen2.5-0.5B Stroop flagged,
   1.00 vs 0.767, p = 0.0105) before the re-scored table is trusted.

Output: results/reanalysis/contamination_rescored.json (all 50 cells).

Usage: COGARENA_ROOT=/path/to/CogArena python3 scripts/reanalysis/rescore_contamination.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("COGARENA_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from scipy.stats import fisher_exact  # noqa: E402

import run_contamination_test as rct  # noqa: E402
from cogarena.scoring import score_static, item_accuracy  # noqa: E402

MODELS = ["deepseek-r1:14b", "gemma2:9b", "qwen2.5:0.5b", "qwen2.5:32b", "qwen2.5:7b"]
TRUNC_CAP = 200  # evaluate_group stored resp[:200]
ALPHA = 0.05


def regenerate_items():
    """Return {paradigm: (classic_items, novel_items)} using the original seeds."""
    # capture the item lists instead of querying a model
    rct.evaluate_group = lambda model_id, items, label: items
    pools = {}
    for paradigm in rct.STATIC_PARADIGMS:
        classic, novel = rct.PARADIGM_TESTS[paradigm](model_id="none", n=30)
        pools[paradigm] = (
            {it.task_id: it for it in classic},
            {it.task_id: it for it in novel},
        )
    return pools


def rescore_cell(details, id_map):
    """Re-score one condition's stored details; returns (n_correct, n, n_kept_stored, n_unmatched)."""
    correct = kept = unmatched = 0
    for entry in details:
        resp = entry["response"]
        if len(resp) >= TRUNC_CAP:
            # response was persisted truncated; the stored verdict was computed
            # on the full text at runtime, so it is the better evidence here
            correct += bool(entry["correct"])
            kept += 1
            continue
        item = id_map.get(entry["task_id"])
        if item is None:
            correct += bool(entry["correct"])
            unmatched += 1
            continue
        correct += item_accuracy(score_static(item, resp)) >= 0.5
    return correct, len(details), kept, unmatched


def fisher(c_correct, c_n, n_correct, n_n):
    _, p = fisher_exact(
        [[c_correct, c_n - c_correct], [n_correct, n_n - n_correct]]
    )
    return p


def main():
    pools = regenerate_items()

    # ── self-check: stored verdicts must reproduce the published table ──────
    stored_flags = []
    for model in MODELS:
        details_path = ROOT / "results" / "contamination_v3" / f"contamination_details_openai_{model}.json"
        for cell in json.loads(details_path.read_text()):
            c = [bool(x["correct"]) for x in cell["classic_details"]]
            n = [bool(x["correct"]) for x in cell["novel_details"]]
            p = fisher(sum(c), len(c), sum(n), len(n))
            if p < ALPHA and sum(c) / len(c) > sum(n) / len(n):
                stored_flags.append((model, cell["paradigm"], round(sum(c) / len(c), 3), round(sum(n) / len(n), 3), round(p, 4)))
    expected = [("qwen2.5:0.5b", "stroop", 1.0, 0.767, 0.0105)]
    assert stored_flags == expected, f"self-check failed: stored verdicts give {stored_flags}, published table was {expected}"
    print(f"self-check OK: stored verdicts reproduce the published table {expected}")

    # ── re-score with the current scorers ───────────────────────────────────
    out = {"models": {}, "n_flagged_uncorrected": 0, "alpha": ALPHA,
           "bonferroni_alpha": ALPHA / 50,
           "provenance": "scripts/reanalysis/rescore_contamination.py on results/contamination_v3 details; "
                         "items regenerated with run_contamination_test.py seeds; scorers = cogarena.scoring.score_static"}
    for model in MODELS:
        details_path = ROOT / "results" / "contamination_v3" / f"contamination_details_openai_{model}.json"
        cells = {}
        for cell in json.loads(details_path.read_text()):
            paradigm = cell["paradigm"]
            classic_map, novel_map = pools[paradigm]
            cc, cn, ck, cu = rescore_cell(cell["classic_details"], classic_map)
            nc, nn, nk, nu = rescore_cell(cell["novel_details"], novel_map)
            p = float(fisher(cc, cn, nc, nn))
            flagged = bool(p < ALPHA and cc / cn > nc / nn)
            cells[paradigm] = {
                "classic_acc": round(cc / cn, 3), "novel_acc": round(nc / nn, 3),
                "fisher_p": round(p, 4), "flagged_uncorrected": flagged,
                "n_truncated_kept_stored": int(ck + nk), "n_unmatched": int(cu + nu),
            }
            out["n_flagged_uncorrected"] += int(flagged)
        out["models"][model] = cells

    dst = ROOT / "results" / "reanalysis" / "contamination_rescored.json"
    dst.write_text(json.dumps(out, indent=1))
    flags = [(m, p, c["classic_acc"], c["novel_acc"], c["fisher_p"])
             for m, cs in out["models"].items() for p, c in cs.items() if c["flagged_uncorrected"]]
    print(f"re-scored table: {out['n_flagged_uncorrected']}/50 cells flagged uncorrected -> {dst}")
    for f in flags:
        print("  flag:", f)


if __name__ == "__main__":
    main()
