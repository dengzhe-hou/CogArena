#!/bin/bash
# Fail-closed full-chain recomputation for the frozen paper statistics.
#
# Run from the repository root of a CLEAN checkout after the final SM overlay
# is adopted (SM_MANIFEST status=final).  OSpan uses the deterministic strict
# v4 parser as primary; canonical whitespace parsing is the co-reported
# scoring-specification sensitivity.  No human adjudication is consumed.
#
#   HEAD=$(git rev-parse HEAD)
#   sbatch --export=ALL,COGARENA_GIT_HEAD="$HEAD",COGARENA_SM_OVERLAY=results/sm_rerun_20260718/sm_scores_overlay.json,COGARENA_WAGER_OVERLAY=results/reanalysis/profile_validity_20260720/wager_replay/wager_accuracy_overlay.json \
#     scripts/reanalysis/run_final_chain.sbatch
#
# Execution graph (primary-estimand propagation; ORDER MATTERS):
#   phase 1  SM + wagering overlays -> corrected matrix -> BASE construct
#            matrix (both under the overlays) -> A+ suite, which consumes the
#            construct matrix it just got and produces
#            strict/canonical matrices + its MANIFEST
#   freeze   COGARENA_PRIMARY_MATRIX := matrix_aplus_strict.csv
#   phase 2  final inference, sensitivity, PC1 validation, joint
#            exclusion, step4/4b (difficulty splits included), reanalysis
#            JSONs, split-half, and ALL figures consume the primary
#            matrix. (two_level_bootstrap.py stays on the corrected
#            matrix by design: its pools mirror the production scorers;
#            the primary two-level bootstrap lives in the A+ MANIFEST.)
#   postflight  assert the A+ MANIFEST's recorded construct hash equals
#            the on-disk construct matrix, verify an EXPLICIT expected-
#            output list, and hash everything into CHAIN_MANIFEST.json.
set -euo pipefail
: "${SLURM_JOB_ID:?run_final_chain.sh must execute inside a Slurm allocation}"
case "$(hostname -s)" in
  c01*) ;;
  *) echo "FATAL: final statistics chain must run on c01, got $(hostname -s)" >&2; exit 2 ;;
esac
cd "$(cd "$(dirname "$0")"/../.. && pwd)"
export COGARENA_ROOT="$(pwd)"
export COGARENA_PRIMARY_CONFIG=aplus_strict
export COGARENA_REQUIRE_ADJ=0

echo "=== preflight"
: "${COGARENA_SM_OVERLAY:?set COGARENA_SM_OVERLAY to the final SM overlay path}"
: "${COGARENA_WAGER_OVERLAY:?set COGARENA_WAGER_OVERLAY to the verified full-pool wagering overlay path}"
: "${COGARENA_GIT_HEAD:?inject the committed source revision at Slurm submission}"
python3 scripts/reanalysis/final_chain_source_manifest.py --verify
export COGARENA_CHAIN_STARTED_NS="$(date +%s%N)"
python3 - <<'EOF'
# fail-closed: explicit raises, not assert (python -O must not weaken gates)
import hashlib, json, os
def req(cond, msg):
    if not cond:
        raise SystemExit(f"PREFLIGHT FAILED: {msg}")
root = os.environ["COGARENA_ROOT"]
root_real = os.path.realpath(root)
rev = os.environ["COGARENA_GIT_HEAD"]
req(len(rev) == 40 and all(c in "0123456789abcdef" for c in rev),
    "COGARENA_GIT_HEAD is not a full commit SHA")
ov_path = os.environ["COGARENA_SM_OVERLAY"]
req(os.path.exists(ov_path), f"overlay missing: {ov_path}")
man = json.load(open(f"{root}/results/sm_rerun_20260718/SM_MANIFEST.json"))
req(man["status"] == "final", f"SM_MANIFEST status is {man['status']!r}, not final")
h = hashlib.sha256(open(ov_path, "rb").read()).hexdigest()
want = man["outputs"]["sm_scores_overlay.json"]
req(h == want, f"overlay hash {h[:12]} != SM_MANIFEST {want[:12]}")
ov = json.load(open(ov_path))
req(len(ov) == 55 and all(len(v) == 50 for v in ov.values()), "overlay is not 55x50")
w_path = os.environ["COGARENA_WAGER_OVERLAY"]
req(os.path.exists(w_path), f"wagering overlay missing: {w_path}")
wman_path = f"{root}/results/reanalysis/profile_validity_20260720/wager_replay/WAGER_REPLAY_MANIFEST.json"
req(os.path.exists(wman_path), f"wagering manifest missing: {wman_path}")
wman = json.load(open(wman_path))
req(wman.get("schema_version") == "cogarena-wager-replay-manifest-v1"
    and wman.get("status") == "final"
    and wman.get("all_gates_passed") is True,
    "wagering replay manifest did not pass all final gates")
req(wman.get("execution", {}).get("git_head") == rev,
    "wagering replay source revision differs from the active chain revision")
req(wman.get("checks", {}).get("wager_construct_overlay_representable") is True
    and wman.get("checks", {}).get("did_bet_difference_count") == 0,
    "accuracy-only wagering overlay cannot represent the construct scorer inputs")
for rel, expected in wman.get("outputs", {}).items():
    path = os.path.realpath(os.path.join(root, rel))
    req(os.path.commonpath([root_real, path]) == root_real,
        f"wagering manifest output escapes repository root: {rel}")
    req(os.path.isfile(path), f"wagering manifest output missing: {rel}")
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    req(got == expected, f"wagering manifest output hash mismatch: {rel}")
wh = hashlib.sha256(open(w_path, "rb").read()).hexdigest()
wrel = os.path.relpath(w_path, root)
req(wh == wman["outputs"].get(wrel),
    "wagering overlay hash differs from replay manifest")
wov = json.load(open(w_path))
req(len(wov) == 55 and all(len(v) == 50 for v in wov.values()),
    "wagering overlay is not 55x50")
req(all(isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1
        for vals in wov.values() for x in vals.values()),
    "wagering overlay contains an invalid score")
ap = f"{root}/results/reanalysis/aplus_20260718"
blind = f"{root}/results/reanalysis/ospan_blind_20260718"
stale = [p for p in (
    f"{ap}/adjudications.json",
    f"{ap}/ADJUDICATION_MANIFEST.json",
    f"{ap}/matrix_aplus_adjudicated.csv",
    f"{ap}/matrix_aplus_adjudicated_exclamb.csv",
    f"{ap}/matrix_construct_aplus_adjudicated.csv",
    f"{blind}/annotator_A.json",
    f"{blind}/annotator_B.json",
    f"{blind}/agreement_report.json",
    f"{blind}/disagreement_manifest.json",
) if os.path.exists(p)]
req(not stale, f"strict-primary freeze refuses stale adjudication artifacts: {stale}")
req(os.environ.get("COGARENA_PRIMARY_CONFIG") == "aplus_strict",
    "formal freeze requires COGARENA_PRIMARY_CONFIG=aplus_strict")
print("preflight OK: SM and wagering overlays pinned 55x50, strict primary frozen")
EOF

run() { echo "=== $1"; shift; "$@"; }

# ---- phase 1: overlay-consistent base, construct BEFORE A+ ---------------
run "corrected matrix + headline (build_and_recompute)" \
    python3 results/recompute_20260703/build_and_recompute.py
run "base construct-native matrix (pre-A+, overlay-consistent)" \
    python3 results/construct_native_20260711/build_construct_matrix.py
run "A+ estimand suite (strict primary; canonical sensitivity)" \
    python3 scripts/reanalysis/aplus_rescore_20260718.py

# ---- freeze the primary matrices -----------------------------------------
export COGARENA_PRIMARY_MATRIX="results/reanalysis/aplus_20260718/matrix_aplus_strict.csv"
export COGARENA_PRIMARY_CONSTRUCT_MATRIX="results/reanalysis/aplus_20260718/matrix_construct_aplus_strict.csv"
[ -f "${COGARENA_PRIMARY_MATRIX}" ] || { echo "FATAL: primary matrix not produced"; exit 1; }
[ -f "${COGARENA_PRIMARY_CONSTRUCT_MATRIX}" ] || { echo "FATAL: primary construct matrix not produced"; exit 1; }
echo "=== primary matrices frozen:"
sha256sum "${COGARENA_PRIMARY_MATRIX}" "${COGARENA_PRIMARY_CONSTRUCT_MATRIX}"

# ---- phase 2: everything downstream consumes the primary matrices --------
run "final inference (permutation + bootstrap CIs)" \
    python3 results/recompute_20260703/final_inference.py
run "sensitivity analyses" \
    python3 results/recompute_20260703/sensitivity.py
run "PC1 validation simulations (self-check on base, analyses on primary)" \
    python3 results/pc1_validation_20260711/validate_pc1.py
run "joint validity-threat exclusion" \
    python3 results/pc1_validation_20260711/joint_exclusion.py
run "difficulty-stratified robustness" \
    python3 results/recompute_20260703/difficulty_robustness.py
run "two-level bootstrap (corrected-matrix arm, by design)" \
    python3 results/twolevel_bootstrap_20260712/two_level_bootstrap.py
run "step4 paper numbers (incl. difficulty splits)" \
    python3 results/recompute_20260703/step4_numbers.py
run "step4b paper numbers" \
    python3 results/recompute_20260703/step4b_numbers.py
run "reanalysis JSON regeneration (fails closed)" \
    python3 scripts/reanalysis/apply_corrected_results.py
run "split-half reliability" \
    python3 scripts/reanalysis/split_half_reliability.py
run "paper figures (generate_all, Type 42 fonts)" \
    python3 paper/figures/generate_all.py
run "headline manifold figure" \
    python3 paper/figures/fig_manifold.py

echo "=== postflight"
run "postflight (coherence gates + CHAIN_MANIFEST)" \
    python3 scripts/reanalysis/chain_postflight.py

echo "=== FULL CHAIN COMPLETE"
