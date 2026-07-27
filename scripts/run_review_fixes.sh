#!/bin/bash
# Auto-review-loop round-1 fixes: regenerate b2_expanded.json (cosmetic flag fix,
# headline must be UNCHANGED) + recompile main.tex and supplementary.tex.
#
# FAIL-CLOSED (2026-07-19, round 19.1): no `|| true` on any compile step, source
# files are prechecked, stale PDFs/logs are deleted before compiling, and the
# fresh outputs are verified after. NOTE: the local paper/main.tex and
# paper/supplementary.tex were quarantined as *.stale-* (see paper/AUTHORITY.md);
# this script therefore refuses to run until a fresh Overleaf clone restores
# authoritative sources -- that refusal is intentional.
set -euo pipefail
cd "$(cd "$(dirname "$0")"/.. && pwd)"

echo "===================== STEP 1: regenerate b2_expanded.json ====================="
python3 scripts/compute_b2_expanded.py > /tmp/b2_regen.log 2>&1 \
  || { echo "B2 FAILED"; tail -30 /tmp/b2_regen.log; exit 1; }
python3 - <<'PY'
import json
d = json.load(open("results/reanalysis/b2_expanded.json"))
print("repro20_matches_paper:", d.get("repro20_matches_paper"))
print("paper_target:", d.get("paper_target"))
print("repro20 delta/p:", d["repro20"]["delta"], d["repro20"]["p_value"])
e = d["exp55"]
print("HEADLINE exp55  within/cross/delta/p:", e["within_mean"], e["cross_mean"], e["delta"], e["p_value"])
print("scaling conf_cal pearson_r:", d["scaling"]["confidence_calibration"]["pearson_r"])
PY

compile_one() {
  # compile_one <basename>: precheck source, purge stale outputs, compile
  # without error suppression, then verify the fresh PDF exists.
  local base="$1"
  if [ ! -f "${base}.tex" ]; then
    echo "FATAL: ${base}.tex missing. Local copies were quarantined as stale;"
    echo "       restore authoritative sources from a fresh Overleaf clone"
    echo "       (see paper/AUTHORITY.md) before compiling."
    exit 1
  fi
  rm -f "${base}.pdf" "${base}.log" "${base}.aux" "${base}.fls" "${base}.fdb_latexmk"
  latexmk -pdf -interaction=nonstopmode "${base}.tex" > "/tmp/${base}_compile.log" 2>&1 \
    || { echo "COMPILE FAILED: ${base}.tex"; tail -40 "/tmp/${base}_compile.log"; exit 1; }
  [ -f "${base}.pdf" ] || { echo "FATAL: ${base}.pdf not produced"; exit 1; }
  echo "${base}.pdf pages:"; pdfinfo "${base}.pdf" | grep -i pages
  echo "${base} undefined refs/citations:"
  grep -i "LaTeX Warning: Reference\|Citation.*undefined\|There were undefined" "${base}.log" | head -20 || true
  echo "${base} overfull (>15pt):"
  grep -o "Overfull \\\\hbox ([0-9.]*pt" "${base}.log" | awk -F'[(.]' '{if ($2+0>15) print}' | head -20 || true
}

echo ""
echo "===================== STEP 2: compile main.tex ====================="
cd paper
compile_one main

echo ""
echo "===================== STEP 3: compile supplementary.tex ====================="
compile_one supplementary

echo ""
echo "===================== DONE ====================="
