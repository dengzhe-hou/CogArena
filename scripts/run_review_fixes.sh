#!/bin/bash
# Auto-review-loop round-1 fixes: regenerate b2_expanded.json (cosmetic flag fix,
# headline must be UNCHANGED) + recompile main.tex and supplementary.tex.
set -e
cd "$(cd "$(dirname "$0")"/.. && pwd)"

echo "===================== STEP 1: regenerate b2_expanded.json ====================="
python3 scripts/compute_b2_expanded.py > /tmp/b2_regen.log 2>&1 || { echo "B2 FAILED"; tail -30 /tmp/b2_regen.log; }
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

echo ""
echo "===================== STEP 2: compile main.tex ====================="
# ensure a TeX distribution (pdflatex/latexmk) is on your PATH
cd "$(cd "$(dirname "$0")"/.. && pwd)"/paper
latexmk -pdf -interaction=nonstopmode main.tex > /tmp/main_compile.log 2>&1 || true
echo "main.pdf pages:"; pdfinfo main.pdf 2>/dev/null | grep -i pages
echo "main undefined refs/citations:"; grep -c -i "undefined" main.log || echo 0
grep -i "LaTeX Warning: Reference\|Citation.*undefined\|There were undefined" main.log | head -20 || true
echo "main overfull (>15pt):"; grep -o "Overfull \\\\hbox ([0-9.]*pt" main.log | awk -F'[(.]' '{if ($2+0>15) print}' | head -20 || true

echo ""
echo "===================== STEP 3: compile supplementary.tex ====================="
latexmk -pdf -interaction=nonstopmode supplementary.tex > /tmp/supp_compile.log 2>&1 || true
echo "supplementary.pdf pages:"; pdfinfo supplementary.pdf 2>/dev/null | grep -i pages
echo "supp undefined refs/citations:"; grep -i "LaTeX Warning: Reference\|Citation.*undefined\|There were undefined" supplementary.log | head -20 || true

echo ""
echo "===================== DONE ====================="
