#!/bin/bash
# Full CogArena evaluation: 20 text LLMs + 4 VLMs + 4 Agents
# Run from project root: bash scripts/run_full_eval.sh

set -e
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://localhost:11434/v1}

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_DIR="results/full_eval_${TIMESTAMP}"
N_ITEMS=50  # items per paradigm
SEED=42

echo "============================================"
echo "CogArena Full Evaluation"
echo "Output: ${OUT_DIR}"
echo "Items per paradigm: ${N_ITEMS}"
echo "Started: $(date)"
echo "============================================"

# ── Phase 1: 20 Text LLMs (static only first, multi-turn later) ──
TEXT_MODELS=(
  "openai/tinyllama:1.1b"
  "openai/qwen2.5:0.5b"
  "openai/qwen2.5:1.5b"
  "openai/gemma2:2b"
  "openai/llama3.2:1b"
  "openai/qwen2.5:3b"
  "openai/llama3.2:3b"
  "openai/qwen2.5:7b"
  "openai/mistral:7b"
  "openai/llama3.1:8b"
  "openai/deepseek-r1:7b"
  "openai/gemma2:9b"
  "openai/qwen2.5:14b"
  "openai/phi3:14b"
  "openai/deepseek-r1:14b"
  "openai/gemma2:27b"
  "openai/qwen2.5:32b"
  "openai/mixtral:8x7b"
  "openai/yi:34b"
  "openai/command-r:35b"
)

echo ""
echo "=== PHASE 1: Text LLMs (static, ${#TEXT_MODELS[@]} models) ==="
for model in "${TEXT_MODELS[@]}"; do
  echo ""
  echo ">>> ${model} ($(date +%H:%M:%S))"
  python scripts/run_unified.py \
    --model "${model}" \
    --mode text \
    --n-items ${N_ITEMS} \
    --seed ${SEED} \
    --static-only \
    --output-dir "${OUT_DIR}" 2>&1 | tail -12
done

echo ""
echo "=== PHASE 1 COMPLETE: Text LLMs ==="
echo "Time: $(date)"

# ── Phase 2: 4 VLMs ──
VLM_MODELS=(
  "openai/qwen2.5vl:7b"
  "openai/llava:7b"
  "openai/gemma3:4b"
  "openai/moondream:1.8b"
)

echo ""
echo "=== PHASE 2: VLMs (${#VLM_MODELS[@]} models) ==="
for model in "${VLM_MODELS[@]}"; do
  echo ""
  echo ">>> ${model} ($(date +%H:%M:%S))"
  python scripts/run_unified.py \
    --model "${model}" \
    --mode image \
    --n-items ${N_ITEMS} \
    --seed ${SEED} \
    --output-dir "${OUT_DIR}" 2>&1 | tail -10
done

echo ""
echo "=== PHASE 2 COMPLETE: VLMs ==="

# ── Phase 3: 4 Agents ──
AGENT_MODELS=(
  "openai/qwen2.5:7b"
  "openai/qwen2.5:32b"
  "openai/tinyllama:1.1b"
  "openai/deepseek-r1:14b"
)

echo ""
echo "=== PHASE 3: Agents (${#AGENT_MODELS[@]} models) ==="
for model in "${AGENT_MODELS[@]}"; do
  echo ""
  echo ">>> ${model} agent ($(date +%H:%M:%S))"
  python scripts/run_unified.py \
    --model "${model}" \
    --mode agent \
    --n-items 2 \
    --seed ${SEED} \
    --output-dir "${OUT_DIR}" 2>&1 | tail -10
done

echo ""
echo "============================================"
echo "FULL EVALUATION COMPLETE"
echo "Output: ${OUT_DIR}"
echo "Finished: $(date)"
echo "============================================"

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="
find "${OUT_DIR}" -name "aggregate.json" | sort | while read f; do
  echo "--- ${f} ---"
  python -c "
import json
d = json.load(open('${f}'))
print(f'  Model: {d[\"model\"]} | Mode: {d[\"mode\"]} | Items: {d[\"n_results\"]}')
for p, v in sorted(d.get('paradigms', {}).items()):
    print(f'    {p:25s} {v[\"correct\"]}/{v[\"count\"]} = {v[\"accuracy\"]:.0%}')
" 2>/dev/null
done
