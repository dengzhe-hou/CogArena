#!/bin/bash
#SBATCH --job-name=cog_fp16
#SBATCH --partition=batch
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/full_eval_fp16/slurm_%j.out
#
# fp16 de-confounding run for CogArena §5.6 predictive validity.
# Re-runs the STATIC (single-turn) battery at fp16 for a 6-model subset, so the
# §5.6 / C4 correlations can be recomputed on matched-precision scores.
# Static-only is deliberate: the multi-turn paradigms (n-back/CVLT) are already
# disclosed as scorer-sensitive; the robust single-turn paradigms cover the full
# Control/ToM/Meta domains.  Submit AFTER the login-node pre-pull finishes.

cd "$(cd "$(dirname "$0")"/.. && pwd)" || exit 9
mkdir -p results/full_eval_fp16
PORT=11500
export OLLAMA_HOST=127.0.0.1:${PORT}
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://127.0.0.1:${PORT}/v1

echo "node=$(hostname)  job=${SLURM_JOB_ID}  $(date)"
command -v ollama >/dev/null || { echo "FATAL: ollama not on PATH"; exit 1; }
nvidia-smi -L || echo "(no nvidia-smi)"

# start a node-local Ollama (separate port from any shared :11434)
ollama serve > results/full_eval_fp16/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
for i in $(seq 1 90); do
  curl -s http://127.0.0.1:${PORT}/api/tags >/dev/null 2>&1 && break
  sleep 2
done
if ! curl -s http://127.0.0.1:${PORT}/api/tags >/dev/null 2>&1; then
  echo "FATAL: ollama did not become ready"; kill ${OLLAMA_PID} 2>/dev/null; exit 1
fi
echo "ollama ready on :${PORT}"

MODELS=(
  qwen2.5:0.5b-instruct-fp16
  qwen2.5:7b-instruct-fp16
  qwen2.5:14b-instruct-fp16
  gemma2:9b-instruct-fp16
  llama3.1:8b-instruct-fp16
  mistral:7b-instruct-fp16
)

for m in "${MODELS[@]}"; do
  echo ""
  echo ">>> ${m}  $(date +%H:%M:%S)"
  ollama pull "$m" >/dev/null 2>&1 || { echo "SKIP ${m} (pull failed)"; continue; }
  python scripts/run_unified.py \
    --model "openai/${m}" \
    --mode text \
    --n-items 50 \
    --seed 42 \
    --static-only \
    --output-dir results/full_eval_fp16 2>&1 | tail -15
  ollama stop "$m" 2>/dev/null || true
done

kill ${OLLAMA_PID} 2>/dev/null || true
echo ""
echo "FP16 EVAL DONE  $(date)"
find results/full_eval_fp16 -name aggregate.json | sort
