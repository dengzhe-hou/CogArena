#!/bin/bash
#SBATCH --job-name=cog_large
#SBATCH --partition=batch
#SBATCH --array=0-1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=results/full_eval_expansion/slurm_large_%A_%a.out
#
# Larger open-model extension. This tests whether the
# positive manifold / dimensional null / scaling hold beyond 47B?
# 3 models, one per array task, 2x A100-80GB each (Mixtral-8x22B Q4 ~80GB needs >1 card).
# Writes into the SAME dirs as the 32-model expansion so compute_b2_expanded.py picks
# them up once they are added to expansion_modellist.txt (re-aggregated from details.json).
# Each extends an existing family (llama3.1 / qwen2.5 / mixtral) -> strengthens within-family scaling.

cd "$(cd "$(dirname "$0")"/.. && pwd)" || exit 9
mkdir -p results/full_eval_expansion results/multiturn_expansion
MODELS=(llama3.1:70b mixtral:8x22b)
m="${MODELS[$SLURM_ARRAY_TASK_ID]}"
PORT=$((11600 + SLURM_ARRAY_TASK_ID))
export OLLAMA_HOST=127.0.0.1:${PORT}
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://127.0.0.1:${PORT}/v1
export OLLAMA_CONTEXT_LENGTH=4096   # cap KV cache so 70B+ fit fully on one 96GB GPU (short CogArena prompts)

echo "task=${SLURM_ARRAY_TASK_ID} model=${m} node=$(hostname) port=${PORT} $(date)"
command -v ollama >/dev/null || { echo "FATAL: no ollama"; exit 1; }
nvidia-smi -L || true

ollama serve > results/full_eval_expansion/ollama_large_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
OLLAMA_PID=$!
for i in $(seq 1 120); do curl -s http://127.0.0.1:${PORT}/api/tags >/dev/null 2>&1 && break; sleep 2; done
curl -s http://127.0.0.1:${PORT}/api/tags >/dev/null 2>&1 || { echo "FATAL: ollama not ready"; kill ${OLLAMA_PID} 2>/dev/null; exit 1; }
echo "ollama ready on :${PORT}"

echo ">>> pull ${m} $(date +%H:%M:%S)"   # no-op if already in the shared store
ollama pull "${m}" 2>&1 | tail -2 || { echo "PULL FAIL ${m}"; kill ${OLLAMA_PID} 2>/dev/null; exit 1; }

# --- GPU-readiness gate: force-load and ABORT FAST if ollama fell back to CPU ---
# (ollama GPU-discovery can time out under node contention; never waste walltime on a CPU run)
echo ">>> gpu-gate ${m} $(date +%H:%M:%S)"
timeout 600 ollama run "${m}" "hi" >/dev/null 2>&1 || true
ollama ps || true
if ollama ps 2>/dev/null | grep -q "100% CPU"; then   # abort only on FULL CPU fallback, not minor spillover
  echo "FATAL: ${m} loaded on CPU (GPU discovery failed) -- aborting task to avoid burning walltime"
  ollama stop "${m}" 2>/dev/null || true; kill ${OLLAMA_PID} 2>/dev/null || true; exit 2
fi
echo "GPU OK for ${m}"

echo ">>> static ${m} $(date +%H:%M:%S)"
python scripts/run_unified.py --model "openai/${m}" --mode text --static-only \
    --n-items 50 --seed 42 --output-dir results/full_eval_expansion 2>&1 | tail -4 || echo "STATIC FAIL ${m}"

echo ">>> multiturn ${m} $(date +%H:%M:%S)"
python scripts/run_unified.py --model "openai/${m}" --mode text --dimensions working_memory episodic_memory \
    --n-items 50 --seed 42 --output-dir results/multiturn_expansion 2>&1 | tail -4 || echo "MT FAIL ${m}"

ollama stop "${m}" 2>/dev/null || true
kill ${OLLAMA_PID} 2>/dev/null || true
echo "DONE ${m} $(date)"
