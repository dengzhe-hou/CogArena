#!/bin/bash
#SBATCH --job-name=cog_exp
#SBATCH --partition=batch
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/full_eval_expansion/slurm_%A_%a.out
#
# Model-pool expansion eval: 32 new family-diverse models x 2 passes (static + multi-turn),
# round-robin across 4 array tasks. Output re-aggregated later from details.json (run_unified
# inline aggregate undercounts dict-scored paradigms). Submit AFTER models are pre-pulled.

cd "$(cd "$(dirname "$0")"/.. && pwd)" || exit 9
mkdir -p results/full_eval_expansion results/multiturn_expansion
PORT=$((11500 + SLURM_ARRAY_TASK_ID))
export OLLAMA_HOST=127.0.0.1:${PORT}
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://127.0.0.1:${PORT}/v1

echo "task=${SLURM_ARRAY_TASK_ID} node=$(hostname) port=${PORT} $(date)"
command -v ollama >/dev/null || { echo "FATAL: no ollama"; exit 1; }
nvidia-smi -L || true

ollama serve > results/full_eval_expansion/ollama_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
OLLAMA_PID=$!
for i in $(seq 1 90); do curl -s http://127.0.0.1:${PORT}/api/tags >/dev/null 2>&1 && break; sleep 2; done
curl -s http://127.0.0.1:${PORT}/api/tags >/dev/null 2>&1 || { echo "FATAL: ollama not ready"; kill ${OLLAMA_PID} 2>/dev/null; exit 1; }
echo "ollama ready on :${PORT}"

mapfile -t ALL < results/reanalysis/expansion_modellist.txt
N=${#ALL[@]}
for ((idx=SLURM_ARRAY_TASK_ID; idx<N; idx+=4)); do
  m="${ALL[$idx]}"
  [ -z "$m" ] && continue
  echo ""; echo ">>> [${idx}] ${m}  $(date +%H:%M:%S)"
  # static (10 single-turn paradigms)
  python scripts/run_unified.py --model "openai/${m}" --mode text --static-only \
      --n-items 50 --seed 42 --output-dir results/full_eval_expansion 2>&1 | tail -4 || echo "STATIC FAIL ${m}"
  # multi-turn (n-back / op-span / cvlt via WM+Episodic dimensions)
  python scripts/run_unified.py --model "openai/${m}" --mode text --dimensions working_memory episodic_memory \
      --n-items 50 --seed 42 --output-dir results/multiturn_expansion 2>&1 | tail -4 || echo "MT FAIL ${m}"
  ollama stop "$m" 2>/dev/null || true
done

kill ${OLLAMA_PID} 2>/dev/null || true
echo "TASK ${SLURM_ARRAY_TASK_ID} DONE  $(date)"
