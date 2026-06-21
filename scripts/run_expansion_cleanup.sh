#!/bin/bash
#SBATCH --job-name=cog_exp_cl
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --exclude=c02
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=results/full_eval_expansion/slurm_cleanup_%j.out
#
# Re-run the models the timed-out V100 task missed (reads expansion_cleanup_list.txt).

cd "$(cd "$(dirname "$0")"/.. && pwd)" || exit 9
PORT=11600
export OLLAMA_HOST=127.0.0.1:${PORT}
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://127.0.0.1:${PORT}/v1
echo "node=$(hostname) port=${PORT} $(date)"; nvidia-smi -L || true
ollama serve > results/full_eval_expansion/ollama_cleanup_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
for i in $(seq 1 90); do curl -s http://127.0.0.1:${PORT}/api/tags >/dev/null 2>&1 && break; sleep 2; done
curl -s http://127.0.0.1:${PORT}/api/tags >/dev/null 2>&1 || { echo "FATAL ollama"; kill ${OLLAMA_PID} 2>/dev/null; exit 1; }
mapfile -t MODELS < results/reanalysis/expansion_cleanup_list.txt
for m in "${MODELS[@]}"; do
  [ -z "$m" ] && continue
  echo ""; echo ">>> ${m} $(date +%H:%M:%S)"
  python scripts/run_unified.py --model "openai/${m}" --mode text --static-only \
      --n-items 50 --seed 42 --output-dir results/full_eval_expansion 2>&1 | tail -3 || echo "STATIC FAIL ${m}"
  python scripts/run_unified.py --model "openai/${m}" --mode text --dimensions working_memory episodic_memory \
      --n-items 50 --seed 42 --output-dir results/multiturn_expansion 2>&1 | tail -3 || echo "MT FAIL ${m}"
  ollama stop "$m" 2>/dev/null || true
done
kill ${OLLAMA_PID} 2>/dev/null || true
echo "CLEANUP DONE $(date)"
