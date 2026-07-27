#!/bin/bash
#SBATCH -J sm_rerun
#SBATCH -p batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH -o results/sm_rerun_20260718/slurm_%j.log

# Source Monitoring rerun: the 11 dedup-fixed episodes x 55 models (605 runs).
# See results/reanalysis/sm_20260718/ for the exclusions and rerun manifests.
# Submit from the repository root; -o above is submit-dir relative.
#
# Serving env mirrors scripts/run_large_eval.sh: OLLAMA_CONTEXT_LENGTH=4096
# caps the KV cache so 70B+ models fit fully on one GPU (the first attempt,
# job 6803, let ollama default to n_ctx=131072 on llama3.1:70b, spilled KV
# to CPU, and returned HTTP 500s at the 5-minute mark). Flash attention is
# left at the default, as in the original evals.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
D=results/sm_rerun_20260718
PORT=11700
export OLLAMA_HOST=127.0.0.1:${PORT}
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://127.0.0.1:${PORT}/v1
export OLLAMA_KEEP_ALIVE=15m
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=4096

MODELLIST="${1:-${D}/modellist.txt}"
echo "node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-unset}  modellist: ${MODELLIST}  $(date)"
env | grep '^OLLAMA' | sort
command -v ollama >/dev/null || { echo "FATAL: no ollama"; exit 1; }
nvidia-smi -L || true

ollama serve > ${D}/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
trap "kill ${OLLAMA_PID} 2>/dev/null || true" EXIT
for i in $(seq 1 90); do curl -s http://127.0.0.1:${PORT}/api/tags >/dev/null 2>&1 && break; sleep 2; done
curl -s http://127.0.0.1:${PORT}/api/tags >/dev/null 2>&1 || { echo "FATAL: ollama not ready"; exit 1; }
echo "ollama ready on :${PORT}"

readiness() {
  # One tiny generation proves the model actually serves before the 11 items.
  curl -s --max-time 360 http://127.0.0.1:${PORT}/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":8,\"temperature\":0}" \
    | grep -q '"content"'
}

while read -r m; do
  [ -z "$m" ] && continue
  echo ">>> ${m}  $(date +%H:%M:%S)"
  ollama show "$m" >/dev/null 2>&1 || ollama pull "$m" || { echo "PULL FAIL ${m}"; continue; }
  if ! readiness "$m"; then
    echo "readiness retry ${m}"
    if ! readiness "$m"; then
      echo "READY FAIL ${m}"
      ollama stop "$m" 2>/dev/null || true
      continue
    fi
  fi
  python3 ${D}/run_sm_rerun.py "$m" || echo "EVAL FAIL ${m}"
  ollama stop "$m" 2>/dev/null || true
done < "${MODELLIST}"

python3 ${D}/summarize_sm_rerun.py
echo "DONE $(date)"
