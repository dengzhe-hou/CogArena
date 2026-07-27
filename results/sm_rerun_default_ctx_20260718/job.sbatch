#!/bin/bash
#SBATCH -J sm_rerun_dctx
#SBATCH -p batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH -o results/sm_rerun_default_ctx_20260718/slurm_%j.log

# Serving-alignment arm: the 21 ordinary resumed models rerun WITHOUT the
# context override, matching the original evaluation's serving path (only
# llama3.1:70b and mixtral:8x22b ran under OLLAMA_CONTEXT_LENGTH=4096 in
# the original evals; the other resumed models used ollama defaults).
# The 4096 results in results/sm_rerun_20260718/ stay as the sensitivity
# arm. Per-model serving state (ollama ps: actual n_ctx, VRAM split) and
# model digests (/api/tags) are captured for the SM manifest.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
D=results/sm_rerun_default_ctx_20260718
R=results/sm_rerun_20260718
PORT=11710
export OLLAMA_HOST=127.0.0.1:${PORT}
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://127.0.0.1:${PORT}/v1
export OLLAMA_KEEP_ALIVE=15m
export OLLAMA_NUM_PARALLEL=1
# NO OLLAMA_CONTEXT_LENGTH: defaults on purpose.

MODELLIST="${1:-${D}/modellist_default_ctx.txt}"
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
curl -sf http://127.0.0.1:${PORT}/api/tags > ${D}/api_tags_${SLURM_JOB_ID}.json || true

readiness() {
  curl -sf --max-time 360 http://127.0.0.1:${PORT}/v1/chat/completions \
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
  echo "--- serving state for ${m}:"
  ollama ps || true
  python3 ${R}/run_sm_rerun.py "$m" "${D}" || echo "EVAL FAIL ${m}"
  ollama stop "$m" 2>/dev/null || true
done < "${MODELLIST}"

python3 ${D}/compare_serving_arms.py
echo "DONE $(date)"
