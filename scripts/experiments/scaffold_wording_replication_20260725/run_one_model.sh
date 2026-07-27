#!/bin/bash
set -euo pipefail

MODEL="${1:?model required}"
: "${SLURM_JOB_ID:?must run under Slurm}"
: "${CUDA_VISIBLE_DEVICES:?GPU allocation missing}"
: "${COGARENA_GIT_HEAD:?inject the committed revision with sbatch --export}"

ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
cd "${ROOT}"
RESULT_ROOT="${ROOT}/results/scaffold_wording_replication_20260725"
SPEC_PATH="${ROOT}/scripts/experiments/scaffold_wording_replication_20260725/SPEC.json"
mkdir -p "${RESULT_ROOT}/slurm" "${RESULT_ROOT}/ollama"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PORT=$((14000 + (SLURM_JOB_ID % 1000) * 20 + TASK_ID))
export OLLAMA_HOST="127.0.0.1:${PORT}"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=ollama
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=1
export COGARENA_ROOT="${ROOT}"
export COGARENA_CAUSAL_SPEC="${SPEC_PATH}"
export COGARENA_CAUSAL_RESULTS_ROOT="${RESULT_ROOT}"
export COGARENA_REASONING_EFFORT=none
export COGARENA_STOP_MODE=format_routed

python -m scripts.experiments.scaffold_wording_replication_20260725.preflight \
  --model "${MODEL}"
command -v ollama >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi -L

SAFE_MODEL="${MODEL//:/__}"
OLLAMA_LOG="${RESULT_ROOT}/ollama/formal_${SAFE_MODEL}_${SLURM_JOB_ID}.log"
ollama serve >"${OLLAMA_LOG}" 2>&1 &
OLLAMA_PID=$!
cleanup() {
  ollama stop "${MODEL}" >/dev/null 2>&1 || true
  kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "http://127.0.0.1:${PORT}/api/tags" >/dev/null
ollama show "${MODEL}" >/dev/null 2>&1

curl -fsS --max-time 600 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK <END_COGARENA_RESPONSE>\"}],\"temperature\":0,\"max_tokens\":8,\"reasoning_effort\":\"none\",\"stop\":[\"<END_COGARENA_RESPONSE>\"]}" \
  >/dev/null
ollama ps

python -m scripts.experiments.scaffold_wording_replication_20260725.run_model \
  --profile formal --model "${MODEL}"
python -m scripts.experiments.causal_selectivity_20260720.verify_model \
  --profile formal --model "${MODEL}"
echo "WORDING REPLICATION MODEL PASS model=${MODEL} job=${SLURM_JOB_ID}"
