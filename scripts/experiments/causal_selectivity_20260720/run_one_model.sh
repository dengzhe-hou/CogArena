#!/bin/bash
# Shared compute-node body. This file never submits a job and never calls git.
set -euo pipefail

PROFILE="${1:?profile required}"
MODEL="${2:?model required}"
: "${SLURM_JOB_ID:?must run under Slurm}"
: "${CUDA_VISIBLE_DEVICES:?GPU allocation missing}"
: "${COGARENA_GIT_HEAD:?inject the committed 40-hex revision with sbatch --export}"

case "${COGARENA_GIT_HEAD}" in
  *[!0-9a-f]*|'') echo "FATAL: invalid COGARENA_GIT_HEAD" >&2; exit 2 ;;
esac
[ "${#COGARENA_GIT_HEAD}" -eq 40 ] || { echo "FATAL: COGARENA_GIT_HEAD is not 40 hex" >&2; exit 2; }

ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
cd "${ROOT}"
RESULT_ROOT="results/causal_selectivity_20260720"
mkdir -p "${RESULT_ROOT}/slurm" "${RESULT_ROOT}/ollama"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PORT=$((12000 + (SLURM_JOB_ID % 2000) * 20 + TASK_ID))
export OLLAMA_HOST="127.0.0.1:${PORT}"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=ollama
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=1
export COGARENA_ROOT="${ROOT}"
export COGARENA_REASONING_EFFORT=none
export COGARENA_STOP_MODE=format_routed

echo "profile=${PROFILE} model=${MODEL} job=${SLURM_JOB_ID} task=${TASK_ID} node=$(hostname) port=${PORT}"
echo "source_revision=${COGARENA_GIT_HEAD} cuda=${CUDA_VISIBLE_DEVICES} context=${OLLAMA_CONTEXT_LENGTH} reasoning_effort=${COGARENA_REASONING_EFFORT} stop_mode=${COGARENA_STOP_MODE}"
python -m scripts.experiments.causal_selectivity_20260720.preflight \
  --profile "${PROFILE}" --model "${MODEL}"
command -v ollama >/dev/null || { echo "FATAL: ollama unavailable" >&2; exit 3; }
command -v nvidia-smi >/dev/null || { echo "FATAL: nvidia-smi unavailable" >&2; exit 3; }
nvidia-smi -L

SAFE_MODEL="${MODEL//:/__}"
OLLAMA_LOG="${RESULT_ROOT}/ollama/${PROFILE}_${SAFE_MODEL}_${SLURM_JOB_ID}.log"
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
curl -fsS "http://127.0.0.1:${PORT}/api/tags" >/dev/null || {
  echo "FATAL: node-local Ollama server did not become ready" >&2; exit 5;
}
# A missing local tag is a hard failure. Formal jobs never pull or silently
# substitute a model revision. The node-local server must be ready first.
ollama show "${MODEL}" >/dev/null 2>&1 || {
  echo "FATAL: ${MODEL} is not present in the node-visible Ollama store; no automatic pull allowed" >&2
  exit 4
}

# Force-load before the Python runner captures the selected model's exact
# PROCESSOR and CONTEXT columns plus model digest. The table printed here is
# diagnostic; Python fails unless this model's PROCESSOR equals `100% GPU`.
curl -fsS --max-time 600 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK <END_COGARENA_RESPONSE>\"}],\"temperature\":0,\"max_tokens\":8,\"reasoning_effort\":\"${COGARENA_REASONING_EFFORT}\",\"stop\":[\"<END_COGARENA_RESPONSE>\"]}" \
  >/dev/null
ollama ps

python -m scripts.experiments.causal_selectivity_20260720.run_model \
  --profile "${PROFILE}" --model "${MODEL}"
python -m scripts.experiments.causal_selectivity_20260720.verify_model \
  --profile "${PROFILE}" --model "${MODEL}"
echo "ALL MODEL GATES PASSED profile=${PROFILE} model=${MODEL} $(date --iso-8601=seconds)"
