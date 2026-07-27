#!/bin/bash
set -euo pipefail
LABEL="${1:?pass pro6000}"
: "${SLURM_JOB_ID:?must run under Slurm}"
: "${CUDA_VISIBLE_DEVICES:?GPU allocation missing}"
: "${COGARENA_GIT_HEAD:?inject committed revision at submission}"
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
cd "${ROOT}"
MODEL=qwen2.5:32b
PORT=$((53000 + (SLURM_JOB_ID % 500) * 4))
export OLLAMA_HOST="127.0.0.1:${PORT}"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=ollama
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=1
export COGARENA_ROOT="${ROOT}"
export COGARENA_REASONING_EFFORT=none
export COGARENA_STOP_MODE=format_routed
mkdir -p results/causal_selectivity_20260720/{slurm,ollama}
LOG="results/causal_selectivity_20260720/ollama/capacity_${LABEL}_${SLURM_JOB_ID}.log"
ollama serve >"${LOG}" 2>&1 &
PID=$!
trap 'ollama stop "${MODEL}" >/dev/null 2>&1 || true; kill "${PID}" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "http://127.0.0.1:${PORT}/api/tags" >/dev/null || {
  echo "FATAL: node-local Ollama server did not become ready" >&2; exit 5;
}
ollama show "${MODEL}" >/dev/null 2>&1 || {
  echo "FATAL: ${MODEL} absent; capacity jobs never pull" >&2; exit 4;
}
curl -fsS --max-time 600 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK <END_COGARENA_RESPONSE>\"}],\"temperature\":0,\"max_tokens\":8,\"reasoning_effort\":\"${COGARENA_REASONING_EFFORT}\",\"stop\":[\"<END_COGARENA_RESPONSE>\"]}" \
  >/dev/null
python -m scripts.experiments.causal_selectivity_20260720.capacity_probe --hardware-label "${LABEL}"
