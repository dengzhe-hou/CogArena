#!/bin/bash
#SBATCH --job-name=cog-rep-transport-diag
#SBATCH --partition=batch
#SBATCH --nodelist=c04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=results/scaffold_wording_replication_20260725/slurm/transport_diag_%j.log

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
cd "${ROOT}"
export OLLAMA_HOST="127.0.0.1:18110"
export OPENAI_BASE_URL="http://127.0.0.1:18110/v1"
export OPENAI_API_KEY=ollama
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=1
export COGARENA_ROOT="${ROOT}"
export COGARENA_CAUSAL_SPEC="${ROOT}/scripts/experiments/scaffold_wording_replication_20260725/SPEC.json"
export COGARENA_CAUSAL_RESULTS_ROOT="${ROOT}/results/scaffold_wording_replication_20260725"
export COGARENA_REASONING_EFFORT=none
export COGARENA_STOP_MODE=format_routed

ollama serve >"${ROOT}/results/scaffold_wording_replication_20260725/ollama/transport_diag_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
cleanup() {
  ollama stop "olmo2:7b" >/dev/null 2>&1 || true
  kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:18110/api/tags" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "http://127.0.0.1:18110/api/tags" >/dev/null
ollama show "olmo2:7b" >/dev/null
curl -fsS --max-time 600 "http://127.0.0.1:18110/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"olmo2:7b","messages":[{"role":"user","content":"Reply OK <END_COGARENA_RESPONSE>"}],"temperature":0,"max_tokens":8,"reasoning_effort":"none","stop":["<END_COGARENA_RESPONSE>"]}' \
  >/dev/null
ollama ps
python -m scripts.experiments.scaffold_wording_replication_20260725.diagnose_transport
