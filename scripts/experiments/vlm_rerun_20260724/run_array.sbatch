#!/bin/bash
#SBATCH --job-name=cog-vlm-rerun
#SBATCH --partition=batch
#SBATCH --nodelist=c03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%2
#SBATCH --output=results/vlm_rerun_20260724_authority/slurm/model_%A_%a.log

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
cd "${ROOT}"
test -n "${SLURM_JOB_ID:-}" || { echo "FATAL no Slurm job"; exit 1; }
case "$(hostname)" in login*|head*) echo "FATAL login node"; exit 1;; esac

if [ -n "${COGARENA_OLLAMA_BIN_DIR:-}" ]; then
  export PATH="${COGARENA_OLLAMA_BIN_DIR}:${PATH}"
fi
export COGARENA_ROOT="${ROOT}"
export COGARENA_VLM_RUN_ROOT="${COGARENA_VLM_RUN_ROOT:-${ROOT}/results/vlm_rerun_20260724_authority}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${COGARENA_VLM_RUN_ROOT}/mplconfig"
export OPENAI_API_KEY=ollama
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
unset OLLAMA_CONTEXT_LENGTH
unset OLLAMA_FLASH_ATTENTION
if [ -z "${COGARENA_GIT_HEAD:-}" ]; then
  COGARENA_GIT_HEAD="$(git rev-parse HEAD)"
fi
[[ "${COGARENA_GIT_HEAD}" =~ ^[0-9a-f]{40}$ ]] || { echo "FATAL COGARENA_GIT_HEAD is not a commit hash"; exit 1; }
export COGARENA_GIT_HEAD

mapfile -t MODELS < scripts/experiments/vlm_rerun_20260724/models.txt
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
API_MODEL="${MODEL#openai/}"
PORT=$((19000 + (SLURM_ARRAY_TASK_ID * 100) + (SLURM_JOB_ID % 97)))
export OLLAMA_HOST="127.0.0.1:${PORT}"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"

D="${COGARENA_VLM_RUN_ROOT}"
mkdir -p "${D}/ollama" "${D}/slurm" "${MPLCONFIGDIR}"
python - <<'PY'
from scripts.experiments.vlm_rerun_20260724.common import load_image_manifest
manifest, _ = load_image_manifest(verify_images=True)
if manifest.get("design_id") != "balanced_montage_v2":
    raise SystemExit("FATAL authority job requires balanced_montage_v2")
print("AUTHORITY DESIGN VERIFIED balanced_montage_v2")
PY
echo "MODEL=${MODEL} NODE=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-unset} JOB=${SLURM_JOB_ID}"
env | grep '^OLLAMA' | sort
nvidia-smi -L
ollama --version
python -m scripts.experiments.vlm_rerun_20260724.verify_cache --model "${MODEL}"

ollama serve >"${D}/ollama/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
OLLAMA_PID=$!
cleanup() {
  ollama stop "${API_MODEL}" >/dev/null 2>&1 || true
  kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "http://127.0.0.1:${PORT}/api/tags" >/dev/null
ollama show "${API_MODEL}" >/dev/null || {
  echo "FATAL cached tag unavailable: ${API_MODEL}; refusing to pull or mutate provenance"
  exit 1
}

python -m scripts.experiments.vlm_rerun_20260724.run_model --model "${MODEL}"
python -m scripts.experiments.vlm_rerun_20260724.verify --model "${MODEL}"
echo "MODEL DONE ${MODEL} $(date --iso-8601=seconds)"
