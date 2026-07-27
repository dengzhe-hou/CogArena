#!/bin/bash
#SBATCH --job-name=cog-vlm-finalize
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/vlm_rerun_20260724_authority/slurm/finalize_%j.log

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
cd "${ROOT}"
export COGARENA_ROOT="${ROOT}"
export COGARENA_VLM_RUN_ROOT="${COGARENA_VLM_RUN_ROOT:-${ROOT}/results/vlm_rerun_20260724_authority}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [ -z "${COGARENA_GIT_HEAD:-}" ]; then
  COGARENA_GIT_HEAD="$(git rev-parse HEAD)"
fi
[[ "${COGARENA_GIT_HEAD}" =~ ^[0-9a-f]{40}$ ]] || { echo "FATAL COGARENA_GIT_HEAD is not a commit hash"; exit 1; }
export COGARENA_GIT_HEAD
python -m scripts.experiments.vlm_rerun_20260724.verify
echo "DONE $(date --iso-8601=seconds)"
