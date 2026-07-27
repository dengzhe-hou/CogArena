#!/usr/bin/env bash
#SBATCH --job-name=cog-bifactor
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/reanalysis/exploratory_bifactor_20260727/slurm-%j.out
#SBATCH --error=results/reanalysis/exploratory_bifactor_20260727/slurm-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR}"
cd "${ROOT}"
PYTHON="${COGARENA_PYTHON:-python3}"
DEPS="${SLURM_TMPDIR:-/tmp}/cogarena_pyro_1_9_1"
mkdir -p "${DEPS}"

if ! PYTHONPATH="${DEPS}" "${PYTHON}" -c "import pyro" 2>/dev/null; then
  "${PYTHON}" -m pip install \
    --disable-pip-version-check \
    --target "${DEPS}" \
    --no-deps \
    "pyro-ppl==1.9.1" \
    "pyro-api==0.1.2" \
    "opt-einsum==3.4.0"
fi

export PYTHONPATH="${DEPS}:${ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

nvidia-smi
read -r -a EXTRA_ARGS <<< "${COGARENA_BIFACTOR_ARGS:-}"
"${PYTHON}" scripts/reanalysis/exploratory_bayesian_bifactor_20260727.py \
  --device cuda \
  "${EXTRA_ARGS[@]}"
