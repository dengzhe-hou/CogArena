#!/bin/bash
#SBATCH --job-name=cog-causal-close
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=results/causal_selectivity_20260720/slurm/close_%j.log

set -euo pipefail
: "${COGARENA_GIT_HEAD:?inject the same frozen revision used by the raw array}"
PROFILE="${1:?pass pilot or formal}"
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
srun python -m scripts.experiments.causal_selectivity_20260720.verify_run --profile "${PROFILE}"
