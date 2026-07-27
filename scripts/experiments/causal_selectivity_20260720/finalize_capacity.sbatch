#!/bin/bash
#SBATCH --job-name=cog-capacity-close
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=results/causal_selectivity_20260720/slurm/capacity_close_%j.log

set -euo pipefail
: "${COGARENA_GIT_HEAD:?inject the same frozen revision used by all capacity probes}"
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
srun python -m scripts.experiments.causal_selectivity_20260720.verify_capacity
