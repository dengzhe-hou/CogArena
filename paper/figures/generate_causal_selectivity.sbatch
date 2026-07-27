#!/bin/bash
#SBATCH --job-name=cog-causal-figure
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=results/causal_selectivity_20260720/slurm/figure_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export COGARENA_ROOT="${SLURM_SUBMIT_DIR}"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: figure generation must run under Slurm"; exit 1; }
[[ "$(hostname -s)" == c01* ]] || { echo "FATAL: figure generation must run on c01"; exit 1; }
srun python paper/figures/generate_causal_selectivity.py
