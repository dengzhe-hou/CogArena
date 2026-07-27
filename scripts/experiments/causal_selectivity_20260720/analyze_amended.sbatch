#!/bin/bash
#SBATCH --job-name=cog-causal-analysis-amended
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/causal_selectivity_20260720/slurm/analysis_amended_%j.log

set -euo pipefail
: "${COGARENA_GIT_HEAD:?inject the immutable formal raw-run revision}"
: "${COGARENA_ANALYSIS_GIT_HEAD:?inject the post-freeze amendment revision}"
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export COGARENA_ROOT="${SLURM_SUBMIT_DIR}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: amended analysis must run under Slurm"; exit 1; }
[[ "$(hostname -s)" == c01* ]] || { echo "FATAL: amended analysis must run on c01"; exit 1; }
srun python -m scripts.experiments.causal_selectivity_20260720.analyze_amended
