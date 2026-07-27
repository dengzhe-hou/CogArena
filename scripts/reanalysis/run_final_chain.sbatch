#!/bin/bash
#SBATCH --job-name=cog-final-chain
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/reanalysis/aplus_20260718/final_chain_%j.log

# CPU-only scorer replay/statistical/figure chain.  Submit after the formal
# wagering replay has emitted its hash-bound full 55x50 overlay.  No analysis
# runs on the login node.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the CogArena repository root}"
cd "$ROOT"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: not inside Slurm"; exit 1; }
[[ "$(hostname -s)" == c01* ]] || { echo "FATAL: final chain requires c01"; exit 1; }
: "${COGARENA_GIT_HEAD:?submit with a frozen full commit SHA}"

export COGARENA_ROOT="$ROOT"
export COGARENA_SM_OVERLAY="${COGARENA_SM_OVERLAY:-results/sm_rerun_20260718/sm_scores_overlay.json}"
export COGARENA_WAGER_OVERLAY="${COGARENA_WAGER_OVERLAY:-results/reanalysis/profile_validity_20260720/wager_replay/wager_accuracy_overlay.json}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

bash scripts/reanalysis/run_final_chain.sh
