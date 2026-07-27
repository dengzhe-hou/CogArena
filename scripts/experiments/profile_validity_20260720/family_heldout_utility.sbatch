#!/bin/bash
#SBATCH --job-name=cog_profileutil
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=results/reanalysis/profile_validity_20260720/family_heldout_%j.out

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the CogArena repository root}"
cd "$ROOT"
mkdir -p results/reanalysis/profile_validity_20260720
export COGARENA_ROOT="$ROOT"
: "${COGARENA_GIT_HEAD:?submit with --export=ALL,COGARENA_GIT_HEAD=<frozen commit>}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

srun python3 -m pytest -q tests/test_family_heldout_utility.py
srun python3 scripts/experiments/profile_validity_20260720/family_heldout_utility.py
echo "ALL GATES PASSED"
