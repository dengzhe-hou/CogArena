#!/bin/bash
#SBATCH --job-name=cog-ext-checks
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/reanalysis/profile_validity_20260720/development_checks_%j.out

# Development-only closure for the prospective extension analyses.  It runs
# scorer replay and statistics only inside the c01 Slurm allocation; none of
# these reduced-bootstrap artifacts is a formal result.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the CogArena repository root}"
cd "$ROOT"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: not inside Slurm"; exit 1; }
[[ "$(hostname -s)" == c01* ]] || { echo "FATAL: development checks require c01"; exit 1; }

mkdir -p results/reanalysis/profile_validity_20260720
export COGARENA_ROOT="$ROOT"
export COGARENA_GIT_HEAD="${COGARENA_GIT_HEAD:-uncommitted-development-run}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun python3 -m pytest -q \
  tests/test_family_structure.py \
  tests/test_family_heldout_utility.py \
  tests/test_test_retest_profile_validity.py \
  tests/test_wager_replay_audit.py \
  tests/test_wager_overlay_chain.py \
  tests/test_final_chain_source_manifest.py

srun python3 scripts/experiments/profile_validity_20260720/family_structure.py \
  --n-boot 200 \
  --output results/reanalysis/profile_validity_20260720/family_structure_development_v2.json

echo "ALL DEVELOPMENT GATES PASSED"
