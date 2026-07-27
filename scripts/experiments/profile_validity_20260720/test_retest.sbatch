#!/bin/bash
#SBATCH --job-name=cog_testretest
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/reanalysis/profile_validity_20260720/test_retest_%j.out

# CPU-only scorer replay and clustered bootstrap.  Submit with sbatch from the
# repository root; never invoke the Python analysis directly on a login node.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the CogArena repository root}"
cd "$ROOT"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: not inside a Slurm allocation"; exit 1; }
[[ "$(hostname -s)" == c01* ]] || { echo "FATAL: test-retest analysis must run on c01"; exit 1; }

mkdir -p results/reanalysis/profile_validity_20260720/test_retest
export COGARENA_ROOT="$ROOT"
: "${COGARENA_GIT_HEAD:?submit with --export=ALL,COGARENA_GIT_HEAD=<frozen commit>}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun python3 scripts/experiments/profile_validity_20260720/test_retest.py \
  --bootstrap-reps 20000 \
  --seed 42

test -s results/reanalysis/profile_validity_20260720/test_retest/TEST_RETEST_MANIFEST.json
grep -q '"all_gates_passed": true' \
  results/reanalysis/profile_validity_20260720/test_retest/TEST_RETEST_MANIFEST.json
echo "ALL GATES PASSED"
