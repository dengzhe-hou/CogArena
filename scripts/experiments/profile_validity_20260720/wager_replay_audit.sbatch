#!/bin/bash
#SBATCH --job-name=cog_wager_replay
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/reanalysis/profile_validity_20260720/wager_replay_%j.out

# CPU-only scorer replay. Submit from the repository root with:
#   sbatch --export=ALL,COGARENA_GIT_HEAD=$(git rev-parse HEAD) <this-file>
# Do not invoke the Python audit on a login node.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the CogArena repository root}"
cd "$ROOT"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: not inside Slurm"; exit 1; }
[[ "$(hostname -s)" == c01* ]] || { echo "FATAL: wagering replay must run on c01"; exit 1; }
: "${COGARENA_GIT_HEAD:?submit with a frozen full git SHA}"

mkdir -p results/reanalysis/profile_validity_20260720/wager_replay
export COGARENA_ROOT="$ROOT"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun python3 -m pytest -q tests/test_wager_replay_audit.py tests/test_wager_overlay_chain.py
srun python3 scripts/experiments/profile_validity_20260720/wager_replay_audit.py

MANIFEST=results/reanalysis/profile_validity_20260720/wager_replay/WAGER_REPLAY_MANIFEST.json
test -s "$MANIFEST"
grep -q '"all_gates_passed": true' "$MANIFEST"
echo "ALL GATES PASSED"
