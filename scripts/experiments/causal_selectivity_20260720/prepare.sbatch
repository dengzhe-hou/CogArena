#!/bin/bash
#SBATCH --job-name=cog-causal-prepare
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/causal_selectivity_20260720/slurm/prepare_%j.log

# Zero-inference preparation. This job generates no model responses and must
# run on c01 rather than the login node.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the CogArena repository root}"
cd "${ROOT}"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: not inside Slurm"; exit 1; }
[[ "$(hostname -s)" == c01* ]] || { echo "FATAL: prepare must run on c01"; exit 1; }

mkdir -p results/causal_selectivity_20260720/slurm
export COGARENA_ROOT="${ROOT}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

python -m scripts.experiments.causal_selectivity_20260720.build_item_manifests
FORMAL_1="$(sha256sum results/causal_selectivity_20260720/item_manifest_formal.json | cut -d' ' -f1)"
PILOT_1="$(sha256sum results/causal_selectivity_20260720/item_manifest_pilot.json | cut -d' ' -f1)"
PROV_1="$(sha256sum results/causal_selectivity_20260720/ITEM_PROVENANCE.json | cut -d' ' -f1)"

python -m scripts.experiments.causal_selectivity_20260720.build_item_manifests
test "${FORMAL_1}" = "$(sha256sum results/causal_selectivity_20260720/item_manifest_formal.json | cut -d' ' -f1)"
test "${PILOT_1}" = "$(sha256sum results/causal_selectivity_20260720/item_manifest_pilot.json | cut -d' ' -f1)"
test "${PROV_1}" = "$(sha256sum results/causal_selectivity_20260720/ITEM_PROVENANCE.json | cut -d' ' -f1)"

python -m pytest -q scripts/experiments/causal_selectivity_20260720/tests
echo "ALL PREP GATES PASSED"
