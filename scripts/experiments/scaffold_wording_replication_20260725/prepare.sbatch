#!/bin/bash
#SBATCH --job-name=cog-scaffold-rep-prep
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=results/scaffold_wording_replication_20260725/slurm/prepare_%j.log

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
cd "${ROOT}"
export COGARENA_ROOT="${ROOT}"
export COGARENA_CAUSAL_SPEC="${ROOT}/scripts/experiments/scaffold_wording_replication_20260725/SPEC.json"
export COGARENA_CAUSAL_RESULTS_ROOT="${ROOT}/results/scaffold_wording_replication_20260725"
mkdir -p "${COGARENA_CAUSAL_RESULTS_ROOT}/slurm"

python -m scripts.experiments.scaffold_wording_replication_20260725.build_spec
python -m scripts.experiments.causal_selectivity_20260720.build_item_manifests
python -m scripts.experiments.scaffold_wording_replication_20260725.preflight \
  --manifest-only
echo "WORDING REPLICATION PREP PASS"
