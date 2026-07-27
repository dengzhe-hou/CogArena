#!/bin/bash
#SBATCH --job-name=cog-scaffold-rep-final
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/scaffold_wording_replication_20260725/slurm/finalize_%j.log

set -euo pipefail
: "${COGARENA_GIT_HEAD:?inject the same revision used by the GPU array}"
: "${COGARENA_ANALYSIS_ADAPTER_HEAD:?inject the revision containing the replication analysis adapter}"
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
cd "${ROOT}"
export COGARENA_ROOT="${ROOT}"
export COGARENA_CAUSAL_SPEC="${ROOT}/scripts/experiments/scaffold_wording_replication_20260725/SPEC.json"
export COGARENA_CAUSAL_RESULTS_ROOT="${ROOT}/results/scaffold_wording_replication_20260725"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun python -m scripts.experiments.causal_selectivity_20260720.verify_run \
  --profile formal
srun python -m scripts.experiments.scaffold_wording_replication_20260725.analyze
echo "WORDING REPLICATION FINALIZE PASS"
