#!/bin/bash
#SBATCH --job-name=cog-causal-target-baseline
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/causal_selectivity_20260720/slurm/target_baseline_%j.log

set -euo pipefail
: "${COGARENA_GIT_HEAD:?inject the immutable formal raw-run revision}"
: "${COGARENA_TARGET_BASELINE_GIT_HEAD:?inject the committed post-hoc analysis revision}"
: "${COGARENA_FROZEN_RUNTIME_ROOT:?prepare and inject the frozen-source runtime}"
: "${COGARENA_TARGET_BASELINE_RUNTIME_MANIFEST_SHA256:?inject the runtime-manifest SHA-256}"
AUTHORITY_ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
EXPECTED_RUNTIME="${AUTHORITY_ROOT}/results/causal_selectivity_20260720/runtime/formal_${COGARENA_GIT_HEAD}__analysis_${COGARENA_TARGET_BASELINE_GIT_HEAD}"
[[ "$(realpath "${COGARENA_FROZEN_RUNTIME_ROOT}")" == "$(realpath "${EXPECTED_RUNTIME}")" ]] || {
  echo "FATAL: frozen runtime is not the canonical formal/analysis revision path"
  exit 1
}
[[ -L "${COGARENA_FROZEN_RUNTIME_ROOT}/results" ]] || {
  echo "FATAL: frozen runtime results path is not a symlink"
  exit 1
}
[[ "$(realpath "${COGARENA_FROZEN_RUNTIME_ROOT}/results")" == "$(realpath "${AUTHORITY_ROOT}/results")" ]] || {
  echo "FATAL: frozen runtime results symlink misses authority tree"
  exit 1
}
[[ "$(sha256sum "${COGARENA_FROZEN_RUNTIME_ROOT}/TARGET_BASELINE_RUNTIME_MANIFEST.json" | awk '{print $1}')" == "${COGARENA_TARGET_BASELINE_RUNTIME_MANIFEST_SHA256}" ]] || {
  echo "FATAL: frozen runtime manifest hash mismatch"
  exit 1
}
cd "${COGARENA_FROZEN_RUNTIME_ROOT}"
export COGARENA_ROOT="${COGARENA_FROZEN_RUNTIME_ROOT}"
export COGARENA_ANALYSIS_AUTHORITY_ROOT="${AUTHORITY_ROOT}"
export PYTHONPATH="${COGARENA_FROZEN_RUNTIME_ROOT}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "FATAL: target-baseline analysis must run under Slurm"
  exit 1
}
[[ "$(hostname -s)" == c01* ]] || {
  echo "FATAL: target-baseline analysis must run on c01"
  exit 1
}
srun python -m scripts.experiments.causal_selectivity_20260720.analyze_target_baseline
