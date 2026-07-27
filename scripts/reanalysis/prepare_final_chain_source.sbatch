#!/bin/bash
#SBATCH --job-name=cog-source-freeze
#SBATCH --partition=batch
#SBATCH --nodelist=c01
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/reanalysis/aplus_20260718/source_freeze_%j.log

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
cd "${ROOT}"
export COGARENA_ROOT="${ROOT}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

srun python3 scripts/reanalysis/final_chain_source_manifest.py --write
ONE="$(sha256sum results/reanalysis/aplus_20260718/FINAL_CHAIN_SOURCE_MANIFEST.json | cut -d' ' -f1)"
srun python3 scripts/reanalysis/final_chain_source_manifest.py --write
test "${ONE}" = "$(sha256sum results/reanalysis/aplus_20260718/FINAL_CHAIN_SOURCE_MANIFEST.json | cut -d' ' -f1)"
srun python3 scripts/reanalysis/final_chain_source_manifest.py --verify
srun python3 -m pytest -q tests/test_final_chain_source_manifest.py tests/test_wager_overlay_chain.py
echo "ALL SOURCE FREEZE GATES PASSED"
