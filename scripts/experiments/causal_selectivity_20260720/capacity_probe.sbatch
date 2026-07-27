#!/bin/bash
#SBATCH --job-name=cog-32b-capacity
#SBATCH --partition=batch
#SBATCH --nodelist=c04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=results/causal_selectivity_20260720/slurm/capacity_%j.log

set -euo pipefail
bash scripts/experiments/causal_selectivity_20260720/run_capacity_probe.sh "${1:?pass pro6000}"
