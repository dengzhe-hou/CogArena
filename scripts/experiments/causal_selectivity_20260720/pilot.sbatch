#!/bin/bash
#SBATCH --job-name=cog-causal-pilot
#SBATCH --partition=batch
#SBATCH --nodelist=c04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2%1
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=results/causal_selectivity_20260720/slurm/pilot_%A_%a.log

set -euo pipefail
MODELS=(qwen2.5:1.5b llama3.2:3b gemma3:1b)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
bash scripts/experiments/causal_selectivity_20260720/run_one_model.sh pilot "${MODEL}"
