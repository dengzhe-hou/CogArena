#!/bin/bash
#SBATCH --job-name=cog-scaffold-rep
#SBATCH --partition=batch
#SBATCH --nodelist=c04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-11%2
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=results/scaffold_wording_replication_20260725/slurm/formal_%A_%a.log

set -euo pipefail
MODELS=(
  qwen2.5:3b qwen2.5:14b
  gemma2:2b gemma2:9b
  llama2:7b llama2:13b
  gemma3:12b gemma3:27b
  falcon3:7b falcon3:10b
  olmo2:7b olmo2:13b
)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
bash scripts/experiments/scaffold_wording_replication_20260725/run_one_model.sh \
  "${MODEL}"
