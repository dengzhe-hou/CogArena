#!/bin/bash
#SBATCH --job-name=cog-causal-formal
#SBATCH --partition=batch
#SBATCH --nodelist=c04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-11%2
#SBATCH --gres=gpu:pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=results/causal_selectivity_20260720/slurm/formal_%A_%a.log

# FORMAL HARD GATE: preflight refuses execution unless the completed pilot and
# capacity closures remain hash-bound in the frozen specification. The exact
# post-freeze commit is injected at submission rather than embedded circularly.
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
bash scripts/experiments/causal_selectivity_20260720/run_one_model.sh formal "${MODEL}"
