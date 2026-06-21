#!/bin/bash
#SBATCH --job-name=cog_rescore
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:50:00
#SBATCH --output=results/reanalysis/A_compute_%j.out
cd "$(cd "$(dirname "$0")"/.. && pwd)" || exit 9
export MPLBACKEND=Agg
echo "=== (1) rescore conf_cal (fixed scorer, stored responses) ==="
python scripts/rescore_conf_cal.py
echo "=== (2) B2 + scaling (corrected conf_cal via override) ==="
python scripts/compute_b2_expanded.py 2>&1 | grep -iE "REPRO 20|EXP 5|n_models_total|within_mean|cross_mean|\"delta\"|p_value|STRONG|NEAR-ZERO|4-scaler|matches_paper"
echo "=== (3) PCA (corrected) ==="
python scripts/reanalysis/pca_partialcorr.py 2>&1 | grep -iE "pc1_variance|\"delta\"|p_value|within|cross|interpretation"
echo "=== (4) regenerate Fig 1/2/S1/S2/S3 (Fig 2/S2/S3 use corrected conf_cal) ==="
python paper/figures/generate_all.py 2>&1 | grep -iE "Figure|corrected|Loaded|Merged"
echo "=== (5) regenerate Fig 3 manifold (corrected) ==="
python paper/figures/fig_manifold.py 2>&1 | tail -1
echo "=== DONE A-compute ==="
