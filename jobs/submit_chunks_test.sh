#!/bin/bash
#SBATCH --job-name=plasticc_gp_test
#SBATCH --array=0-20                  # just first 21 chunks for now
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/chunk_test_%a.out
#SBATCH --error=logs/chunk_test_%a.err
#SBATCH --partition=regularshort

# ── environment ──────────────────────────────────────────────────────────────
source ~/miniconda/etc/profile.d/conda.sh
conda activate project

mkdir -p logs

echo "Starting chunk $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"

scripts/run_chunk_test \
    --chunk $SLURM_ARRAY_TASK_ID \
    --num-chunks 500

echo "Finished chunk $SLURM_ARRAY_TASK_ID at $(date)"