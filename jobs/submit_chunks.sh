#!/bin/bash
#SBATCH --job-name=plasticc_gp
#SBATCH --array=0-99                  # one job per chunk (adjust if NUM_CHUNKS != 100)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # GP fitting benefits from a few cores
#SBATCH --mem=16G                     # adjust to your dataset size
#SBATCH --time=00:30:00               # adjust — run one chunk manually first to time it
#SBATCH --output=logs/chunk_%a.out
#SBATCH --error=logs/chunk_%a.err
#SBATCH --partition=regularshort    

# ── environment ──────────────────────────────────────────────────────────────
source ~/miniconda/etc/profile.d/conda.sh
conda activate project

mkdir -p logs

echo "Starting chunk $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"

scripts/run_chunk \
    --chunk $SLURM_ARRAY_TASK_ID \
    --num-chunks 100

echo "Finished chunk $SLURM_ARRAY_TASK_ID at $(date)"