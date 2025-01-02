#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -p newnodes
#SBATCH --mem=50GB
#SBATCH --array=0-19 #20 jobs 
hostname

source ~/.bashrc
conda activate omnicell

DATASET="repogle_k562_essential_raw"
TOTAL_JOBS=20

python -m scripts.compute_DEGs.get_pert_counts_per_cell_type \
    --dataset "${DATASET}" \
    --job_id "${SLURM_ARRAY_TASK_ID}" \
    --total_jobs "${TOTAL_JOBS}"