#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -p newnodes
#SBATCH --mem=50GB
#SBATCH --array=0-3 # 4 datasets
hostname

source ~/.bashrc
conda activate omnicell

DATASETS=("satija_IFNB_raw" "repogle_k562_essential_raw" "kang" "essential_gene_knockouts_raw")

python -m scripts.compute_DEGs.combine_parts \
    --dataset "${DATASET}" \
    --job_id "${SLURM_ARRAY_TASK_ID}" \
    --total_jobs "${TOTAL_JOBS}"