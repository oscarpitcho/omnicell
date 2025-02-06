#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -p newnodes
#SBATCH --mem=50GB
#SBATCH --array=0-19 #20 jobs 
echo hostname

source ~/.bashrc
echo "Starting job $SLURM_ARRAY_TASK_ID"
conda activate omnicell
echo "Activated conda environment"

DATASET="PapalexiSatija2021_eccite_arrayed_RNA"
TOTAL_JOBS=20

echo "Running python script"

python -m scripts.compute_DEGs.get_pert_counts_per_cell_type \
    --dataset "${DATASET}" \
    --job_id "${SLURM_ARRAY_TASK_ID}" \
    --total_jobs "${TOTAL_JOBS}"

echo "Job finished"
