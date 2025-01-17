#!/bin/bash
#SBATCH -t 12:00:00          # walltime = 2 hours
#SBATCH --ntasks-per-node=4  # 4 CPU cores
#SBATCH --mem=128GB          # memory per node
#SBATCH --array=0-2         # 3 jobs: one for each model type

hostname

source ~/.bashrc
conda activate sandbox

# Array of model result directories
declare -a MODEL_DIRS=(
    "BioBERT/nearest-neighbor_pert_emb_substitute"
    "nearest-neighbor_gene_dist_substitute"
    "UCE_ESM2_Mean_Imputed/nearest-neighbor_pert_emb_substitute"
)

# Get the current model directory
MODEL_DIR="${MODEL_DIRS[$SLURM_ARRAY_TASK_ID]}"

echo "Processing evaluations for model: ${MODEL_DIR}"

# Generate evaluations
python generate_evaluations.py --root_dir "./results/essential_gene_knockouts_raw/${MODEL_DIR}"

echo "Completed processing for model: ${MODEL_DIR}"