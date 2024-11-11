#!/bin/bash
#SBATCH -t 48:00:00          # walltime = 48 hours
#SBATCH --ntasks-per-node=4  # 4 CPU cores
#SBATCH --mem=512GB          # memory per node
#SBATCH --array=0-4          # Job array with indices 0 to 4 (5 splits)

hostname                     # Print the hostname of the compute node

# ===== CONFIGURATION =====
# Set your config paths here
PREPROCESSING_CONFIG="configs/ETL/preprocess_no_embedding.yaml"
MODEL_CONFIG="configs/models/nearest-neighbor/nearest-neighbor_gene_dist_substitute.yaml"
# =======================

source ~/.bashrc
conda activate sandbox

# Define the base directory for configs
CONFIG_BASE_DIR="configs"
SPLIT_BASE_DIR="${CONFIG_BASE_DIR}/satija_IFNB_raw/random_splits/acrossP_ood_ss:ns-10:5"

# Use the SLURM_ARRAY_TASK_ID directly for split number
SPLIT_DIR="split_${SLURM_ARRAY_TASK_ID}"

echo "Processing split: ${SPLIT_DIR}"
echo "Using preprocessing config: ${PREPROCESSING_CONFIG}"
echo "Using model config: ${MODEL_CONFIG}"

# Run training
python train.py --etl_config ${PREPROCESSING_CONFIG} \
               --datasplit_config ${SPLIT_BASE_DIR}/${SPLIT_DIR}/split_config.yaml \
               --eval_config ${SPLIT_BASE_DIR}/${SPLIT_DIR}/eval_config.yaml \
               --model_config ${MODEL_CONFIG} -l DEBUG

# Generate evaluations
python generate_evaluations.py \
    --root_dir ./results/satija_IFNB_raw/nearest-neighbor_gene_dist_substitute/ \

echo "All jobs finished for ${SPLIT_DIR}"