#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -n 4
#SBATCH --mem=250GB
#SBATCH --gres=gpu:h100:1 # 1 h100 GPU
#SBATCH -p ou_bcs_low
#SBATCH --array=0-1 # 2 splits = 2 combinations
hostname

# ===== CONFIGURATION =====
DATASET="repogle_k562_essential_raw"
CONFIG_BASE_DIR="configs"
ETL_CONFIG="${CONFIG_BASE_DIR}/ETL/no_preprocs.yaml"
SPLIT_BASE_DIR="${CONFIG_BASE_DIR}/splits/${DATASET}/random_splits/rs_accP_k562_ood_ss:ns_20_2_most_pert_0.1"
MODEL_CONFIG="${CONFIG_BASE_DIR}/models/scot/proportional_scot.yaml"

# Define splits array
SPLITS=(0 1)

# Get current split
split_idx=${SLURM_ARRAY_TASK_ID}
SPLIT_DIR="split_${SPLITS[split_idx]}"

source ~/.bashrc
conda activate omnicell

echo "Processing split: ${SPLIT_DIR}"

# Run synthetic data generation
python -m scripts.synthetic_data.generate_synthetic_proportional \
--dataset ${DATASET} \
--etl_config ${ETL_CONFIG} \
--datasplit_config ${SPLIT_BASE_DIR}/${SPLIT_DIR}/split_config.yaml \
--model_config ${MODEL_CONFIG}

echo "Job finished for split ${SPLIT_DIR}"