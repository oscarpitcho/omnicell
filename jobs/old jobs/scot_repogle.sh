#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -n 4
#SBATCH --mem=250GB
#SBATCH --gres=gpu:h100:1  # 1 h100 GPU
#SBATCH -p ou_bcs_low
#SBATCH --array=0-3        # 1 ETL x 2 Models x 2 Splits = 4 combinations

hostname

CONFIG_BASE_DIR="configs"
ETL_BASE_DIR="configs/ETL"
MODEL_BASE_DIR="${CONFIG_BASE_DIR}/models"

# ===== CONFIGURATION =====
DATASET="repogle_k562_essential_raw"
SPLIT_BASE_DIR="${CONFIG_BASE_DIR}/splits/${DATASET}/random_splits/rs_accP_k562_ood_ss:ns_20_2_most_pert_0.1"

# Define configs and splits
ETL_CONFIGS=("no_preproc_drop_unmatched")
MODELS=("scot" "proportional_scot")
SPLITS=(0 1)

# Calculate indices
total_splits=${#SPLITS[@]}
model_idx=$((SLURM_ARRAY_TASK_ID / total_splits))
split_idx=$((SLURM_ARRAY_TASK_ID % total_splits))

# Get current config, model and split
ETL_CONFIG="${ETL_BASE_DIR}/${ETL_CONFIGS[0]}.yaml"
MODEL_NAME="${MODELS[$model_idx]}"
MODEL_CONFIG="${MODEL_BASE_DIR}/scot/${MODEL_NAME}.yaml"
SPLIT_DIR="split_${split_idx}"

source ~/.bashrc
conda activate omnicell

echo "Processing ETL config: ${ETL_CONFIG}"
echo "Processing model: ${MODEL_NAME}"
echo "Processing split: ${SPLIT_DIR}"

# Run training
python train.py \
    --etl_config ${ETL_CONFIG} \
    --datasplit_config ${SPLIT_BASE_DIR}/${SPLIT_DIR}/split_config.yaml \
    --eval_config ${SPLIT_BASE_DIR}/${SPLIT_DIR}/eval_config.yaml \
    --model_config ${MODEL_CONFIG} \
    --slurm_id ${SLURM_ARRAY_JOB_ID} \
    --slurm_array_task_id ${SLURM_ARRAY_TASK_ID} \
    -l DEBUG

# Generate evaluations
python generate_evaluations.py \
    --root_dir ./results/${DATASET}/${ETL_CONFIGS[0]}/${MODEL_NAME}
    
echo "All jobs finished for ${MODEL_NAME} - ${SPLIT_DIR}"