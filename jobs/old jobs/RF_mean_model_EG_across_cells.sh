#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -n 1      
#SBATCH --cpus-per-task=8
#SBATCH --mem=900GB
#SBATCH -p ou_bcs_low
#SBATCH --array=0-1        # CHANGE HERE TO MATCH SIZE OF CROSS PRODUCT: 1 Gene Embeddings x 1 Embeding x 2 Splits = 2 combinations 


hostname


CONFIG_BASE_DIR="configs"
ETL_BASE_DIR="configs/ETL"
EMB_BASE_DIR="configs/embeddings"

### CHANGE HERE FOR THE CORRECT MODEL CONFIG ###
MODEL_CONFIG="${CONFIG_BASE_DIR}/models/RF_mean_model.yaml"
MODEL_NAME="RF_mean_model"

### CHANGE HERE TO SELECT ONLY THE RELEVANT ETL CONFIGS UNDER ${ETL_BASE_DIR} ###
ETL_CONFIGS=("no_preproc_drop_unmatched")

### CHANGE HERE TO SELECT ONLY THE RELEVANT EMBEDDING CONFIGS UNDER ${EMB_BASE_DIR} ###
EMB_CONFIGS=("pemb_GenePT")

### CHANGE HERE TO ONLY SELECT ONE OF THE RANDOM SPLITS ###
SPLITS=("hepg2" "jurkat") # 2 splits (0) or (1)





# ===== CONFIGURATION =====
DATASET="essential_gene_knockouts_raw"
SPLIT_NAME="rs_accC_hepg2_jurkat_ood_ss:ns_20_2_most_pert_0.1"
SPLIT_BASE_DIR="${CONFIG_BASE_DIR}/splits/${DATASET}/random_splits/${SPLIT_NAME}"



# Calculate indices for 3 dimensions
total_splits=${#SPLITS[@]}
total_emb=${#EMB_CONFIGS[@]}
total_etl=${#ETL_CONFIGS[@]}

# Calculate indices properly
etl_config_idx=$((SLURM_ARRAY_TASK_ID / (total_emb * total_splits)))
remaining=$((SLURM_ARRAY_TASK_ID % (total_emb * total_splits)))
emb_config_idx=$((remaining / total_splits))
split_idx=$((remaining % total_splits))


# Get current configs and split
ETL_CONFIG="${ETL_BASE_DIR}/${ETL_CONFIGS[$etl_config_idx]}.yaml"
EMB_CONFIG="${EMB_BASE_DIR}/${EMB_CONFIGS[$emb_config_idx]}.yaml"
SPLIT_DIR="split_${SPLITS[$split_idx]}"

ETL_NAME=${ETL_CONFIGS[$etl_config_idx]}
EMBEDDING_NAME=${EMB_CONFIGS[$emb_config_idx]}
SPLIT=${SPLITS[$split_idx]}

echo "Processing:"
echo "- Dataset: ${DATASET}"
echo "- ETL: ${ETL_NAME}"
echo "- Embedding: ${EMBEDDING_NAME}"
echo "- Cell Type: ${SPLIT}"


source ~/.bashrc
conda activate omnicell


# Run training
python train.py \
    --etl_config ${ETL_CONFIG} \
    --datasplit_config ${SPLIT_BASE_DIR}/${SPLIT_DIR}/split_config.yaml \
    --eval_config ${SPLIT_BASE_DIR}/${SPLIT_DIR}/eval_config.yaml \
    --model_config ${MODEL_CONFIG} \
    --embedding_config ${EMB_CONFIG} \
    --slurm_id ${SLURM_ARRAY_JOB_ID} \
    --slurm_array_task_id ${SLURM_ARRAY_TASK_ID} \
    -l DEBUG


# Generate evaluations
python generate_evaluations.py \
    --root_dir ./results/${DATASET}/${EMBEDDING_NAME}/${ETL_NAME}/${MODEL_NAME}/${SPLIT_NAME}/${SPLIT_NAME}-split_${SPLIT}
    
echo "All jobs finished for ${SPLIT_DIR}"