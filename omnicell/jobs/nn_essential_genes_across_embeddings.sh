#!/bin/bash
#SBATCH -t 8:00:00          # walltime = 8 hours
#SBATCH --ntasks-per-node=4  # 4 CPU cores
#SBATCH --gres=gpu:1 --constraint=high-capacity  # 1 non-A100 GPU 
#SBATCH --mem=200GB          # memory per node
#SBATCH --array=0-9          # Job array with indices 0 to 4 (for 5 splits)

hostname                     # Print the hostname of the compute node

source ~/.bashrc
conda activate sandbox

# Define the base directory for configs
CONFIG_BASE_DIR="configs"
SPLIT_BASE_DIR="${CONFIG_BASE_DIR}/essential_gene_knockouts_raw/random_splits/acrossP_ood_ss:ns-20:10"

# Use the SLURM_ARRAY_TASK_ID to select the split
SPLIT_DIR="split_${SLURM_ARRAY_TASK_ID}"

# Define the combinations of models and embeddings
declare -A MODEL_EMBEDDING_COMBOS
MODEL_EMBEDDING_COMBOS=(
    ["nearest-neighbor/nearest-neighbor_pert_space_substitute"]="preprocess_and_UCE_embedding preprocess_and_BioBert"
    ["nearest-neighbor/nearest-neighbor_gene_dist_substitute"]="preprocess_no_embedding"
)

# Function to run a job
run_job() {
    local MODEL=$1
    local EMBEDDING=$2
    
    ETL_CONFIG="${CONFIG_BASE_DIR}/ETL/${EMBEDDING}.yaml"
    MODEL_CONFIG="${CONFIG_BASE_DIR}/models/${MODEL}.yaml"
    
    echo "Running job for split ${SLURM_ARRAY_TASK_ID} with embedding: ${EMBEDDING} and model: ${MODEL}"

    python train.py --etl_config ${ETL_CONFIG} \
     --datasplit_config ${SPLIT_BASE_DIR}/${SPLIT_DIR}/split_config.yaml \
     --eval_config ${SPLIT_BASE_DIR}/${SPLIT_DIR}/eval_config.yaml \
     --model_config ${MODEL_CONFIG} -l DEBUG


    echo "Finished job for split ${SLURM_ARRAY_TASK_ID} with embedding: ${EMBEDDING} and model: ${MODEL}"
}

# Loop through the combinations
for MODEL in "${!MODEL_EMBEDDING_COMBOS[@]}"; do
    for EMBEDDING in ${MODEL_EMBEDDING_COMBOS[$MODEL]}; do
        run_job "$MODEL" "$EMBEDDING"
        python generate_evaluations.py --root_dir ./results/essential_gene_knockouts_raw/
    done
done


echo "All jobs finished for Split ${SLURM_ARRAY_TASK_ID}"