#!/bin/bash
#SBATCH -t 8:00:00          # walltime = 48 hours
#SBATCH -n 4                 # 4 CPU cores
#SBATCH --gres=gpu:1 --constraint=high-capacity  # 1 non-A100 GPU 
#SBATCH --mem=256G           # memory per node
hostname                     # Print the hostname of the compute node

# Define arrays for holdouts and targets



source ~/.bashrc
conda activate sandbox

python train.py --etl_config configs/ETL/preprocess_no_embedding.yaml \
 --model_config configs/models/nearest-neighbor/nearest-neighbor_pert_emb_substitute_pca.yaml \
 --datasplit_config configs/satija_IFNB_raw/random_splits/acrossC_ood_ss\:10/split_A549/split_config.yaml \
 --eval_config configs/satija_IFNB_raw/random_splits/acrossC_ood_ss\:10/split_A549/eval_config.yaml \
 -l DEBUG

 


python generate_evalutions.py --root_dir ./results/nn/satija_across_genes

