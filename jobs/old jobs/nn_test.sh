#!/bin/bash
#SBATCH -t 48:00:00          # walltime = 48 hours
#SBATCH --ntasks-per-node=4  # 4 CPU cores
#SBATCH --gres=gpu:1 --constraint=high-capacity  # 1 non-A100 GPU 
#SBATCH --mem=200G           # memory per node
hostname                     # Print the hostname of the compute node

# Define arrays for holdouts and targets



source ~/.bashrc
conda activate sandbox


python train.py \
 --etl_config configs/ETL/preprocess_and_BioBert.yaml \
 --datasplit_config configs/satija_IFNB_raw/random_splits/acrossP_ood_ss:ns-10:5/split_0/split_config.yaml \
 --eval_config configs/satija_IFNB_raw/random_splits/acrossP_ood_ss:ns-10:5/split_0/eval_config.yaml \
 --model_config configs/models/nearest-neighbor/nearest-neighbor_pert_emb_substitute.yaml -l DEBUG

# python generate_evaluations.py --root_dir ./results/nn/satija_across_genes

echo "Job Finished"