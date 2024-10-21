#!/bin/bash
#SBATCH -t 48:00:00          # walltime = 48 hours
#SBATCH --ntasks-per-node=4  # 4 CPU cores
#SBATCH --gres=gpu:1 --constraint=high-capacity  # 1 non-A100 GPU 
#SBATCH --mem=128G           # memory per node
hostname                     # Print the hostname of the compute node

# Define arrays for holdouts and targets



source ~/.bashrc
conda activate sandbox

# Run the training script
python train.py --data_config configs/splits/satija_raw/satija_across_genes_hvg.yaml --model_config configs/models/flow.yaml \
 --eval_config configs/evals/satija_raw/ev_satija_across_genes_hvg.yaml -l DEBUG


# python generate_evaluations.py --root_dir ./results/nn/satija_across_genes

echo "Job Finished"