#!/bin/bash
#SBATCH -t 48:00:00          # walltime = 48 hours
#SBATCH -n 4                 # 4 CPU cores
#SBATCH --gres=gpu:1 --constraint=high-capacity  # 1 non-A100 GPU 
#SBATCH --mem=256G           # memory per node
hostname                     # Print the hostname of the compute node

# Define arrays for holdouts and targets



source ~/.bashrc
mamba activate sandbox

# Run the training script
python train.py --task_config configs/tasks/essential_genes/raw/essential_genes_across_cells_basic_ood_no_norm.yaml \
    --model_config configs/models/vae.yaml -l DEBUG

echo "Generating evaluations"

python generate_evaluations.py --model_name vae --task_name essential_genes_across_cells_basic_ood_no_norm

echo "End of bash script"
