#!/bin/bash
#SBATCH -t 48:00:00          # walltime = 48 hours
#SBATCH -n 4                 # 4 CPU cores
#SBATCH --gres=gpu:1 --constraint=high-capacity  # 1 non-A100 GPU 
#SBATCH --mem=128G           # memory per node
hostname                     # Print the hostname of the compute node

# Define arrays for holdouts and targets



source ~/.bashrc
mamba activate sandbox

# Run the training script
python train.py --task_config "configs/tasks/satija_ho_A549_basic.yaml" \
    --model_config "configs/models/vae.yaml" -l DEBUG


python generate_evaluations.py --model_name "vae" --task_name "satija_ho_A549_basic" 

