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
python train.py --etl_config configs/kang/ETL/normalize_and_log1p.yaml \
 --datasplit_config configs/kang/splits/ho_CD4T.yaml \
 --eval_config configs/kang/evals/ev_CD4T.yaml \
 --model_config configs/models/vae.yaml -l DEBUG


python generate_evaluations.py --root_dir ./ho_CD4T

echo "Job Finished"