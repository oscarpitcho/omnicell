#!/bin/bash
#SBATCH -t 8:00:00          # walltime = 48 hours
#SBATCH -n 4                 # 4 CPU cores
#SBATCH --gres=gpu:1 --constraint=high-capacity  # 1 non-A100 GPU 
#SBATCH --mem=128G           # memory per node
hostname                     # Print the hostname of the compute node

# Define arrays for holdouts and targets



source ~/.bashrc
conda activate sandbox

python generate_evaluations.py --root_dir ./results/satija_IFNB_raw/split_0_ss:ns-10:5/BioBERT

# python generate_evaluations.py --root_dir ./results/nn/kang_ho_CD4T

echo "Job Finished"