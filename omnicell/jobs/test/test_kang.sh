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
python train.py --task_config "configs/tasks/kang/kang_ho_CD4T_iid.yaml" \
    --model_config "configs/models/test.yaml" -l DEBUG


echo "Running evaluation script"

python generate_evaluations.py --model_name test --task_name "kang_ho_CD4T_iid" 

echo "End of bash script"

