#!/bin/bash
#SBATCH -t 48:00:00          # walltime = 48 hours
#SBATCH -n 4                 # 4 CPU cores
#SBATCH --mem=256G           # memory per node
hostname                     # Print the hostname of the compute node

# Define arrays for holdouts and targets



source ~/.bashrc
mamba activate sandbox

# Run the training script
python train.py --task_config configs/tasks/essential_genes/raw/essential_genes_across_cells_basic_ood.yaml \
    --model_config configs/models/nearest-neighbor/nearest-neighbor_mean_shift.yaml -l DEBUG

echo "Generating evaluations"

python generate_evaluations.py --model_name nearest-neighbor_mean_shift --task_name essential_genes_across_cells_basic_ood

echo "End of bash script"

