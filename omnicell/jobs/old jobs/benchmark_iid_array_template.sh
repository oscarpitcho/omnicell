#!/bin/bash
#SBATCH -t 48:00:00          # walltime = 48 hours
#SBATCH -n 4                 # 4 CPU cores
#SBATCH --gres=gpu:1 --constraint=high-capacity  # 1 non-A100 GPU 
#SBATCH --array=0-17         # 18 jobs in total (6 holdouts * 3 targets)

hostname                     # Print the hostname of the compute node

# Define arrays for holdouts and targets
holdouts=('A549_IFNB' 'BXPC3_IFNB' 'HAP1_IFNB' 'HT29_IFNB' 'K562_IFNB' 'MCF7_IFNB')
targets=('IFNAR2' 'IFNAR1' 'USP18')

# Calculate the current holdout and target based on the array task ID
holdout_index=$((SLURM_ARRAY_TASK_ID / 3))
target_index=$((SLURM_ARRAY_TASK_ID % 3))

holdout=${holdouts[$holdout_index]}
target=${targets[$target_index]}

echo "Processing holdout: $holdout, target: $target"

source ~/.bashrc
mamba activate cellot

# Run the training script
python train.py --outdir "./results/satja/model-cellot/IFNB/holdout_${holdout}/targ_${target}" \
    --config ./configs/tasks/satija_IFNB_iid.yaml \
    --config ./configs/models/cellot.yaml \
    --config.datasplit.holdout $holdout \
    --config.data.target $target

# Run the evaluation script
# Note it's iid training ood evaluation, only the target pert + ce;; is held out
python evaluate.py --outdir "./results/satja/model-cellot/IFNB/holdout_${holdout}/targ_${target}" \
    --setting ood