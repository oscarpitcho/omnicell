#!/bin/bash
#SBATCH -t 48:00:00          # walltime = 2 hours
#SBATCH --ntasks-per-node=4  # 4 CPU cores
#SBATCH --mem=256GB          # memory per node
hostname

source ~/.bashrc
conda activate sandbox


python -m notebooks.compute_DEGS.generate_DEGs_per_pert --dataset satija_IFNB_raw

echo "Finished successfully"