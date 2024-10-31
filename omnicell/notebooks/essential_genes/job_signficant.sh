#!/bin/bash
#SBATCH -t 167:00:00          # walltime = 2 hours
#SBATCH --ntasks-per-node=4  # 4 CPU cores
#SBATCH --mem=256GB          # memory per node
hostname

source ~/.bashrc
conda activate sandbox


python -m notebooks.essential_genes.generate_DEGs_per_pert --signifcant

echo "Finished successfully"