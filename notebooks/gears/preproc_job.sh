#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_low
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=200GB
echo hostname

source ~/.bashrc
conda activate omnicell


python -m notebooks.gears.test_preproc_script

echo "Job finished"