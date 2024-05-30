#!/bin/bash
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=dlesman@hms.harvard.edu
#SBATCH --mem=64G              # memory per node
#SBATCH --time=04:00:00

source /cm/shared/openmind8/anaconda/3-2022.10/etc/profile.d/conda.sh
cd /orcd/archive/abugoot/001/Projects/dlesman/scripts
conda activate /orcd/archive/abugoot/001/Projects/dlesman/scvi-env 
python eval_script.py -d Satija_TGFB_HVG -l eval_locations.json 