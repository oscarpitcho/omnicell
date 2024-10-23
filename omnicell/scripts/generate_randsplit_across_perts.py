import scanpy as sc
import numpy as np
import argparse
from omnicell.data.catalogue import Catalogue
import yaml
import os


def template_split_config(name, mode, dataset_name, holdout_cells, holdout_perts) -> str:
   return {
        "name": name,
        "mode": mode,
        "dataset_name": dataset_name,
        "holdout_cells": holdout_cells,
        "holdout_perts": holdout_perts,
        }

def template_eval_config(name, dataset, evaluation_targets) -> str:
   
   return {
        "name": name,
        "dataset": dataset,
        "evaluation_targets": evaluation_targets,
        }



#Do we handle the across

def main():
    parser = argparse.ArgumentParser(description='Analysis settings.')
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument('--split_mode', type=str, default='ood', help='Mode of the split config')
    parser.add_argument('--split_size', type=int, help='Size of the split, # of perts for evaluation')
    parser.add_argument('--number_splits', type=int, help='Number of splits')
    parser.add_argument('--target_cell', type=str, default='MAX', help='Target cell for the split, if MAX then the cell with the most observations is selected')


    args = parser.parse_args()





    catalogue = Catalogue("configs/catalogue")

    ds_details = catalogue.get_dataset_details(args.dataset)
    
    ds_path = ds_details.path

    print(f"Loading dataset from {ds_path}")
    with open(ds_path, 'rb') as f:
        adata = sc.read_h5ad(f)
    


    
    perts = [x for x in adata.obs[ds_details.pert_key].unique() if x != ds_details.control]
    cells = adata.obs[ds_details.cell_key].unique()

    print(f"Loaded dataset with {len(perts)} non control perts")
    



    path = f"configs/{args.dataset}/random_splits/acrossP_{args.split_mode}_ss:ns-{args.split_size}:{args.number_splits}"



    target_cell = args.target_cell

    #Select the cell type with the most observations
    if target_cell == "MAX":
        target_cell = adata.obs[ds_details.cell_key].value_counts().idxmax()

    else:
        target_cell = str(target_cell)

    print(f"Selected cell {target_cell}")

    for i in range(args.number_splits):

        #Select a random subset of perturbations
        perts = [str(x) for x in np.random.choice(perts, args.split_size, replace=False)]

        #Create the split config

        split_config = template_split_config(f"split_{i}_ss:ns-{args.split_size}:{args.number_splits}", args.split_mode, args.dataset, [], perts)


        eval_config = template_eval_config(f"eval_{i}_ss:ns-{args.split_size}:{args.number_splits}", args.dataset, [[str(target_cell), str(pert)] for pert in perts])

        path_split = f"{path}/split_{i}"

        #make the directory
        os.makedirs(path_split, exist_ok=True)

        #Save the split config
        with open(f"{path_split}/split_config.yaml", 'w+') as f:
            yaml.dump(split_config, f)
        
        #Save the eval config
        with open(f"{path_split}/eval_config.yaml", 'w+') as f:
            yaml.dump(eval_config, f)




        #Save the configs







    #Saving the configs

    #If we holdout cells we need to make sure that we evaluate across several perts on the heldout cells

















   


if __name__ == "__main__":
    main()