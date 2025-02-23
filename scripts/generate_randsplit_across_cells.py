import scanpy as sc
import numpy as np
import argparse
from omnicell.data.catalogue import Catalogue
import yaml
import os
import json
import pandas as pd


def template_split_config(name, mode, dataset_name, holdout_cells, holdout_perts) -> str:
   return {
        "name": name,
        "mode": mode,
        "dataset": dataset_name,
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
    parser.add_argument('--most_perturbative', type=float, default=0.1, help='If set to a fraction then the heldout perts are sampled from the most perturbative perts')
    parser.add_argument('--target_cells', nargs="+", help='Cell type to hold out, if an int we randomly take that many cells, one heldout per split, if a list we hold out those cells each one being a different split, if ALL we do one split per cell')


    args = parser.parse_args()


    ds_details = Catalogue.get_dataset_details(args.dataset)
    
    ds_path = ds_details.path

    print(f"Loading dataset from {ds_path}")

    with open(ds_path, 'rb') as f:
        adata = sc.read_h5ad(f)
    


        
    cells_types = adata.obs[ds_details.cell_key].unique()
   

    if len(args.target_cells) == 1 and args.target_cells[0].upper() == "ALL":
        # If the user passed "ALL" (e.g. --target_cells ALL)
        print("Holding out all cell types")
        target_cells = [str(c) for c in cells_types]
    elif len(args.target_cells) == 1 and args.target_cells[0].isdigit():
        # If the user passed a single integer, e.g. --target_cells 3
        # This means we randomly pick that many cell types.
        target_cell_count = int(args.target_cells[0])
        assert target_cell_count <= len(cells_types), \
            "Number of held-out cells is greater than the number of cell types"
        target_cells = list(np.random.choice(cells_types, target_cell_count, replace=False))
        print(f"Randomly selected {target_cell_count} cell types to hold out: {target_cells}")
    else:
        # Otherwise, assume the user passed explicit cell-type names
        # e.g. --target_cells B T NK
        target_cells = args.target_cells
        assert len(target_cells) <= len(cells_types), \
            "Number of held-out cells is greater than the number of cell types"
        print(f"Explicitly selected cell types to hold out: {target_cells}")

    print(f"Selected cells {target_cells}")

    perts_in_ds = [x for x in adata.obs[ds_details.pert_key].unique() if x != ds_details.control]

    print(f"Loaded dataset with {len(perts_in_ds)} non control perts")
    

    split_name = f"rs_accC_{'_'.join(target_cells)}_{args.split_mode}_ss:ns_{args.split_size}_{len(target_cells)}"
    if args.most_perturbative is not None:
        assert ds_details.precomputed_DEGs is not None, "DEGs must be computed for this option"
        assert 0 < args.most_perturbative < 1, "most_perturbative must be a fraction between 0 and 1"

        split_name += f"_most_pert_{args.most_perturbative}"


    


    splits_path = f"configs/splits/{args.dataset}/random_splits/{split_name}"




    for c in target_cells:

        #Select a random subset of perturbations
        candidate_perts = None
        if args.most_perturbative is not None:
            DEGs_path = f"{ds_details.folder_path}/DEGs.json"

            print()
            DEGs_all = json.load(open(DEGs_path, 'r'))




            #TODO: Standardize this logic, used in both scripts, also used in NN 
            DEGs_target = {}
            for pert in DEGs_all[c]:
                if DEGs_all[c][pert] is not None:
                    df = pd.DataFrame.from_dict(DEGs_all[c][pert], orient='index')
                    df = df[df['pvals_adj'] < 0.05]
                    DEGs_target[pert] = df

            


            number_DEGs_per_pert = {}
            for pert in DEGs_target:
                number_DEGs_per_pert[pert] = len(DEGs_target[pert])
    

            #Select the most perturbative perts based on the number of DEGs
            number_perts = int(len(DEGs_all[c]) * args.most_perturbative)
            candidate_perts = sorted(number_DEGs_per_pert, key=number_DEGs_per_pert.get, reverse=True)[:number_perts]
        
        else:
            candidate_perts = perts_in_ds


        eval_perts = [str(x) for x in np.random.choice(candidate_perts, args.split_size, replace=False)]

        #Create the split config

        #Separating split_name and split number with hyphen ensures the splits will be grouped together (and only together) in the results directory
        split_config = template_split_config(f"{split_name}-split_{c}", args.split_mode, args.dataset, [str(c)], [])


        eval_config = template_eval_config(f"{split_name}-eval_{c}", args.dataset, [[str(c), str(pert)] for pert in eval_perts])

        split_path = f"{splits_path}/split_{c}"

        #make the directory
        os.makedirs(split_path, exist_ok=True)

        print(f"Saving split {c} to {split_path}")
        
        #Save the split config
        with open(f"{split_path}/split_config.yaml", 'w+') as f:
            yaml.dump(split_config, f)
        
        #Save the eval config
        with open(f"{split_path}/eval_config.yaml", 'w+') as f:
            yaml.dump(eval_config, f)





if __name__ == "__main__":
    main()