import scanpy as sc
import numpy as np
import anndata
import pandas as pd
from anndata import AnnData
import scipy.sparse as sp
import anndata as ad
import os
import argparse
from omnicell.data.catalogue import Catalogue, DatasetDetails


def shift_prediction(control_adata, sum_control, sum_perturbed):
    """
    Distribute the global per-gene difference (sum_diff[g]) across cells in proportion
    to the cell's existing counts for that gene. 
    """


    #Compute sum_diff
    sum_diff =  sum_perturbed - sum_control # shape: (n_genes,)
    print(type(sum_diff))

    sum_diff = np.squeeze(sum_diff)


    new_X = control_adata.X.toarray() # shape: (n_cells, n_genes)

    n_cells, n_genes = new_X.shape

    #For each gene, distribute sum_diff[g] using a single multinomial draw
    for g in range(n_genes):
        diff = int(sum_diff[g])
        if diff == 0:
            continue  

        # Current counts for this gene across cells
        gene_counts = new_X[:, g]

        current_total = gene_counts.sum()
        if current_total <= 0:
            continue

        # Probabilities ~ gene_counts / current_total
        p = gene_counts.astype(np.float64) / current_total.astype(np.float64) 


        # Normalize probabilities explicitly to ensure they sum to 1
        p = p / p.sum()  # Additional normalization step to handle floating point issues



        if diff > 0:
            # We want to add `diff` counts
            draws = np.random.multinomial(diff, p)  # shape: (n_cells,)
            new_X[:, g] = gene_counts + draws
        else:
            # We want to remove `abs(diff)` counts
            amt_to_remove = abs(diff)

            to_remove = min(amt_to_remove, current_total)

            draws = np.random.multinomial(to_remove, p)
            # Subtract, then clamp
            updated = gene_counts - draws
            updated[updated < 0] = 0
            new_X[:, g] = updated


    return new_X


# Load the data
parser = argparse.ArgumentParser(description='Create dataset using Jason Method of Shift distribution')

parser.add_argument('--dataset', type=str, help='Name of the dataset')
parser.add_argument('--job_id', type=int)
parser.add_argument('--total_jobs', type=int)

args = parser.parse_args()


dd = Catalogue.get_dataset_details(args.dataset)



adata = sc.read(dd.path)



unique_genes = np.unique(adata.obs[dd.pert_key])
unique_genes = [g for g in unique_genes if g!= dd.control]
unique_cells = np.unique(adata.obs[dd.cell_key])

list_adata_adjusted = []

for uc in unique_cells:
    print(uc)
    adata_ctrl = adata[(adata.obs[dd.pert_key] == dd.control) & (adata.obs[dd.cell_key] == uc)]

    sum_ctrl = adata_ctrl.X.sum(axis=0)
    sum_ctrl = np.asarray(sum_ctrl).ravel()
    tot_ctrl = sum_ctrl.sum()
    
    for i, ug in enumerate(unique_genes):
        adata_pert = adata[(adata.obs[dd.pert_key] == ug) & (adata.obs[dd.cell_key] == uc)]

        if i % args.total_jobs == args.job_id and adata_pert.shape[0] > 0:
            print(ug)


            sum_pert = adata_pert.X.sum(axis=0)
            sum_pert = np.asarray(sum_pert).ravel()

            tot_pert = sum_pert.sum()
            
             
            # If tot2 == 0 for some reason, avoid division by zero
            if tot_pert > 0:
                sum_pert = ((tot_ctrl / tot_pert) * sum_pert)


            predicted_shift = shift_prediction(adata_ctrl, sum_ctrl, sum_pert)


            adata_shifted = adata_ctrl.copy()
            adata_shifted.X = predicted_shift
            adata_shifted.obs[dd.pert_key] = ug

            list_adata_adjusted.append(adata_shifted)


    





    
    # The second loop (for "NT" vs. "NT") 
    for i, uc in enumerate(unique_cells):
        if i % args.total_jobs == args.job_id:
            print(uc)
            adata_ctrl = adata[(adata.obs[dd.pert_key] == dd.control) & (adata.obs[dd.cell_key] == uc)]

        
            list_adata_adjusted.append(adata_ctrl)
    



adata_combined = ad.concat(list_adata_adjusted)

path = f"{dd.folder_path}/Jason_Shifted"

if not os.path.exists(path):
    os.makedirs(path)


file_name = f"{path}/{args.job_id}_of_{args.total_jobs}.h5ad"

adata.write(file_name)

print(f"Saved {file_name}")


