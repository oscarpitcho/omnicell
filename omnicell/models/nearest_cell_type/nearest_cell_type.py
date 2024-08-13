from pytorch_lightning.callbacks import TQDMProgressBar
from datamodules import FMDataset, fm_collate, CFMDataset, SCFMDataset, cfm_collate, StratifiedBatchSampler
from torch.utils.data import RandomSampler
from sc_etl_utils import *
from arch import *
from flow_utils import compute_conditional_flow
import json

import scvi
import torch
import numpy as np
import pytorch_lightning as pl
import os

import scanpy as sc

from torchcfm.conditional_flow_matching import *
import scanpy as sc
import hashlib

import argparse
parser = argparse.ArgumentParser(description='Fit fm.')
parser.add_argument('--dataset', help='Dataset adata file', type=str, required=True)
parser.add_argument('--control_pert', help='Name of control in perturbation column', type=str, required=True)
parser.add_argument('--holdout_cells', help='Name of hold out cell types in cell type column', nargs='+', required=True)
parser.add_argument('--holdout_perts', help='Name of hold out perturbations in perturbation column', nargs='+', required=True)
parser.add_argument('--cell_col', help='Name of cell type column', type=str, default="cell_type")
parser.add_argument('--pert_col', help='Name of perturbation column', type=str, default="pert_type")
parser.add_argument('-e', '--embedding', help='Name of embedding', type=str, default="standard")

args = parser.parse_args()
print(args)

args.model_type = "nearest_cell_type"

cell_col, pert_col = args.cell_col, args.pert_col
control_pert, holdout_cells, holdout_perts = args.control_pert, args.holdout_cells, args.holdout_perts
dataset = args.dataset
embedding = args.embedding

hash_dir = hashlib.sha256(json.dumps(args.__dict__, sort_keys=True).encode()).hexdigest()
save_path = f"{dataset}/{hash_dir}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(f"{save_path}/config.json", 'w') as f:
    json.dump(args.__dict__, f, indent=2)

print("Loading dataset")
# load some data
adata = sc.read_h5ad(f'/orcd/archive/abugoot/001/Projects/njwfish/datasets/{dataset}.h5ad')

print("Splitting dataset")



def train(config):
    task_config = config['task_config']
    control_pert = task_config['control_pert']
    holdout_cells = task_config['holdout_cells']
    holdout_perts = task_config['holdout_perts']
    cell_col = task_config['cell_col']
    pert_col = task_config['pert_col']
    embedding = task_config['embedding']

    

        
control_idx, pert_idx, eval_idx, eval_cell_idx, eval_pert_idx = get_train_eval_idxs(
    adata, control_pert, holdout_cells, holdout_perts, cell_col=cell_col, pert_col=pert_col
)

pert_ids, pert_mat, cell_types = get_identity_features(
    adata, cell_col=cell_col, pert_col=pert_col
)

standard = adata.X.astype(float)
if hasattr(standard, 'toarray'):
    standard = standard.toarray().astype(np.float32)

adata.obsm["standard"] = standard.astype(np.float32)

X = adata.obsm[embedding]
if hasattr(X, 'toarray'):
    X = adata.obsm[embedding] = X.toarray().astype(np.float32)

control_train, pert_train, pert_ids_train, control_cell_types, pert_cell_types, control_eval, pert_eval, pert_ids_eval = get_train_eval(
    X, pert_ids, cell_types, control_idx, pert_idx, eval_idx, eval_cell_idx, eval_pert_idx
)

print("Computing predictions")

cell_type_names = adata.obs[cell_col]
pert_type_names = adata.obs[pert_col]
# Save the predicted perturbation
control_train_types = np.unique(control_cell_types)
train_means = np.zeros((control_train_types.shape[0], X.shape[1]))
for i, cell_type in enumerate(control_train_types):
    train_means[i] = control_train[control_cell_types == cell_type].mean(axis=0)
    
for cell_type, pert_type in zip(holdout_cells, holdout_perts):
    torch.cuda.empty_cache()
    control_eval = adata.obsm[embedding][cell_type_names == cell_type]
    pert_id = pert_ids[(pert_type_names == pert_type) & (cell_type_names == cell_type)][0]
    control_eval_mean = control_eval.mean(axis=0)
    # todo: add more metrics
    mean_distances = np.mean((train_means - control_eval_mean[None, :])**2, axis=1)
    nearest_cell_type = mean_distances.argmin()
    pred_pert = pert_train[pert_cell_types == nearest_cell_type]
    print(pred_pert.shape)
    
    print(f"Saving {pert_type} predictions")
    np.savez(
        f"{save_path}/pred_{pert_type}_{cell_type}.npz", 
        pred_pert=pred_pert, 
        true_pert=adata.obsm["standard"][(pert_type_names == pert_type) & (cell_type_names == cell_type)], 
        control=adata.obsm["standard"][cell_type_names == cell_type],
        true_pert_embedding=adata.obsm[embedding][(pert_type_names == pert_type) & (cell_type_names == cell_type)], 
        control_embedding=adata.obsm[embedding][cell_type_names == cell_type]
    )