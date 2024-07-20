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

from Encode import Encode
from Decode import Decode

import argparse
parser = argparse.ArgumentParser(description='Fit fm.')
parser.add_argument('-b','--batch_size',help='Batch size', type=int, default=32)
parser.add_argument('-m','--max_epochs',help='Max epochs', type=int, default=100)
parser.add_argument('-k', '--model_kwargs', help='Json formatted dict of model kwargs', type=json.loads)
parser.add_argument('--dataset', help='Dataset adata file', type=str, required=True)
parser.add_argument('--control_pert', help='Name of control in perturbation column', type=str, required=True)
parser.add_argument('--holdout_cells', help='Name of hold out cell types in cell type column', nargs='+', required=True)
parser.add_argument('--holdout_perts', help='Name of hold out perturbations in perturbation column', nargs='+', required=True)
parser.add_argument('--cell_col', help='Name of cell type column', type=str, default="cell_type")
parser.add_argument('--pert_col', help='Name of perturbation column', type=str, default="pert_type")
parser.add_argument('-s', help='Stratify sample', action='store_true')
parser.add_argument('--exclude_ct', help='Exclude cell type from conditioning features', action='store_true')
parser.add_argument('-e', '--embedding', help='Name of embedding', type=str, default="vae")
parser.add_argument('-a', '--arch', help='Name of arch', type=str, default="cmlp")
parser.add_argument('--fm', help='Type of flow matching', type=str, default="dcfm")

args = parser.parse_args()
print(args)

args.model_type = "latent_flow"
batch_size = args.batch_size
max_epochs = args.max_epochs
cell_col, pert_col = args.cell_col, args.pert_col
control_pert, holdout_cells, holdout_perts = args.control_pert, args.holdout_cells, args.holdout_perts
embedding = args.embedding
strat = args.s
dataset = args.dataset # 'Seurat_object_TGFB_Perturb_seq.h5ad'
arch = args.arch
cell_type_features = not args.exclude_ct
model_kwargs = args.model_kwargs

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
control_idx, pert_idx, eval_idx, eval_cell_idx, eval_pert_idx = get_train_eval_idxs(
    adata, control_pert, holdout_cells, holdout_perts, cell_col=cell_col, pert_col=pert_col
)

pert_ids, pert_mat, cell_types = get_identity_features(
    adata, cell_col=cell_col, pert_col=pert_col, cell_type_features=cell_type_features
)

standard = adata.X.astype(float)
if hasattr(standard, 'toarray'):
    standard = standard.toarray().astype(np.float32)

adata.obsm["standard"] = standard.astype(np.float32)

X = adata.obsm["standard"]
if hasattr(X, 'toarray'):
    X = adata.obsm["standard"] = X.toarray().astype(np.float32)

inp_mean, inp_var = Encode(X, model_name=f"{dataset}_vae")
X = adata.obsm[embedding] = np.hstack([inp_mean, inp_var])

control_train, pert_train, pert_ids_train, control_cell_types, pert_cell_types, control_eval, pert_eval, pert_ids_eval = get_train_eval(
    X, pert_ids, cell_types, control_idx, pert_idx, eval_idx, eval_cell_idx, eval_pert_idx
)

control_train, control_train_var = control_train[:, :inp_mean.shape[1]], control_train[:, inp_mean.shape[1]:]
pert_train, pert_train_var = pert_train[:, :inp_mean.shape[1]], pert_train[:, inp_mean.shape[1]:]

control_eval, control_eval_var = control_eval[:, :inp_mean.shape[1]], control_eval[:, inp_mean.shape[1]:]
pert_eval, pert_eval_var = pert_eval[:, :inp_mean.shape[1]], pert_eval[:, inp_mean.shape[1]:]

X = X[:, :inp_mean.shape[1]]

print("Creating dataloader")
if strat:
    # set up data processing for cfm
    dset = SCFMDataset(
        control_train, pert_train, 
        pert_ids_train, pert_mat, 
        control_cell_types, pert_cell_types,
        batch_size=batch_size, size=X.shape[0]
    )
    dl = torch.utils.data.DataLoader(
        dset, collate_fn=cfm_collate, 
        batch_sampler=StratifiedBatchSampler(
            RandomSampler(dset), batch_size=batch_size, drop_last=True, 
            probs=dset.probs, num_strata=dset.num_strata
        )
    )
else:
    dset = CFMDataset(
        control_train, pert_train, 
        pert_ids_train, pert_mat, 
        size=X.shape[0]
    )
    dl = torch.utils.data.DataLoader(dset, batch_size=batch_size, collate_fn=cfm_collate)


print("Training model")
# Train the model
trainer = pl.Trainer(
    accelerator='gpu', devices=1,  # Specify the number of GPUs to use
    max_epochs=max_epochs,  # Specify the maximum number of training epochs
    default_root_dir=save_path,
    callbacks=[TQDMProgressBar(refresh_rate=100)]
)

if arch.lower() == 'cmlp':
    model = CMLP(training_module=CFM, feat_dim=X.shape[1], cond_dim=pert_mat.shape[1], time_varying=True, **model_kwargs)
elif arch.lower() == 'cmha':
    model = CMHA(training_module=CFM, feat_dim=X.shape[1], cond_dim=pert_mat.shape[1], time_varying=True, **model_kwargs)
else:
    raise NotImplemented
    
    
trainer.fit(model, dl)

print("Computing predictions")

cell_type_names = adata.obs[cell_col]
pert_type_names = adata.obs[pert_col]
# Save the predicted perturbation
for cell_type, pert_type in zip(holdout_cells, holdout_perts):
    torch.cuda.empty_cache()
    control_eval = adata.obsm[embedding][cell_type_names == cell_type][:, :inp_mean.shape[1]]
    control_eval_var = adata.obsm[embedding][cell_type_names == cell_type][:, inp_mean.shape[1]:]
    pert_id = pert_ids[(pert_type_names == pert_type) & (cell_type_names == cell_type)][0]
    traj = compute_conditional_flow(
        model, 
        control_eval, 
        np.repeat(pert_id, control_eval.shape[0]), 
        pert_mat,
        n_batches = 10
    )  
    print(f"Saving {pert_type} predictions")
    np.savez(
        f"{save_path}/pred_{pert_type}_{cell_type}.npz", 
        pred_pert=Decode(traj[-1, :, :], control_eval_var[:traj.shape[1], :], model_name=f"{dataset}_vae"), 
        true_pert=adata.obsm["standard"][(pert_type_names == pert_type) & (cell_type_names == cell_type)], 
        control=adata.obsm["standard"][cell_type_names == cell_type],
        true_pert_embedding=adata.obsm[embedding][(pert_type_names == pert_type) & (cell_type_names == cell_type)], 
        control_embedding=adata.obsm[embedding][cell_type_names == cell_type]
    )
    del traj