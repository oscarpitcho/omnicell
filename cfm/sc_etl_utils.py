import pandas as pd
import numpy as np

def get_train_eval_idxs(
    adata, control_pert, holdout_cells, holdout_perts, 
    cell_col='cell_type', pert_col='pert_type', verbose=2
):
    # here we set up the train/eval and control/pert sets
    # set the idx of the controls
    control_idx = adata.obs[pert_col] == control_pert
    # set the idx of the perts (currently just "all not control")
    pert_idx = adata.obs[pert_col] != control_pert
    # set the hold out cell-type/pert
    eval_cell_idx = adata.obs[cell_col].isin(holdout_cells)
    eval_pert_idx = adata.obs[pert_col].isin(holdout_perts)
    eval_idx = eval_cell_idx & eval_pert_idx
    if verbose > 0:
        print(f"Controls: {(control_idx & ~eval_idx).sum()}, Perturbations: {(pert_idx & ~eval_idx).sum()},  Eval: {eval_idx.sum()}")
    return control_idx, pert_idx, eval_idx, eval_cell_idx, eval_pert_idx

def get_identity_features(adata, cell_col='cell_type', pert_col='perturb', cell_type_features=True):
    perts = pd.get_dummies(adata.obs[pert_col]).values.astype(float)
    cell_types = pd.get_dummies(adata.obs[cell_col]).values
    if cell_type_features:
        combo = pd.get_dummies(adata.obs[cell_col].astype(str) + adata.obs[pert_col].astype(str)).values
        idx = (combo!=0).argmax(axis=0)
        pert_mat = np.hstack([cell_types, perts])[idx, :].astype('float32')
    else:
        combo = perts
        idx = (combo!=0).argmax(axis=0)
        pert_mat = perts[idx, :].astype('float32')
    
    pert_ids = combo.argmax(axis=1)
    cell_types = cell_types.argmax(axis=1)
    return pert_ids, pert_mat, cell_types

def get_train_eval(
    X, pert_ids, cell_types, control_idx, pert_idx, eval_idx, eval_cell_idx, eval_pert_idx
):
    # set train and eval split
    control_train = X[control_idx & ~eval_idx]
    pert_train = X[pert_idx & ~eval_idx]
    pert_ids_train =  pert_ids[pert_idx & ~eval_idx]
    control_cell_types = cell_types[control_idx & ~eval_idx]
    pert_cell_types = cell_types[pert_idx & ~eval_idx]

    control_eval = X[control_idx & eval_cell_idx]
    pert_eval = X[eval_idx]
    pert_ids_eval = pert_ids[eval_idx]
    return control_train, pert_train, pert_ids_train, control_cell_types, pert_cell_types, control_eval, pert_eval, pert_ids_eval