import pandas as pd
import numpy as np
import scanpy as sc
from typing import List

from ..constants import PERT_KEY, CELL_TYPE_KEY, CONTROL_PERT


"""


                Task Definition



Predict
    \
     \           Pert 1                        Pert 2
      \
   On  \
        \
         \__________________________________________________________
         |                              |                           |
         |                              | • We see Pert 2           |
         |  Same logic as bottom right  |   at training but         |
  Cell A |  Quadrant                    |   never on cell A         |
         |                              | • We never see Pert 2     |
         |                              |   at training             |
         |                              |                           |
         |______________________________|___________________________|
         |                              |                           |
         | • We see unperturbed         | • We never see the        |
         |   Cell B at training         |   combination Cell B +    |
         | • We never see Cell B        |   Gene 2 at training but  |
  Cell B |   at training                |   we see them separately  |
         |                              | • We see Unperturbed      |
         |                              |   Cell B but never see    |
         |                              |   Gene 2 (and reverse)    |
         |                              | • We never see Pert 1     |
         |                              |   or Cell B               |
         |                              |                           |
         |______________________________|___________________________|

Bullet points are sorted by increasing difficulty


"""

def generate_split_across_genes(adata: sc.AnnData, holdout_perts: str | List[str], allow_in_distribution: bool, ctrl_frac_eval: float = 0.1):
    """
    Generate a split across genes for the holdout perturbations

    Perturbations that are not in the holdout set are in the training set.
    ctrl_frac_eval is the fraction of the control perturbations to be used for evaluation

    returns - AnnData objects:
      train_pert, ctrl_train, ctrl_eval, holdouts
    """

    holdout_perts = [holdout_perts] if isinstance(holdout_perts, str) else holdout_perts
    
    
    holdout_idx = adata.obs[PERT_KEY].isin(holdout_perts)
    ctrl_idx = adata.obs[PERT_KEY] == CONTROL_PERT

    ctrl_eval_idx = ctrl_idx & np.random.choice([True, False], size=ctrl_idx.shape, p=[ctrl_frac_eval, 1-ctrl_frac_eval])
    ctrl_train_idx = ctrl_idx & ~ctrl_eval_idx

    train_pert_idx = ~holdout_idx & ~ctrl_eval_idx

    return adata[train_pert_idx], adata[ctrl_train_idx], adata[ctrl_eval_idx], adata[holdout_idx]


def generate_random_split_across_genes(adata, folds = 5, ctrl_frac_eval = 0.1):
    """
    Generate a random split across genes for the holdout perturbations

    Folds are generated with replacement

    returns - Array which contains a dict for each fold, with keys:
        {
            "fold": heldout_perts,
            "data": [train_pert, ctrl_train, ctrl_eval, holdouts]
        }
       
    """

    perturbations = adata.obs[PERT_KEY].unique()

    folds = [np.random.choice(perturbations, size=int(len(perturbations) / folds), replace=False) for _ in range(folds)]

    return [{ 'fold' : fold, 'data' : generate_split_across_genes(adata, fold, ctrl_frac_eval)} for fold in folds]


def generate_splt_across_cells(adata, houldout_cells: str | List[str], allow_in_distribution: bool):

    return













    



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
    eval_idx = control_idx & False

    #What does the zipping do? 
    #This piece of code is in fact useless
    for holdout_cell, holdout_pert in zip(holdout_cells, holdout_perts):
        eval_idx |= (adata.obs[cell_col] == holdout_cell) & (adata.obs[pert_col] == holdout_pert)

    
    eval_cell_idx = adata.obs[cell_col].isin(holdout_cells)
    eval_pert_idx = adata.obs[pert_col].isin(holdout_perts)

    #The & operator is crucial here, evaluation indices are those that are both in the holdout cells and holdout perts
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