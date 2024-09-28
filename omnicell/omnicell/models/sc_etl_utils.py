import pandas as pd
import numpy as np
from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY
from omnicell.models.datamodules import  SCFMDataset, cfm_collate, StratifiedBatchSampler, ot_collate
import torch


def get_identity_features(adata, cell_type_features=True):
    perts = pd.get_dummies(adata.obs[PERT_KEY]).values.astype(float)
    cell_types = pd.get_dummies(adata.obs[CELL_KEY]).values
    if cell_type_features:
        combo = pd.get_dummies(adata.obs[CELL_KEY].astype(str) + adata.obs[PERT_KEY].astype(str)).values
        idx = (combo!=0).argmax(axis=0)
        pert_mat = np.hstack([cell_types, perts])[idx, :].astype('float32')
    else:
        combo = perts
        idx = (combo!=0).argmax(axis=0)
        pert_mat = perts[idx, :].astype('float32')
    
    pert_ids = combo.argmax(axis=1)
    cell_types = cell_types.argmax(axis=1)
    return pert_ids, pert_mat, cell_types


def get_dataloader(
        adata, batch_size=512, embedding="standard", verbose=0, pert_reps=None, pert_ids=None
):
        control_idx = adata.obs[PERT_KEY] == CONTROL_PERT
        pert_idx = adata.obs[PERT_KEY] != CONTROL_PERT
        cell_types = adata.obs[CELL_KEY].values

        if pert_reps is None:
            pert_ids, pert_reps, cell_types = get_identity_features(adata)

        X = adata.obsm[embedding]

        control_train = X[control_idx]
        pert_train = X[pert_idx]
        pert_ids_train =  pert_ids[pert_idx]
        control_cell_types = cell_types[control_idx]
        pert_cell_types = cell_types[pert_idx]
        
        dset = SCFMDataset(
            control_train, pert_train, 
            pert_ids_train, pert_reps, 
            control_cell_types, pert_cell_types, size=X.shape[0]
        )
        ns = np.array([[t.shape[0] for t in ts] for ts in dset.target])
        dl = torch.utils.data.DataLoader(
            dset, collate_fn=ot_collate, 
            batch_sampler=StratifiedBatchSampler(
                ns=ns, batch_size=batch_size
            )
        )
        return dl