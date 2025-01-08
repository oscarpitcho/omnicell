#!/usr/bin/env python

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import scanpy as sc
import anndata
import pandas as pd
from scipy.sparse import issparse
import scipy

##############################################
# Helper Classes/Functions for Sparse Batch
##############################################
class SparseSingleCellDataset(Dataset):
    def __init__(self, adata_ctrl, adata_pert, whichpert, row_indices):
        super().__init__()
        self.adata_ctrl = adata_ctrl
        self.adata_pert = adata_pert
        self.whichpert = whichpert
        self.row_indices = row_indices

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, idx):
        i_ = self.row_indices[idx]
        ctrl_row = self.adata_ctrl.X[i_]
        pert_row = self.adata_pert.X[i_]
        wpert = self.whichpert[i_]
        return (ctrl_row, pert_row, wpert)

def collate_fn_sparse(batch):

    ctrl_arrays = []
    pert_arrays = []
    wpert_list = []

    for (ctrl_sp, pert_sp, wpert) in batch:
        # Densify each row (1, G) => (G,)
        if issparse(ctrl_sp):
            ctrl_arr = ctrl_sp.toarray().ravel()
        else:
            ctrl_arr = np.array(ctrl_sp).ravel()

        if issparse(pert_sp):
            pert_arr = pert_sp.toarray().ravel()
        else:
            pert_arr = np.array(pert_sp).ravel()

        np.log1p(ctrl_arr, out=ctrl_arr)
        np.log1p(pert_arr, out=pert_arr)

        ctrl_arrays.append(ctrl_arr)
        pert_arrays.append(pert_arr)
        wpert_list.append(wpert)


    ctrl_mat = np.stack(ctrl_arrays, axis=0)
    pert_mat = np.stack(pert_arrays, axis=0)

    mask_mat = (np.expm1(ctrl_mat) > 0)  

    x_ctrl_log  = torch.from_numpy(ctrl_mat).float()   # (B, G)
    x_pert_log  = torch.from_numpy(pert_mat).float()   # (B, G)
    whichpert_t = torch.tensor(wpert_list).long()      # (B,)
    x_ctrl_mask = torch.from_numpy(mask_mat).bool()    # (B, G)

    return x_ctrl_log, x_pert_log, whichpert_t, x_ctrl_mask

##############################################
# Holdout params
##############################################
#   'cell' => hold out all data for holdout_cell (except NT),
#             then predict holdout_pert for that cell
#   'pert' => hold out a specific perturbation for all cells,
#             then predict holdout_pert for holdout_cell
#   'both' => only hold out the specific combination of (holdout_cell, holdout_pertname)
#
# Regardless of mode, our OOD set is ALWAYS the single combo:
#   (cell_type == holdout_cell) & (gene == holdout_pertname)
#
# Also, we never hold out "NT" data in training.
cell_or_pert_holdout = 'pert'   # 'cell', 'pert', 'both'
holdout_cell = 'A549'
holdout_pertname = 'USP18'  # e.g. 'IFNAR2'
holdout_dataset = '0'

##############################################
# Load "True" data (for final evaluation / DEGs)
##############################################
dataset_true_pert = 'satija_IFN_adjusted_truepert'
dataset_true_ctrl = 'satija_IFN_adjusted_tructrl'
adata_true_pert = sc.read(f'input/{dataset_true_pert}.h5ad')
adata_true_ctrl = sc.read(f'input/{dataset_true_ctrl}.h5ad')

# We create a gene_map with an extra entry for 'NT'
gene_map = {k: i for i, k in enumerate(adata_true_ctrl.var['gene'])}
gene_map = gene_map | {'NT': max(gene_map.values()) + 1}
gene_unmap = {v: k for k, v in gene_map.items()}

holdoutpert = gene_map[holdout_pertname]  # integer index for holdout_pert

adata_true_ctrl.obs_names_make_unique()
adata_true_pert.obs_names_make_unique()


# Subset "true" data for final evaluation
adata_true_ctrl = adata_true_ctrl[(adata_true_ctrl.obs['cell_type'] == holdout_cell) &
                                  (adata_true_ctrl.obs['dataset'] == holdout_dataset)]
adata_true_pert = adata_true_pert[
    (adata_true_pert.obs['gene'] == holdout_pertname) & 
    (adata_true_pert.obs['cell_type'] == holdout_cell) &
    (adata_true_pert.obs['dataset'] == holdout_dataset)
]



def to_dense(X):
    if issparse(X):
        return X.toarray()
    else:
        return np.asarray(X)

def get_eval(true_adata, pred_adata, DEGs, DEG_vals, pval_threshold):
    results_dict = {}
    
    true_mean = to_dense(true_adata.X).mean(axis = 0)
    true_var = to_dense(true_adata.X).var(axis = 0)
    
    pred_mean = to_dense(pred_adata.X).mean(axis = 0)
    pred_var = to_dense(pred_adata.X).var(axis = 0)
    
    true_corr_mtx = np.corrcoef(to_dense(true_adata.X), rowvar=False).flatten()
    true_cov_mtx = np.cov(to_dense(true_adata.X), rowvar=False).flatten()
        
    pred_corr_mtx = np.corrcoef(to_dense(pred_adata.X), rowvar=False).flatten()
    pred_cov_mtx = np.cov(to_dense(pred_adata.X), rowvar=False).flatten()

    results_dict['all_genes_mean_R2'] = scipy.stats.pearsonr(true_mean, pred_mean)[0]**2
    results_dict['all_genes_var_R2'] = scipy.stats.pearsonr(true_var, pred_var)[0]**2
    results_dict['all_genes_mean_MSE'] = (np.square(true_mean - pred_mean)).mean(axis=0)
    results_dict['all_genes_var_MSE'] = (np.square(true_var - pred_var)).mean(axis=0)
   
    corr_nas = np.logical_or(np.isnan(true_corr_mtx), np.isnan(pred_corr_mtx))
    cov_nas = np.logical_or(np.isnan(true_cov_mtx), np.isnan(pred_cov_mtx))
        
    results_dict['all_genes_corr_mtx_R2'] = scipy.stats.pearsonr(
        true_corr_mtx[~corr_nas], pred_corr_mtx[~corr_nas]
    )[0]**2
    results_dict['all_genes_cov_mtx_R2'] = scipy.stats.pearsonr(
        true_cov_mtx[~cov_nas], pred_cov_mtx[~cov_nas]
    )[0]**2
    results_dict['all_genes_corr_mtx_MSE'] = (
        np.square(true_corr_mtx[~corr_nas] - pred_corr_mtx[~corr_nas])
    ).mean(axis=0)
    results_dict['all_genes_cov_mtx_MSE'] = (
        np.square(true_cov_mtx[~cov_nas] - pred_cov_mtx[~cov_nas])
    ).mean(axis=0)

    significant_DEGs = DEGs[DEGs['pvals_adj'] < pval_threshold]
    num_DEGs = len(significant_DEGs)
    DEG_vals.insert(0, num_DEGs)
    
    for val in DEG_vals:
        if (val == 0) or (val > num_DEGs):
            results_dict[f'Top_{val}_DEGs_mean_R2'] = None
            results_dict[f'Top_{val}_DEGs_var_R2'] = None
            results_dict[f'Top_{val}_DEGs_mean_MSE'] = None
            results_dict[f'Top_{val}_DEGs_var_MSE'] = None
            results_dict[f'Top_{val}_DEGs_corr_mtx_R2'] = None
            results_dict[f'Top_{val}_DEGs_cov_mtx_R2'] = None
            results_dict[f'Top_{val}_DEGs_corr_mtx_MSE'] = None
            results_dict[f'Top_{val}_DEGs_cov_mtx_MSE'] = None
        else:
            top_DEGs = significant_DEGs.iloc[:val].index
        
            true_mean = to_dense(true_adata[:, top_DEGs].X).mean(axis = 0)
            true_var = to_dense(true_adata[:, top_DEGs].X).var(axis = 0)
            true_corr_mtx = np.corrcoef(to_dense(true_adata[:, top_DEGs].X), rowvar=False).flatten()
            true_cov_mtx = np.cov(to_dense(true_adata[:, top_DEGs].X), rowvar=False).flatten()

            pred_mean = to_dense(pred_adata[:, top_DEGs].X).mean(axis = 0)
            pred_var = to_dense(pred_adata[:, top_DEGs].X).var(axis = 0)
            pred_corr_mtx = np.corrcoef(to_dense(pred_adata[:, top_DEGs].X), rowvar=False).flatten()
            pred_cov_mtx = np.cov(to_dense(pred_adata[:, top_DEGs].X), rowvar=False).flatten()

            results_dict[f'Top_{val}_DEGs_mean_R2'] = scipy.stats.pearsonr(true_mean, pred_mean)[0]**2
            results_dict[f'Top_{val}_DEGs_var_R2'] = scipy.stats.pearsonr(true_var, pred_var)[0]**2
            results_dict[f'Top_{val}_DEGs_mean_MSE'] = (np.square(true_mean - pred_mean)).mean(axis=0)
            results_dict[f'Top_{val}_DEGs_var_MSE'] = (np.square(true_var - pred_var)).mean(axis=0)
            
            corr_nas = np.logical_or(np.isnan(true_corr_mtx), np.isnan(pred_corr_mtx))
            cov_nas = np.logical_or(np.isnan(true_cov_mtx), np.isnan(pred_cov_mtx))
            
            results_dict[f'Top_{val}_DEGs_corr_mtx_R2'] = scipy.stats.pearsonr(
                true_corr_mtx[~corr_nas], pred_corr_mtx[~corr_nas]
            )[0]**2
            results_dict[f'Top_{val}_DEGs_cov_mtx_R2'] = scipy.stats.pearsonr(
                true_cov_mtx[~cov_nas], pred_cov_mtx[~cov_nas]
            )[0]**2
            results_dict[f'Top_{val}_DEGs_corr_mtx_MSE'] = (
                np.square(true_corr_mtx[~corr_nas] - pred_corr_mtx[~corr_nas])
            ).mean(axis=0)
            results_dict[f'Top_{val}_DEGs_cov_mtx_MSE'] = (
                np.square(true_cov_mtx[~cov_nas] - pred_cov_mtx[~cov_nas])
            ).mean(axis=0)

    return results_dict    

def get_DEGs(control_adata, target_adata):
    temp_concat = anndata.concat([control_adata, target_adata], label='batch')
    sc.tl.rank_genes_groups(
        temp_concat, 'batch', method='wilcoxon', groups=['1'], ref='0', rankby_abs=True
    )
    rankings = temp_concat.uns['rank_genes_groups']
    result_df = pd.DataFrame(
        {
            'scores': rankings['scores']['1'],
            'pvals_adj': rankings['pvals_adj']['1']
        },
        index=rankings['names']['1']
    )
    return result_df

def get_DEG_with_direction(gene, score):
    if score > 0:
        return(f'{gene}+')
    else:
        return(f'{gene}-')

def get_DEGs_overlaps(true_DEGs, pred_DEGs, DEG_vals, pval_threshold):
    true_DEGs_for_comparison = [
        get_DEG_with_direction(gene, score)
        for gene, score in zip(true_DEGs.index, true_DEGs['scores'])
    ]
    pred_DEGs_for_comparison = [
        get_DEG_with_direction(gene, score)
        for gene, score in zip(pred_DEGs.index, pred_DEGs['scores'])
    ]

    significant_DEGs = true_DEGs[true_DEGs['pvals_adj'] < pval_threshold]
    num_DEGs = len(significant_DEGs)
    DEG_vals.insert(0, num_DEGs)
    
    results = {}
    for val in DEG_vals:
        if val > num_DEGs:
            results[f'Overlap_in_top_{val}_DEGs'] = None
        else:
            # Intersection of top 'val' with direction
            results[f'Overlap_in_top_{val}_DEGs'] = len(
                set(true_DEGs_for_comparison[:val]).intersection(
                    set(pred_DEGs_for_comparison[:val])
                )
            )
    return results

# "True" sets for final eval
true_pert = anndata.AnnData(X=adata_true_pert.X)
true_pert.obs['condition_key'] = 'perturbed'
control = anndata.AnnData(X=adata_true_ctrl.X)
control.obs['condition_key'] = 'control'
sc.pp.normalize_total(control, target_sum=1e5)
sc.pp.normalize_total(true_pert, target_sum=1e5)
sc.pp.log1p(control)
sc.pp.log1p(true_pert)
true_DEGs_df = get_DEGs(control, true_pert)


dataset_ctrl = 'satija_IFN_adjusted_ctrl'
dataset_pert = 'satija_IFN_adjusted_pert'

adata_ctrl = sc.read(f'input/{dataset_ctrl}.h5ad')
adata_pert = sc.read(f'input/{dataset_pert}.h5ad')
adata_ctrl.obs_names_make_unique()
adata_pert.obs_names_make_unique()

# Mark the perturbed gene index for each cell
adata_pert.obs['pert_type'] = adata_pert.obs['gene'].map(gene_map)

gene_series = adata_pert.obs['gene']
cell_series = adata_pert.obs['cell_type']

# OOD: always the single combination (holdout_cell, holdout_pertname)
ood_condition = (
    (cell_series == holdout_cell) &
    (gene_series == holdout_pertname)
)

# Training holdout depends on mode, but skip if "NT"
if cell_or_pert_holdout == 'cell':
    train_holdout_condition = (
        (cell_series == holdout_cell) &
        (gene_series != 'NT')
    )
elif cell_or_pert_holdout == 'pert':
    if holdout_pertname != 'NT':
        train_holdout_condition = (gene_series == holdout_pertname)
    else:
        train_holdout_condition = np.zeros(len(gene_series), dtype=bool)
elif cell_or_pert_holdout == 'both':
    if holdout_pertname != 'NT':
        train_holdout_condition = (
            (cell_series == holdout_cell) &
            (gene_series == holdout_pertname)
        )
    else:
        train_holdout_condition = np.zeros(len(gene_series), dtype=bool)
else:
    raise ValueError("cell_or_pert_holdout must be 'cell', 'pert', or 'both'.")

train_condition = ~train_holdout_condition & ~ood_condition

trainidx = np.where(train_condition)[0]
holdoutidx = np.where(ood_condition)[0]

whichpert = np.array(adata_pert.obs['pert_type'])


train_dataset = SparseSingleCellDataset(
    adata_ctrl, 
    adata_pert, 
    whichpert, 
    row_indices=trainidx
)

holdout_dataset = SparseSingleCellDataset(
    adata_ctrl,
    adata_pert,
    whichpert,
    row_indices=holdoutidx
)

n_total = len(train_dataset)
n_train = int(0.999 * n_total)
n_val = n_total - n_train

# We do a random split for train/val
train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42))


batch_size = 16
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn_sparse
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn_sparse
)

holdout_loader = DataLoader(
    holdout_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn_sparse
)


class BigPertModel(nn.Module):
    """
    Architecture:
      1) control_encoder: (b, G) => (b, enc_dim)
      2) pert_embedding: (b,) => (b, enc_dim)
      3) gene_embedding: (G, enc_dim)
      4) For each gene => combine => decode to produce (b, G) for:
         - pred_ctrl
         - pred_delta
    """
    def __init__(
        self,
        num_genes: int,
        enc_dim: int = 340,
        hidden_enc_1: int = 3402,
        hidden_dec_1: int = 340
    ):
        super().__init__()
        self.num_genes = num_genes

        # Encode Control
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, hidden_enc_1),
            nn.ReLU(),
            nn.Linear(hidden_enc_1, enc_dim)
        )

        # Pert Embedding
        self.shared_embedding = nn.Embedding(num_genes + 1, enc_dim)

        # We'll decode for pred_ctrl and pred_delta using
        # a single hidden layer, but then 2 heads:
        #   head_ctrl => (hidden_dec_1->1)
        #   head_delta => (hidden_dec_1->1)
        self.hidden_layer = nn.Linear(enc_dim*3, hidden_dec_1)
        self.head_ctrl    = nn.Linear(hidden_dec_1, 1)  
        self.head_delta   = nn.Linear(hidden_dec_1, 1) 

    def forward(self, x_ctrl_log, whichpert_idx):
        """
        Inputs:
          x_ctrl_log: (b, G)
          whichpert_idx: (b,)
        Returns:
          pred_ctrl:  (b, G)
          pred_delta: (b, G)
        """
        b, G = x_ctrl_log.size()
        ctrl_embed = self.encoder(x_ctrl_log)          # (b, enc_dim)
        pert_embed = self.shared_embedding(whichpert_idx)  # (b, enc_dim)

        combined_cp = torch.cat([ctrl_embed, pert_embed], dim=1)  # (b, 2*enc_dim)

        # Gene embeddings => shape(G, enc_dim)
        gene_emb_all = self.shared_embedding.weight[:G]

        # Expand => shape(b, G, 2*enc_dim)
        ccp_expanded = combined_cp.unsqueeze(1).expand(b, G, -1)
        # shape(b, G, enc_dim)
        ge_expanded  = gene_emb_all.unsqueeze(0).expand(b, G, -1)

        # Concat => shape(b, G, 3*enc_dim)
        dec_in = torch.cat([ccp_expanded, ge_expanded], dim=2)
        # Flatten => shape(bG, 3*enc_dim)
        dec_in_flat = dec_in.view(b*G, -1)

        # Single hidden => shape(bG, hidden_dec_1)
        x_hidden = F.relu(self.hidden_layer(dec_in_flat))

        # We produce 2 outputs => pred_ctrl_flat, pred_delta_flat
        pred_ctrl_flat  = self.head_ctrl(x_hidden)   # (bG,1)
        pred_delta_flat = self.head_delta(x_hidden)  # (bG,1)

        # reshape => (b, G)
        pred_ctrl  = pred_ctrl_flat.view(b, G)
        pred_delta = pred_delta_flat.view(b, G)

        return pred_ctrl, pred_delta

##############################################
# Loss Functions for pred_ctrl + pred_delta
##############################################
def masked_mse_dual(pred_ctrl, pred_delta, true_ctrl, true_pert, mask):
    """
    We want to measure errors in predicting:
      - pred_ctrl vs. true_ctrl
      - pred_delta vs. (true_pert - true_ctrl)

    pred_ctrl, pred_delta, true_ctrl, true_pert: (b, G)
    mask: (b, G) => from original ctrl>0
    """
    true_delta = true_pert - true_ctrl

    # MSE on pred_ctrl vs. true_ctrl
    diff_sq_ctrl = (pred_ctrl - true_ctrl)**2
    diff_sq_ctrl_masked = diff_sq_ctrl[mask]

    # MSE on pred_delta vs. true_delta
    diff_sq_delta = (pred_delta - true_delta)**2
    diff_sq_delta_masked = diff_sq_delta[mask]

    if diff_sq_ctrl_masked.numel()==0:
        loss_ctrl = diff_sq_ctrl.mean()
    else:
        loss_ctrl = diff_sq_ctrl_masked.mean()

    if diff_sq_delta_masked.numel()==0:
        loss_delta = diff_sq_delta.mean()
    else:
        loss_delta = diff_sq_delta_masked.mean()

    # Weighted sum or direct sum
    loss_total = 0.9*loss_ctrl + 0.1*loss_delta
    return loss_total, loss_ctrl.item(), loss_delta.item()

##############################################
# Evaluate (predict ctrl+delta => compare)
##############################################
def evaluate_with_preds_dual_in_dist(model, loader, device):
    """
    In-dist: we do final_pert = (pred_ctrl + pred_delta)
    We'll accumulate MSE vs. (x_pert_batch).
    """
    model.eval()
    total_loss=0.0
    total_count=0

    all_pred_ctrl=[]
    all_pred_pert=[]
    all_ctrl=[]
    all_pert=[]

    with torch.no_grad():
        for x_ctrl_batch,x_pert_batch,whichpert_batch,mask_batch in loader:
            x_ctrl_batch   = x_ctrl_batch.to(device)
            x_pert_batch   = x_pert_batch.to(device)
            whichpert_batch= whichpert_batch.to(device)
            mask_batch     = mask_batch.to(device)

            pred_ctrl, pred_delta= model(x_ctrl_batch, whichpert_batch)
            pred_ctrl  = pred_ctrl * mask_batch.float()
            pred_delta = pred_delta* mask_batch.float()

            final_pert= (pred_ctrl + pred_delta)
            diff_sq= (final_pert- x_pert_batch)**2
            diff_sq_masked= diff_sq[mask_batch]

            if diff_sq_masked.numel()==0:
                loss= diff_sq.mean()
            else:
                loss= diff_sq_masked.mean()

            bs_= x_ctrl_batch.size(0)
            total_loss+= loss.item()*bs_
            total_count+= bs_

            all_pred_ctrl.append(pred_ctrl.cpu())
            all_pred_pert.append(final_pert.cpu())
            all_ctrl.append(x_ctrl_batch.cpu())
            all_pert.append(x_pert_batch.cpu())

    mse_val= total_loss/ total_count if total_count>0 else 0.0
    full_pred_ctrl= torch.cat(all_pred_ctrl, dim=0)
    full_pred_pert= torch.cat(all_pred_pert, dim=0)
    full_ctrl     = torch.cat(all_ctrl, dim=0)
    full_pert     = torch.cat(all_pert, dim=0)
    return mse_val, full_pred_ctrl, full_pred_pert, full_ctrl, full_pert


# -- CHANGED FOR GT CONTROL IN OOD -- 
def evaluate_with_preds_dual_ood(model, loader, device):
    """
    OOD: we do final_pert = (ground-truth ctrl + pred_delta)
    We'll still do MSE vs x_pert_batch, but
    we ignore model's predicted ctrl for the final.

    We'll store pred_ctrl if you want to look at it, but final_pert is
    (true_ctrl + pred_delta).

    We'll accumulate average MSE on that final_pert vs actual x_pert_batch.
    """
    model.eval()
    total_loss=0.0
    total_count=0

    all_pred_ctrl=[]
    all_pred_pert=[]
    all_ctrl=[]
    all_pert=[]

    with torch.no_grad():
        for x_ctrl_batch,x_pert_batch,whichpert_batch,mask_batch in loader:
            x_ctrl_batch   = x_ctrl_batch.to(device)
            x_pert_batch   = x_pert_batch.to(device)
            whichpert_batch= whichpert_batch.to(device)
            mask_batch     = mask_batch.to(device)

            pred_ctrl, pred_delta= model(x_ctrl_batch, whichpert_batch)
            # we won't use pred_ctrl for final. We'll do final_pert= (x_ctrl_batch + pred_delta).
            # We do still mask them for the loss, if needed
            pred_ctrl  = pred_ctrl* mask_batch.float()
            pred_delta = pred_delta*mask_batch.float()

            # final_pert = (actual ground-truth ctrl + predicted delta)
            final_pert= x_ctrl_batch + pred_delta

            diff_sq= (final_pert- x_pert_batch)**2
            diff_sq_masked= diff_sq[mask_batch]

            if diff_sq_masked.numel()==0:
                loss= diff_sq.mean()
            else:
                loss= diff_sq_masked.mean()

            bs_= x_ctrl_batch.size(0)
            total_loss+= loss.item()*bs_
            total_count+= bs_

            all_pred_ctrl.append(pred_ctrl.cpu())
            all_pred_pert.append(final_pert.cpu())
            all_ctrl.append(x_ctrl_batch.cpu())
            all_pert.append(x_pert_batch.cpu())

    mse_val= total_loss/ total_count if total_count>0 else 0.0
    full_pred_ctrl= torch.cat(all_pred_ctrl, dim=0)
    full_pred_pert= torch.cat(all_pred_pert, dim=0)   # note: "pert" is ctrl + delta
    full_ctrl     = torch.cat(all_ctrl, dim=0)
    full_pert     = torch.cat(all_pert, dim=0)
    return mse_val, full_pred_ctrl, full_pred_pert, full_ctrl, full_pert


def print_sample_rows_dual(X_ctrl, X_pred_ctrl, X_pred_pert, X_pert, num_print=10):
    """
    We'll pick random (row,col) where X_ctrl>0 and X_ctrl!=X_pert
    Then print ctrl, pred_ctrl, pred_pert, gt_pert
    """
    diff_thresh=1e-5
    mask= (X_ctrl>0)&((X_ctrl-X_pert).abs()>diff_thresh)
    nonzero= mask.nonzero(as_tuple=False)
    if len(nonzero)==0:
        print("No indices found with X_ctrl>0 && X_ctrl!=X_pert!")
        return

    chosen= np.random.choice(len(nonzero), size=min(num_print,len(nonzero)), replace=False)
    print(f"--- Printing {len(chosen)} random (row,col) from those conditions ---")

    for idx in chosen:
        row= nonzero[idx,0].item()
        col= nonzero[idx,1].item()
        c_val = X_ctrl[row,col].item()
        pc_val= X_pred_ctrl[row,col].item()
        pt_val= X_pred_pert[row,col].item()
        gt_val= X_pert[row,col].item()
        print(f"  (row={row}, col={col}): ctrl_log={c_val:.4f}, pred_ctrl={pc_val:.4f}, "
              f"pred_pert={pt_val:.4f}, gt_pert={gt_val:.4f}")


num_genes = train_dataset[0][0].shape[1]
model = BigPertModel(num_genes=num_genes, enc_dim=340, hidden_enc_1=3402, hidden_dec_1=340)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4)
ckpt_path = "model_latest.pth"
start_epoch = 0
num_epochs = 30

print_interval = 100
big_print_interval = 10*print_interval


import time
start_time= time.time()
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    num_samples = 0

    for batch_idx, (x_ctrl_batch, x_pert_batch, whichpert_batch, mask_batch) in enumerate(train_loader):
        x_ctrl_batch   = x_ctrl_batch.to(device)
        x_pert_batch   = x_pert_batch.to(device)
        whichpert_batch= whichpert_batch.to(device)
        mask_batch     = mask_batch.to(device)

        optimizer.zero_grad()

        # CHANGED FOR DUAL PREDICTION
        pred_ctrl, pred_delta = model(x_ctrl_batch, whichpert_batch)
        pred_ctrl  = pred_ctrl  * mask_batch.float()
        pred_delta = pred_delta * mask_batch.float()

        # Compute dual masked MSE
        loss_total, loss_ctrl_val, loss_delta_val = masked_mse_dual(
            pred_ctrl, pred_delta,
            x_ctrl_batch, x_pert_batch,
            mask_batch
        )

        loss_total.backward()
        optimizer.step()

        bs_= x_ctrl_batch.size(0)
        running_loss += loss_total.item()*bs_
        num_samples+= bs_

        iteration= batch_idx + 1 + epoch*len(train_loader)
        if iteration% print_interval==0:
            curr_mse= running_loss/ num_samples
            elapsed= time.time()- start_time
            print(f"[Epoch {epoch+1}, Iter {batch_idx+1}/{len(train_loader)}] "
                  f"Train MSE so far={curr_mse:.6f} (ctrl+delta)   "
                  f"Time last {print_interval} iters: {elapsed:.2f}s")
            running_loss= 0.0
            num_samples= 0
            start_time= time.time()

        if iteration% big_print_interval==0:
            # Evaluate
            val_loss, val_pred_ctrl, val_pred_pert, val_ctrl, val_targ= evaluate_with_preds_dual_in_dist(
                model, val_loader, device
            )
            # Evaluate => OOD => ground-truth ctrl + predicted delta
            ood_loss, ood_pred_ctrl, ood_pred_pert, ood_ctrl, ood_targ= evaluate_with_preds_dual_ood(
                model, holdout_loader, device
            )
            print(f"---- Validation MSE={val_loss:.6f} | OOD MSE={ood_loss:.6f} ----")

            # We do DE analysis on predicted => pred_ctrl + pred_delta
            predicted = anndata.AnnData(X=np.expm1(ood_pred_pert.cpu().detach().numpy()))
            predicted.obs['condition_key'] = 'predicted'
            sc.pp.normalize_total(predicted, target_sum=1e5)
            sc.pp.log1p(predicted)

            # Compare DE
            pred_DEGs_df= get_DEGs(control, predicted)
            DEGs_overlaps= get_DEGs_overlaps(true_DEGs_df, pred_DEGs_df, [100,50,20], 0.05)
            print(DEGs_overlaps)


    train_mse= running_loss/ num_samples if num_samples>0 else 0.0
    print(f"[Epoch {epoch+1}] Train MSE: {train_mse:.6f}")

    # Save checkpoint
    checkpoint_data= {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hyperparams": {
            "num_genes": num_genes
        }
    }
    torch.save(checkpoint_data, ckpt_path)
    print(f"Checkpoint saved at epoch {epoch+1} to {ckpt_path}.\n")
