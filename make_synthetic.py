import scanpy as sc
import numpy as np
import anndata
import pandas as pd
from anndata import AnnData
import scipy.sparse as sp
import anndata as ad


def shift_prediction(control_adata, perturbed_adata, sum_control, sum_perturbed):
    """
    Distribute the global per-gene difference (sum_diff[g]) across cells in proportion
    to the cell's existing counts for that gene. 
    """


    #Compute sum_diff
    sum_diff = sum_perturbed - sum_control  # shape: (n_genes,)


    new_X = np.array(control_adata.X.copy().todense())  # shape: (n_cells, n_genes)
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
        p = gene_counts / current_total


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


    new_adata = AnnData(
        X=new_X,
        obs=control_adata.obs.copy(),
        var=control_adata.var.copy(),
        uns=control_adata.uns.copy()
    )
    return new_adata


def get_DEGs(control_adata, target_adata):
    temp_concat = anndata.concat([control_adata, target_adata], label = 'batch')
    sc.tl.rank_genes_groups(temp_concat, 'batch', method='wilcoxon', 
                                groups = ['1'], ref = '0', rankby_abs = True)

    rankings = temp_concat.uns['rank_genes_groups']
    result_df = pd.DataFrame({'scores': rankings['scores']['1'],
                     'pvals_adj': rankings['pvals_adj']['1']},
                    index = rankings['names']['1'])
    return result_df

def get_DEG_with_direction(gene, score):
    if score > 0:
        return(f'{gene}+')
    else:
        return(f'{gene}-')
def get_DEGs_overlaps(true_DEGs, pred_DEGs, DEG_vals, pval_threshold):
    true_DEGs_for_comparison = [get_DEG_with_direction(gene,score) for gene, score in zip(true_DEGs.index, true_DEGs['scores'])]   
    pred_DEGs_for_comparison = [get_DEG_with_direction(gene,score) for gene, score in zip(pred_DEGs.index, pred_DEGs['scores'])]

    significant_DEGs = true_DEGs[true_DEGs['pvals_adj'] < pval_threshold]
    num_DEGs = len(significant_DEGs)
    DEG_vals.insert(0, num_DEGs)
    
    results = {}
    for val in DEG_vals:
        if val > num_DEGs:
            results[f'Overlap_in_top_{val}_DEGs'] = None
        else:
            results[f'Overlap_in_top_{val}_DEGs'] = len(set(true_DEGs_for_comparison[0:val]).intersection(set(pred_DEGs_for_comparison[0:val])))

    return results






S_IFNog = sc.read(f'input/satija_IFN.h5ad')
S_TGF = sc.read(f'input/satija_TGF.h5ad')
S_IFNog.var_names = S_IFNog.var['gene']
S_TGF.var['gene'] = S_TGF.var_names
S_TGF.X = S_TGF.layers['counts']


S_merged = ad.concat(
    [S_IFNog, S_TGF],
    axis=0,            # stack along obs
    join='outer',      # keep the union of all var_names
    fill_value=0,      # fill missing counts with 0
    index_unique=None,  # don't rename var_names from the 2nd object
    label='dataset'
)

S_merged.var['gene'] = S_merged.var_names

list_adata1_adjusted = []
list_adata2_adjusted = []
list_adata3_adjusted = []
list_adata4_adjusted = []

unique_datasets = np.unique(S_merged.obs['dataset'])

for dset in unique_datasets:
    
    S_IFN = S_merged[S_merged.obs['dataset']==dset]

    unique_genes = np.unique(S_IFN.obs['gene'])
    unique_genes = [g for g in unique_genes if g!='NT']
    unique_cells = np.unique(S_IFN.obs['cell_type'])
    
    
    for uc in unique_cells:
        print(uc)
        S_ctrl = S_IFN[(S_IFN.obs['gene'] == 'NT') & (S_IFN.obs['cell_type'] == uc)].copy()
        list_adata4_adjusted.append(S_ctrl.copy())
        
        for ug in unique_genes:
            print(ug)
    
            S_IFNAR2 = S_IFN[(S_IFN.obs['gene'] == ug) & (S_IFN.obs['cell_type'] == uc)].copy()
        
            adata1 = S_ctrl
            adata2 = S_IFNAR2
            
            list_adata3_adjusted.append(adata2.copy())
            
            if sp.issparse(adata1.X):
                adata1_sum = adata1.X.sum(axis=0).A1
            else:
                adata1_sum = adata1.X.sum(axis=0)
    
            if sp.issparse(adata2.X):
                adata2_sum = adata2.X.sum(axis=0).A1
            else:
                adata2_sum = adata2.X.sum(axis=0)
    
            tot1 = adata1_sum.sum()
            tot2 = adata2_sum.sum()
    
            # Scale adata2 counts by ratio of total counts; cast to int
            if tot2 > 0:
                adata2_sum = ((tot1 / tot2) * adata2_sum).astype(np.int32)
            else:
                # If tot2 == 0 for some reason, avoid division by zero
                adata2_sum = adata2_sum.astype(np.int32)
    
            
            

            predicted_shift = shift_prediction(adata1, adata2, adata1_sum, adata2_sum)
    
            random_indices = np.random.choice(predicted_shift.n_obs,
                                              size=min(S_ctrl.shape[0], S_IFNAR2.shape[0]),
                                              replace=False)
    
            predicted_shift = predicted_shift[random_indices, :].copy()
            adata1 = adata1[random_indices, :].copy()
            adata2 = adata2[:min(S_ctrl.shape[0], S_IFNAR2.shape[0]), :].copy()
    
            # Overwrite adata2.X with the predicted values
            adata2.X = predicted_shift.X
    
    
            list_adata1_adjusted.append(adata1.copy())
            list_adata2_adjusted.append(adata2.copy())
    
    
    # The second loop (for "NT" vs. "NT") 
    for uc in unique_cells:
        print(uc)
        S_ctrl = S_IFN[(S_IFN.obs['gene'] == 'NT') & (S_IFN.obs['cell_type'] == uc)].copy()
        S_IFNAR2 = S_IFN[(S_IFN.obs['gene'] == 'NT') & (S_IFN.obs['cell_type'] == uc)].copy()
        
        adata1 = S_ctrl
        adata2 = S_IFNAR2
    
        pert_strength = 0
        adata2.obs['pert_strength'] = pert_strength
    
        list_adata1_adjusted.append(adata1.copy())
        list_adata2_adjusted.append(adata2.copy())



adata1_combined = anndata.concat(list_adata1_adjusted, join='outer', axis=0)
adata2_combined = anndata.concat(list_adata2_adjusted, join='outer', axis=0)
adata3_combined = anndata.concat(list_adata3_adjusted, join='outer', axis=0)
adata4_combined = anndata.concat(list_adata4_adjusted, join='outer', axis=0)


adata1_combined.var['gene'] = adata1.var['gene']
adata2_combined.var['gene'] = adata2.var['gene']
adata3_combined.var['gene'] = adata2.var['gene']
adata4_combined.var['gene'] = adata2.var['gene']




del adata1_combined['orig.ident']
del adata2_combined.obs['orig.ident']
del adata3_combined.obs['orig.ident']
del adata4_combined.obs['orig.ident']
# Write results
adata1_combined.write('input/satija_IFN_adjusted_ctrl.h5ad')
adata2_combined.write('input/satija_IFN_adjusted_pert.h5ad')
adata3_combined.write('input/satija_IFN_adjusted_truepert.h5ad')
adata4_combined.write('input/satija_IFN_adjusted_tructrl.h5ad')

