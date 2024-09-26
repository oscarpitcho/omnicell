import torch
import sys
import numpy as np
import scanpy as sc
import anndata
import pandas as pd  



holdoutpert = 'IFNAR2'
num_of_degs = 20
pvalcut = 0.05

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datasetso = 'Seurat_IFNB.h5ad'

inp = sc.read(f'/orcd/archive/abugoot/001/Projects/dlesman/datasets/{datasetso}')

print(f"Input data shape: {inp.shape}")



#WHEN WE REFER TO THIS THERE ARE TARGET PERTS IN THIS CODE 
inp = inp[inp.obs['cell_type']=='A549']

print(f"# Cells with A549 cell type: {inp.shape}")
inp.var_names = inp.var['gene']

sc.pp.normalize_total(inp,target_sum=10000)

sc.pp.log1p(inp)


# Extract the unique genes
unique_genes = inp.obs['gene'].unique()
unique_genes = [i for i in unique_genes if i!='NT']


print(f" # Control A549 cells: {len(inp[inp.obs['gene']=='NT'])}")

DEGSlist = []
GTOlist = []

unique_genes_noholdout = [ug for ug in unique_genes if ug!=holdoutpert]

print(f"Lenght of unique genes_no_ho: {len(unique_genes_noholdout)}")

for ug in unique_genes_noholdout:
    cont = np.array(inp[inp.obs['gene'] == 'NT'].X.todense())
    pert = np.array(inp[inp.obs['gene'] == ug].X.todense())
    print(f'ug: {ug} - Control shape: {cont.shape}, Perturbed shape: {pert.shape}')

    control = anndata.AnnData(X=cont)
    
    control.obs['condition_key'] = 'control'
    
    true_pert = anndata.AnnData(X=pert)
    
    true_pert.obs['condition_key'] = 'perturbed'
    
    
    control.obs_names = control.obs_names+'-1'
    control.X[0,(control.X.var(axis=0)==0)] += np.amin(control.X[np.nonzero(control.X)])
    true_pert.X[0,(true_pert.X.var(axis=0)==0)] += np.amin(true_pert.X[np.nonzero(true_pert.X)])
    
    temp_concat = anndata.concat([control, true_pert], label = 'batch')
    sc.tl.rank_genes_groups(temp_concat, 'batch', method='wilcoxon', groups = ['1'], ref = '0')
    
    rankings = temp_concat.uns['rank_genes_groups']
    result_df = pd.DataFrame({'pva': rankings['pvals_adj']['1'], 'pvals_adj': rankings['scores']['1']}, index = rankings['names']['1'])
    result_df.index = result_df.index.astype(np.int32)
    result_df = result_df.sort_index()
    
    GTO = torch.from_numpy(np.array(result_df['pvals_adj']).astype(np.float32))
    
    DEGs = torch.argsort(torch.abs(GTO))
    

    print(f'ug: {ug} - DEGs shape: {DEGs.shape}, GTO shape: {GTO.shape}')
    DEGSlist.append(DEGs)
    GTOlist.append(GTO)
    
significant_reducers = []
significant_reducers_pval = []
print(f"DEG list length: {len(DEGSlist)}")
print(f"GTO list length: {len(GTOlist)}")
for genno in unique_genes_noholdout:
    gto_gene = GTOlist[unique_genes_noholdout.index(genno)]
    rank = gto_gene[torch.where(DEGSlist[unique_genes_noholdout.index(genno)] == np.where(inp.var_names==holdoutpert)[0][0])[0][0].item()]
    if ((0 <= rank) and (rank <= pvalcut)):
       significant_reducers.append(genno)
       significant_reducers_pval.append(rank) 

reduced_DEGs = []    
for sr in significant_reducers:
    reduced_DEGs.append(DEGSlist[unique_genes_noholdout.index(sr)][-num_of_degs:])
reduced_DEGs = torch.cat(reduced_DEGs)   

# Count the occurrences of each element
unique_elements, counts = torch.unique(reduced_DEGs, return_counts=True)

# Filter to keep only elements that repeat more than once
repeated_elements = unique_elements[counts > 1]

# Mask the original tensor to keep only repeated elements
duplicated_DEGs = torch.unique(reduced_DEGs[torch.isin(reduced_DEGs, repeated_elements)])

bestnnscore = 0
bestpval = 0
for jq, sr in enumerate(significant_reducers):
    nnscore = len(np.intersect1d(duplicated_DEGs.cpu().detach().numpy(),DEGSlist[unique_genes_noholdout.index(sr)][-num_of_degs:].cpu().detach().numpy()))
    
        
    if nnscore > bestnnscore:
        nnbr = sr
        bestnnscore = nnscore
        bestpval = significant_reducers_pval[jq]
    #pval as tie-break
    if nnscore == bestnnscore:
        if bestpval > significant_reducers_pval[jq]:
            nnbr = sr
            bestnnscore = nnscore
            bestpval = significant_reducers_pval[jq]
            


print('Nearest neighbor perturbation is: '+nnbr)
    
    
