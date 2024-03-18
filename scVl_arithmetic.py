import scanpy as sc
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
from torchclustermetrics import silhouette
import torch
import warnings

warnings.filterwarnings("ignore")

import anndata
import numpy as np
import scanpy as sc
import scvi
from sklearn.ensemble import RandomForestClassifier
import requests
import sys

# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Analysis settings.')
parser.add_argument('--dataset', type=str, default='Nault_single', help='Dataset name.')

device = torch.device("cpu")

# Suppress warnings
warnings.filterwarnings('ignore')

# Parse arguments from command line
args = parser.parse_args()

# Assign parsed arguments to variables
dataset_name = args.dataset


# Create a timestamp for output files
timestamp = datetime.today().strftime('%Y%m%d%H%M%S')

# Define path to dataset
data_folder_path = f'Datasets/{dataset_name}/'
datafilename = [f for f in os.listdir(data_folder_path+'Data/') if f[-14:]=='ensemblid.h5ad']
notensemblprocessed = False
if len(datafilename)==0:
    datafilename = [f for f in os.listdir(data_folder_path+'Data/') if f[-4:]=='h5ad']
    notensemblprocessed = True
datafilename = datafilename[0]
filnametrunc = datafilename[:-5]
# Load dataset
adata = sc.read(f'{data_folder_path}Data/{datafilename}')

def get_ensembl_ids_for_gene_names(gene_names):
    server = "https://rest.ensembl.org"
    ext = "/lookup/symbol/homo_sapiens"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    max_elements_per_request = 1000  # API limit

    # Initialize ensembl_ids with gene name mapping to itself
    ensembl_ids = {gene_name: gene_name for gene_name in gene_names}

    # Split gene_names into smaller chunks
    chunks = [gene_names[i:i + max_elements_per_request] for i in range(0, len(gene_names), max_elements_per_request)]
    counter = 0
    print('Fetching ensembl ids...')
    for chunk in chunks:
        print(str(counter)+'/'+str(len(gene_names)))
        counter+=1000
        payload = {"symbols": chunk}
        response = requests.post(f"{server}{ext}", json=payload, headers=headers)
        if not response.ok:
            # Log error and continue with the next chunk
            print("Failed to retrieve data for a chunk:", response.text)
            continue

        data = response.json()

        # Update ensembl_ids with actual Ensembl IDs where found
        for gene_name in chunk:
            # Check if gene_name is in the response, and if so, update its Ensembl ID
            if gene_name in data and 'id' in data[gene_name]:
                ensembl_ids[gene_name] = data[gene_name]['id']
            # If a gene_name is not found or there's no 'id', it will keep its original mapping
    print('Finished fetching ensembl ids.')
    return ensembl_ids

if dataset_name == 'Kang_et_al':
    if notensemblprocessed:
        # Example usage with a mix of gene names and identifiers
        gene_names = [x for x in adata.var['gene_symbol'].to_numpy().tolist()]
        #gene_names = [g.split('.')[0] for g in gene_names]
        ensembl_ids = get_ensembl_ids_for_gene_names(gene_names)

        adata.var['ensembl_id'] = ensembl_ids
        
        adata.write(f'{data_folder_path}Data/{filnametrunc}_ensemblid.h5ad')
    adata.obs['perturbed'] = adata.obs['condition'] == 'stimulated'
    adata.var['gene_symbols'] = adata.var['gene_symbol']
    pathways = ['All']
    pathway_unperturbed = 'None'
    pathways = [p for p in pathways if p != pathway_unperturbed]
    adata.obs['pathway'] = 'All'
    
if dataset_name == 'Nault_single':
    adata.var.index = [f.upper() for f in adata.var.index]
    if notensemblprocessed:
        
        # Example usage with a mix of gene names and identifiers
        gene_names = [x for x in adata.var_names.to_numpy().tolist()]
        #gene_names = [g.split('.')[0] for g in gene_names]
        ensembl_ids = get_ensembl_ids_for_gene_names(gene_names)

        adata.var['ensembl_id'] = ensembl_ids
        
        adata.write(f'{data_folder_path}Data/{filnametrunc}_ensemblid.h5ad')
    adata.obs['perturbed'] = adata.obs.Dose >= 30
    adata.var['gene_symbols'] = adata.var_names
    pathways = ['All']
    pathway_unperturbed = 'None'
    pathways = [p for p in pathways if p != pathway_unperturbed]
    adata.obs['pathway'] = 'All'
    adata.obs['cell_type'] = adata.obs['celltype']
    
if dataset_name == 'srivatsan':
        
    adata.var['ensembl_id'] = [f.split('.')[0] for f in adata.var['id'].tolist()]
    
    adata.obs['perturbed'] = adata.obs.dose >= 10000
    adata.obs['unperturbed'] = adata.obs.dose == 0
    adata = adata[adata.obs['perturbed'] | adata.obs['unperturbed']]
    adata.var['gene_symbols'] = adata.var['gene_short_name']
    adata.obs['pathway'] = adata.obs['pathway_level_1']
    pathways = np.unique(adata.obs['pathway_level_1'].to_numpy())
    pathway_unperturbed = 'Vehicle'
    pathways = [p for p in pathways if p != pathway_unperturbed]


adata.obs['perturbed'] = adata.obs['perturbed'].astype('category')
adata.X.data = adata.X.data.astype(np.float32)

adata.obs["joinid"] = list(range(adata.n_obs))
# initialize the batch to be unassigned. This could be any dummy value.
adata.obs["batch"] = "unassigned"
adata.var_names = adata.var['ensembl_id']

model_filename = f"scVI/"
scvi.model.SCVI.prepare_query_anndata(adata, model_filename)

vae_q = scvi.model.SCVI.load_query_data(
    adata,
    model_filename,
)

# This allows for a simple forward pass
vae_q.is_trained = True
latent = vae_q.get_latent_representation()
adata.obsm["scvi"] = latent

# filter out missing features
adata = adata[:, adata.var["gene_symbols"].notnull().values].copy()
adata.var.set_index("gene_symbols", inplace=True)


# Prepare output directory
output_path = Path(f'{data_folder_path}scvi/{timestamp}')
output_path.mkdir(parents=True, exist_ok=True)

cell_types = list(adata.obs['cell_type'].cat.categories)
allscores = []
for ct in cell_types:
    for pt in pathways:
        curdata = adata[adata.obs['cell_type'] == ct].copy()
        curdata = curdata[(curdata.obs['pathway']==pt) | (curdata.obs['pathway']==pathway_unperturbed)]
        sc.pp.neighbors(curdata, n_neighbors=15, use_rep='scvi')
        curdata.obsm["X_pca"] = sc.tl.pca(curdata.obsm["scvi"])
        sc.tl.umap(curdata, min_dist=0.1, random_state=42)        
        with plt.rc_context():  
            sc.pl.umap(curdata, color=["perturbed", "cell_type"], frameon=False, show=False)
            plt.savefig(str(output_path) + '/'+str(ct)+'_umap.png', bbox_inches="tight")
            sc.pl.pca(curdata, color=["perturbed", "cell_type"], frameon=False, show=False)
            plt.savefig(str(output_path) + '/'+str(ct)+'_pca.png', bbox_inches="tight")
        curdataX_torch = torch.from_numpy(curdata.obsm['X_umap'].astype(np.float32))
        curdataX_torch = curdataX_torch.to(device)
        condition_torch = torch.from_numpy(curdata.obs["perturbed"].to_numpy().astype(np.float32))
        condition_torch = condition_torch.to(device)
        asw = silhouette.score(curdataX_torch, condition_torch)
        ctl = torch.sum(condition_torch==0).item()
        pert = torch.sum(condition_torch==1).item()
        print(f'Silhouette Score is {asw}, ctl={ctl}, pert={pert} for {ct}_{pt}')
        with open(str(output_path) + '/asw.txt', "a") as f:
            f.write(f"{ct}_{pt} = {asw}\n")