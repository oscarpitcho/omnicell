from geneformer import EmbExtractor
from geneformer import TranscriptomeTokenizer
import scanpy as sc
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.sparse as sp
import sys
import pickle
from datasets import Dataset
from torchclustermetrics import silhouette
import torch
import requests
import json
import pandas as pd

# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Analysis settings.')
parser.add_argument('--dataset', type=str, default='Nault_single', help='Dataset name.')


device = torch.device("cpu")

gene_median_file="Geneformer/gene_median_dictionary.pkl"
token_dictionary_file="Geneformer/token_dictionary.pkl"
with open(gene_median_file, "rb") as f:
    gene_median_dict = pickle.load(f)
with open(token_dictionary_file, "rb") as f:
    gene_token_dict = pickle.load(f)
gene_keys = list(gene_median_dict.keys())
genelist_dict = dict(zip(gene_keys, [True] * len(gene_keys)))
chunk_size=65536
target_sum=10_000
nproc = 16
use_generator=False
keep_uncropped_input_ids=False
model_input_size=2048

# load token dictionary (Ensembl IDs:token)
with open(token_dictionary_file, "rb") as f:
    gene_token_dict = pickle.load(f)

def rank_genes(gene_vector, gene_tokens):
    """
    Rank gene expression vector.
    """
    # sort by median-scaled gene values
    sorted_indices = np.argsort(-gene_vector)
    return gene_tokens[sorted_indices]




# Suppress warnings
warnings.filterwarnings('ignore')



# Parse arguments from command line
args = parser.parse_args()

# Assign parsed arguments to variables
dataset_name = args.dataset


# Create a timestamp for output files
timestamp = datetime.today().strftime('%Y%m%d%H%M%S')

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
    adata.obs['cell'] = adata.obs['cell_type']
    
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
adata.obs["joinid"] = list(range(adata.n_obs))



adata.X.data = adata.X.data.astype(np.float32)

adata.var['in_geneformer'] = np.array([genelist_dict.get(i, False) for i in adata.var["ensembl_id"]])
adata = adata[:, adata.var.in_geneformer]
adata.obs['n_counts'] = adata.X.sum(axis=1)[:,0]
adata.obs['enoughcounts'] = adata.obs['n_counts']>0
adata = adata[adata.obs.enoughcounts, :]

#tokenize cells
custom_attr_name_dict = {"joinid": "joinid"}
andata = adata

if custom_attr_name_dict is not None:
    file_cell_metadata = {
        attr_key: [] for attr_key in custom_attr_name_dict.keys()
    }
coding_miRNA_loc = np.where(
    [genelist_dict.get(i, False) for i in andata.var["ensembl_id"]]
)[0]
norm_factor_vector = np.array(
    [
        gene_median_dict[i]
        for i in andata.var["ensembl_id"][coding_miRNA_loc]
    ]
)
coding_miRNA_ids = andata.var["ensembl_id"][coding_miRNA_loc]
coding_miRNA_tokens = np.array(
    [gene_token_dict[i] for i in coding_miRNA_ids]
)

try:
    _ = andata.obs["filter_pass"]
except KeyError:
    var_exists = False
else:
    var_exists = True

if var_exists:
    filter_pass_loc = np.where([i == 1 for i in andata.obs["filter_pass"]])[0]
elif not var_exists:
    print(
        f"adata has no column attribute 'filter_pass'; tokenizing all cells."
    )
    filter_pass_loc = np.array([i for i in range(andata.shape[0])])

tokenized_cells = []

#bigram

for i in range(0, len(filter_pass_loc), chunk_size):
    print(i)
    idx = filter_pass_loc[i : i + chunk_size]

    n_counts = andata[idx].obs["n_counts"].values[:, None]
    X_view = andata[idx, coding_miRNA_loc].X
    X_norm = X_view / n_counts * target_sum / norm_factor_vector
    X_norm = sp.csr_matrix(X_norm)

    tokenized_cells += [
        rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices])
        for i in range(X_norm.shape[0])
    ]

    # add custom attributes for subview to dict
    if custom_attr_name_dict is not None:
        for k in file_cell_metadata.keys():
            file_cell_metadata[k] += andata[idx].obs[k].tolist()
    else:
        file_cell_metadata = None
        
        
#make dataset from tokenized

dataset_dict = {"input_ids": tokenized_cells}
if custom_attr_name_dict is not None:
    dataset_dict.update(file_cell_metadata)

# create dataset
if use_generator:

    def dict_generator():
        for i in range(len(tokenized_cells)):
            yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}

    output_dataset = Dataset.from_generator(dict_generator, num_proc=nproc)
else:
    output_dataset = Dataset.from_dict(dataset_dict)

def format_cell_features(example):
    # Store original uncropped input_ids in separate feature
    if keep_uncropped_input_ids:
        example["input_ids_uncropped"] = example["input_ids"]
        example["length_uncropped"] = len(example["input_ids"])


    # Truncate/Crop input_ids to input size
    example["input_ids"] = example["input_ids"][0 : model_input_size]
    example["length"] = len(example["input_ids"])

    return example

output_dataset_truncated = output_dataset.map(
    format_cell_features, num_proc=nproc
)

output_path = "Geneformer/temp.dataset"
output_dataset_truncated.save_to_disk(output_path)



embex = EmbExtractor(model_type="Pretrained",
                     max_ncells=None,
                     emb_layer=0,
                     emb_label=['joinid'],
                     forward_batch_size=100,
                     nproc=16)
embs = embex.extract_embs("Geneformer/geneformer-12L-30M",
                          output_path,
                          "Geneformer/",
                          "temp")

embs = embs.sort_values("joinid")
adata.obsm["geneformer"] = embs.drop(columns="joinid").to_numpy()




# Prepare output directory
output_path = Path(f'{data_folder_path}geneformer/{timestamp}')
output_path.mkdir(parents=True, exist_ok=True)

cell_types = list(adata.obs['cell_type'].cat.categories)
allscores = []
for ct in cell_types:
    for pt in pathways:
        curdata = adata[adata.obs['cell_type'] == ct].copy()
        curdata = curdata[(curdata.obs['pathway']==pt) | (curdata.obs['pathway']==pathway_unperturbed)]
        sc.pp.neighbors(curdata, n_neighbors=15, use_rep='geneformer')
        curdata.obsm["X_pca"] = sc.tl.pca(curdata.obsm["geneformer"])
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



