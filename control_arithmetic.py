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
# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Analysis settings.')
parser.add_argument('--dataset', type=str, default='Kang_et_al', help='Dataset name.')

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
datafilename = [f for f in os.listdir(data_folder_path+'Data/') if f[-1]=='d'][0]

# Load dataset
adata = sc.read(f'{data_folder_path}Data/{datafilename}')

if dataset_name == 'Kang_et_al':
    adata.obs['perturbed'] = adata.obs['condition'] == 'stimulated'
    
adata.X.data = adata.X.data.astype(np.float32)





# Prepare output directory
output_path = Path(f'{data_folder_path}control/{timestamp}')
output_path.mkdir(parents=True, exist_ok=True)

cell_types = list(adata.obs['cell_type'].cat.categories)
allscores = []
for ct in cell_types:
    curdata = adata[adata.obs['cell_type'] == ct].copy()
    sc.pp.neighbors(curdata, n_neighbors=15)
    adata.obsm["X_pca"] = sc.tl.pca(adata.X.toarray())
    sc.tl.umap(curdata, min_dist=0.1, random_state=42)        
    with plt.rc_context():  
        sc.pl.umap(curdata, color=["condition", "cell_type"], frameon=False, show=False)
        plt.savefig(str(output_path) + '/'+str(ct)+'_umap.png', bbox_inches="tight")
        sc.pl.pca(curdata, color=["condition", "cell_type"], frameon=False, show=False)
        plt.savefig(str(output_path) + '/'+str(ct)+'_pca.png', bbox_inches="tight")
    curdataX_torch = torch.from_numpy(curdata.obsm['X_umap'].astype(np.float32))
    curdataX_torch = curdataX_torch.to(device)
    condition_torch = torch.from_numpy(curdata.obs["perturbed"].to_numpy().astype(np.float32))
    condition_torch = condition_torch.to(device)
    asw = silhouette.score(curdataX_torch, condition_torch)
    ctl = torch.sum(condition_torch==0).item()
    pert = torch.sum(condition_torch==1).item()
    print(f'Silhouette Score is {asw}, ctl={ctl}, pert={pert} for {ct}')
    with open(str(output_path) + '/asw.txt', "a") as f:
        f.write(f"{ct} = {asw}\n")



