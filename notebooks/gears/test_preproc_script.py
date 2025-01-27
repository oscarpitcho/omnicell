import sys

from omnicell.models.gears.pertdata import PertData
from omnicell.data.catalogue import Catalogue
import scanpy as sc
dd = Catalogue.get_dataset_details('repogle_k562_essential_raw')
adata = sc.read(dd.path)

adata.obs["condition"] = adata.obs["gene"]
perts = [p for p in adata.obs["condition"].unique() if p != dd.control]
adata.obs["condition"] = adata.obs["condition"].replace({dd.control:"ctrl"})
adata.obs["condition"] = adata.obs["condition"].replace({p:p+"+ctrl" for p in perts})

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata,n_top_genes=5000, subset=True)



pert_data = PertData('./data') # specific saved folder
pert_data.new_data_process(dataset_name = 'repogle', adata = adata) # specific dataset name and adata object
print(f"Data processed and saved in {pert_data.data_path}")
pert_data.load(data_path = './data/repogle') # load the processed data, the path is saved folder + dataset_name
print(f"Data loaded from {pert_data.data_path}")
pert_data.prepare_split(split = 'simulation', seed = 1) # get data split with seed
print(f"Data split with seed 1")
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128) # prepare data loader
print(f"Data loader prepared")