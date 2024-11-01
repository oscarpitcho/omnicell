import scanpy as sc
import json
from omnicell.evaluation.utils import get_DEGs
from omnicell.data.catalogue import Catalogue
import argparse

parser = argparse.ArgumentParser(description='Analysis settings.')

parser.add_argument('--signifcant', action='store_true', help="Whether to filter for dsignifcant DEGs")

args = parser.parse_args()

DATA_PATH = '/om/group/abugoot/Projects/Omnicell_datasets/essential_gene_knockouts_raw/essential_gene_knockouts_raw.h5ad'

catalogue = Catalogue('configs/catalogue')
dd = catalogue.get_dataset_details('essential_gene_knockouts_raw')


with open(DATA_PATH, 'rb') as f:
    adata = sc.read_h5ad(f)


sc.pp.normalize_total(adata, target_sum=10_000)
sc.pp.log1p(adata)


cell_types = adata.obs[dd.cell_key].unique()

results = {}

for cell_type in cell_types:
    results[cell_type] = {}
    perts = [x for x in  adata.obs[(adata.obs[dd.cell_key] == cell_type)][dd.pert_key].unique() if x != dd.control]

    adata_control = adata[(adata.obs[dd.cell_key] == cell_type) & (adata.obs[dd.pert_key] == dd.control)]

    for pert in perts:
        adata_pert = adata[(adata.obs[dd.cell_key] == cell_type) & (adata.obs[dd.pert_key] == pert)]

        if adata_pert.shape[0] < 2:
            results[cell_type][pert] = []
        else:
            DEGs = get_DEGs(adata_control, adata_pert)

            if args.signifcant:
                DEGs = DEGs[DEGs['pvals_adj'] < 0.05]

            results[cell_type][pert] = list(DEGs.index)



file_name = 'DEGs_per_pert_significant.json' if args.signifcant else 'DEGs_per_pert_ALL.json'

with open(file_name, 'w') as f:
    json.dump(results, f)
