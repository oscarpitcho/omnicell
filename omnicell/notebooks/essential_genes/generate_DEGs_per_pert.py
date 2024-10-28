import scanpy as sc
import json
from omnicell.evaluation.utils import get_DEGs
from omnicell.data.catalogue import Catalogue
import time
from datetime import timedelta

DATA_PATH = '/om/group/abugoot/Projects/Omnicell_datasets/essential_gene_knockouts_raw/essential_gene_knockouts_raw.h5ad'

catalogue = Catalogue('configs/catalogue')
dd = catalogue.get_dataset_details('essential_gene_knockouts_raw')

print("Loading data...")
with open(DATA_PATH, 'rb') as f:
    adata = sc.read_h5ad(f)

sc.pp.normalize_total(adata, target_sum=10_000)
sc.pp.log1p(adata)

cell_types = adata.obs[dd.cell_key].unique()

results = {}
total_time = 0
number_perts_computed = 0

print("Starting DEG analysis with time tracking...")

for cell_type in cell_types:
    results[cell_type] = {}
    perts = [x for x in adata.obs[(adata.obs[dd.cell_key] == cell_type)][dd.pert_key].unique() if x != dd.control]
    
    adata_control = adata[(adata.obs[dd.cell_key] == cell_type) & (adata.obs[dd.pert_key] == dd.control)]
    
    print(f"\nProcessing cell type: {cell_type}")
    print(f"Number of perturbations to process: {len(perts)}")
    
    for pert in perts:
        start_time = time.time()
        
        adata_pert = adata[(adata.obs[dd.cell_key] == cell_type) & (adata.obs[dd.pert_key] == pert)]

        if len(adata_pert) > 1:
            DEGs = get_DEGs(adata_control, adata_pert)
            results[cell_type][pert] = list(DEGs.index)
            
            # Update timing statistics
            iteration_time = time.time() - start_time
            number_perts_computed += 1
            total_time += iteration_time
            avg_time = total_time / number_perts_computed
            
            # Format times for printing
            iteration_time_str = str(timedelta(seconds=int(iteration_time)))
            avg_time_str = str(timedelta(seconds=int(avg_time)))
            total_time_str = str(timedelta(seconds=int(total_time)))
            
            print(f"\rPerturbation {number_perts_computed}: {pert}")
            print(f"Time for this perturbation: {iteration_time_str}")
            print(f"Cumulative average time per perturbation: {avg_time_str}")
            print(f"Total time elapsed: {total_time_str}")

        else:
            print(f"\rPerturbation {number_perts_computed}: {pert} - Not enough cells to compute DEGs")
            results[cell_type][pert] = []
            number_perts_computed += 1

print("\nAnalysis complete!")
print(f"Final Statistics:")
print(f"Total perturbations processed: {number_perts_computed}")
print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
print(f"Average time per perturbation: {str(timedelta(seconds=int(total_time/number_perts_computed)))}")

with open('DEGs_per_pert_ALL.json', 'w') as f:
    json.dump(results, f)