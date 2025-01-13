import scanpy as sc
from omnicell.data.catalogue import Catalogue
from omnicell.evaluation.utils import get_DEGs
import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--job_id', type=int)
    parser.add_argument('--total_jobs', type=int)

    args = parser.parse_args()
    
    print(f"Starting job {args.job_id} of {args.total_jobs} for dataset {args.dataset}")

     # Get dataset
    dd = Catalogue.get_dataset_details(args.dataset)
    if dd.precomputed_DEGs:
        print("Precomputed DEGs found - Terminating")
        return


    adata = sc.read(dd.path)
    print(f"Loaded dataset {args.dataset} with {adata.shape[0]} cells and {adata.shape[1]} genes")
    
    # Preprocess
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)

    print("Preprocessing complete")
    
    results = {}
    
    total_cell_types = len(adata.obs[dd.cell_key].unique())

    # Process pairs assigned to this job
    for cell_idx, cell_type in enumerate(adata.obs[dd.cell_key].unique()):
        print(f"Processing cell type {cell_type} ({cell_idx + 1}/{total_cell_types})")
        current_idx = 0
        

        if cell_type not in results:
            results[cell_type] = {}
            
        control_cells = adata[(adata.obs[dd.cell_key] == cell_type) & 
                            (adata.obs[dd.pert_key] == dd.control)]
        

        #This order does not change --> Unique returns in order of appearance
        perts = [x for x in adata.obs[
            (adata.obs[dd.cell_key] == cell_type)][dd.pert_key].unique() 
            if x != dd.control]
        
        print(f"Found {len(perts)} perturbations for {cell_type}")

        perts_processed = 0
        
        for pert in perts:
            if current_idx % args.total_jobs == args.job_id:
                if perts_processed % 50 == 0:
                    print(f"Job {args.job_id}: Processed {perts_processed} perturbations for {cell_type}")
                
                pert_cells = adata[(adata.obs[dd.cell_key] == cell_type) & 
                                 (adata.obs[dd.pert_key] == pert)]
                
                if pert_cells.shape[0] >= 2:
                    DEGs = get_DEGs(control_cells, pert_cells)
                    results[cell_type][pert] = DEGs.to_dict(orient='index')
                else:
                    results[cell_type][pert] = None
                    
                perts_processed += 1
                    
            current_idx += 1
        
        print(f"Job {args.job_id}: Completed {perts_processed} perturbations for {cell_type}")
    
    with open(f'{dd.folder_path}/DEGs-{args.job_id}_of_{args.total_jobs}.json', 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()