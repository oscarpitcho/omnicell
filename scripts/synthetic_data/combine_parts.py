import json
import glob
import argparse\
import scanpy as sc
import os

from anndata import AnnData
from omnicell.data.catalogue import Catalogue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    dd = Catalogue.get_dataset_details(args.dataset)
    combined_results = {}

    # Find all partial result files
    part_files = glob.glob(f'{dd.folder_path}/Jason_Shifted/*_of_*.h5ad')
    if not part_files:
        raise ValueError(f"No partial results found in {dd.folder_path}")

    # Combine partial results
    adata_parts = []
    for filename in part_files:
        partial_adata = sc.read(filename)
        adata_parts.append(partial_adata)

    adata = AnnData.concatenate(*adata_parts, axis = 0)

    
    # Save combined results
    output_file = f'{dd.folder_path}/Jason_Shifted/combined.h5ad'
    adata.write(output_file)

    
    print(f"Combined results saved to {output_file}")

    # Remove partial results
    for filename in part_files:
        os.remove(filename)

    print(f"Removed {len(part_files)} partial result files")
    print("Done")