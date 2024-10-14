
import scanpy as sc
from omnicell.data.loader import DataLoader, DatasetDetails
import torch 
import argparse
from omnicell.constants import DATA_CATALOGUE_PATH
import json
from omnicell.data.catalogue import Catalogue, DatasetDetails

import logging

logger = logging.getLogger(__name__)



def main():

    parser = argparse.ArgumentParser(description='Generate static embedding')

    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--static_embedding_path', type=str, help='Path to the static embedding')
    parser.add_argument('--embedding_name', type=str, help='Name of the embedding, this will be used to save the embedding')



    #Can be None or Mean
    parser.add_argument('--imputing_method', choices=['None', 'Mean'], help='Imputing method to use, if None, perts with not data will be left out, if Mean, the mean of all pert embeddings will be used.')

    args = parser.parse_args()

    assert args.dataset_name is not None, "Please provide a dataset name"
    assert args.static_embedding_path is not None, "Please provide a path to the static embedding"
    assert args.embedding_name is not None, "Please provide a name for the embedding"
    assert args.imputing_method is not None, "Please provide an imputing method"



    catalogue = Catalogue(DATA_CATALOGUE_PATH)


    #Getting the dataset details from the data_catalogue.json

    ds_details = catalogue.get_dataset_details(args.dataset_name)

    pert_key = ds_details.pert_key
    control_pert = ds_details.control

    #Loading the dataset
    adata = sc.read(ds_details.path, backed='r')



    #Selecting all non Control Perts

    perts = [x for x in adata.obs[pert_key].unique() if x != control_pert]



    #Getting the static embedding

    static_embedding = torch.load(args.static_embedding_path)

    all_embeddings = torch.stack(list(static_embedding.values()))

    print(f"Shape of static embedding: {all_embeddings.shape}")

    mean_embedding = torch.mean(all_embeddings, axis = 0)

    embedding = {}

    coverage = 0
    for pert in perts:

        if pert in static_embedding:
            embedding[pert] = static_embedding[pert]
            coverage += 1
        else:

            if args.imputing_method == 'None':
                embedding[pert] = None

            elif args.imputing_method == 'Mean':
                embedding[pert] = mean_embedding

            else:
                raise ValueError(f"Unrecognized imputing method: {args.imputing_method}")



    #Overwrites any existing file with the same name
    torch.save(embedding, f"{ds_details.folder_path}/{args.embedding_name}.pt")

    print(f'Total coverage: {coverage}/{len(perts)}')
    print(f"Size of the embedding: {len(embedding)}")


    #Register the new embedding in the catalogue, This modifies the underlying yaml file
    catalogue.register_new_pert_embedding(args.dataset_name, args.embedding_name)




if __name__ == '__main__':
    main()