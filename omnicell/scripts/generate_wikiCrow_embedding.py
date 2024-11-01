import os
import sys
import re
import json
import argparse
import pickle
from google.cloud import storage
import numpy as np
from tqdm import tqdm
from omnicell.data.catalogue import Catalogue

WIKICROW_PATH = "/om/group/abugoot/Projects/Omnicell_datasets/wikicrow2"
def main():

    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process files in GCS bucket.')
    parser.add_argument('--dataset', required=True, help='Name of the dataset.')
    parser.add_argument('--prefix', required=False, help='Prefix')
    parser.add_argument('--max', required=False, help='Limit to this count')
    args = parser.parse_args()


    catalogue = Catalogue('configs/catalogue')
    dd = catalogue.get_dataset_details(args.dataset)


    prefix = args.prefix

    # Initialize the GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket('fh-public')

    # List all blobs in the bucket
    blobs = list(bucket.list_blobs(prefix=prefix))
    print(f"Blobs in the bucket: {blobs}")

    if not blobs:
        print("No files found in the bucket.")
        return

    # Filter blobs based on prefix
    if prefix:
        filtered_blobs = [blob for blob in blobs if blob.name.startswith(prefix)]
    else:
        filtered_blobs = blobs

    if not filtered_blobs:
        print("No files matched the prefix.")
        return

    if args.max:
        filtered_blobs = filtered_blobs[:int(args.max)]

    gene_names = []
    texts = []

    # Load cache dictionary if it exists
    if os.path.exists('cache.pkl'):
        with open('cache.pkl', 'rb') as f:
            cache_dict = pickle.load(f)
    else:
        cache_dict = {}

    # Download files with a progress bar
    for i, blob in tqdm(enumerate(filtered_blobs), desc="Processing files"):
        blob_name = blob.name

        if blob_name in cache_dict:
            content = cache_dict[blob_name]
        else:
            # Read the content of the file as text
            content = blob.download_as_text()

            # Extract only overview
            try:
                content = content.split('## Overview')[1].split('##')[0]
            except IndexError:
                # Skip if '## Overview' section is not found
                continue

            # Save processed content to cache
            cache_dict[blob_name] = content

            # Save cache dictionary to disk after every 100 files
            if i % 100 == 0:
                with open('cache.pkl', 'wb') as f:
                    pickle.dump(cache_dict, f)

        # Store content and file path
        texts.append(content)
        gene_names.append(blob_name.split("/")[-1].split(".txt")[0])

    if not texts:
        print("No valid content found after processing.")
        return

    # Compute embeddings with batching and progress bar
    embeddings = embed_texts(texts)
    embeddings = np.array(embeddings)

    # Save embeddings
    np.save('embeddings.npy', embeddings)

    # Save texts and gene_names using pickle
    with open('texts.pkl', 'wb') as f:
        pickle.dump(texts, f)

    with open('gene_names.pkl', 'wb') as f:
        pickle.dump(gene_names, f)


if __name__ == "__main__":
    main()