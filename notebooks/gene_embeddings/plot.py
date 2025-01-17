import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import umap

def main():
    # Load data
    embedding_path = "notebooks/gene_embeddings/gene_representations_llamaPMC.pt"
    generep_names = torch.load(embedding_path)
    gene_representations = generep_names['repr'][:5000]

    # Convert to numpy and standardize
    data = gene_representations.cpu().numpy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # PCA reduction
    print("Running PCA...")
    pca = PCA(n_components=100)
    data_pca = pca.fit_transform(data_scaled)
    print("PCA DONE")

    # Calculate variance explained
    var_explained = pca.explained_variance_ratio_.cumsum()
    print(f"Variance explained by 100 PCs: {var_explained[-1]:.3f}")

    # UMAP reduction
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
        low_memory=True
    )
    embedding = reducer.fit_transform(data_pca)
    print("UMAP DONE")

    # Create and save plot
    plt.figure(figsize=(12, 10))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=1,
        alpha=0.5,
        c='blue'
    )
    plt.title('UMAP visualization of gene representations (PCA -> UMAP)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    sns.despine()
    plt.tight_layout()

    # Save plot in same directory as embedding
    output_dir = os.path.dirname(embedding_path)
    output_path = os.path.join(output_dir, 'gene_representations_umap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()