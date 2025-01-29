from omnicell.constants import PERT_KEY, GENE_EMBEDDING_KEY, CONTROL_PERT, CELL_KEY

import scanpy as sc
import pandas as pd 
import numpy as np
import scipy

# After the existing imports, add:
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from pathlib import Path


def expected_distribute_shift(ctrl_cells, shift_pred):
    cell_fractions = ctrl_cells.sum(axis=1) / ctrl_cells.sum()
    Z = ctrl_cells + shift_pred[None, :] * cell_fractions[:, None] * ctrl_cells.shape[0]
    return Z

def distribute_shift(ctrl_cells, mean_shift):
    """
    Distribute the global per-gene difference (sum_diff[g]) across cells in proportion
    to the cell's existing counts for that gene. 
    """ 
    ctrl_cells = ctrl_cells.copy()
    sum_shift = (mean_shift * ctrl_cells.shape[0]).astype(int)

    n_cells, n_genes = ctrl_cells.shape

    #For each gene, distribute sum_diff[g] using a single multinomial draw
    for g in range(n_genes):
        diff = int(sum_shift[g])
        if diff == 0:
            continue  

        # Current counts for this gene across cells
        gene_counts = ctrl_cells[:, g].astype(np.float64)

        current_total = gene_counts.sum().astype(np.float64)
        

        # Probabilities ~ gene_counts / current_total
        p = gene_counts / current_total


        if diff > 0:
            # We want to add `diff` counts
            draws = np.random.multinomial(diff, p)  # shape: (n_cells,)
            
            ctrl_cells[:, g] = gene_counts + draws
        else:
            if current_total <= 0:
                continue

            # We want to remove `abs(diff)` counts
            amt_to_remove = abs(diff)

            to_remove = min(amt_to_remove, current_total)
            draws = np.random.multinomial(to_remove, p)
            # Subtract, then clamp
            updated = gene_counts - draws
            updated[updated < 0] = 0
            ctrl_cells[:, g] = updated

    return ctrl_cells

def fit_supervised_model(X, Y, model_type='linear', **kwargs):
    """
    Fit a supervised model based on the specified model type.
    
    Args:
        X: Input features (gene embeddings)
        Y: Target values (perturbation effects)
        model_type: Type of model to fit ('linear', 'ridge', 'lasso', 'elastic_net', 'rf', 'svr')
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        fitted model, training MSE, R2 score
    """
    models = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
        'rf': RandomForestRegressor,
        'svr': SVR
    }
    
    if model_type not in models:
        raise ValueError(f"Model type {model_type} not supported. Choose from {list(models.keys())}")
    
    model = models[model_type](**kwargs)
    model.fit(X, Y)
    
    # Make predictions and calculate metrics
    Y_pred = model.predict(X)
    mse = mean_squared_error(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)
    
    return model, mse, r2

def compute_cell_type_means(adata, cell_type):
    """Compute perturbation effect embeddings for a specific cell type"""
    # Filter data for this cell type
    cell_type_data = adata[adata.obs[CELL_KEY] == cell_type]
    
    # Compute control mean for this cell type
    ctrl_mean = np.mean(
        cell_type_data[cell_type_data.obs[PERT_KEY] == CONTROL_PERT].X, axis=0
    )
    
    # Convert to dense array if sparse
    X = cell_type_data.X.toarray() if scipy.sparse.issparse(cell_type_data.X) else cell_type_data.X
    
    # Create dataframe
    df = pd.DataFrame(X, index=cell_type_data.obs.index)
    df['perturbation'] = cell_type_data.obs[PERT_KEY].values
    
    # Compute means per perturbation
    pert_means = df.groupby('perturbation').mean()
    
    # Compute deltas from control
    pert_deltas = pd.DataFrame(pert_means.values - ctrl_mean, index=pert_means.index)
    pert_deltas_dict = {
        pert: np.array(means) 
        for pert, means in pert_deltas.iterrows() 
        if pert != CONTROL_PERT
    }
    
    return ctrl_mean, pert_deltas_dict

class MeanPredictor():

    def __init__(self, model_config: dict, pert_rep_map: dict):
        self.model = None
        self.model_type = model_config['model_type']
        self.pca_pert_embeddings = model_config['pca_pert_embeddings']
        self.pca_pert_embeddings_components = model_config['pca_pert_embeddings_components']
        self.pert_rep_map = pert_rep_map

    def train(self, adata: sc.AnnData, model_savepath: Path):
        if self.pca_pert_embeddings:
            pca = PCA(n_components=self.pca_pert_embeddings_components)
            pert_emb_temp = pca.fit_transform(np.array(list(self.pert_rep_map.values())))
            self.pert_rep_map = {pert : pert_emb_temp[i] for i, pert in enumerate(self.pert_rep_map.keys())}

        # Get unique cell types
        cell_types = adata.obs[CELL_KEY].unique()

        # Compute embeddings for each cell type
        Xs = []
        Ys = []
        for cell_type in cell_types:
            ctrl_mean, pert_deltas_dict = compute_cell_type_means(adata, cell_type)
            
            # Get perturbation IDs for this cell type
            idxs = pert_deltas_dict.keys()
            
            # Create feature matrix X and target matrix Y
            Y = np.array([pert_deltas_dict[pert] for pert in idxs])
            X = np.array([self.pert_rep_map[g] for g in idxs])

            # Store the embeddings
            Xs.append(X)
            Ys.append(Y)

        # Now you can train a model for each cell type
        X = np.concatenate(Xs)
        Y = np.concatenate(Ys)
        self.model, mse, r2 = fit_supervised_model(X, Y, model_type=self.model_type)
        
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        ctrl_cells = adata[(adata.obs[PERT_KEY] == CONTROL_PERT) & (adata.obs[CELL_KEY] == cell_type)].X.toarray()
        X_new = np.array(self.pert_rep_map[pert_id].reshape(1, -1))
        shift_pred = np.array(self.model.predict(X_new)).flatten()
        return distribute_shift(ctrl_cells, shift_pred)