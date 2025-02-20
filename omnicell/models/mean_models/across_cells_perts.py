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
from omnicell.models.mean_models.L4Regressor import L4Regressor
from sklearn.model_selection import GridSearchCV

from omnicell.models.utils.distribute_shift import sample_pert, get_proportional_weighted_dist

import logging
logger = logging.getLogger(__name__)

def fit_supervised_model(X, Y, model_type='linear', param_grid=None, **kwargs):
    """
    Fit a supervised model, optionally performing hyperparameter tuning with GridSearchCV.
    
    Args:
        X: Input features.
        Y: Target values.
        model_type: Type of model to fit.
        param_grid: Hyperparameters grid for GridSearchCV.
        **kwargs: Fixed parameters for the model.
        
    Returns:
        Fitted model, training MSE, R2 score.
    """
    models = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
        'rf': RandomForestRegressor,
        'svr': SVR,
        'l4': L4Regressor
    }
    
    if model_type not in models:
        raise ValueError(f"Model type {model_type} not supported.")
    
    model_class = models[model_type]
    base_model = model_class(n_jobs=8, **kwargs)
    
    if param_grid:
        logger.debug(f"Performing hyperparameter tuning with {model_type}, on param grid: {param_grid} with 2-fold CV")
        grid_search = GridSearchCV(base_model, param_grid, cv=2, n_jobs = 8, scoring='neg_mean_squared_error')
        grid_search.fit(X, Y)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        best_model = base_model.fit(X, Y)
    
    Y_pred = best_model.predict(X)
    mse = mean_squared_error(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)
    return best_model, mse, r2

def compute_cell_type_means(adata, cell_type):
    """Compute perturbation effect embeddings for a specific cell type"""
    
    # Filter data for this cell type
    cell_type_data = adata[adata.obs[CELL_KEY] == cell_type]
    
    # Compute control mean for this cell type
    ctrl_mean = np.mean(
        cell_type_data[cell_type_data.obs[PERT_KEY] == CONTROL_PERT].X, axis=0
    )
    
    # Convert to dense array if sparse
    X = cell_type_data.X
    
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

class CellxPertEmbMeanPredictor():

    def __init__(self, model_config: dict, pert_embedding: dict):
        self.model = None
        self.model_type = model_config['model_type']
        self.pca_pert_embeddings = model_config['pca_pert_embeddings']
        self.pca_pert_embeddings_components = model_config['pca_pert_embeddings_components']
        self.pca_cell_embeddings_components = model_config['pca_cell_emb_components']
        self.pert_embedding = pert_embedding
        self.cell_embeddings = {}
        self.param_grid = model_config.get('param_grid', None)


    def train(self, adata: sc.AnnData, **kwargs):
        if self.pca_pert_embeddings:
            pca = PCA(n_components=self.pca_pert_embeddings_components)
            pert_emb_temp = pca.fit_transform(np.array(list(self.pert_embedding.values())))
            self.pert_embedding = {pert : pert_emb_temp[i] for i, pert in enumerate(self.pert_embedding.keys())}

        
        #Generating cell embeddings with PCA

        self.pca_model_cells = PCA(n_components=self.pca_cell_embeddings_components)

        X = adata.X
        logger.debug(f"Training PCA model for cell embeddings with {X.shape[0]} cells")
        self.pca_model_cells.fit(X)

        logger.debug("Computing cell embeddings")
        for cell_type in adata.obs[CELL_KEY].unique():
            cell_data = adata[(adata.obs[CELL_KEY] == cell_type) & (adata.obs[PERT_KEY] == CONTROL_PERT)].X
            self.cell_embeddings[cell_type] = np.mean(self.pca_model_cells.transform(cell_data), axis=0)


        # Get unique cell types
        cell_types = adata.obs[CELL_KEY].unique()

        # Compute embeddings for each cell type
        Xs = []
        Ys = []

        #We append the data for each cell type, 
        for cell_type in cell_types:
            logger.debug(f"Creating training data for {cell_type}")
            ctrl_mean, pert_deltas_dict = compute_cell_type_means(adata, cell_type)
            
            
            # Get perturbation IDs for this cell type
            idxs = pert_deltas_dict.keys()
            
            # Create feature matrix X and target matrix Y
            Y = np.array([pert_deltas_dict[pert] for pert in idxs])
            X = np.array([np.concatenate([self.pert_embedding[g], self.cell_embeddings[cell_type]])  for g in idxs])

            Xs.append(X)
            Ys.append(Y)

        # Now you can train a model for each cell type
        X = np.concatenate(Xs)
        Y = np.concatenate(Ys)
        self.model, mse, r2 = fit_supervised_model(X, Y, model_type=self.model_type, param_grid=self.param_grid)
        
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        ctrl_cells = adata[(adata.obs[PERT_KEY] == CONTROL_PERT) & (adata.obs[CELL_KEY] == cell_type)].X

        cell_embedding = np.mean(self.pca_model_cells.transform(ctrl_cells), axis=0)

        X_new = np.concatenate([self.pert_embedding[pert_id], cell_embedding])
        X_new = X_new.reshape(1, -1)
        
        mean_shift_pred = np.array(self.model.predict(X_new)).flatten().astype(np.float32)
        weighted_dist = get_proportional_weighted_dist(ctrl_cells)
        samples = sample_pert(ctrl_cells, weighted_dist, mean_shift_pred)
        return samples
