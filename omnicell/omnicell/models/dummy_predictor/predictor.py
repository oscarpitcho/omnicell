
import scanpy as sc
import numpy as np

from omnicell.constants import *


"""
Predictor to test the rest of the pipeline. 

It takes the entire dataset at construction and then uses the ground truth to make predictions. Thus it is a perfect predictor.

"""
class TestPredictor():


    def __init__(self, adata: sc.AnnData):
        self.model = None
        self.total_adata = adata.copy()

    

    def train(self, adata: sc.AnnData):
        """Does nothing because we are going to cheat"""
        pass



    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:

        number_preds = len(adata)

        ground_truth = self.total_adata[(self.total_adata.obs[PERT_KEY] == pert_id) & (self.total_adata.obs[CELL_KEY] == cell_type)]

        #sampled_gt = sc.pp.subsample(ground_truth, n_obs=number_preds, replace=True, copy=True)


        return ground_truth.X
