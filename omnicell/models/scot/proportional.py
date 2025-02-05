import sys
import yaml
import torch
import logging
from pathlib import Path
import time

# Add the path to the directory containing the omnicell package
# Assuming the omnicell package is in the parent directory of your notebook

import numpy as np
import scanpy as sc
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT, logger
from omnicell.models.distribute_shift import sample_pert, get_proportional_weighted_dist
from omnicell.processing.utils import to_dense

import logging

logger = logging.getLogger(__name__)
class ProportionalSCOT():

    def __init__(self, adata: sc.AnnData):
        self.model = None
        self.total_adata = adata

    def train(self, adata: sc.AnnData):
        """Does nothing because we are going to cheat"""
        pass

    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        X_ctrl = to_dense(self.total_adata[(self.total_adata.obs[PERT_KEY] == CONTROL_PERT) & (self.total_adata.obs[CELL_KEY] == cell_type)].X.toarray())
        X_pert = to_dense(self.total_adata[(self.total_adata.obs[PERT_KEY] == pert_id) & (self.total_adata.obs[CELL_KEY] == cell_type)].X.toarray())
        mean_ctrl = X_ctrl.mean(axis=0)
        mean_pert = X_pert.mean(axis=0)
        
        mean_shift = mean_pert - mean_ctrl
        logger.debug(f"Mean shift shape: {mean_shift.shape}")

        weighted_dist = get_proportional_weighted_dist(X_ctrl)
        weighted_dist = weighted_dist.astype(np.float64)
        s = weighted_dist.sum(axis=0)
        weighted_dist[:, s > 0] /= s[s > 0]
        
        preds = sample_pert(X_ctrl, weighted_dist, mean_shift, max_rejections=100)
        return preds
    


    def generate_synthetic_counterfactuals(self, dset: torch.utils.data.Dataset, stratum, batch_size, i) -> np.ndarray:

        source_batch = {} 
        synthetic_counterfactual_batch = {}

            source_batch[stratum] = X_ctrl = dset.source[stratum][i:i+batch_size]
            synthetic_counterfactual_batch[stratum] = {}

            mean_ctrl = X_ctrl.mean(axis=0)
            
            # Time the weighted dist calculation
            dist_start = time.time()
            weighted_dist = get_proportional_weighted_dist(X_ctrl)
            weighted_dist = weighted_dist.astype(np.float64)
            s = weighted_dist.sum(axis=0)
            weighted_dist[:, s > 0] /= s[s > 0]
            
            dist_time = time.time() - dist_start
            logger.info(f"Weighted dist calculation took: {dist_time:.2f}s")

            for j, pert in enumerate(dset.unique_pert_ids):
                if j % 10 == 0:
                    pert_start = time.time()
                    logger.info(f"{j} / {len(dset.unique_pert_ids)}")
                
                X_pert = dset.target[stratum][pert]
                mean_pert = X_pert.mean(axis=0)
                mean_shift = mean_pert - mean_ctrl
                
                # Time the sample_pert call
                preds = sample_pert(
                    X_ctrl, 
                    weighted_dist, 
                    mean_shift, 
                    max_rejections=100, 
                    # num_threads=2
                )
                
                synthetic_counterfactual_batch[stratum][pert] = preds.astype(np.int16)
                
                if (j + 1) % 10 == 0:
                    pert_time = time.time() - pert_start
                    logger.info(f"Perturbation {j} took: {pert_time:.2f}s")
            
        # Save timing data along with results
        data_dict = {
            'synthetic_counterfactuals': synthetic_counterfactual_batch,
            'source': source_batch,
            'unique_pert_ids': dset.unique_pert_ids,
            'strata': dset.strata,
        }

        return data_dict
            