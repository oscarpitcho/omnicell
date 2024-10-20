import numpy
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT
import logging
from typing import Optional, Tuple, Dict
import pickle

from omnicell.models.metric_fns import distance_metrics


logger = logging.getLogger(__name__)

class NearestNeighborPredictor():
    def __init__(self, config, pert_rep=None, pert_map=None, cell_rep=None):
        self.config = config
        self.train_adata = None
        self.seen_cell_types = None
        self.seen_perts = None
        self.mean_shift = config['mean_shift']

        self.pert_dist_fn = distance_metrics[config['pert_dist_metric']]
        self.cell_dist_fn = distance_metrics[config['cell_dist_metric']]

        self.cell_rep = cell_rep
        self.pert_rep = pert_rep
        self.pert_map = pert_map

    def train(self, adata):
        """
        Trains the model on the given data.

        Parameters
        ----------
        adata : AnnData
                
        pert_embedding : Tuple[Dict, np.ndarray]
            Tuple containing the perturbation embedding dictionary and the perturbation embedding matrix
        """
        self.train_adata = adata
        self.seen_cell_types = adata.obs[CELL_KEY].unique()
        self.seen_perts = [pert for pert in adata.obs[PERT_KEY].unique() if pert != CONTROL_PERT]

        if self.cell_rep is None:
            cell_rep = []
            for cell_type in self.seen_cell_types:
                cell_rep.append(self.train_adata[(self.train_adata.obs[CELL_KEY] == cell_type) & (self.train_adata.obs[PERT_KEY] == CONTROL_PERT)].obsm['embedding'].mean(axis=0))
            cell_rep = np.squeeze(np.array(cell_rep))
        self.cell_rep = cell_rep
    
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        assert self.train_adata is not None, "Model has not been trained yet"
        if cell_type in self.seen_cell_types:
            if pert_id in self.seen_perts:
                raise NotImplementedError(f"Both cell type: {cell_type} and perturbation: {pert_id} are in the training data, in distribution prediction not implemented yet")
            else:
                return self._predict_across_pert(adata, pert_id, cell_type)
        else:
            if pert_id in self.seen_perts:
                return self._predict_across_cell(adata, pert_id, cell_type)
            else:
                raise NotImplementedError(f"Neither cell type: {cell_type} and perturbation: {pert_id}, out of distribution prediction not implemented yet")
               

    def _predict_across_cell(self, heldout_cell_adata: sc.AnnData, target_pert: str, cell_id: str) -> np.ndarray:
        """
        Makes a prediction for a seen target perturbation given some unseen cell type. 
        
        We find the closest cell type in the control state and apply apply the average effect of the perturbation on the neighboring cell type to our heldout cell data.

        Parameters
        ----------
        heldout_cell_adata : AnnData
            The AnnData object containing the unseen cell type (and only that cell type) with control perturbation
        target_pert : str
            The target perturbation ID to predict

        Returns
        -------
        np.ndarray
            The predicted perturbation for the target using the control perturbation

        """
        assert self.train_adata is not None, "Model has not been trained yet"
        assert heldout_cell_adata.obs[CELL_KEY].nunique() == 1, "Heldout cell data must contain only one cell type"
        assert heldout_cell_adata.obs[CELL_KEY].unique()[0] == cell_id, "Heldout cell data must contain only one cell type"
        assert heldout_cell_adata.obs[PERT_KEY].nunique() == 1, "Heldout cell data must contain only control data"
        assert heldout_cell_adata.obs[PERT_KEY].unique()[0] == CONTROL_PERT, "Heldout cell data must contain only control data"

        #Mean control state of the heldout cell
        heldout_cell_rep = heldout_cell_adata.obsm['embedding'].mean(axis=0)
    
        #Computing distances
        # diffs = train_cell_rep - heldout_cell_rep
        #Applying L2 distance, could be changed to L1
        # squared_diffs = np.square(diffs)

        # distances_to_heldout = np.sum(squared_diffs, axis=1)

        distances_to_heldout = self.cell_dist_fn(self.cell_rep, heldout_cell_rep)
        closest_cell_type_idx = np.argmin(distances_to_heldout)

        logger.debug(f"Closest cell type to evaluated cell_type {cell_id} is {self.seen_cell_types[closest_cell_type_idx]}")
        closest_cell_type = self.seen_cell_types[closest_cell_type_idx]

        if self.mean_shift:
            perturbed_closest_cell_type = self.train_adata[(self.train_adata.obs[CELL_KEY] == closest_cell_type) & (self.train_adata.obs[PERT_KEY] == target_pert)].obsm['embedding'].mean(axis=0)
            pert_effect = perturbed_closest_cell_type - self.cell_rep[closest_cell_type_idx]
            pert_effect_norm = np.linalg.norm(pert_effect)

            logger.debug(f"Perturbation effect norm {pert_effect_norm}")
            #Apply the perturbation effect to the heldout cell data
            predicted_perts = heldout_cell_adata.obsm['embedding'] + pert_effect
            return predicted_perts
        else:
            adata_nbr = self.train_adata[(self.train_adata.obs[CELL_KEY] == closest_cell_type) & (self.train_adata.obs[PERT_KEY] == target_pert)]
            #TODO: FIX - It might cause issue with the predict method returns a different number of cells than the heldout cell data
            #res = sc.pp.subsample(adata_nbr, n_obs=len(heldout_cell_adata), copy=True)
            res = adata_nbr #sc.pp.subsample(adata_nbr, n_obs=len(adata), replace=True, copy=True)
            return res.obsm['embedding']

    #SO I want to predict across genes --> Two options either we provide the data or we don't provide the data on which the prediction is made
    def _predict_across_pert(self, adata: sc.AnnData, target_pert: str, cell_type: str) -> np.ndarray:
        """
        Makes a prediction for an unseen perturbation.
        
        Takes the perturbation effect in the training data which is closest to the heldout perturbation and applies it to the given control data

        Parameters
        ----------
        target : str
            The target perturbation to predict

        cell_id : str
            The cell type of the data on which the prediction is done

        adata : AnnData
            The AnnData object control data on the cell type of the prediction.

        Returns
        -------
        np.ndarray
            The predicted perturbation for the target using the control perturbation datapoints of the training.

        """

        assert self.train_adata is not None, "Model has not been trained yet"
        assert target_pert not in self.train_adata.obs[PERT_KEY].unique(), "Target perturbation is already in the training data"
        assert adata.obs[CELL_KEY].nunique() == 1, "Heldout cell data must contain only one cell type"
        assert adata.obs[CELL_KEY].unique()[0] == cell_type, "Heldout cell data must contain only one cell type"
        assert adata.obs[PERT_KEY].nunique() == 1, "Heldout cell data must contain only control data"
        assert adata.obs[PERT_KEY].unique()[0] == CONTROL_PERT, "Heldout cell data must contain only control data"
        logger.debug(f'Predicting unseen perturbation {target_pert} using all training data')

        
        cell_type_idx = np.where(self.seen_cell_types == cell_type)[0][0]
        
        
        # Computing distances
        distances_to_target = self.pert_dist_fn(self.pert_rep[list(map(self.pert_map.get, self.seen_perts))], self.pert_rep[self.pert_map[target_pert]])
        
        # Sort perturbations by distance
        sorted_indices = np.argsort(distances_to_target)
        sorted_perts = [self.seen_perts[i] for i in sorted_indices]

        # Log the ranking of perturbations
        num_to_log = min(10, len(sorted_perts))  # Log top 10 or all if less than 10
        logger.debug(f'Top {num_to_log} closest perts to {target_pert}: {sorted_perts[:num_to_log]}')

        closest_pert = sorted_perts[0]
        logger.debug(f'Nearest neighbor perturbation of {target_pert} is {closest_pert}')


        #Mean control state of each cell type
        if self.mean_shift:
            logger.debug("Running shift method")
            perturbed_closest_pert_type = self.train_adata[(self.train_adata.obs[CELL_KEY] == cell_type) & (self.train_adata.obs[PERT_KEY] == target_pert)].obsm['embedding'].mean(axis=0)
            pert_effect = perturbed_closest_pert_type - self.cell_rep[cell_type_idx]
            selected_cell_control_mean = self.train_adata[(self.train_adata.obs[CELL_KEY] == cell_type) & (self.train_adata.obs[PERT_KEY] == CONTROL_PERT)].obsm['embedding'].mean(axis=0)
            selected_cell_nbr_pert_mean = self.train_adata[(self.train_adata.obs[CELL_KEY] == cell_type) & (self.train_adata.obs[PERT_KEY] == closest_pert)].obsm['embedding'].mean(axis=0)
            pert_effect = selected_cell_nbr_pert_mean - selected_cell_control_mean
            predictions = adata.copy()
            predictions.obsm['embedding'] = pert_effect + predictions.obsm['embedding']
            return predictions.obsm['embedding']
        else:
            logger.debug("Running substitution method")
            adata_nbr = self.train_adata[(self.train_adata.obs[CELL_KEY] == cell_type) & (self.train_adata.obs[PERT_KEY] == closest_pert)]
            logger.debug(f"Number of cells with cell_id {cell_type} and perturbation {closest_pert} in training data {len(adata_nbr)}")
            res = adata_nbr #sc.pp.subsample(adata_nbr, n_obs=len(adata), replace=True, copy=True)
            return res.obsm['embedding']
        




    


