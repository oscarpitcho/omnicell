from ..constants import PERT_KEY, CELL_TYPE_KEY, CONTROL_PERT
import numpy as np
import scanpy as sc
from typing import List, Tuple, Dict





#TODO: Lotsa implementation details to be filled in here
"""
- Requirements:
    - Supporting random folds
    - Supporting undefined tasks, i.e. because we might need to split the data and are just interested in training, not evaluations yet

"""
class Splitter:

    def __init__(self, task_config):
        self.config = task_config

        self.holdout_cells = self.config.get('cell_holdout', [])
        self.holdout_perts = self.config.get('pert_holdout', [])


    def split(self, data: sc.AnnData)-> List[Tuple[sc.AnnData, sc.AnnData, List[str], List[str]]]:
        """
        Splits the given adata acording to the config and task specifications

        Parameters
        ----------
        data : AnnData
            The AnnData object to split, should have been preprocessed with the correct key names for perturbations and cell types already.

        
        Returns
        -------

        A fold composed of 4-tuples made of 
        (adata_train, adata_eval, holdout_genes, holdout_cells)

        adata_train : AnnData
            The training data

        adata_eval : AnnData
            The evaluation data, i.e. any datapoint that contained a holdout perturbation or cell type

        holdout_perts : List[str]
            The perturbations that were held out
        
        holdout_cells : List[str]
            The cell types that were held out
        
        """


        #TODO: Implement random folds

        train_mask = (data.obs[PERT_KEY] not in self.holdout_perts) & (data.obs[CELL_TYPE_KEY] not in self.holdout_cells)
        adata_train = data.obs[train_mask]
        adata_eval = data.obs[~train_mask]


        return [(adata_train, adata_eval, self.holdout_perts, self.holdout_cells)]



