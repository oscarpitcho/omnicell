from ..constants import PERT_KEY, CELL_TYPE_KEY, CONTROL_PERT
import numpy as np
import scanpy as sc






class Splitter:

    def __init__(self, config):
        self.config = config

        self.holdout_cells = None
        self.holdout_perts = None


    def split(self, data: sc.AnnData):
        """
        Splits the given adata acording to the config and task specifications

        Parameters
        ----------
        data : AnnData
            The AnnData object to split, should have been preprocessed with the correct key names for perturbations and cell types already.

        
        Returns
        -------
        train : AnnData
            The training data


        
        
        
        
        """

        train_mask = (data.obs[PERT_KEY] not in self.holdout_perts) & (data.obs[CELL_TYPE_KEY] not in self.holdout_genes)
        train = data.obs[train_mask]

        #For now forget random splits and folds --> across 


