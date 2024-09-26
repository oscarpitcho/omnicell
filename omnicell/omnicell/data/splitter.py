from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT
import numpy as np
import scanpy as sc
from typing import List, Tuple, Dict
from omnicell.config.config import Config
import logging

"""


                Task Definition



Predict
    \
     \           Pert 1                        Pert 2
      \
   On  \
        \
         \__________________________________________________________
         |                              |                           |
         |                              | • We see Pert 2           |
         |  Same logic as bottom right  |   at training but         |
  Cell A |  Quadrant                    |   never on cell A         |
         |                              | • We never see Pert 2     |
         |                              |   at training             |
         |                              |                           |
         |______________________________|___________________________|
         |                              |                           |
         | • We see unperturbed         | • We never see the        |
         |   Cell B at training         |   combination Cell B +    |
         | • We never see Cell B        |   Gene 2 at training but  |
  Cell B |   at training                |   we see them separately  |
         |                              | • We see Unperturbed      |
         |                              |   Cell B but never see    |
         |                              |   Gene 2 (and reverse)    |
         |                              | • We never see Pert 1     |
         |                              |   or Cell B               |
         |                              |                           |
         |______________________________|___________________________|

Bullet points are sorted by increasing difficulty


"""





#TODO: Lotsa implementation details to be filled in here
"""
- Requirements:
    - Supporting random folds
    - Supporting undefined tasks, i.e. because we might need to split the data and are just interested in training, not evaluations yet

"""
logger = logging.getLogger(__name__)    
class Splitter:
    """
    If ood is on any observation with a heldout perturbation or cell type is considered as eval data

    If iid is on only observations with heldout _perturbed_ cells or observations with heldout perturbations _on heldout cells_ are considered as eval data
    
    
    """

    def __init__(self, config: Config):
        self.config = config

        self.holdout_cells = self.config.get_heldout_cells()
        self.holdout_perts = self.config.get_heldout_perts()

        self.eval_targets = self.config.get_eval_targets()

        self.mode = self.config.get_mode()

        assert self.mode == "iid" or self.mode == "ood", f"Unsupported mode {self.mode} specified, only iid and ood are supported"


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

        #Consumes more mem
        data = data.copy()


        if self.mode == "ood":
            logger.info("Doing OOD split")
            train_mask = (~data.obs[PERT_KEY].isin(self.holdout_perts)) & (~data.obs[CELL_KEY].isin(self.holdout_cells))


        #IID, we include unperturbed holdouts cells
        else:

            logger.info("Doing IID split")
            heldout_cell_mask = (data.obs[CELL_KEY].isin(self.holdout_cells)) & (data.obs[PERT_KEY] != CONTROL_PERT)

            heldout_pert_mask = (data.obs[PERT_KEY].isin(self.holdout_perts)) & (~data.obs[CELL_KEY].isin(self.holdout_cells))

            holdout_mask = heldout_cell_mask | heldout_pert_mask

            train_mask = ~holdout_mask

        adata_train = data[train_mask]
        adata_eval = data[~train_mask]

        #TODO: Implement count normalizations?



        return [(adata_train, adata_eval, self.holdout_perts, self.holdout_cells, self.eval_targets)]



