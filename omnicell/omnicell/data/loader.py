import scanpy as sc
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from omnicell.config.config import Config
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT
from omnicell.data.catalogue import DatasetDetails, Catalogue
import torch

import logging

logger = logging.getLogger(__name__)


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


#We define an enum which is either Training or Evaluation
#We can then use this to determine which data to load



#TODO: Want to include generic dataset caching, we might starting having many datasets involved in training, not just one

class DataLoader:
    def __init__(self, config: Config, data_catalogue: Catalogue):
        self.config = config
        self.data_catalogue = data_catalogue
        self.training_dataset_details: DatasetDetails = data_catalogue.get_dataset_details(config.get_training_dataset_name())

        logger.debug(f"Training dataset details: {self.training_dataset_details}")

        self.pert_embedding_name: Optional[str] = config.get_pert_embedding_name()

        self.cell_embedding_name: Optional[str] = config.get_cell_embedding_name()
        
        #TODO: Handle
        self.pert_embedding_details: Optional[dict] = None

        self.eval_dataset_details: DatasetDetails = None

        #We only store the data once it has been preprocessed
        self.complete_training_adata: Optional[sc.AnnData] = None
        self.complete_eval_adata: Optional[sc.AnnData] = None
        

    # TODO: This processing should be common between the training and the eval data
    # Mutates the adata object
    def preprocess_data(self, adata: sc.AnnData, training: bool) -> sc.AnnData:

        # Standardize column names and key values
        condition_key = self.training_dataset_details.pert_key if training else self.eval_dataset_details.pert_key
        cell_key = self.training_dataset_details.cell_key if training else self.eval_dataset_details.cell_key
        control = self.training_dataset_details.control if training else self.eval_dataset_details.control

        #TODO: If we could rename the columns it would be better
        adata.obs[PERT_KEY] = adata.obs[condition_key]
        adata.obs[CELL_KEY] = adata.obs[cell_key]
        adata.obs[PERT_KEY] = adata.obs[PERT_KEY].cat.rename_categories({control: CONTROL_PERT})

        if (self.config.get_cell_embedding_name() is not None) & (self.config.get_apply_normalization() | self.config.get_apply_log1p()):
            raise ValueError("Cannot both apply cell embedding and normalization/log1p transformation")
        
        elif self.config.get_cell_embedding_name() is not None:

            if self.config.get_cell_embedding_name() not in self.training_dataset_details.cell_embeddings:
                raise ValueError(f"Cell embedding {self.config.get_cell_embedding_name()} not found in embeddings available for dataset {self.training_dataset_details.name}")
            
            #We replace the data matrix with the cell embeddings
            adata.X = adata.obsm[self.config.get_cell_embedding_name()]

        else:

            adata.X = adata.X.toarray().astype('float32')
            # Set gene names
            if self.training_dataset_details.var_names_key:
                adata.var_names = adata.var[self.training_dataset_details.var_names_key]

            # Apply normalization and log1p if needed
            if self.config.get_apply_normalization() & (not self.training_dataset_details.count_normalized):
                sc.pp.normalize_total(adata, target_sum=10_000)
            elif not self.config.get_apply_normalization() & self.training_dataset_details.count_normalized:
                raise ValueError("Specified dataset is count normalized, but normalization is turned off in the config")
            
            if self.config.get_apply_log1p() & (not self.training_dataset_details.log1p_transformed):
                sc.pp.log1p(adata)
            elif not self.config.get_apply_log1p() & self.training_dataset_details.log1p_transformed:
                raise ValueError("Specified dataset is log1p transformed, but log1p transformation is turned off in the config")


        return adata

    def get_training_data(self) -> Tuple[sc.AnnData, Optional[dict]]:
        """
        Returns the training data according to the config.
        If an pert embedding is specified then it is also returned
        """

        pert_embedding = None 
        if self.pert_embedding_name is not None:
            if self.pert_embedding_name not in self.training_dataset_details.pert_embeddings:
                print(self.training_dataset_details.pert_embeddings)
                raise ValueError(f"Perturbation embedding {self.pert_embedding_name} not found in embeddings available for dataset {self.training_dataset_details.name}")
            else:
                logger.info(f"Loading perturbation embedding from {self.training_dataset_details.folder_path}/{self.pert_embedding_name}.pt")
                pert_embedding = torch.load(f"{self.training_dataset_details.folder_path}/{self.pert_embedding_name}.pt")

        # Checking if we have already a cached version of the training data
        if self.complete_training_adata is None:

            logger.info(f"Loading training data at path: {self.training_dataset_details.path}")
            adata = sc.read(self.training_dataset_details.path)


            logger.info("Preprocessing training data")
            adata = self.preprocess_data(adata, training=True)

            self.complete_training_adata = adata

            logger.debug(f"Loaded complete data, # of data points: {len(adata)}, # of genes: {len(adata.var)}, # of conditions: {len(adata.obs[PERT_KEY].unique())}")


        if self.config.get_mode() == "ood":
            logger.info("Doing OOD split")

            # Taking cells for training where neither the cell nor the perturbation is held out
            holdout_mask = (adata.obs[PERT_KEY].isin(self.config.get_heldout_perts())) | (adata.obs[CELL_KEY].isin(self.config.get_heldout_cells()))
            train_mask = ~holdout_mask

        # IID, we include unperturbed holdouts cells
        else:
            logger.info("Doing IID split")
            # Holding out only cells that have heldout perturbations AND cell. Thus:
            # A perturbation will be included on the non holdout cell type eg
            # Control of heldout cell type will be included
            holdout_mask = (adata.obs[CELL_KEY].isin(self.config.get_heldout_cells())) & (adata.obs[PERT_KEY].isin(self.config.get_heldout_perts()))
            train_mask = ~holdout_mask

        adata_train = adata[train_mask]

        return adata_train, pert_embedding


    def get_complete_training_dataset(self) -> sc.AnnData:
        """
        Returns the entire dataset for training, including the heldout cells and perturbations.
        Data is preprocessed according to config
        """
        if self.complete_training_adata is None:
            adata = sc.read(self.training_dataset_details.path)
            logger.info("Preprocessing training data")
            self.complete_training_adata = self.preprocess_data(adata, training=True)
 
        return self.complete_training_adata

    def get_eval_data(self):
        self.eval_dataset_details = self.data_catalogue.get_dataset_details(self.config.get_eval_dataset_name())

        if self.eval_dataset_details.name == self.training_dataset_details.name:
            self.complete_eval_adata = self.get_complete_training_dataset()

        else:

            logger.info(f"Loading evaluation data at path: {self.eval_dataset_details.path}")
            adata = sc.read(self.eval_dataset_details.path)

            logger.info("Preprocessing evaluation data")
            adata = self.preprocess_data(adata, training=False)
            self.complete_eval_adata = adata


        logger.debug(f"Eval targets are {self.config.get_eval_targets()}")
        for cell_id, pert_id in self.config.get_eval_targets():
            gt_data = self.complete_eval_adata[(adata.obs[PERT_KEY] == pert_id) & (self.complete_eval_adata.obs[CELL_KEY] == cell_id)]
            ctrl_data = self.complete_eval_adata[(adata.obs[CELL_KEY] == cell_id) & (self.complete_eval_adata.obs[PERT_KEY] == CONTROL_PERT)]

            assert len(gt_data) > 0, f"No data found for {cell_id} and {pert_id} in {self.eval_dataset_details.name}"
            assert len(ctrl_data) > 0, f"No control data found for {cell_id} in {self.eval_dataset_details.name}"
            
            yield cell_id, pert_id, ctrl_data, gt_data



        