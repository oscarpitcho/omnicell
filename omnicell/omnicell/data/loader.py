import scanpy as sc
from typing import Optional, List
from dataclasses import dataclass, field
from omnicell.config.config import Config
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT

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

@dataclass
class DatasetDetails:
    name: str
    path: str
    meta_data_path: str
    cell_key: str
    condition_key: str
    control_pert: str
    perturbation_key: str
    var_names_key: str
    HVG: bool
    log1p_transformed: bool
    count_normalized: bool
    description: Optional[str] = None
    pert_embeddings: List[str] = field(default_factory=list)
    cell_embeddings: List[str] = field(default_factory=list)

class DataLoader:
    def __init__(self, config: Config, catalogue):
        self.config = config
        self.dataset_details: DatasetDetails = self._get_dataset_details(config.get_dataset_name(), catalogue)
        self.adata: Optional[sc.AnnData] = None
        self.preprocessed: bool = False

    @staticmethod
    def _get_dataset_details(dataset_name: str, catalogue) -> DatasetDetails:
        details = next((x for x in catalogue if x['name'] == dataset_name), None)
        if not details:
            raise ValueError(f"Dataset {dataset_name} not found in catalogue")
        return DatasetDetails(**details)

    def load_data(self) -> sc.AnnData:
        if self.adata is None:
            self.adata = sc.read(self.dataset_details.path)
        return self.adata


    def preprocess_data(self) -> sc.AnnData:
        if not self.preprocessed:
            adata = self.load_data()
            
            # Standardize column names and key values
            adata.obs[PERT_KEY] = adata.obs[self.dataset_details.condition_key]
            adata.obs[CELL_KEY] = adata.obs[self.dataset_details.cell_key]
            adata.obs[PERT_KEY] = adata.obs[PERT_KEY].cat.rename_categories({self.dataset_details.control_pert: CONTROL_PERT})


            if (self.config.get_cell_embedding_name() is not None) & (self.config.get_apply_normalization() | self.config.get_apply_log1p()):
                raise ValueError("Cannot both apply cell embedding and normalization/log1p transformation")
            
            elif self.config.get_cell_embedding_name() is not None:

                if self.config.get_cell_embedding_name() not in self.dataset_details.cell_embeddings:
                    raise ValueError(f"Cell embedding {self.config.get_cell_embedding_name()} not found in embeddings available for dataset {self.dataset_details.name}")
                
                
                #We replace the data matrix with the cell embeddings
                adata.X = adata.obsm[self.config.get_cell_embedding_name()]

            else:


                # Set gene names
                if self.dataset_details.var_names_key:
                    adata.var_names = adata.var[self.dataset_details.var_names_key]

                # Apply normalization and log1p if needed
                if self.config.get_apply_normalization() & (not self.dataset_details.count_normalized):
                    sc.pp.normalize_total(adata, target_sum=10_000)
                elif not self.config.get_apply_normalization() & self.dataset_details.count_normalized:
                    raise ValueError(f"Specified dataset is count normalized, but normalization is turned off in the config")
                
                if self.config.get_apply_log1p() & (not self.dataset_details.log1p_transformed):
                    sc.pp.log1p(adata)
                elif not self.config.get_apply_log1p() & self.dataset_details.log1p_transformed:
                    raise ValueError(f"Specified dataset is log1p transformed, but log1p transformation is turned off in the config")

        self.adata = adata
        self.preprocessed = True

        return self.adata

    def get_training_data(self) -> sc.AnnData:
        """
        Returns the training data according to the config.
        """

        adata = self.preprocess_data()



        if self.config.get_mode() == "ood":
            logger.info("Doing OOD split")

            #Taking cells for training where neither the cell nor the perturbation is held out
            holdout_mask = (adata.obs[PERT_KEY].isin(self.config.get_heldout_perts())) | (adata.obs[CELL_KEY].isin(self.config.get_heldout_cells()))
            train_mask = ~holdout_mask


        #IID, we include unperturbed holdouts cells
        else:
            logger.info("Doing IID split")


            #Holding out only cells that have heldout perturbations AND cell. Thus:
            # A perturbation will be included on the non holdout cell type eg
            #Control of heldout cell type will be included
            holdout_mask = (adata.obs[CELL_KEY].isin(self.config.get_heldout_cells())) & (adata.obs[PERT_KEY].isin(self.config.get_heldout_perts()))
            train_mask = ~holdout_mask


        adata_train = adata[train_mask]
        return adata_train


    def get_eval_data(self):
        adata = self.preprocess_data()
        for cell_id, pert_id in self.config.get_eval_targets():
            gt_data = adata[(adata.obs[PERT_KEY] == pert_id) & (adata.obs[CELL_KEY] == cell_id)]
            ctrl_data = adata[(adata.obs[CELL_KEY] == cell_id) & (adata.obs[PERT_KEY] == CONTROL_PERT)]
           
            yield cell_id, pert_id, ctrl_data, gt_data



        