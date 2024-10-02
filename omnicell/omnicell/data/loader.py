import scanpy as sc
from omnicell.config.config import Config
from typing import Iterable, Tuple
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT
from omnicell.data.utils import get_pert_cell_data, get_cell_ctrl_data

import logging
logger = logging.getLogger(__name__)

"""
Wrapper class for loading and preprocessing the data.

Will load the data from disk as soon as the object is created.

"""
class Loader:

    def __init__(self, config: Config, catalogue):
        self.config = config
        self.catalogue = catalogue

        self.dataset_name = config.get_dataset_name()
        self.dataset_details = Loader._get_dataset_details(self.dataset_name, catalogue)

        self.adata = sc.read(self.dataset_details['path'])




    


    def get_training_data(self):

        dl_config = self.config.get_data_loader_config()

        data_path = self.dataset_details['path']

        logger.info(f"Reading data from {data_path}")
        adata = sc.read(data_path)

        adata = self._preprocess(adata, self.config, self.dataset_details)


        prepare_data_fn = None

        if dl_config['type'] == 'adata':
            prepare_data_fn = None
        elif dl_config['type'] == 'dataloader':
            prepare_data_fn = None
        else:


        pass


    def get_eval_data(self) -> sc.AnnData:
        """
        Returns the evaluation data associated with the task

        Returns
        -------
        Iterable[Tuple[cell_id, pert_id, control_data: sc.AnnData, gt_data: sc.AnnData]]
            A list of tuples of the name of the dataset and the evaluation data
        """


        for cell_id, pert_id, in self.config.get_eval_targets():
            
            gt_data = get_pert_cell_data(cell_id, pert_id)
            ctrl_data = get_cell_ctrl_data(cell_id)
            
            yield cell_id, pert_id, ctrl_data, gt_data



    def _get_dataset_details(dataset_name, catalogue):

        dataset_details = [x for x in catalogue if x['name'] == dataset_name][0]

    def get_preprocessed_data(self) -> sc.AnnData:
        """
        Preprocesses the data according to the config and the dataset details contained in the catalogue.

        The processing steps are as follows, standardzing column names and pert_names, applying log1p and normalization if required.

        Returns
        -------

        adata : AnnData
            The preprocessed AnnData object

        """


        data_path = self.dataset_details['path']
        logger.info(f"Reading data from {data_path}")
        adata = sc.read(data_path)
    
        

        adata = adata.copy()

        logger.debug(f"Data shape before preprocessing: {adata.shape}")
        #Standardizing column names and key values
        adata.obs[PERT_KEY] = adata.obs[config.get_pert_key()]
        adata.obs[CELL_KEY] = adata.obs[config.get_cell_key()]
        adata.obs[PERT_KEY] = adata.obs[PERT_KEY].cat.rename_categories({config.get_control_pert() : CONTROL_PERT})



        #Setting the names of the genes as varnames
        if config.get_var_names_key() is not None:
            adata.var_names = adata.var[config.get_var_names_key()]
        
        #adata.var_names = adata.var[config.get_var_gene_key()]

            #Making it faster for testing
        if config.get_test_mode():
            logger.debug("Running in test mode, capping dataset at 10_000")
            adata_first = adata[:10000]
            adata_ctrl = adata[adata.obs[PERT_KEY] == CONTROL_PERT]
            adata = adata_first.concatenate(adata_ctrl)



        if config.get_apply_normalization():
            sc.pp.normalize_total(adata, target_sum=10_000)
        if config.get_apply_log1p():
            sc.pp.log1p(adata)

        logger.debug(f"Data shape after preprocessing: {adata.shape}")

        return adata

        



        