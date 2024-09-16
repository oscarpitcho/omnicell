#TODO: Gotta fill that up

import scanpy as sc
from typing import List, Tuple, Dict, Union
from omnicell.config.config import Config
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT
import logging

logger = logging.getLogger(__name__)
def preprocess(adata: sc.AnnData, config: Config) -> sc.AnnData:
    """
    Preprocesses the data according to the config

    Parameters
    ----------
    adata : AnnData
        The AnnData object to preprocess

    config : Config
        The configuration object to use for preprocessing

    Returns
    -------

    adata : AnnData
        The preprocessed AnnData object

    """


        
        

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