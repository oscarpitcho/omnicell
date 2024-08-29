#TODO: Gotta fill that up

import scanpy as sc
from typing import List, Tuple, Dict, Union
from omnicell.config.config import Config



def preprocess(adata: sc.AnnData, config: Config) -> sc.AnnData:
    """
    Preprocesses the data according to the config

    Parameters
    ----------
    adata : AnnData
        The AnnData object to preprocess, should have been preprocessed with the correct key names for perturbations and cell types already.

    config : Config
        The configuration object to use for preprocessing

    Returns
    -------

    adata : AnnData
        The preprocessed AnnData object

    """

    #Making it faster for testing
    if config.get_test_mode():
        adata = adata[:10000]

    #Standardizing column names and key values
    adata.obs[config.get_pert_key()] = adata.obs[config.get_pert_key()]
    adata.obs[config.get_cell_key()] = adata.obs[config.get_cell_key()]
    adata.obs[config.get_pert_key()] = adata.obs[config.get_pert_key()].cat.rename_categories({config.get_control_pert() : config.get_control_pert()})

    return adata