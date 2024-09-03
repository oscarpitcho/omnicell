#TODO: Gotta fill that up

import scanpy as sc
from typing import List, Tuple, Dict, Union
from omnicell.config.config import Config
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT


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


        
        

    #Standardizing column names and key values
    adata.obs[PERT_KEY] = adata.obs[config.get_pert_key()]
    adata.obs[CELL_KEY] = adata.obs[config.get_cell_key()]
    adata.obs[PERT_KEY] = adata.obs[PERT_KEY].cat.rename_categories({config.get_control_pert() : CONTROL_PERT})

        #Making it faster for testing
    if config.get_test_mode():
        adata_first = adata[:10000]
        adata_ctrl = adata[adata.obs[PERT_KEY] == CONTROL_PERT]

        print("adta control", adata_ctrl)

        adata = adata_first.concatenate(adata_ctrl)

    return adata