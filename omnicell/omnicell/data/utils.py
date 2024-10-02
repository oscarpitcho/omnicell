import scanpy as sc

#Importing union type
from typing import List, Tuple, Union
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT


def prediction_filename(pert, cell):
    return f'pred_{pert}_{cell}'


def get_pert_cell_data(adata: sc.AnnData, pert: str, cell: str) -> sc.AnnData:
    """
    Returns a view of the anndata object with only the specified perturbation on the specified cell type.
    """

    return adata[(adata.obs[PERT_KEY] == pert) & (adata.obs[CELL_KEY] == cell)]


def get_cell_ctrl_data(adata: sc.AnnData, cell: str) -> sc.AnnData:
    """
    Returns a view of the anndata object with only the specified control data for the specified cell type. 
    """

    return adata[(adata.obs[CELL_KEY] == cell) & (adata.obs[PERT_KEY] == CONTROL_PERT)]