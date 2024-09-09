import scanpy as sc

#Importing union type
from typing import List, Tuple, Union
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT


def prediction_filename(pert, cell):
    return f'pred_{pert}_{cell}'


def split_anndata(adata: sc.AnnData, holdout_perts: Union[str, List[str]], holdout_cells: Union[str, List[str]], pert_key: str, cell_key: str) -> Tuple[sc.AnnData, sc.AnnData]:
    """
    Splits anndata object into two based on holdout perturbations and holdout cell types
    
    If a cell has either a holdout perturbation or a holdout cell type, it is put into the evaluation set
    
    returns a tuple of two anndata objects (train, eval) which are views of the original anndata object"""


    holdout_perts = [holdout_perts] if isinstance(holdout_perts, str) else holdout_perts
    holdout_cells = [holdout_cells] if isinstance(holdout_cells, str) else holdout_cells

    idx_hold_perts = adata.obs[pert_key].isin(holdout_perts)
    idx_hold_cells = adata.obs[cell_key].isin(holdout_cells)

    idx_eval = idx_hold_perts | idx_hold_cells
    idx_train = ~idx_eval

    return adata[idx_train], adata[idx_eval]


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