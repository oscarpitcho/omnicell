import scanpy as sc
from pathlib import Path
from omnicell.constants import *

class GEARSPredictor():

    def __init__():

    


    def train(adata: sc.AnnData, path: Path):


        adata.var["gene_name"] = adata.var_names
        adata.obs["condition"] = adata.obs[PERT_KEY]

        perts = [p for p in adata.obs["condition"].unique() if p != CONTROL_PERT]
        adata.obs["condition"] = adata.obs["condition"].cat.rename_categories({CONTROL_PERT:"ctrl"})
        adata.obs["condition"] = adata.obs["condition"].cat.rename_categories({p:p+"+ctrl" for p in perts})






    def make_predict()