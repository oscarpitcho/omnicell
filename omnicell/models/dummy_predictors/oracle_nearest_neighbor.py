
from turtle import distance
import scanpy as sc
import pandas as pd
import numpy as np

from omnicell.constants import *

import logging

logger = logging.getLogger(__name__)

class OracleNNPredictor():

    def __init__(self, model_config: dict):

        self.model = None
        self.model_config = model_config

        self.DEGs = None 
        self.number_top_DEGs_overlap = model_config["number_top_DEGs_overlap"]
        self.seen_pert_per_cell = None
        self.training_adata = None

    def train(self, adata: sc.AnnData, model_savepath: Path):
        self.seen_pert_per_cell = {}
        for cell in adata.obs[CELL_KEY].unique():
            cell_mask = adata.obs[CELL_KEY] == cell
            perts_in_cell = adata[cell_mask].obs[PERT_KEY].unique()
            self.seen_pert_per_cell[cell] = [p for p in perts_in_cell if p != CONTROL_PERT]

        self.training_adata = adata
        self.DEGs = adata.uns['DEGs']

        for cell in self.DEGs:
            for pert in self.DEGs[cell]:
                df = pd.DataFrame.from_dict(self.DEGs[cell][pert], orient='index')
                df = df[df['pvals_adj'] <self.model_config['p_threshold']]
                self.DEGs[cell][pert] = df



    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:


        seen_cell_types = self.seen_pert_per_cell.keys()

        #We are making a prediction across cells,
        # we will fetch the perturbation that has the most overlapping DEGs all seen perturbations on the all seen cell types
        if cell_type not in seen_cell_types:
            # Looking for the seen (cell, pert) pair with the highest overlap with the target pert
            logger.debug("Running NN Oracle across cells")
            DEG_overlaps = {}
            DEG_target = self.DEGs[cell_type][pert_id]

            for c in seen_cell_types:
                for p in self.seen_pert_per_cell[c]:
                    DEG_p = self.DEGs[c][p]
                    DEG_overlaps[(c, p)] = len(DEG_target.index[:self.number_top_DEGs_overlap].intersection(DEG_p.index[:self.number_top_DEGs_overlap]))
                
            
            closest_cell, closest_pert = max(DEG_overlaps, key=DEG_overlaps.get)
            logger.info(f"Selected perturbation {closest_pert} in cell type {closest_cell} as closest to {pert_id} in cell type {cell_type}, with {DEG_overlaps[(closest_cell, closest_pert)]} overlapping DEGs")


            nn_population = self.training_adata[(self.training_adata.obs[PERT_KEY] == closest_pert) & (self.training_adata.obs[CELL_KEY] == closest_cell)]

            return nn_population.X

        #We are making a prediction within a cell type, across perts, finding seen pert with highest overlap with target pert
        else:
            # Looking for the seen perturbation with the highest overlap with the target pert

            logger.debug("Running NN Oracle across perts")

            DEGs_per_pert = self.DEGs[cell_type]
            DEGs_target = self.DEGs[cell_type][pert_id]

            non_target_perts_in_cell = [p for p in DEGs_per_pert.keys() if p != pert_id]

            DEG_overlaps = {}

            for p in non_target_perts_in_cell:
                if p in self.seen_pert_per_cell[cell_type]:
                    DEGs_p = DEGs_per_pert[p]
                    DEG_overlaps[p] = len(DEGs_target.index[:self.number_top_DEGs_overlap].intersection(DEGs_p.index[:self.number_top_DEGs_overlap]))

            closest_pert = max(DEG_overlaps, key=DEG_overlaps.get)

        
            logger.info(f"Selected perturbation {closest_pert} as closest to {pert_id} in cell type {cell_type}, with {DEG_overlaps[closest_pert]} overlapping DEGs")

            nn_population = self.training_adata[(self.training_adata.obs[PERT_KEY] == closest_pert) & (self.training_adata.obs[CELL_KEY] == cell_type)]
            return nn_population.X
    
