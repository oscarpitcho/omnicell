
from turtle import distance
import scanpy as sc
import pandas as pd
import numpy as np

from omnicell.constants import *

import logging

logger = logging.getLogger(__name__)

class OracleNNPredictor():

    def __init__(self, adata_complete: sc.AnnData, model_config: dict):

        self.model = None
        self.model_config = model_config
        self.adata_complete = adata_complete

        temp = adata_complete.uns['DEGs']

        self.DEGs = {}


        for cell in self.DEGs:
            if cell not in self.DEGs:
                self.DEGs[cell] = {}
            for pert in self.DEGs[cell]:
                if self.DEGs[cell][pert] is not None:
                    df = pd.DataFrame.from_dict(temp[cell][pert], orient='index')
                    df = df[df['pvals_adj'] < self.model_config['p_threshold']]
                    self.DEGs[cell][pert] = df

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
            
            #We average the number of overlapping DEGs by the number of DEGs per cell_type to get a most similar perturbation

            candidate_perts = [p for c, p in DEG_overlaps]
            mean_overlaps = {}
            for p in candidate_perts:
                overlaps = []
                for c in seen_cell_types:
                    if (c, p) in DEG_overlaps:
                        overlaps.append(DEG_overlaps[(c, p)])
                
                mean_overlap = np.mean(overlaps)
                mean_overlaps[p] = mean_overlap
            
            closest_pert = max(mean_overlaps, key=mean_overlaps.get)



            logger.info(f"Selected perturbation {closest_pert} as closest pert, with {mean_overlaps[closest_pert]} average overlapping DEGs on seen cell types")


            #We return the population of the most similar pert, what if it is not in the training set? TODO
            nn_population = self.adata_complete[(self.adata_complete.obs[PERT_KEY] == closest_pert) & (self.adata_complete.obs[CELL_KEY] == cell_type)]
            
            logger.debug(f"Returning population of {len(nn_population)} cells")
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
    
