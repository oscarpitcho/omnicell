import numpy
import scanpy as sc
import np
from ...constants import PERT_KEY, CELL_TYPE_KEY, CONTROL_PERT



class NearestNeighborPredictor():
    def __init__(self, config):
        self.mode = config['mode']
        self.train_adata = None





    def train(self, adata):

        self.train_adata = adata

        #Mode across cells
        cell_types = adata.obs['cell_type'].unique()
        train_cell_type_means = []

        for cell_type in cell_types:
            train_cell_type_means.append(adata[adata.obs['cell_type'] == cell_type].X.mean(axis=0))


        train_cell_type_ctrl_means = np.array(train_cell_type_means)


        #Mode across perturbations

    
    def predict_across_cell(self, heldout_cell_adata: sc.AnnData, target_pert: str) -> np.ndarray:
        """
        Makes a prediction for a seen target perturbation given some unseen cell type. 
        
        We find the closest cell type in the control state and apply apply the average effect of the perturbation on the neighboring cell type to our heldout cell data.

        Parameters
        ----------
        heldout_cell_adata : AnnData
            The AnnData object containing the unseen cell type (and only that cell type) with control and target perturbations
        target_pert : str
            The target perturbation to predict

        Returns
        -------
        np.ndarray
            The predicted perturbation for the target using the control perturbation

        """

        assert self.train_adata is not None, "Model has not been trained yet"
        assert heldout_cell_adata.obs['cell_type'].nunique() == 1, "Heldout cell data must contain only one cell type"

        cell_types = self.train_adata.obs['cell_type'].unique()

        train_cell_type_ctrl_means = []

        for cell_type in cell_types:
            train_cell_type_ctrl_means.append(self.train_adata[(self.train_adata.obs[CELL_TYPE_KEY] == cell_type) & (self.train_adata.obs[PERT_KEY] == CONTROL_PERT)].X.mean(axis=0))

        
        train_cell_type_ctrl_means = np.array(train_cell_type_ctrl_means)




        #Mean control state of the heldout cell
        heldout_cell_ctrl_mean = heldout_cell_adata.obs[heldout_cell_adata.obs[PERT_KEY] == CONTROL_PERT].X.mean(axis=0)

        closest_cell_type_idx = np.argmin(np.sum((train_cell_type_ctrl_means - heldout_cell_mean)**2, axis=1))
        closest_cell_type = cell_types[closest_cell_type_idx]

        perturbed_closest_cell_type = self.train_adata[(self.train_adata.obs[CELL_TYPE_KEY] == closest_cell_type) & (self.train_adata.obs[PERT_KEY] == target_pert)].X.mean(axis=0)

        pert_effect = perturbed_closest_cell_type - train_cell_type_ctrl_means[closest_cell_type_idx]

        predicted_perts = heldout_cell_adata.obs[heldout_cell_adata.obs[PERT_KEY] == CONTROL_PERT].X + pert_effect

        return predicted_perts
    


    def predict_across_gene(self, heldout_pert_adata: sc.AnnData)
        

    


