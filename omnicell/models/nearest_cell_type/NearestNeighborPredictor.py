import numpy
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from ...constants import PERT_KEY, CELL_TYPE_KEY, CONTROL_PERT



class NearestNeighborPredictor():
    def __init__(self, config):
        self.mode = config['mode']
        self.config = config
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

        closest_cell_type_idx = np.argmin(np.sum((train_cell_type_ctrl_means - heldout_cell_ctrl_mean)**2, axis=1))
        closest_cell_type = cell_types[closest_cell_type_idx]

        perturbed_closest_cell_type = self.train_adata[(self.train_adata.obs[CELL_TYPE_KEY] == closest_cell_type) & (self.train_adata.obs[PERT_KEY] == target_pert)].X.mean(axis=0)

        pert_effect = perturbed_closest_cell_type - train_cell_type_ctrl_means[closest_cell_type_idx]

        predicted_perts = heldout_cell_adata.obs[heldout_cell_adata.obs[PERT_KEY] == CONTROL_PERT].X + pert_effect

        return predicted_perts
    


    def predict_across_gene(self, heldout_pert_adata: sc.AnnData, cell_type: str) -> np.ndarray:
        """
        Makes a prediction on a seen target cell type given some unseen perturbation.
        
        Takes the perturbation effect which is closest to the heldout perturbation and applies it to the heldout cell data.

        """


        assert self.train_adata is not None, "Model has not been trained yet"
        assert heldout_pert_adata.obs['pert_type'].nunique() == 1, "Heldout pert data must contain only one perturbation"
        assert heldout_pert_adata.obs['cell_type'].unique() not in self.train_adata.obs['cell_type'].unique(), "Heldout cell type must be unseen in training data"


        holdout_pert_id = heldout_pert_adata.obs['pert_type'].unique()[0]

        unique_perturbations = self.train_adata.obs['pert_type'].unique()
        unique_genes_noholdout = unique_perturbations[unique_perturbations != CONTROL_PERT]

        DEGSlist = []
        GTOlist = []

        inp = self.train_adata
        for ug in unique_genes_noholdout:
            cont = np.array(inp[inp.obs[PERT_KEY] == CONTROL_PERT].X.todense())
            pert = np.array(inp[inp.obs[PERT_KEY] == ug].X.todense())
            
            control = sc.AnnData(X=cont)
            
            control.obs['condition_key'] = 'control'
            
            true_pert = sc.AnnData(X=pert)
            
            true_pert.obs['condition_key'] = 'perturbed'
            
            
            control.obs_names = control.obs_names+'-1'
            control.X[0,(control.X.var(axis=0)==0)] += np.amin(control.X[np.nonzero(control.X)])
            true_pert.X[0,(true_pert.X.var(axis=0)==0)] += np.amin(true_pert.X[np.nonzero(true_pert.X)])
            
            temp_concat = sc.concat([control, true_pert], label = 'batch')
            sc.tl.rank_genes_groups(temp_concat, 'batch', method='wilcoxon', groups = ['1'], ref = '0')
            
            rankings = temp_concat.uns['rank_genes_groups']
            result_df = pd.DataFrame({'pva': rankings['pvals_adj']['1'], 'pvals_adj': rankings['scores']['1']}, index = rankings['names']['1'])
            result_df.index = result_df.index.astype(np.int32)
            result_df = result_df.sort_index()
            
            GTO = torch.from_numpy(np.array(result_df['pvals_adj']).astype(np.float32))
            
            DEGs = torch.argsort(torch.abs(GTO))
            
            DEGSlist.append(DEGs)
            GTOlist.append(GTO)
            
        significant_reducers = []
        for genno in unique_genes_noholdout:
            rank = GTOlist[unique_genes_noholdout.index(genno)][torch.where(DEGSlist[unique_genes_noholdout.index(genno)] == np.where(inp.var_names==holdout_pert_id)[0][0])[0][0].item()]
            if ((0 <= rank) and (rank <= self.config['pvalcut'])):
                significant_reducers.append(genno) 

        reduced_DEGs = []    
        for sr in significant_reducers:
            reduced_DEGs.append(DEGSlist[unique_genes_noholdout.index(sr)][-num_of_degs:])
        reduced_DEGs = torch.cat(reduced_DEGs)   

        # Count the occurrences of each element
        unique_elements, counts = torch.unique(reduced_DEGs, return_counts=True)

        # Filter to keep only elements that repeat more than once
        repeated_elements = unique_elements[counts > 1]

        # Mask the original tensor to keep only repeated elements
        duplicated_DEGs = torch.unique(reduced_DEGs[torch.isin(reduced_DEGs, repeated_elements)])

        bestnnscore = 0
        for sr in significant_reducers:
            nnscore = len(np.intersect1d(duplicated_DEGs.cpu().detach().numpy(),DEGSlist[unique_genes_noholdout.index(sr)][-num_of_degs:].cpu().detach().numpy()))
            if nnscore >= bestnnscore:
                nnbr = sr
                bestnnscore = nnscore



        return predicted_perts
        

    


