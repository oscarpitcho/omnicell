import numpy
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT
import logging

logger = logging.getLogger(__name__)

class NearestNeighborPredictor():
    def __init__(self, config):
        self.config = config
        self.train_adata = None
        self.seen_cell_types = None
        self.seen_perts = None


        #TODO: We can compute means here for some extra perf

    def train(self, adata):

        self.train_adata = adata

        self.seen_cell_types = adata.obs[CELL_KEY].unique()
        self.seen_perts = adata.obs[PERT_KEY].unique()

        #Mode across perturbations

    
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        assert self.train_adata is not None, "Model has not been trained yet"
    
        if cell_type in self.seen_cell_types:
            if pert_id in self.seen_perts:
                raise NotImplementedError("Both cell type and perturbation are in the training data, in distribution prediction not implemented yet")
            else:
                return self._predict_across_pert(adata, pert_id, cell_type)
        else:
            if pert_id in self.seen_perts:
                return self._predict_across_cell(adata, pert_id, cell_type)
            else:
                raise NotImplementedError("Both cell type and perturbation are not in the training data, out of distribution prediction not implemented yet")
               

    def _predict_across_cell(self, heldout_cell_adata: sc.AnnData, target_pert: str, cell_id: str) -> np.ndarray:
        """
        Makes a prediction for a seen target perturbation given some unseen cell type. 
        
        We find the closest cell type in the control state and apply apply the average effect of the perturbation on the neighboring cell type to our heldout cell data.

        Parameters
        ----------
        heldout_cell_adata : AnnData
            The AnnData object containing the unseen cell type (and only that cell type) with control perturbation
        target_pert : str
            The target perturbation ID to predict

        Returns
        -------
        np.ndarray
            The predicted perturbation for the target using the control perturbation

        """


        assert self.train_adata is not None, "Model has not been trained yet"
        assert heldout_cell_adata.obs[CELL_KEY].nunique() == 1, "Heldout cell data must contain only one cell type"
        assert heldout_cell_adata.obs[PERT_KEY].nunique() == 1, "Heldout cell data must contain only control data"
        assert heldout_cell_adata.obs[PERT_KEY].unique()[0] == CONTROL_PERT, "Heldout cell data must contain only control data"


        cell_types = self.train_adata.obs[CELL_KEY].unique()

        logger.debug(f"Cell types in training data {cell_types}")

        train_cell_type_ctrl_means = []

        for cell_type in cell_types:
            train_cell_type_ctrl_means.append(self.train_adata[(self.train_adata.obs[CELL_KEY] == cell_type) & (self.train_adata.obs[PERT_KEY] == CONTROL_PERT)].X.mean(axis=0))

        
        train_cell_type_ctrl_means = np.squeeze(np.array(train_cell_type_ctrl_means))

        logger.debug(f"train_cell_type_ctrl_means shape {train_cell_type_ctrl_means.shape}")




        #Mean control state of the heldout cell
        heldout_cell_ctrl_mean = heldout_cell_adata.X.mean(axis=0)


        logger.debug(f"train_cell_type_ctrl_means shape {train_cell_type_ctrl_means.shape}")
        logger.debug(f"heldout_cell_ctrl_mean shape {heldout_cell_ctrl_mean.shape}")
        diffs = train_cell_type_ctrl_means - heldout_cell_ctrl_mean

        #Applying L2 distance, could be changed to L1
        squared_diffs = np.square(diffs)



        distances_to_heldout = np.sum(squared_diffs, axis=1)

        closest_cell_type_idx = np.argmin(distances_to_heldout)

        logger.debug(f"Closest cell type to evaluated cell_type {cell_id} is {cell_types[closest_cell_type_idx]}")

        closest_cell_type = cell_types[closest_cell_type_idx]

        perturbed_closest_cell_type = self.train_adata[(self.train_adata.obs[CELL_KEY] == closest_cell_type) & (self.train_adata.obs[PERT_KEY] == target_pert)].X.mean(axis=0)


        pert_effect = perturbed_closest_cell_type - train_cell_type_ctrl_means[closest_cell_type_idx]


        #Apply the perturbation effect to the heldout cell data
        predicted_perts = heldout_cell_adata.X + pert_effect

        return predicted_perts
    


    #SO I want to predict across genes --> Two options either we provide the data or we don't provide the data on which the prediction is made
    def _predict_across_pert(self, adata: sc.AnnData, target: str, cell_id: str) -> np.ndarray:
        """
        Makes a prediction for an unseen perturbation.
        
        Takes the perturbation effect in the training data which is closest to the heldout perturbation and applies it to the given control data

        Parameters
        ----------
        target : str
            The target perturbation to predict

        cell_id : str
            The cell type of the data on which the prediction is done

        adata : AnnData
            The AnnData object control data on the cell type of the prediction.

        Returns
        -------
        np.ndarray
            The predicted perturbation for the target using the control perturbation datapoints of the training.

        """


        assert self.train_adata is not None, "Model has not been trained yet"
        assert target not in self.train_adata.obs[PERT_KEY].unique(), "Target perturbation is already in the training data"
        logger.debug(f'Predicting unseen perturbation {target} using all training data')
        
        num_of_degs = self.config['num_of_degs']

        unique_perturbations = self.train_adata.obs[PERT_KEY].unique()
        unique_genes_noholdout = [ug for ug in unique_perturbations if (ug!=target and ug!=CONTROL_PERT)]

        DEGSlist = []
        GTOlist = []

        inp = self.train_adata[self.train_adata.obs[CELL_KEY] == cell_id].copy()
        logger.debug(f' # Of cell with type {cell_id} in training data {len(inp)}')
        logger.debug(f'Finding nearest neighbor perturbation for {target}')
        for ug in unique_genes_noholdout:
            cont = np.array(inp[inp.obs[PERT_KEY] == CONTROL_PERT].X.todense())
            pert = np.array(inp[inp.obs[PERT_KEY] == ug].X.todense())
            
            logger.debug(f'Finding nearest neighbor perturbation for {target} using {ug}')
            logger.debug(f'Control shape {cont.shape}, pert shape {pert.shape}')
            
            control = sc.AnnData(X=cont)
                        
            true_pert = sc.AnnData(X=pert)
        
            
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
        significant_reducers_pval = []
        for genno in unique_genes_noholdout:
            rank = GTOlist[unique_genes_noholdout.index(genno)][torch.where(DEGSlist[unique_genes_noholdout.index(genno)] == np.where(inp.var_names==target)[0][0])[0][0].item()]
            if ((0 <= rank) and (rank <= self.config['pvalcut'])):
                significant_reducers.append(genno)
                significant_reducers_pval.append(rank) 

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
        bestpval = 0
        for jq, sr in enumerate(significant_reducers):
            nnscore = len(np.intersect1d(duplicated_DEGs.cpu().detach().numpy(),DEGSlist[unique_genes_noholdout.index(sr)][-num_of_degs:].cpu().detach().numpy()))
            
                
            if nnscore > bestnnscore:
                nnbr = sr
                bestnnscore = nnscore
                bestpval = significant_reducers_pval[jq]
            #pval as tie-break
            if nnscore == bestnnscore:
                if bestpval > significant_reducers_pval[jq]:
                    nnbr = sr
                    bestnnscore = nnscore
                    bestpval = significant_reducers_pval[jq]
                    

        logger.debug(f'Nearest neighbor perturbation of {target} is {nnbr}')
        #We have the neighboring perturbation, now we find the effect of this perturbation on each cell type and then apply the corresponding to each cell in the heldout data.
        nnbr_pert = nnbr


        #Mean control state of each cell type



        cell_types = self.train_adata.obs[CELL_KEY].unique()

        cell_type_to_index = {cell_type: i for i, cell_type in enumerate(cell_types)}


        train_cell_type_ctrl_means = []

        for cell_type in cell_types:
            train_cell_type_ctrl_means.append(self.train_adata[(self.train_adata.obs[CELL_KEY] == cell_type) & (self.train_adata.obs[PERT_KEY] == CONTROL_PERT)].X.mean(axis=0))

        
        nbr_pert_effect_per_cell_type = []
        for cell_type in cell_types:
            perturbed_cell_type = self.train_adata[(self.train_adata.obs[CELL_KEY] == cell_type) & (self.train_adata.obs[PERT_KEY] == nnbr_pert)].X.mean(axis=0)
            pert_effect = perturbed_cell_type - train_cell_type_ctrl_means[cell_type_to_index[cell_type]]
            nbr_pert_effect_per_cell_type.append(pert_effect)

        
        nbr_pert_effect_per_cell_type = np.array(nbr_pert_effect_per_cell_type)
        

        #We now have the effect of the neighboring perturbation on each cell type. We can now apply this to the heldout cell data, choosing the correct effect based on the cell type of the heldout data.

        predictions = adata.copy()

        for cell_type in cell_types:
            cell_type_effect = nbr_pert_effect_per_cell_type[cell_type_to_index[cell_type]]
            predictions[predictions.obs[CELL_KEY] == cell_type].X += cell_type_effect


        return predictions.X

        

    


