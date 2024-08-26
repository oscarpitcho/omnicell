import scanpy as sc
import torch
import yaml
import numpy as np

class LLMPredictor():


    def __init__(self, config, model):
        self.config = config


    #Should take care of saving the model under some results/model/checkpoints in 
    #BTW I think hidden dirs fuck with with the cluster, so don't call it .checkpoint
    def train(the_data_which_is_nicely_split):

        
        return preds
    

    #I mean to we need to evaluate anything? 
    def make_predict(self, adata: sc.AnnData | torch.DataLoader, pert_id: str, cell_type: str) -> np.ndarray:
        pass 
