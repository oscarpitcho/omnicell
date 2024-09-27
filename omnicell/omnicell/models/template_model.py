import torch 
import scanpy as sc
import numpy as np
#Bit of a guide so I don't forget what I'm doing

class TemplateModel():

    def __init__(self, config):
        self.config = config
        self.model = None



    def train(adata, pairing):
        """
        Training method of the model, takes andata without the evaluation data + Pairing and does whatever it needs
        to do such that the model is "trained" and ready to make predictions / encode / decode.
        
        """
        pass
    

    #I mean to we need to evaluate anything? 
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        """
        Makes prediction on the passed adata object, for the given pert and cell_type.

        In charge of checking whether this is across cells / perts and whether the model supports this
        """
        pass


    def encode(self, adata) -> np.ndarray:        
        pass

    def decode(self, X_embed) -> np.ndarray:
        pass
        


