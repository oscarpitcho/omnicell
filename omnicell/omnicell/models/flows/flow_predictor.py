import scanpy as sc
import torch

import numpy as np

from omnicell.models.datamodules import get_dataloader

from omnicell.models.flows.arch import CMHA, CMLP, CFM
from omnicell.models.flows.flow_utils import compute_conditional_flow
from pytorch_lightning.callbacks import TQDMProgressBar
import pytorch_lightning as pl

import scanpy as sc

from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY


class FlowPredictor():
    def __init__(self, config, input_size, device, pert_rep=None, pert_map=None, cell_rep=None):
        self.model_config = config['model']
        self.trainig_config = config['training']
        self.device = device

        self.max_epochs = self.trainig_config['max_epochs']

        self.pert_map = None
        self.pert_rep = None

        if self.arch.lower() == self.model_config['arch']:
            self.model = CMLP(training_module=CFM, feat_dim=xt.shape[1], cond_dim=pert_rep.shape[1], time_varying=True, **self.model_config)
        elif self.arch.lower() == self.model_config['arch']:
            self.model = CMHA(training_module=CFM, feat_dim=xt.shape[1], cond_dim=pert_rep.shape[1], time_varying=True, **self.model_config)
        else:
            raise NotImplementedError(f"Model architecture {self.model_config['arch']} not implemented")
        

    #Should take care of saving the model under some results/model/checkpoints in 
    #BTW I think hidden dirs fuck with with the cluster, so don't call it .checkpoint
    def train(self, adata):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        adata.X = adata.X.toarray()
        adata.X = adata.X / adata.X.sum(axis=1)[:, None]
        adata.obsm["standard"] = adata.X

        dl = get_dataloader(adata, pert_ids=self.pert_ids, pert_reps=self.pert_reps)

        print("Training model")
        # Train the model
        trainer = pl.Trainer(
            accelerator='gpu', devices=1,  # Specify the number of GPUs to use
            max_epochs=self.max_epochs,  # Specify the maximum number of training epochs
            # default_root_dir=save_path,
            callbacks=[TQDMProgressBar(refresh_rate=100)]
        )

        _, xt, _, pert_rep = next(iter(dl))

        
        self.model = self.model.to(device)            
        trainer.fit(self.model, dl)
    

    #I mean to we need to evaluate anything? 
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        X = adata.obsm["standard"]
        X = X.toarray()
        X = X / X.sum(axis=1)[:, None]

        cell_types = adata.obs[CELL_KEY].values
        control_eval = adata.obsm[self.embedding][cell_types == cell_type]
        traj = compute_conditional_flow(
            self.model, 
            control_eval, 
            np.repeat(pert_id, control_eval.shape[0]), 
            self.pert_rep[pert_id],
            n_batches = 5 
        )  
        return traj[-1, :, :]
