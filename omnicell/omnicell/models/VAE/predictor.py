from torch import nn, optim
import numpy as np
import scanpy as sc
import torch
import logging
from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY
from omnicell.models.VAE.vae import Net
from omnicell.models.early_stopping import EarlyStopper

logger = logging.getLogger(__name__)


class VAEPredictor():
    def __init__(self, config, input_dim, device, pert_ids):
        self.training_config = config['training']
        self.model_config = config['model']
        self.input_dim = input_dim
        self.device = device
        self.model = Net(input_dim, self.model_config['hidden_dim'],
                         self.model_config['latent_dim'], 
                         self.training_config['alpha'],
                         self.model_config['dropout_rate'],
                         self.training_config['learning_rate'])
        
        self.epochs = self.training_config['epochs']
        self.batsize = self.training_config['batsize']

        self.model_eval = Net(input_dim, self.model_config['hidden_dim'],
                         self.model_config['latent_dim'], 
                         self.training_config['alpha'],
                         self.model_config['dropout_rate'],
                         self.training_config['learning_rate'])
        
        self.model.to(device)

        #Stores the latent space shift associated which each perturbation
        self.perts = pert_ids
        self.deltas =  None
        self.perts_to_idx = {pert: idx for idx, pert in enumerate(pert_ids)}


    #Note this model needs the entire data or sth like that. 
    #The mean operations are computed on the entire dataset.
    def train(self, adata):
        device = self.device
        epochs = self.epochs
        batsize = self.batsize



        #TODO bad if we start passing batches to the model, it will see only part of the data and the len of the data will be wrong.
        datalen = len(adata)
        indices = np.random.permutation(datalen)
        train = adata[indices[:np.int32(datalen*0.9)]]  # 90% training data
        valid = adata[indices[np.int32(datalen*0.9):]]         

        trainlen = train.shape[0]
        validlen = valid.shape[0]

        net = self.model
        net.to(device)  # Model for training
        neteval = self.model_eval  # Model for evaluation
        neteval.to(device)
        neteval.eval()  # Set evaluation mode

        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        running_loss = 0

        train = torch.from_numpy(train.X.astype(np.float32)).to(device)  # Convert train data to torch tensor and move to device
        valid = torch.from_numpy(valid.X.astype(np.float32)).to(device)  # Convert validation data to torch tensor and move to device
        
        # Training loop
        for e in range(epochs):
            running_loss = 0
        
            for lower in range(0, trainlen, batsize):
                upper = min(lower + batsize, trainlen)
                lower = min(trainlen - batsize, lower)
                batch = train[lower:upper, :]
                optimizer.zero_grad()  
                out, mu, logvar = net(batch)  # Forward pass
                loss = net.loss_function(out, batch, mu, logvar) 
                loss.backward() 
                running_loss += loss.item()
                optimizer.step() 
            logger.info(f'Epoch {e+1}/{epochs}')
            logger.info(f'Train loss: {running_loss/1000000}')
        
            running_loss = 0
            state_dict = net.state_dict()
            neteval.load_state_dict(state_dict)  # Update evaluation model state
            
            # Validation loop
            with torch.no_grad():
                for lower in range(0, validlen, batsize):
                    upper = min(lower + batsize, validlen)
                    lower = min(validlen - batsize, lower)
                    batch = valid[lower:upper, :]
                    out, mu, logvar = neteval(batch)
                    loss = neteval.loss_function(out, batch, mu, logvar)
                    running_loss += loss.item()
                logger.info(f'Valid loss: {running_loss/1000000}')

            #Early stopping
            early_stopper = EarlyStopper(patience=10, min_delta=0.01)

            if early_stopper.early_stop(loss.item()):
                logger.info('Early stopping')
                break


        total_enc = net(adata)
        mean_enc = total_enc.mean(axis=0)

        pert_means = []
        for i in range(len(self.perts)):
            pert = self.perts[i]
            pert_adata = adata[adata.obs['perturbation'] == pert]
            pert_enc = net(pert_adata)
            pert_mean = pert_enc.mean(axis=0)
            pert_means.append(pert_mean)

        self.deltas = np.array(pert_means) - mean_enc


    #Predicting perturbations --> How do we compute the means? 
    #--> We need a delta for each perturbation
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        
        assert pert_id in self.perts, f'{pert_id} not in the list of perturbations, this model cannot transfer across perturbations'
        assert adata.obs[CELL_KEY].unique() == cell_type, f'Cell type {cell_type} not in the provided data'
        assert adata.obs[PERT_KEY].unique() == CONTROL_PERT, f'Input data contains non control perturbations'

        logger.info(f'Predicting seen perturbation {pert_id} for unseen cell type {cell_type}')

        data = torch.from_numpy(adata.X.astype(np.float32)).to(self.device)
        encodings = self.model.encode(data)

        pert_delta = self.deltas[self.perts_to_idx[pert_id]]

        pert_enc = encodings + pert_delta

        return self.model.decode(pert_enc).cpu().detach().numpy()

