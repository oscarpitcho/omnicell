import math
from tqdm import tqdm  # For progress bar visualization
import torch
import numpy as np
import os
from torch import nn
import torch.nn.functional as F
import scanpy as sc  # For single-cell data analysis

import scanpy as sc  # Duplicate import, can be removed
from datamodules import SCFMDataset, cfm_collate, StratifiedBatchSampler, ot_collate  # Custom data modules
from torch.utils.data import RandomSampler
from sc_etl_utils import *  # Custom utility functions
from arch import *  # Custom architectures
import json

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
import os
import scipy
import scanpy as sc  # Duplicate import, can be removed
from scipy.sparse import issparse
from torchcfm.conditional_flow_matching import *  # Import for conditional flow matching
import scanpy as sc  # Duplicate import
import hashlib
from llm import MAE  # Importing MAE model from llm module
import time
import sys
import anndata

def get_DEGs(control_adata, target_adata):
    temp_concat = anndata.concat([control_adata, target_adata], label = 'batch')
    sc.tl.rank_genes_groups(temp_concat, 'batch', method='wilcoxon', 
                                groups = ['1'], ref = '0', rankby_abs = True)

    rankings = temp_concat.uns['rank_genes_groups']
    result_df = pd.DataFrame({'scores': rankings['scores']['1'],
                     'pvals_adj': rankings['pvals_adj']['1']},
                    index = rankings['names']['1'])
    return result_df

def get_eval(true_adata, pred_adata, DEGs, DEG_vals, pval_threshold):
        
    results_dict =  {}
    
    true_mean = to_dense(true_adata.X).mean(axis = 0)
    true_var = to_dense(true_adata.X).var(axis = 0)
    
    pred_mean = to_dense(pred_adata.X).mean(axis = 0)
    pred_var = to_dense(pred_adata.X).var(axis = 0)
    
    true_corr_mtx = np.corrcoef(to_dense(true_adata.X), rowvar=False).flatten()
    true_cov_mtx = np.cov(to_dense(true_adata.X), rowvar=False).flatten()
        
    pred_corr_mtx = np.corrcoef(to_dense(pred_adata.X), rowvar=False).flatten()
    pred_cov_mtx = np.cov(to_dense(pred_adata.X), rowvar=False).flatten()

    results_dict['all_genes_mean_R2'] = scipy.stats.pearsonr(true_mean, pred_mean)[0]**2
    results_dict['all_genes_var_R2'] = scipy.stats.pearsonr(true_var, pred_var)[0]**2
    results_dict['all_genes_mean_MSE'] = (np.square(true_mean - pred_mean)).mean(axis=0)
    results_dict['all_genes_var_MSE'] = (np.square(true_var - pred_var)).mean(axis=0)
   
    corr_nas = np.logical_or(np.isnan(true_corr_mtx), np.isnan(pred_corr_mtx))
    cov_nas = np.logical_or(np.isnan(true_cov_mtx), np.isnan(pred_cov_mtx))
        
    results_dict['all_genes_corr_mtx_R2'] = scipy.stats.pearsonr(true_corr_mtx[~corr_nas], pred_corr_mtx[~corr_nas])[0]**2
    results_dict['all_genes_cov_mtx_R2'] = scipy.stats.pearsonr(true_cov_mtx[~cov_nas], pred_cov_mtx[~cov_nas])[0]**2
    results_dict['all_genes_corr_mtx_MSE'] = (np.square(true_corr_mtx[~corr_nas] - pred_corr_mtx[~corr_nas])).mean(axis=0)
    results_dict['all_genes_cov_mtx_MSE'] = (np.square(true_cov_mtx[~cov_nas] - pred_cov_mtx[~cov_nas])).mean(axis=0)

    significant_DEGs = DEGs[DEGs['pvals_adj'] < pval_threshold]
    num_DEGs = len(significant_DEGs)
    DEG_vals.insert(0, num_DEGs)
    
    for val in DEG_vals:
        if ((val > num_DEGs) or (val == 0)):
            results_dict[f'Top_{val}_DEGs_mean_R2'] = None
            results_dict[f'Top_{val}_DEGs_var_R2'] = None
            results_dict[f'Top_{val}_DEGs_mean_MSE'] = None
            results_dict[f'Top_{val}_DEGs_var_MSE'] = None
                        
            results_dict[f'Top_{val}_DEGs_corr_mtx_R2'] = None
            results_dict[f'Top_{val}_DEGs_cov_mtx_R2'] = None
            results_dict[f'Top_{val}_DEGs_corr_mtx_MSE'] = None
            results_dict[f'Top_{val}_DEGs_cov_mtx_MSE'] = None
        
        else:
            top_DEGs = significant_DEGs[0:val].index
        
            true_mean = to_dense(true_adata[:,top_DEGs].X).mean(axis = 0)
            true_var = to_dense(true_adata[:,top_DEGs].X).var(axis = 0)
            true_corr_mtx = np.corrcoef(to_dense(true_adata[:,top_DEGs].X), rowvar=False).flatten()
            true_cov_mtx = np.cov(to_dense(true_adata[:,top_DEGs].X), rowvar=False).flatten()

            pred_mean = to_dense(pred_adata[:,top_DEGs].X).mean(axis = 0)
            pred_var = to_dense(pred_adata[:,top_DEGs].X).var(axis = 0)
            pred_corr_mtx = np.corrcoef(to_dense(pred_adata[:,top_DEGs].X), rowvar=False).flatten()
            pred_cov_mtx = np.cov(to_dense(pred_adata[:,top_DEGs].X), rowvar=False).flatten()

            results_dict[f'Top_{val}_DEGs_mean_R2'] = scipy.stats.pearsonr(true_mean, pred_mean)[0]**2
            results_dict[f'Top_{val}_DEGs_var_R2'] = scipy.stats.pearsonr(true_var, pred_var)[0]**2
            results_dict[f'Top_{val}_DEGs_mean_MSE'] = (np.square(true_mean - pred_mean)).mean(axis=0)
            results_dict[f'Top_{val}_DEGs_var_MSE'] = (np.square(true_var - pred_var)).mean(axis=0)
            
            corr_nas = np.logical_or(np.isnan(true_corr_mtx), np.isnan(pred_corr_mtx))
            cov_nas = np.logical_or(np.isnan(true_cov_mtx), np.isnan(pred_cov_mtx))
            
            results_dict[f'Top_{val}_DEGs_corr_mtx_R2'] = scipy.stats.pearsonr(true_corr_mtx[~corr_nas], pred_corr_mtx[~corr_nas])[0]**2
            results_dict[f'Top_{val}_DEGs_cov_mtx_R2'] = scipy.stats.pearsonr(true_cov_mtx[~cov_nas], pred_cov_mtx[~cov_nas])[0]**2
            results_dict[f'Top_{val}_DEGs_corr_mtx_MSE'] = (np.square(true_corr_mtx[~corr_nas] - pred_corr_mtx[~corr_nas])).mean(axis=0)
            results_dict[f'Top_{val}_DEGs_cov_mtx_MSE'] = (np.square(true_cov_mtx[~cov_nas] - pred_cov_mtx[~cov_nas])).mean(axis=0)

    return results_dict

def get_DEG_Coverage_Recall(true_DEGs, pred_DEGs, p_cutoff):
    sig_true_DEGs = true_DEGs[true_DEGs['pvals_adj'] < p_cutoff]
    true_DEGs_with_direction = [get_DEG_with_direction(gene,score) for gene, score in zip(sig_true_DEGs.index, sig_true_DEGs['scores'])]
    sig_pred_DEGs = pred_DEGs[pred_DEGs['pvals_adj'] < p_cutoff]
    pred_DEGs_with_direction = [get_DEG_with_direction(gene,score) for gene, score in zip(sig_pred_DEGs.index, sig_pred_DEGs['scores'])]
    num_true_DEGs = len(true_DEGs_with_direction)
    num_pred_DEGs = len(pred_DEGs_with_direction)
    num_overlapping_DEGs = len(set(true_DEGs_with_direction).intersection(set(pred_DEGs_with_direction)))
    if num_true_DEGs > 0: 
        COVERAGE = num_overlapping_DEGs/num_true_DEGs
    else:
        COVERAGE = None
    if num_pred_DEGs > 0:
        RECALL = num_overlapping_DEGs/num_pred_DEGs
    else:
        RECALL = None
    return COVERAGE, RECALL

def get_DEGs_overlaps(true_DEGs, pred_DEGs, DEG_vals, pval_threshold):
    true_DEGs_for_comparison = [get_DEG_with_direction(gene,score) for gene, score in zip(true_DEGs.index, true_DEGs['scores'])]   
    pred_DEGs_for_comparison = [get_DEG_with_direction(gene,score) for gene, score in zip(pred_DEGs.index, pred_DEGs['scores'])]

    significant_DEGs = true_DEGs[true_DEGs['pvals_adj'] < pval_threshold]
    num_DEGs = len(significant_DEGs)
    DEG_vals.insert(0, num_DEGs)
    
    results = {}
    for val in DEG_vals:
        if val > num_DEGs:
            results[f'Overlap_in_top_{val}_DEGs'] = None
        else:
            results[f'Overlap_in_top_{val}_DEGs'] = len(set(true_DEGs_for_comparison[0:val]).intersection(set(pred_DEGs_for_comparison[0:val])))

    return results

def get_DEG_with_direction(gene, score):
    if score > 0:
        return(f'{gene}+')
    else:
        return(f'{gene}-')
        
def to_dense(X):
    if issparse(X):
        return X.toarray()
    else:
        return np.asarray(X)


start_time = time.time()  # Start time for tracking execution time

dataset_ctrl = 'satija_IFN_adjusted_ctrl'  # Dataset  name
dataset_pert = 'satija_IFN_adjusted_pert'  # Dataset  name
# Load data using Scanpy
adata_ctrl = sc.read(f'input/{dataset_ctrl}.h5ad')  # Read the dataset
adata_pert = sc.read(f'input/{dataset_pert}.h5ad') 
 # Set gene names as variable index
# Create a mapping from gene names to indices
gene_map = {k: i for i, k in enumerate(adata_ctrl.var['gene'])}
gene_map = gene_map | {'NT': max(gene_map.values()) + 1}  # Add 'NT' (non-targeting) as a control
gene_unmap = {gene_map[k]: k for k in gene_map}  # Reverse mapping from indices to gene names
perts = adata_pert.obs.gene.unique().map(gene_map)  # Map unique perturbations to indices
adata_pert.obs['pert_type'] = adata_pert.obs.gene.map(gene_map)  # Map perturbations in observations

holdoutcell = 'A549'
holdoutpert = gene_map['USP18']
holdoutidx = np.where((adata_pert.obs['pert_type']==holdoutpert)&(adata_pert.obs['cell_type']==holdoutcell))
trainidx = np.where((adata_pert.obs['pert_type']!=holdoutpert)|(adata_pert.obs['cell_type']!=holdoutcell))

X_ctrl = torch.from_numpy(np.log1p(adata_ctrl[trainidx].X)).float()
X_pert = torch.from_numpy(np.log1p(adata_pert[trainidx].X)).float()
whichpert = torch.from_numpy(np.array(adata_pert[trainidx].obs['pert_type']))

X_ctrl_h = torch.from_numpy(np.log1p(adata_ctrl[holdoutidx].X)).float()
X_pert_h = torch.from_numpy(np.log1p(adata_pert[holdoutidx].X)).float()
whichpert_h = torch.from_numpy(np.array(adata_pert[holdoutidx].obs['pert_type']))




# Define gene embedding module
class GeneEmbedding(torch.nn.Module):
    def __init__(self, input_dim, emb_dim=128):
        super().__init__()
        # Initialize positional embeddings for genes
        self.pos = torch.nn.Parameter(torch.zeros(1, input_dim, emb_dim))
        nn.init.normal_(self.pos)  # Initialize with normal distribution

# Define perturbation embedder module
class PertEmbedder(torch.nn.Module):
    def __init__(self, gene_embedding):
        super().__init__()
        _, input_dim, emb_dim = gene_embedding.pos.shape
        self.gene_embedding = gene_embedding
        # Define perturbation token as learnable parameter
        self.pert_token = torch.nn.Parameter(torch.zeros(emb_dim))
        nn.init.normal_(self.pert_token)  # Initialize with normal distribution
        
    def forward(self, pert_index, pert_expression):
        # Get gene embeddings for perturbation indices
        pert_pos = self.gene_embedding.pos[:, pert_index][0]
        # Combine perturbation embeddings with perturbation token and expression
        pert_embed_and_expr = torch.cat(
            (
                pert_pos + self.pert_token, 
                pert_expression.unsqueeze(-1)
            ), dim=-1
        )
        return pert_embed_and_expr.unsqueeze(1)  # Return with batch dimension

# Define cell encoder module
class CellEncoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim  
        self.latent_dim = latent_dim  # Dimension of the latent space
        self.dropout_rate = dropout_rate  
        
        # Encoder network definitions
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim*8, bias=False) 
        self.encoder_bn1 = nn.BatchNorm1d(hidden_dim*2)  
        self.encoder_fc2 = nn.Linear(hidden_dim*8, latent_dim, bias=False) 
        self.encoder_bn2 = nn.BatchNorm1d(hidden_dim) 
        self.fc_mu = nn.Linear(hidden_dim, latent_dim, bias=True) 
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim, bias=True) 

    # Encoder function
    def forward(self, x):
        # Pass input through encoder network
        h = F.leaky_relu(self.encoder_fc1(x))  
        # Return encoded features with ELU activation
        return self.encoder_fc2(h) # , self.fc_logvar(h)  

class ExprPred(torch.nn.Module):
    def __init__(self, gene_embedding, ff_dim=128) -> None:
        super().__init__()
        
        # Extract the embedding dimension from the gene embeddings
        _, _, emb_dim = gene_embedding.pos.shape
        self.gene_embedding = gene_embedding

        # Define the neural network for predicting continuous expression values
        self.pred_expr = torch.nn.Sequential(
            torch.nn.Linear(emb_dim + emb_dim, ff_dim),  # Input layer combining cell and gene embeddings
            torch.nn.SELU(),                             # Activation function
            torch.nn.Linear(ff_dim, ff_dim),             # Hidden layer
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),             # Additional hidden layers
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, 1), 
            torch.nn.ELU()    
             # Output layer for expression value
        ) 
        
        # Define the neural network for predicting binary expression (active/inactive)
        self.pred_bin = torch.nn.Sequential(
            torch.nn.Linear(emb_dim + emb_dim, ff_dim),  # Input layer combining cell and gene embeddings
            torch.nn.SELU(),                             # Activation function
            torch.nn.Linear(ff_dim, ff_dim),             # Hidden layer
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),             # Additional hidden layers
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, 1),                  # Output layer for binary prediction
            torch.nn.Sigmoid()                           # Sigmoid activation for binary output
        ) 

    def forward(self, cell_embedding, pred_idx):
        # Combine cell embedding with gene embeddings for prediction
        # Expand cell embeddings to match the number of genes
        #cell_embedding.shape = torch.Size([128, 256])
        #pred_idx = tensor([   0,    1,    2,  ..., 2051, 2052, 2053], device='cuda:0')
        #idx = tensor([   0,    1,    2,  ..., 2051, 2052, 2053], device='cuda:0')
        #for every geen, copy across eevry cell in the batch, for every cell, copy across every gene
        expanded_cell_embed = torch.tile(
            cell_embedding.unsqueeze(1),                # Add a dimension for concatenation
            (1, pred_idx.shape[0], 1)                   # Repeat for each gene index
        )
        #expanded_cell_embed.shape = torch.Size([128, 2054, 256])
        #(expanded_cell_embed[:,0]==expanded_cell_embed[:,1]).all() = True
        # Select and expand gene embeddings to match the batch size of cells
        selected_gene_embed = torch.tile(
            self.gene_embedding.pos[:, pred_idx],       # These are just random normal gene embeddings
            (cell_embedding.shape[0], 1, 1)             # Repeat for each cell in the batch
        )
        #self.gene_embedding.pos.shape = torch.Size([1, 2054, 256])
        
        
        
        # Concatenate cell and gene embeddings along the last dimension
        embed_and_cell_embed = torch.cat(
            (expanded_cell_embed, selected_gene_embed),
            dim=-1
        )
        #embed_and_cell_embed.shape = torch.Size([128, 2054, 512])
        # Predict continuous expression values and adjust output
        pred_expr = self.pred_expr(embed_and_cell_embed) + 1  # Shift outputs to ensure positivity
        
        # Predict binary expression states (active/inactive)
        pred_bin = self.pred_bin(embed_and_cell_embed)
        
        # Remove unnecessary singleton dimensions and return predictions
        return pred_expr.squeeze(), pred_bin.squeeze()


class InfMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim   
        
        # Encoder network definitions
        self.encoder_fc1 = nn.Linear(input_dim, input_dim//2, bias=False) 
        self.encoder_fc2 = nn.Linear(input_dim//2, input_dim//2, bias=False) 

    # Encoder function
    def forward(self, x):
        # Pass input through encoder network
        h = F.leaky_relu(self.encoder_fc1(x))  
        # Return encoded features with ELU activation
        return self.encoder_fc2(h) # , self.fc_logvar(h) 

# Define Bernoulli sampling layer for sparsity
class BernoulliSampleLayer(nn.Module):
    def __init__(self):
        super(BernoulliSampleLayer, self).__init__()

    def forward(self, probs):
        # Sample from Bernoulli distribution
        sample = torch.bernoulli(probs)
        # Return differentiable sample
        return sample + probs - probs.detach()    

# Define MAE model (Masked Autoencoder)
class MAE(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 ff_dim=128,
                 emb_dim=128,
                 encoder_layer=6,
                 ) -> None:
        super().__init__()
        
        self.gene_embedding = GeneEmbedding(input_dim=input_dim, emb_dim=emb_dim)
        self.pert_embedding = PertEmbedder(self.gene_embedding)
        self.encoder = CellEncoder(
            input_dim, emb_dim, hidden_dim=ff_dim
        )
        self.recon = ExprPred(self.gene_embedding)
        self.sparse_sampler = BernoulliSampleLayer()
        
        # Define flow model for conditional flow matching
        self.inf = InfMLP(2*emb_dim)

    def forward(self, expr):
        # Encode the input expressions
        #expr.shape = torch.Size([128, 2054])
        cell_embedding = self.encoder(expr)
        #cell_embedding.shape = torch.Size([128, 256])
        # cell_embedding = features.mean(axis=1)
        return cell_embedding  # Return cell embeddings

    def sparsify(self, pred_expr, pred_bin):
        # Apply sparsity by sampling from Bernoulli distribution
        sparsity = self.sparse_sampler(pred_bin)
        pred_expr *= sparsity
        return pred_expr

    def ae_loss(self, batch_emb, batch, gene_ids, lambd=0.5, return_recon=False):
        # Compute autoencoder loss
        batch_bin = (batch > 0).float() # Binary mask of expressed genes
        # Reconstruct expressions and binary states
        #batch_emb.shape = torch.Size([128, 256])
        batch_recon, batch_bin_recon = self.recon(batch_emb, gene_ids)
        #batch_recon.shape = torch.Size([128, 2054])
        # Compute reconstruction loss only on expressed genes
        recon_loss = torch.sum(batch_bin * (batch_recon - batch[:, gene_ids])**2)/torch.sum(batch_bin)
        # Compute binary cross-entropy loss for binary states
        #batch_bin_recon = batch_bin
        bin_loss = F.binary_cross_entropy(batch_bin_recon, batch_bin[:, gene_ids])
        # Combine losses with weighting factor lambd
        loss = lambd * recon_loss + (1 - lambd) * bin_loss
        if return_recon:
            return loss, batch_recon, batch_bin_recon
        return loss
    

    def inf_loss(self, source_emb, target_emb, cond):
        cond_emb = torch.cat((source_emb,target_emb), axis=1)
        pred_emb = self.inf(cond_emb)
        # Compute mean squared error loss
        return pred_emb, torch.nn.functional.mse_loss(pred_emb, target_emb) 
    

device = 'cuda' 

# Initialize the MAE model
model = MAE(
    X_ctrl.shape[1], 
    emb_dim=128, 
    encoder_layer=4,
    ff_dim=256
)

#model = torch.load(f"ae/2")
# model = torch.load(f"llm/v9")  # Optionally load a pre-trained model
model = model.to(device)
model = model.float()

base_learning_rate = 1e-3
optim = torch.optim.Adam(model.parameters(), lr=base_learning_rate)



class MyDataset(Dataset):
    def __init__(self, X_ctrl, X_pert, whichpert):
        self.X_ctrl = X_ctrl
        self.X_pert = X_pert
        self.whichpert = whichpert
        
    def __len__(self):
        return len(self.whichpert)
    
    def __getitem__(self, idx):
        return self.X_ctrl[idx], self.X_pert[idx], self.whichpert[idx]

# Create the dataset
dataset = MyDataset(X_ctrl, X_pert, whichpert)

# Create a DataLoader for random batch sampling
batch_size = 40  # Adjust this as desired
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

idx = torch.arange(X_ctrl.shape[1]).to(device)  # Gene indices

# Example training loop
num_epochs = 50000  # Adjust as needed
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

for epoch in range(num_epochs):
    model.train()
    running_ctrl_loss = 0.0

    running_pert_loss = 0.0
    running_inf_loss = 0.0

    for batch_idx, (x_ctrl_batch, x_pert_batch, whichpert_batch) in enumerate(dataloader):
        # Move batches to the device
        x_ctrl_batch = x_ctrl_batch.to(device)
        x_pert_batch = x_pert_batch.to(device)
        whichpert_batch = whichpert_batch.to(device)

        # Forward pass
        x_ctrl_emb = model(x_ctrl_batch)
        x_pert_emb = model(x_pert_batch)

        # Compute losses
        # Note: 'idx' should be defined elsewhere in your code, as it's being used below.
        ctrl_loss, ctrl_recon, ctrl_bin_recon = model.ae_loss(x_ctrl_emb, x_ctrl_batch, idx, return_recon=True)
        pert_loss, pert_recon, pert_bin_recon = model.ae_loss(x_pert_emb, x_pert_batch, idx, return_recon=True)
        
        cond = model.gene_embedding.pos[:, whichpert_batch][0]
        pred_emb, inf_loss = model.inf_loss(x_ctrl_emb, x_pert_emb, cond)
        inf_loss = 100*inf_loss
        # Combine losses (adjust weighting if desired)
        #total_loss = ctrl_loss + pert_loss + inf_loss
        total_loss = ctrl_loss + pert_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Accumulate for reporting
        running_ctrl_loss += ctrl_loss.item()
        running_pert_loss += pert_loss.item()
        running_inf_loss += inf_loss.item()

        # Print progress every 100 batches (adjust as needed)
        if (batch_idx + 1) % 100 == 0:
            avg_ctrl_loss = running_ctrl_loss / (batch_idx + 1)
            avg_pert_loss = running_pert_loss / (batch_idx + 1)
            avg_inf_loss = running_inf_loss / (batch_idx + 1)
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}]: "
                  f"ctrl_loss={avg_ctrl_loss:.4f}, pert_loss={avg_pert_loss:.4f}, inference_loss={avg_inf_loss:.4f}")

    # After each epoch, you can also report the average loss over the entire epoch
    avg_ctrl_loss = running_ctrl_loss / len(dataloader)
    avg_pert_loss = running_pert_loss / len(dataloader)
    avg_flow_loss = running_inf_loss / len(dataloader)
    print(f"End of Epoch [{epoch+1}/{num_epochs}]: ctrl_loss={avg_ctrl_loss:.4f}, "
          f"pert_loss={avg_pert_loss:.4f}, flow_loss={avg_flow_loss:.4f}")
    torch.save(model, f"ae/{epoch+1}")



    
sys.exit()
model = torch.load(f"ae/1")

true_pert = anndata.AnnData(X=X_pert_h.cpu().detach().numpy())
true_pert.obs['condition_key'] = 'perturbed'

control = anndata.AnnData(X=X_ctrl_h.cpu().detach().numpy())
control.obs['condition_key'] = 'control'




with torch.no_grad():
    model.eval()
    batch = X_pert_h.float()
    batch = batch.to(device)
    idx = torch.arange(batch.shape[1]).to(device)
    batch_emb = model(batch)

    processing_batch_size = 128
    
    # Initialize lists to collect outputs
    batch_recon_outputs = []
    batch_bin_recon_outputs = []
    
    # Total number of samples
    total_samples = batch_emb.size(0)
    
    # Loop over the data in smaller batches
    for start_idx in range(0, total_samples, processing_batch_size):
        end_idx = min(start_idx + processing_batch_size, total_samples)
        
        # Slice the input tensors
        batch_emb_chunk = batch_emb[start_idx:end_idx]
        batch_chunk = batch[start_idx:end_idx]
        
        # Process the chunk through the model
        _, batch_recon_chunk, batch_bin_recon_chunk = model.ae_loss(
            batch_emb_chunk, batch_chunk, idx, return_recon=True
        )
        
        # Collect the outputs
        batch_recon_outputs.append(batch_recon_chunk)
        batch_bin_recon_outputs.append(batch_bin_recon_chunk)
    
    # Concatenate the outputs to reconstruct the full tensors
    batch_recon = torch.cat(batch_recon_outputs, dim=0)
    batch_bin_recon = torch.cat(batch_bin_recon_outputs, dim=0)
    batch_recon = model.sparsify(batch_recon, batch_bin_recon)

predicted = anndata.AnnData(X=batch_recon.cpu().detach().numpy())
predicted.obs['condition_key'] = 'predicted'

# Invert the log1p transform
predicted.X = np.expm1(predicted.X)
control.X = np.expm1(control.X)
true_pert.X = np.expm1(true_pert.X)

# Normalize by total counts
sc.pp.normalize_total(predicted, target_sum=1e5)
sc.pp.normalize_total(control, target_sum=1e5)
sc.pp.normalize_total(true_pert, target_sum=1e5)

# Re-apply log1p
sc.pp.log1p(predicted)
sc.pp.log1p(control)
sc.pp.log1p(true_pert)


            
true_DEGs_df = get_DEGs(control, true_pert)
pred_DEGs_df = get_DEGs(control, predicted)
DEGs_overlaps = get_DEGs_overlaps(true_DEGs_df, pred_DEGs_df, [100,50,20], 0.05)

print(DEGs_overlaps)

sys.exit()

r2_and_mse = get_eval(true_pert, predicted, true_DEGs_df, [100,50,20], 0.05)

print(r2_and_mse)

sys.exit()

true_pert = anndata.AnnData(X=X_pert_h.cpu().detach().numpy())
true_pert.obs['condition_key'] = 'perturbed'

control = anndata.AnnData(X=X_ctrl_h.cpu().detach().numpy())
control.obs['condition_key'] = 'control'



with torch.no_grad():
    model.eval()
    batch = X_ctrl_h.float()
    batch = batch.to(device)
    idx = torch.arange(batch.shape[1]).to(device)
    batch_emb = model(batch)

    processing_batch_size = 128
    
    # Initialize lists to collect outputs
    batch_recon_outputs = []
    batch_bin_recon_outputs = []
    
    # Total number of samples
    total_samples = batch_emb.size(0)
    
    pert_ids = torch.tensor(np.repeat(holdoutpert, processing_batch_size)).to(device)
    
    # Loop over the data in smaller batches
    for start_idx in range(0, total_samples, processing_batch_size):
        end_idx = min(start_idx + processing_batch_size, total_samples)
        
        # Slice the input tensors
        batch_emb_chunk = batch_emb[start_idx:end_idx]
        batch_chunk = batch[start_idx:end_idx]
        whichpert_batch = pert_ids[:batch_emb_chunk.shape[0]]
        dummy_batch = batch[:batch_emb_chunk.shape[0]]

        cond = model.gene_embedding.pos[:, whichpert_batch][0]
        pred_emb, inf_loss = model.inf_loss(batch_emb_chunk, batch_emb_chunk, cond)
        # Compute losses
        # Note: 'idx' should be defined elsewhere in your code, as it's being used below.
        _, pred_recon, pred_bin_recon = model.ae_loss(pred_emb, dummy_batch, idx, return_recon=True)
        
        
        

        
        # Collect the outputs
        batch_recon_outputs.append(pred_recon)
        batch_bin_recon_outputs.append(pred_bin_recon)
    
    # Concatenate the outputs to reconstruct the full tensors
    batch_recon = torch.cat(batch_recon_outputs, dim=0)
    batch_bin_recon = torch.cat(batch_bin_recon_outputs, dim=0)
    batch_recon = model.sparsify(batch_recon, batch_bin_recon)


predicted = anndata.AnnData(X=batch_recon.cpu().detach().numpy())
predicted.obs['condition_key'] = 'predicted'

# Invert the log1p transform
predicted.X = np.expm1(predicted.X)
control.X = np.expm1(control.X)
true_pert.X = np.expm1(true_pert.X)

# Normalize by total counts
sc.pp.normalize_total(predicted, target_sum=1e5)
sc.pp.normalize_total(control, target_sum=1e5)
sc.pp.normalize_total(true_pert, target_sum=1e5)

# Re-apply log1p
sc.pp.log1p(predicted)
sc.pp.log1p(control)
sc.pp.log1p(true_pert)


            
true_DEGs_df = get_DEGs(control, true_pert)
pred_DEGs_df = get_DEGs(control, predicted)
DEGs_overlaps = get_DEGs_overlaps(true_DEGs_df, pred_DEGs_df, [100,50,20], 0.05)

print(DEGs_overlaps)

sys.exit()

model = torch.load(f"ae/0")

batch_size = 16  # Define batch size for training

# Create SCFMDataset instance for conditional flow matching
dset = SCFMDataset(
    control_train, pert_train, 
    pert_ids_train, pert_mat, 
    control_cell_types, pert_cell_types,
    batch_size=batch_size, size=X.shape[0]
)

# Calculate sample sizes for stratified sampling
ns = np.array([[t.shape[0] for t in ts] for ts in dset.target])

# Create DataLoader with stratified batch sampling
dl = torch.utils.data.DataLoader(
    dset, collate_fn=ot_collate, 
    batch_sampler=StratifiedBatchSampler(
        ns=ns, batch_size=16
    )
)


optim.zero_grad()
import gc
torch.cuda.empty_cache()
gc.collect()

use_sparsity_loss = False
use_mask_task = True
use_active_weights = False
lr_step = 32
minibatch_size = 16

step_count = 0
optim.zero_grad()
pert_task = 0
for e in range(5):
    model.train()
    losses = {'loss': [], 'control': [], 'pert': [], 'flow': []}
    for (bcontrol, bpert, bpert_index) in (pbar := tqdm(iter(dl))):
        bcontrol, bpert, bpert_index = bcontrol.squeeze(), bpert.squeeze(), bpert_index.reshape(-1, 1)# , # bpert_expr.squeeze()
        curr_batch_size = bcontrol.shape[0]
        for i in range(curr_batch_size // minibatch_size):
            ctrl = bcontrol[(i * minibatch_size):((i + 1) * minibatch_size)]
            if ctrl.shape[0] == 0:
                continue
            pert = bpert[(i * minibatch_size):((i + 1) * minibatch_size)]
            pert_index = bpert_index[(i * minibatch_size):((i + 1) * minibatch_size)]
            pert_index = pert_index.squeeze()
            
            ctrl = ctrl.float().to(device)
            pert = pert.float().to(device)
            
            idx = torch.arange(ctrl.shape[1]).to(device)
            
            step_count += 1
            
            ctrl_emb = model(ctrl)
            ctrl_loss, ctrl_recon, ctrl_bin_recon = model.ae_loss(ctrl_emb, ctrl, idx, return_recon=True)
            
            pert_emb = model(pert)
            pert_loss, pert_recon, pert_bin_recon = model.ae_loss(pert_emb, pert, idx, return_recon=True)
            
            cond = model.gene_embedding.pos[:, pert_index][0]
            flow_loss = model.flow_loss(ctrl_emb, pert_emb, cond)
            
            loss = ctrl_loss + pert_loss + flow_loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses['loss'].append(loss.item())
            losses['control'].append(ctrl_loss.item())
            losses['pert'].append(pert_loss.item())
            losses['flow'].append(flow_loss.item())
            if step_count % lr_step == 0:
                lr_scheduler.step()
            pbar.set_description(
                f"loss: {np.array(losses['loss'])[-lr_step:].mean():.3f}, tv: {np.array(losses['control'])[-lr_step:].mean():.3f}, ptv: {np.array(losses['pert'])[-lr_step:].mean():.3f}, flow: {np.array(losses['flow'])[-lr_step:].mean():.3f}"
            )
    
    avg_loss = sum(losses['control']) / len(losses['control'])
    torch.save(model, f"e2e/{e}")
    # writer.add_scalar('mae_loss', avg_loss, global_step=e)
    print(f'In epoch {e}, average traning loss is {avg_loss}.')

    
import math
import torch
import numpy as np
from torchdyn.core import NeuralODE
from datamodules import torch_wrapper

def compute_conditional_flow(
    model, control, pert_ids, pert_mat, batch_size=100, num_steps=400, n_batches=1e8, true_bin=None
):
    node = NeuralODE(
        torch_wrapper(model.flow).to(device), solver="dopri5", sensitivity="adjoint"
    )
    n_samples = min(control.shape[0], pert_ids.shape[0])
    n_batches = min(math.ceil(n_samples / batch_size), n_batches)
    preds = np.zeros((min(n_batches * batch_size, n_samples), control.shape[1]))
    with torch.no_grad():
        for i in range(n_batches):
            control_batch = control[batch_size*i:min(batch_size*(i+1), n_samples)]
            pert_batch = pert_mat[pert_ids][batch_size*i:min(batch_size*(i+1), n_samples)]# [:control_batch.shape[0]]
            model.flow.cond = pert_batch.to(device)
            inp = control_batch.float()
            inp = inp.to(device)
            
            idx = torch.arange(control_eval.shape[1]).to(device)
            cell_embedding = model(inp)
            
            outp = node.trajectory(
                cell_embedding,
                t_span=torch.linspace(0, 1, num_steps)
            )
            outp = outp[-1, :, :]
            outp, outb = model.recon(cell_embedding, idx)
            if true_bin:
                outb = true_bin.to(device)
            outp = model.sparsify(outp, outb)
            outp = outp.cpu()
            preds[batch_size*i:batch_size*(i+1), :] = outp.squeeze()
            
    return preds

for cell_type, pert_type in zip(holdout_cells, holdout_perts):
    break

cell_type_names = adata.obs[cell_col]
pert_type_names = adata.obs[pert_col]
control_eval =  torch.tensor(X[(cell_type_names == cell_type) & (pert_type_names == gene_map['NT'])]).float()
true_pert = torch.tensor(X[(pert_type_names == pert_type) & (cell_type_names == cell_type)]).float()

pred = compute_conditional_flow(
    model,
    control_eval, 
    torch.tensor(np.repeat(pert_type, control_eval.shape[0])), 
    model.gene_embedding.pos[0]
)  

pred_predicted = anndata.AnnData(X=pred)
pred_predicted.obs['condition_key'] = 'pred_predicted'

true_pert = anndata.AnnData(X=pert_eval.copy())
true_pert.obs['condition_key'] = 'perturbed'

control = anndata.AnnData(X=control_eval.cpu().detach().numpy())
control.obs['condition_key'] = 'control'

# Invert the log1p transform
pred_predicted.X = np.expm1(pred_predicted.X)
control.X = np.expm1(control.X)
true_pert.X = np.expm1(true_pert.X)

# Normalize by total counts
sc.pp.normalize_total(pred_predicted, target_sum=1e5)
sc.pp.normalize_total(control, target_sum=1e5)
sc.pp.normalize_total(true_pert, target_sum=1e5)

# Re-apply log1p
sc.pp.log1p(pred_predicted)
sc.pp.log1p(control)
sc.pp.log1p(true_pert)


            
true_DEGs_df = get_DEGs(control, true_pert)
pred_DEGs_df = get_DEGs(control, pred_predicted)
DEGs_overlaps = get_DEGs_overlaps(true_DEGs_df, pred_DEGs_df, [100,50,20], 0.05)

print(DEGs_overlaps)



sys.exit()



