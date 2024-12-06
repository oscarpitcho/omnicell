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

datasetso = 'satija_IFN'  # Dataset  name
# Load data using Scanpy
adata = sc.read(f'input/{datasetso}.h5ad')  # Read the dataset
adata.var.index = adata.var['gene']  # Set gene names as variable index
# Create a mapping from gene names to indices
gene_map = {k: i for i, k in enumerate(adata.var.index)}
gene_map = gene_map | {'NT': max(gene_map.values()) + 1}  # Add 'NT' (non-targeting) as a control
gene_unmap = {gene_map[k]: k for k in gene_map}  # Reverse mapping from indices to gene names
perts = adata.obs.gene.unique().map(gene_map)  # Map unique perturbations to indices
adata.obs['pert_type'] = adata.obs.gene.map(gene_map)  # Map perturbations in observations
pert_ids = np.array(adata.obs['pert_type'])  # Get perturbation IDs as numpy array
pert_mat = np.arange(pert_ids.max() + 1)[:, None]  # Create perturbation matrix

# Define column names for cell types and perturbations
cell_col = 'cell_type'
pert_col = 'pert_type'

# Define control perturbation and holdout cells and perturbations for evaluation
control_pert, holdout_cells, holdout_perts = gene_map['NT'], ['HT29'], [gene_map['USP18']]

# Get indices for training and evaluation
control_idx, pert_idx, eval_idx, eval_cell_idx, eval_pert_idx = get_train_eval_idxs(
    adata, control_pert, holdout_cells, holdout_perts, cell_col=cell_col, pert_col=pert_col
)

# Get identity features (e.g., one-hot encoding) for cell types
_, _, cell_types = get_identity_features(
    adata, cell_col=cell_col, pert_col=pert_col, cell_type_features=False
)

# Preprocess expression data
adata.obsm["standard"] = adata.X  # Store expression matrix in 'standard' obsm
X = adata.obsm["standard"]  # Get the expression matrix
X = X.toarray()  # Convert sparse matrix to dense array
X = np.log1p(X)
#X = X / X.sum(axis=1)[:, None]  # Normalize each cell by total counts
#X = np.log(X * 10_000. + 1)  # Log-transform the data

# Split data into training and evaluation sets
control_train, pert_train, pert_ids_train, control_cell_types, pert_cell_types, control_eval, pert_eval, pert_ids_eval = get_train_eval(
    X, pert_ids, cell_types, control_idx, pert_idx, eval_idx, eval_cell_idx, eval_pert_idx
)

# Combine control and perturbation training data
train = np.vstack([control_train, pert_train])

# Create a custom Dataset class for PyTorch
class NumpyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # Return a single data point converted to torch tensor
        return torch.from_numpy(self.data[index]).float()

    def __len__(self):
        # Return the total number of data points
        return len(self.data)

# Create an instance of the dataset with training data
dataset = NumpyDataset(train)



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


# Define conditional MLP for flow matching
class CMLP(pl.LightningModule):
    def __init__(self, feat_dim, cond_dim, out_dim=None, w1=128, w2=128, n_combo_layer=4, n_cond_layer=3, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = feat_dim
        # Define combined network
        self.combo_net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim + (1 if time_varying else 0) + cond_dim, w1), torch.nn.SELU(),
            *([torch.nn.Linear(w1, w1), torch.nn.SELU()] * n_combo_layer),
            torch.nn.Linear(w1, out_dim)
        )
        self.cond = None  # Placeholder for conditioning variable
        
    def forward(self, x, cond=None):
        if cond is None:
            cond = self.cond
        # Concatenate input and condition, and pass through network
        return self.combo_net(torch.cat([x, cond], dim=-1))

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
        self.flow = CMLP(feat_dim=emb_dim, cond_dim=emb_dim, time_varying=True, w1=ff_dim)
        self.cfm_sampler = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)

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
        batch_bin = (batch > 0).half() # Binary mask of expressed genes
        # Reconstruct expressions and binary states
        #batch_emb.shape = torch.Size([128, 256])
        batch_recon, batch_bin_recon = self.recon(batch_emb, gene_ids)
        #batch_recon.shape = torch.Size([128, 2054])
        # Compute reconstruction loss only on expressed genes
        recon_loss = torch.sum(batch_bin * (batch_recon - batch[:, gene_ids])**2)/torch.sum(batch_bin)
        # Compute binary cross-entropy loss for binary states
        batch_bin_recon = batch_bin
        bin_loss = F.binary_cross_entropy(batch_bin_recon, batch_bin[:, gene_ids])
        # Combine losses with weighting factor lambd
        loss = lambd * recon_loss + (1 - lambd) * bin_loss
        if return_recon:
            return loss, batch_recon, batch_bin_recon
        return loss

    def flow_loss(self, source_emb, target_emb, cond):
        # Compute flow matching loss
        t, xt, ut = self.cfm_sampler.sample_location_and_conditional_flow(
            source_emb, target_emb
        )
        # Concatenate time t to input
        inp = torch.cat([xt, t[:, None]], dim=-1)
        vt = self.flow(inp, cond)
        # Compute mean squared error loss
        return torch.nn.functional.mse_loss(vt, ut) 

device = 'cuda' 
# Initialize the MAE model
model = MAE(
    X.shape[1], 
    emb_dim=128, 
    encoder_layer=4,
    ff_dim=256
)

#model = torch.load(f"ae/2")
# model = torch.load(f"llm/v9")  # Optionally load a pre-trained model
 # Use GPU if available
# device = 'cpu'
model = model.to(device)
model = model.float()

# Set up optimizer and learning rate scheduler
base_learning_rate = 2e-4
weight_decay=0.0
total_epoch = 1000
warmup_epoch = 5
optim = torch.optim.Adam(model.parameters(), lr=base_learning_rate)
# Define learning rate schedule with warmup and cosine decay
lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-5), 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

use_sparsity_loss = False
use_mask_task = True
use_active_weights = False
lr_step = 32  # Steps for updating learning rate
minibatch_size = 16  # Minibatch size for training

step_count = 0
optim.zero_grad()
pert_task = 0
idx = torch.arange(train.shape[1]).to(device)  # Gene indices

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

# Training loop
for e in range(1):
    model.train()
    losses = {'loss': [], 'control': [], 'pert': [], 'flow': []}
    for batch in (pbar := tqdm(iter(dataloader))):
        step_count += 1
        batch = batch.float().to(device)
        idx = torch.arange(batch.shape[1]).to(device)
        batch_emb = model(batch)
        # Train to learn identity (autoencoder loss)
        loss, batch_recon, batch_bin_recon = model.ae_loss(batch_emb, batch, idx, return_recon=True)
        loss.backward()
        optim.step()
        optim.zero_grad()
        with torch.no_grad():
            # Apply sparsity to reconstructed expressions
            batch_recon = model.sparsify(batch_recon, batch_bin_recon)
            # Compute reconstruction loss
            recon_loss = torch.mean((batch_recon - batch)**2)
        losses['loss'].append(recon_loss.item())
        if step_count % lr_step == 0:
            lr_scheduler.step()
        pbar.set_description(
            f"loss: {np.array(losses['loss'])[-lr_step:].mean():.3f}"
        )
    
    # Compute average loss for the epoch
    avg_loss = sum(losses['loss']) / len(losses['loss'])
    # Save the model after each epoch
    torch.save(model, f"ae/{e}")
    # writer.add_scalar('mae_loss', avg_loss, global_step=e)
    print(f'In epoch {e}, average training loss is {avg_loss}.')
    
sys.exit()
model = torch.load(f"ae/1")

true_pert = anndata.AnnData(X=pert_eval.copy())
true_pert.obs['condition_key'] = 'perturbed'

control = anndata.AnnData(X=control_eval.copy())
control.obs['condition_key'] = 'control'




with torch.no_grad():
    model.eval()
    batch = torch.from_numpy(pert_eval).float()
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


