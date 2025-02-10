import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import scanpy as sc
import anndata


import pandas as pd
from scipy.sparse import issparse
import scipy
import warnings
PERT_KEY = 'pert'
CELL_KEY = 'cell'
CONTROL_PERT = 'ctrl'
GENE_VAR_KEY = 'gene_name'

warnings.filterwarnings(
    "ignore",
    message="Observation names are not unique. To make them unique, call `.obs_names_make_unique`.",
    category=UserWarning
)



##############################################
# ADDED FOR KNN (Imports for building and vectorizing)
##############################################
from sklearn.neighbors import NearestNeighbors

def build_knn_indices(emb_tensor, k=10):
    """
    emb_tensor: torch.Tensor of shape [N, d], on CPU or GPU
    Returns: knn_list, a list of arrays where knn_list[i] is the (k) neighbors of i.
    """
    # Move to CPU NumPy for scikit-learn
    emb = emb_tensor.detach().cpu().numpy()  # shape [N, d]
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(emb)
    distances, indices = nbrs.kneighbors(emb)
    
    # For each row i, indices[i,0] == i (the point itself)
    # So the kNN excluding the point itself is indices[i, 1..k]
    knn_list = [row[1:] for row in indices]
    return knn_list

def build_neighbor_idx(knn_list):
    """
    knn_list: list of length N, where knn_list[i] is a list/array of k neighbors for node i.
    Returns a LongTensor neighbor_idx of shape [N, k].
    """
    neighbor_tensors = []
    for nbrs in knn_list:
        nbrs_t = torch.tensor(nbrs, dtype=torch.long).unsqueeze(0)  # shape [1, k]
        neighbor_tensors.append(nbrs_t)
    neighbor_idx = torch.cat(neighbor_tensors, dim=0)  # shape [N, k]
    return neighbor_idx

def knn_pull_loss_vec(emb_tensor, neighbor_idx):
    """
    Vectorized pull loss that encourages each node i to be close to its k neighbors.
    emb_tensor: [N, d]
    neighbor_idx: [N, k]
    Returns: scalar 'pull' loss (MSE between each i and neighbor j).
    """
    device = emb_tensor.device
    N, d = emb_tensor.shape
    k = neighbor_idx.shape[1]
    
    # i_idx => shape [N, k]: repeated i for each neighbor
    i_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, k)  # [N, k]
    
    # Gather embeddings
    x_i = emb_tensor[i_idx]              # shape [N, k, d]
    x_j = emb_tensor[neighbor_idx]       # shape [N, k, d]
    
    # Squared distances => [N, k, d]
    dist_sq = (x_i - x_j)**2
    # Sum over the last dimension => [N, k]
    dist_sum = dist_sq.sum(dim=2)
    # Mean over all pairs => scalar
    pull_loss = dist_sum.mean()
    return pull_loss

##############################################
# ADDED FOR NEGATIVE SAMPLING (Push)
##############################################
def sample_negatives(knn_list, N, num_neg=5000, max_tries=500000):
    """
    Sample 'num_neg' random (i, j) pairs that are:
      - i != j
      - j not in knn_list[i]
    We do random i, j until we collect 'num_neg' valid pairs or exceed 'max_tries'.
    
    Returns two LongTensors i_neg, j_neg of shape [num_neg].
    """
    # Build a set of neighbors for quick membership checks
    neighbors_set = []
    for i, nbrs in enumerate(knn_list):
        neighbors_set.append(set(nbrs.tolist()))
    
    i_neg_list = []
    j_neg_list = []
    tries = 0
    
    while len(i_neg_list) < num_neg and tries < max_tries:
        i_ = np.random.randint(0, N)
        j_ = np.random.randint(0, N)
        if j_ == i_:
            tries += 1
            continue
        if j_ in neighbors_set[i_]:
            tries += 1
            continue

        # valid negative pair
        i_neg_list.append(i_)
        j_neg_list.append(j_)
        tries += 1
    
    # If we didn't get enough pairs, replicate
    if len(i_neg_list) < num_neg:
        shortfall = num_neg - len(i_neg_list)
        i_neg_list += i_neg_list[:shortfall]
        j_neg_list += j_neg_list[:shortfall]
    
    i_neg_t = torch.tensor(i_neg_list[:num_neg], dtype=torch.long)
    j_neg_t = torch.tensor(j_neg_list[:num_neg], dtype=torch.long)
    return i_neg_t, j_neg_t

def neg_push_loss(emb_tensor, i_neg, j_neg, margin=1.0):
    """
    Margin-based push loss for negative pairs (i_neg, j_neg).
    If dist(x_i, x_j) < margin, we penalize => max(0, margin - dist).
    
    emb_tensor: [N, d]
    i_neg, j_neg: shape [num_neg]
    margin: float
    Returns a scalar push loss
    """
    device = emb_tensor.device
    x_i = emb_tensor[i_neg]  # shape [num_neg, d]
    x_j = emb_tensor[j_neg]  # shape [num_neg, d]
    
    # L2 distance
    dist = torch.sqrt(((x_i - x_j)**2).sum(dim=1) + 1e-8)  # shape [num_neg]
    
    # margin-based hinge
    push = F.relu(margin - dist)  # shape [num_neg]
    return push.mean()




    
class autoencoder(nn.Module):
    """
    Architecture:
      1) control_encoder: (b, G) => (b, enc_dim)
      2) pert_embedding: (b,) => (b, enc_dim)
      3) gene_embedding: (G, enc_dim)
      4) For each gene => combine => decode to produce (b, G) for:
         - pred_ctrl
         - pred_delta
    """
    def __init__(
        self,
        model_config: dict,
        num_genes: int,
        enc_dim_cell: int = 340,
        enc_dim_pert: int = 80,
        hidden_enc_1: int = 3402,
        hidden_dec_1: int = 340
    ):
        super().__init__()
        self.model=None
        self.num_genes = num_genes

        # Encode Control
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, hidden_enc_1),
            nn.ReLU(),
            nn.Linear(hidden_enc_1, enc_dim_cell)
        )

        # Pert Embedding
        self.shared_embedding = nn.Embedding(num_genes + 1, enc_dim_pert)

        # We'll decode for pred_ctrl and pred_delta using
        # a single hidden layer, but then 2 heads:
        #   head_ctrl => (hidden_dec_1->1)
        #   head_delta => (hidden_dec_1->1)
        self.hidden_layer = nn.Linear(enc_dim_pert*2+enc_dim_cell, hidden_dec_1)
        self.head_ctrl    = nn.Linear(hidden_dec_1, 1)
        self.head_delta   = nn.Linear(hidden_dec_1, 1)

    def forward(self, x_ctrl_log, whichpert_idx, multiplier):
        """
        Inputs:
          x_ctrl_log: (b, G)
          whichpert_idx: (b,)
        Returns:
          pred_ctrl:  (b, G)
          pred_delta: (b, G)
        """
        b, G = x_ctrl_log.size()
        ctrl_embed = self.encoder(x_ctrl_log)  # (b, enc_dim_cell)
        pert_embed = multiplier*self.shared_embedding(whichpert_idx)  # (b, enc_dim_pert)

        # Combine cell + pert embeddings
        combined_cp = torch.cat([ctrl_embed, pert_embed], dim=1)  # (b, enc_dim_cell + enc_dim_pert)

        # Gene embeddings => shape(G, enc_dim_pert)
        gene_emb_all = self.shared_embedding.weight[:G]

        # Expand => shape(b, G, (enc_dim_cell + enc_dim_pert))
        ccp_expanded = combined_cp.unsqueeze(1).expand(b, G, -1)
        # shape(b, G, enc_dim_pert)
        ge_expanded  = gene_emb_all.unsqueeze(0).expand(b, G, -1)

        # Concat => shape(b, G, enc_dim_pert*2 + enc_dim_cell)
        dec_in = torch.cat([ccp_expanded, ge_expanded], dim=2)
        # Flatten => shape(b*G, enc_dim_pert*2 + enc_dim_cell)
        dec_in_flat = dec_in.view(b*G, -1)

        x_hidden = F.relu(self.hidden_layer(dec_in_flat))
        pred_ctrl_flat  = self.head_ctrl(x_hidden)   # (bG,1)
        pred_delta_flat = self.head_delta(x_hidden)  # (bG,1)

        pred_ctrl  = pred_ctrl_flat.view(b, G)
        pred_delta = pred_delta_flat.view(b, G)

        return pred_ctrl, pred_delta
    
    def masked_mse_dual(self, pred_ctrl, pred_delta, true_ctrl, true_pert, mask):
        """
        We want to measure errors in predicting:
          - pred_ctrl vs. true_ctrl
          - pred_delta vs. (true_pert - true_ctrl)
        """
        true_delta = true_pert - true_ctrl
        diff_sq_ctrl = (pred_ctrl - true_ctrl)**2
        diff_sq_ctrl_masked = diff_sq_ctrl[mask]
        diff_sq_delta = (pred_delta - true_delta)**2
        diff_sq_delta_masked = diff_sq_delta[mask]
    
        if diff_sq_ctrl_masked.numel() == 0:
            loss_ctrl = diff_sq_ctrl.mean()
        else:
            loss_ctrl = diff_sq_ctrl_masked.mean()
    
        if diff_sq_delta_masked.numel() == 0:
            loss_delta = diff_sq_delta.mean()
        else:
            loss_delta = diff_sq_delta_masked.mean()
    
        loss_total = 0.9*loss_ctrl + 0.1*loss_delta
        return loss_total, loss_ctrl.item(), loss_delta.item()

    def train(self, dl):
        """
        Custom training loop that moves model & data to the same device.
        """
        # Choose device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Make sure the model and its embeddings are on that device
        self.to(device)
        
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        start_epoch = 0
        num_epochs = 10
        
        print_interval = 5000
        

        KNN_K = 10
        KNN_EMB_LOSS_WEIGHT = 1e-2
        REBUILD_EVERY = 1
        NEG_LOSS_WEIGHT = 1e-2
        NEG_PAIRS = 7000
        MARGIN = 1.0
        
        neg_i = None
        neg_j = None
        
        knn_list = None
        neighbor_idx = None
        
        import time
        start_time= time.time()

        for epoch in range(start_epoch, num_epochs):
        
            ##############################################
            # Rebuild adjacency (knn_list) + neighbor_idx
            # Also sample negative pairs
            ##############################################
            if epoch % REBUILD_EVERY == 0:
                # emb_mat will already be on GPU if self.to(device) was called
                emb_mat = self.shared_embedding.weight  # shape [num_genes+1, enc_dim_pert]
                N = emb_mat.size(0)
                
                # Build k-NN adjacency (assumes these utility funcs can handle CPU/GPU as needed)
                knn_list = build_knn_indices(emb_mat, k=KNN_K)
                # move to device
                neighbor_idx = build_neighbor_idx(knn_list).to(emb_mat.device)
        
                # Sample negative pairs
                i_neg_t, j_neg_t = sample_negatives(knn_list, N, num_neg=NEG_PAIRS)
                neg_i = i_neg_t.to(emb_mat.device)
                neg_j = j_neg_t.to(emb_mat.device)
        
            running_loss = 0.0
            num_samples = 0
        
            for batch_idx, (x_ctrl_batch_in, x_pert_batch_in, whichpert_batch_in) in enumerate(iter(dl)):


                # Move input tensors to the same device
                x_ctrl_batch    = torch.log1p(torch.clone(x_ctrl_batch_in))
                x_pert_batch    = torch.log1p(torch.clone(x_pert_batch_in))
                
                x_ctrl_batch    = x_ctrl_batch.to(device)
                x_pert_batch    = x_pert_batch.to(device)
                whichpert_batch = torch.clone(whichpert_batch_in).to(device)
                mask_batch      = (x_ctrl_batch > 0).to(device)
        
                optimizer.zero_grad()
        
                pred_ctrl, pred_delta = self.forward(x_ctrl_batch, whichpert_batch, multiplier=1)
                pred_ctrl  = pred_ctrl  * mask_batch.float()
                pred_delta = pred_delta * mask_batch.float()
        
                loss_total, loss_ctrl_val, loss_delta_val = self.masked_mse_dual(
                    pred_ctrl, pred_delta,
                    x_ctrl_batch, x_pert_batch,
                    mask_batch
                )
        
                ##############################################
                # Vectorized Pull Loss on embedding
                ##############################################
                if neighbor_idx is not None:
                    emb_mat = self.shared_embedding.weight  # shape [N, d]
                    loss_knn = knn_pull_loss_vec(emb_mat, neighbor_idx)
                else:
                    loss_knn = 0.0
        
                ##############################################
                # Negative Sampling Push Loss
                ##############################################
                loss_neg = 0.0
                if neg_i is not None and neg_j is not None:
                    emb_mat = self.shared_embedding.weight
                    dist_push = neg_push_loss(emb_mat, neg_i, neg_j, margin=MARGIN)
                    loss_neg = dist_push
        
                # Combine everything
                loss_total_with_knn = loss_total
                if loss_knn != 0.0:
                    loss_total_with_knn += KNN_EMB_LOSS_WEIGHT * loss_knn
                if loss_neg != 0.0:
                    loss_total_with_knn += NEG_LOSS_WEIGHT * loss_neg
        
                loss_total_with_knn.backward()
                optimizer.step()
                
                bs_ = x_ctrl_batch.size(0)
                running_loss += loss_total.item() * bs_
                num_samples += bs_
                
                iteration = batch_idx + 1 + epoch*len(dl)
                if iteration % print_interval == 0:
                    curr_mse = running_loss / num_samples
                    elapsed = time.time() - start_time
                    print(f"[Epoch {epoch+1}, Iter {batch_idx+1}] "
                          f"Train MSE so far={curr_mse:.6f} (ctrl+delta)   "
                          f"Time last {print_interval} iters: {elapsed:.2f}s")
                    running_loss= 0.0
                    num_samples= 0
                    start_time= time.time()

    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)  # ensure model is on device

        # Extract control cells
        ctrl_cells = adata[
            (adata.obs[PERT_KEY] == CONTROL_PERT) & (adata.obs[CELL_KEY] == cell_type)
        ].X.toarray().copy()
        ctrl_cells = torch.from_numpy(ctrl_cells).float().to(device)
        ctrl_cells = torch.log1p(ctrl_cells)

        # Build the whichpert index on device
        whichpert_idx = np.where(adata.var[GENE_VAR_KEY] == pert_id)[0][0]
        whichpert = torch.tensor([whichpert_idx] * ctrl_cells.shape[0], dtype=torch.long, device=device)
        batch_size = 32
        pred_ctrl_list = []
        pred_delta_list = []
        
        with torch.no_grad():
            # Loop over the dataset in mini-batches.
            for i in range(0, ctrl_cells.size(0), batch_size):
                # Slice the inputs for the current batch.
                batch_ctrl_cells = torch.clone(ctrl_cells[i:i+batch_size])
                batch_whichpert = torch.clone(whichpert[i:i+batch_size])  # Remove or adjust if not batched
        
                # Forward pass for the batch
                batch_pred_ctrl, batch_pred_delta = self.forward(batch_ctrl_cells, batch_whichpert, multiplier=1)
                
                # Store the batch predictions
                pred_ctrl_list.append(torch.clone(batch_pred_ctrl))
                pred_delta_list.append(torch.clone(batch_pred_delta))
        
        # Concatenate all the batch outputs along the batch dimension.
        pred_ctrl = torch.cat(pred_ctrl_list, dim=0)
        pred_delta = torch.cat(pred_delta_list, dim=0)
        masko = ctrl_cells>0
        pred = ctrl_cells + pred_delta
        pred = masko*pred
        pred = np.expm1(pred.cpu().detach().numpy())
        pred[pred<=0] = 0
        pred = np.round(pred)
        # If you want outputs back on CPU as a NumPy array:
        return pred
