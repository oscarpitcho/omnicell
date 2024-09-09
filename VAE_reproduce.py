#Copyright 2022, Jason Lequyer and Laurence Pelletier, All rights reserved.
#Sinai Health SystemLunenfeld-Tanenbaum Research Institute
#600 University Avenue, Room 1070
#Toronto, ON, M5G 1X5, Canada
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import numpy as np
from pathlib import Path
import time
import scanpy as sc
import anndata as ad
import scipy
from scipy.sparse import issparse
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
    
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



#Assumption: at least 3 cell types
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inp = sc.read("input/kang.h5ad")
numvars = len(inp.var_names)


ctkey = 'cell_type'
ptkey = 'condition_key'
ptctrlkey = 'control'

holdoutcell = 'NK'

holdoutidx = ((inp.obs[ctkey]==holdoutcell) & (inp.obs[ptkey]!=ptctrlkey))
inp_noholdout = inp[~holdoutidx].copy()
inp_holdout = inp[holdoutidx].copy()
datalen = inp_noholdout.shape[0]

indices = np.random.permutation(datalen)


# Subsetting the AnnData object to create train and validation sets
train = inp_noholdout[indices[:np.int32(datalen*0.9)]]
valid = inp_noholdout[np.int32(datalen*0.9):]


goodo=True
counter=0
batsize = 32
valbatsize = 64

dropout_rate=0.2
latent_dim=100
epochs=300


class Net(nn.Module):
    def __init__(self, input_dim, latent_dim=100, dropout_rate=0.2, learning_rate=0.001, alpha=0.001, model_path='./models/net'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.model_path = model_path
        os.makedirs(self.model_path, exist_ok=True)
        
        # Encoder layers
        self.encoder_fc1 = nn.Linear(input_dim, 800, bias=False)
        self.encoder_bn1 = nn.BatchNorm1d(800)
        self.encoder_fc2 = nn.Linear(800, 800, bias=False)
        self.encoder_bn2 = nn.BatchNorm1d(800)
        self.fc_mu = nn.Linear(800, latent_dim, bias=True)
        self.fc_logvar = nn.Linear(800, latent_dim, bias=True)
        
        # Decoder layers
        self.decoder_fc1 = nn.Linear(latent_dim, 800, bias=False)
        self.decoder_bn1 = nn.BatchNorm1d(800)
        self.decoder_fc2 = nn.Linear(800, 800, bias=False)
        self.decoder_bn2 = nn.BatchNorm1d(800)
        self.decoder_fc3 = nn.Linear(800, input_dim, bias=True)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def encode(self, x):
        h = F.leaky_relu(self.encoder_bn1(self.encoder_fc1(x)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.leaky_relu(self.encoder_bn2(self.encoder_fc2(h)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc1(z)
        h = self.decoder_bn1(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = self.decoder_fc2(h)
        h = self.decoder_bn2(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = self.decoder_fc3(h)
        h = torch.relu(h)  # Using ReLU as the final activation
        return h

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x.view(-1, self.input_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.alpha * KLD
    
class NetEval(nn.Module):
    def __init__(self, input_dim, latent_dim=100, dropout_rate=0.2, learning_rate=0.001, alpha=0.00005, model_path='./models/net'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.model_path = model_path
        os.makedirs(self.model_path, exist_ok=True)
        
        # Encoder layers
        self.encoder_fc1 = nn.Linear(input_dim, 800, bias=False)
        self.encoder_bn1 = nn.BatchNorm1d(800)
        self.encoder_fc2 = nn.Linear(800, 800, bias=False)
        self.encoder_bn2 = nn.BatchNorm1d(800)
        self.fc_mu = nn.Linear(800, latent_dim, bias=True)
        self.fc_logvar = nn.Linear(800, latent_dim, bias=True)
        
        # Decoder layers
        self.decoder_fc1 = nn.Linear(latent_dim, 800, bias=False)
        self.decoder_bn1 = nn.BatchNorm1d(800)
        self.decoder_fc2 = nn.Linear(800, 800, bias=False)
        self.decoder_bn2 = nn.BatchNorm1d(800)
        self.decoder_fc3 = nn.Linear(800, input_dim, bias=True)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def encode(self, x):
        h = F.leaky_relu((self.encoder_fc1(x)))
        h = F.leaky_relu((self.encoder_fc2(h)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc1(z)
        h = F.leaky_relu(h)
        h = self.decoder_fc2(h)
        h = F.leaky_relu(h)
        h = self.decoder_fc3(h)
        h = torch.relu(h)  # Using ReLU as the final activation
        return h

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x.view(-1, self.input_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.alpha * KLD

trainlen = train.shape[0]
validlen = valid.shape[0]
num_genes = train.shape[1]
net = Net(num_genes)
net.to(device)
neteval = Net(num_genes)
neteval.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
running_loss = 0


    
trainctrlnhidx = (train.obs[ptkey]==ptctrlkey).values & (train.obs[ctkey]!=holdoutcell).values
holdoutctrlidx = (train.obs[ptkey]==ptctrlkey).values & (train.obs[ctkey]==holdoutcell).values
trainholdoutidx = (train.obs[ctkey]!=holdoutcell).values
pathways_pert = np.unique(train[train.obs[ptkey]!=ptctrlkey].obs[ptkey])
pathways_idx = []
control_holdout = train[holdoutctrlidx].copy().X.todense()

perturbed_holdout = []
for curpt in pathways_pert:
    pathways_idx.append((train.obs[ptkey]==curpt).values)
    perturbed_holdout.append(inp_holdout[inp_holdout.obs[ptkey]==curpt].copy().X.todense())
trainX = torch.from_numpy(train.X.todense()).to(device)
validX = torch.from_numpy(valid.X.todense()).to(device)




trainX_latent = torch.clone(trainX[:,:latent_dim])
trainX_var = torch.clone(trainX[:,:latent_dim])

holdoutX = torch.from_numpy(inp_holdout[inp_holdout.obs[ptkey]!=ptctrlkey].X.todense()).to(device)
holdoutlen = holdoutX.shape[0]
holdoutX_latent = torch.clone(holdoutX[:,:latent_dim])
holdoutX_var = torch.clone(holdoutX[:,:latent_dim])

celltypes = np.unique(train[train.obs[ctkey]!=holdoutcell].obs[ctkey])
ctidxctrl = []
for ct in celltypes:
    ctidxctrl.append((train.obs[ctkey]==ct).values)
ctrlidxos = (train.obs[ptkey]==ptctrlkey).values




for e in range(epochs):
    running_loss = 0

    for lower in range(0, trainlen, batsize):
        upper = min(lower + batsize, trainlen)
        lower = min(trainlen-batsize,lower)
        counter += 1
        batch = trainX[lower:upper,:]
        optimizer.zero_grad()
        out, mu, logvar = net(batch)
        loss = net.loss_function(out,batch,mu,logvar)
        loss.backward()
        running_loss+=loss.item()
        optimizer.step()
    print(f'Epoch {e+1}/{epochs}')
    print(f'Train loss: {running_loss/1000000}')
    running_loss = 0
    state_dict = net.state_dict()
    neteval.load_state_dict(state_dict)
    with torch.no_grad():
        for lower in range(0, validlen, batsize):
            upper = min(lower + batsize, validlen)
            lower = min(validlen-batsize,lower)
            counter += 1
            batch = validX[lower:upper,:]
            out, mu, logvar = neteval(batch)
            loss = neteval.loss_function(out,batch,mu,logvar)
            running_loss+=loss.item()
        print(f'Valid loss: {running_loss/1000000}')
        
   
    #encode training set
    neteval.eval()
    if (e+1) % 50 == 0:
        predicted_holdout = []
        with torch.no_grad():
            for lower in range(0, trainlen, batsize):
                upper = min(lower + batsize, trainlen)
                lower = min(trainlen-batsize,lower)
                counter += 1
                batch = trainX[lower:upper,:]
                mu, logvar = neteval.encode(batch)
                trainX_latent[lower:upper,:] = mu
                trainX_var[lower:upper,:] = logvar
            for lower in range(0, holdoutlen, batsize):
                upper = min(lower + batsize, holdoutlen)
                lower = min(trainlen-batsize,lower)
                counter += 1
                batch = holdoutX[lower:upper,:]
                mu, logvar = neteval.encode(batch)
                holdoutX_latent[lower:upper,:] = mu
                holdoutX_var[lower:upper,:] = logvar
            avgctl = []
            for ctx in ctidxctrl:
                avgctl.append(torch.mean(trainX_latent[ctx & ctrlidxos],axis=0))
            avgctl = torch.stack(avgctl)
            avgctl = torch.mean(avgctl,axis=0)

            ctrl_holdout_latent = trainX_latent[holdoutctrlidx]
            ctrl_holdout_var = trainX_var[holdoutctrlidx]
            for curptidx in range(len(pathways_pert)):
                avgcurpt = []
                for ctx in ctidxctrl:
                    avgcurpt.append(torch.mean(trainX_latent[ctx & pathways_idx[curptidx]],axis=0))
                avgcurpt = torch.stack(avgcurpt)
                avgcurpt = torch.mean(avgcurpt,axis=0)
                curdelta = torch.unsqueeze(avgcurpt - avgctl,0)
                ctrl_holdout_shifted = ctrl_holdout_latent + curdelta
                ctrl_holdout_reparameterized = neteval.reparameterize(ctrl_holdout_shifted,ctrl_holdout_var)
                ctrl_holdout_decoded = neteval.decode(ctrl_holdout_reparameterized )
                predicted_holdout.append(ctrl_holdout_decoded.cpu().detach().numpy())
                
        pert_dan = np.array(perturbed_holdout[0])
        cont_dan = np.array(control_holdout)
        pred_dan = predicted_holdout[0]
        
        #Delete
        #pert_dan = holdoutX_latent.cpu().detach().numpy()
        #cont_dan = ctrl_holdout_latent.cpu().detach().numpy()
        #pred_dan = ctrl_holdout_shifted.cpu().detach().numpy()
        #
        
        
        pred_pert = anndata.AnnData(X=pred_dan)
        sc.pp.normalize_total(pred_pert)
        pred_pert.obs['condition_key'] = 'predicted'
        
        true_pert = anndata.AnnData(X=pert_dan)
        sc.pp.normalize_total(true_pert)
        true_pert.obs['condition_key'] = 'perturbed'
        
        
        control = anndata.AnnData(X=cont_dan)
        sc.pp.normalize_total(control)
        control.obs['condition_key'] = 'control'
        
        allcon = ad.concat([pred_pert,true_pert,control])

        
        control.obs_names = control.obs_names+'-1'
        control.X[0,(control.X.var(axis=0)==0)] += np.amin(control.X[np.nonzero(control.X)])
        pred_pert.X[0,(pred_pert.X.var(axis=0)==0)] += np.amin(pred_pert.X[np.nonzero(pred_pert.X)])
        true_pert.X[0,(true_pert.X.var(axis=0)==0)] += np.amin(true_pert.X[np.nonzero(true_pert.X)])
        
        
        true_DEGs_df = get_DEGs(control, true_pert)
        pred_DEGs_df = get_DEGs(control, pred_pert)

        DEGs_overlaps = get_DEGs_overlaps(true_DEGs_df, pred_DEGs_df, [100,50,20], 0.05)
        
        
        
        print('DEGs: '+str(DEGs_overlaps['Overlap_in_top_20_DEGs'])+' ('+holdoutcell+')')


            
            
            
        
            
        
        
    
    
  
torch.save(net.state_dict(), 'B')
        

