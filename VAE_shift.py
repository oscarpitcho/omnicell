
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
    




#Assumption: at least 3 cell types
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inp = sc.read("input/nault.h5ad")
sc.pp.normalize_total(inp)
numvars = len(inp.var_names)


ctkey = 'cell_type'
ptkey = 'condition_key'
ptctrlkey = 'control'

holdoutcell = 'Macrophage'

holdoutidx = ((inp.obs[ctkey]==holdoutcell) & (inp.obs[ptkey]!=ptctrlkey))
inp_noholdout = inp[~holdoutidx].copy()
inp_holdout = inp[holdoutidx].copy()
datalen = inp_noholdout.shape[0]

indices = np.random.permutation(datalen)


# Subsetting the AnnData object to create train and validation sets
train = inp_noholdout[indices[:np.int32(datalen*0.9)]]
valid = inp_noholdout[indices[np.int32(datalen*0.9):]]


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
    



num_genes = train.shape[1]
net = Net(num_genes)
net.to(device)
neteval = Net(num_genes)
neteval.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
running_loss = 0


    
trainctrlnhidx = (train.obs[ptkey]==ptctrlkey).values & (train.obs[ctkey]!=holdoutcell).values
trainpertnhidx = (train.obs[ptkey]!=ptctrlkey).values & (train.obs[ctkey]!=holdoutcell).values
ctdx = np.where(trainctrlnhidx)[0]
ptdx = np.where(trainpertnhidx)[0]
    
validctrlnhidx = (valid.obs[ptkey]==ptctrlkey).values & (valid.obs[ctkey]!=holdoutcell).values
validpertnhidx = (valid.obs[ptkey]!=ptctrlkey).values & (valid.obs[ctkey]!=holdoutcell).values
ctdx_v = np.where(validctrlnhidx)[0]
ptdx_v = np.where(validpertnhidx)[0]

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


trainlen = min(len(ctdx),len(ptdx))
validlen = min(len(ctdx_v),len(ptdx_v))

totlen = trainX.shape[0]

epochs=500000
running_loss = 0
running_loss_latent = 0
avgcounter = 0
timer = 10
for e in range(epochs):
    for lower in range(0, trainlen, batsize):
        upper = min(lower + batsize, trainlen)
        lower = min(trainlen-batsize,lower)
        counter += 1
        
        batch = trainX[ctdx[lower:upper],:]
        batch_pert = trainX[ptdx[lower:upper],:]
        wholebatch = torch.stack([batch,batch_pert])
        optimizer.zero_grad()
        out, mu, logvar = net(wholebatch)
        out_pert, mu_pert, logvar_pert = out[batsize:], mu[batsize:], logvar[batsize:]
        out, mu, logvar = out[:batsize], mu[:batsize], logvar[:batsize]
        
        
        
        
        loss_latent = torch.mean((mu_pert - mu - 1)**2)
    
        loss_recon = neteval.loss_function(out,batch,mu,logvar) + net.loss_function(out_pert,batch_pert,mu_pert-1,logvar_pert)
        loss = loss_latent + loss_recon
        loss.backward()
        running_loss+=loss_recon.item()
        running_loss_latent+=loss_latent.item()
        avgcounter+=1
        optimizer.step()
    if e % timer == 0:
        running_loss /= avgcounter
        running_loss_latent /= avgcounter
        print(f'Epoch {e+1}/{epochs}')
        print(f'Train loss latent: {running_loss_latent}')
        print(f'Train loss recon: {running_loss}')
        running_loss = 0
        running_loss_latent = 0
        avgcounter = 0
    state_dict = net.state_dict()
    neteval.load_state_dict(state_dict)
    neteval.eval()
    with torch.no_grad():
        for lower in range(0, validlen, batsize):
            upper = min(lower + batsize, validlen)
            lower = min(validlen-batsize,lower)
            counter += 1
            batch = validX[ctdx_v[lower:upper],:]
            batch_pert = validX[ptdx_v[lower:upper],:]
            wholebatch = torch.stack([batch,batch_pert])
            optimizer.zero_grad()
            out, mu, logvar = net(wholebatch)
            out_pert, mu_pert, logvar_pert = out[batsize:], mu[batsize:], logvar[batsize:]
            out, mu, logvar = out[:batsize], mu[:batsize], logvar[:batsize]
            
            loss_latent = torch.mean((mu_pert - mu - 1)**2)
        
            loss_recon = neteval.loss_function(out,batch,mu,logvar) + net.loss_function(out_pert,batch_pert,mu_pert-1,logvar_pert)
            loss = loss_latent + loss_recon
            running_loss+=loss_recon.item()
            running_loss_latent+=loss_latent.item()
            avgcounter+=1
        if e % timer == 0:
            running_loss /= avgcounter
            running_loss_latent /= avgcounter
            print(f'Valid loss latent: {running_loss_latent}')
            print(f'Valid loss recon: {running_loss}')

        
   
    #encode training set
    if e % timer == 0:
        predicted_holdout = []
        with torch.no_grad():
            for lower in range(0, totlen, batsize):
                upper = min(lower + batsize, totlen)
                lower = min(totlen-batsize,lower)
                counter += 1
                batch = trainX[lower:upper,:]
                _, mu, logvar = neteval(batch)
                trainX_latent[lower:upper,:] = mu
                trainX_var[lower:upper,:] = logvar
            for lower in range(0, holdoutlen, batsize):
                upper = min(lower + batsize, holdoutlen)
                lower = min(trainlen-batsize,lower)
                counter += 1
                batch = holdoutX[lower:upper,:]
                _, mu, logvar = neteval(batch)
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
                #curdelta = torch.unsqueeze(avgcurpt - avgctl,0)
                curdelta = 1
                ctrl_holdout_shifted = ctrl_holdout_latent + curdelta
                ctrl_holdout_reparameterized = neteval.reparameterize(ctrl_holdout_shifted,ctrl_holdout_var)
                ctrl_holdout_decoded = neteval.decode(ctrl_holdout_reparameterized )
                predicted_holdout.append(ctrl_holdout_decoded.cpu().detach().numpy())
                
        pert_dan = np.array(perturbed_holdout[0])
        cont_dan = np.array(control_holdout)
        pred_dan = predicted_holdout[0]
        

        
        pred_pert = anndata.AnnData(X=pred_dan)
        pred_pert.obs['condition_key'] = 'predicted'
        
        true_pert = anndata.AnnData(X=pert_dan)
        true_pert.obs['condition_key'] = 'perturbed'
        
        
        control = anndata.AnnData(X=cont_dan)
        control.obs['condition_key'] = 'control'
        
        allcon = ad.concat([pred_pert,true_pert,control])
        
        sc.pp.neighbors(allcon, n_neighbors=15)
        sc.tl.umap(allcon, min_dist=0.1, random_state=42)
        with plt.rc_context({'axes.spines.right': False, 'axes.spines.top': False}):
            # Generate the PCA plot
            ax = sc.pl.umap(allcon, color=["condition_key"], groups=["control", "perturbed","predicted"], show=False)
            #ax = sc.pl.umap(curdata, color=["condition_key"], groups=["perturbed_decoded","perturbed"], show=False)
            
            
    

            
            # Show the plot
            plt.show()

        
        control.obs_names = control.obs_names+'-1'
        control.X[0,(control.X.var(axis=0)==0)] += np.amin(control.X[np.nonzero(control.X)])
        pred_pert.X[0,(pred_pert.X.var(axis=0)==0)] += np.amin(pred_pert.X[np.nonzero(pred_pert.X)])
        true_pert.X[0,(true_pert.X.var(axis=0)==0)] += np.amin(true_pert.X[np.nonzero(true_pert.X)])
        
        temp_concat = anndata.concat([control, true_pert], label = 'batch')
        sc.tl.rank_genes_groups(temp_concat, 'batch', method='wilcoxon', groups = ['1'], ref = '0', rankby_abs = True)
        rankings = temp_concat.uns['rank_genes_groups']
        result_df = pd.DataFrame({'pvals_adj': rankings['pvals_adj']['1']}, index = rankings['names']['1'])
        degsGT = result_df[:20].index.astype(np.int32)
        
        temp_concat = anndata.concat([control, pred_pert], label = 'batch')
        sc.tl.rank_genes_groups(temp_concat, 'batch', method='wilcoxon', groups = ['1'], ref = '0', rankby_abs = True)
        rankings = temp_concat.uns['rank_genes_groups']
        result_df = pd.DataFrame({'pvals_adj': rankings['pvals_adj']['1']}, index = rankings['names']['1'])
        degsP = result_df[:20].index.astype(np.int32)
        
        
        
        
        print('myDEGs: '+str(np.intersect1d(degsGT,degsP).shape[0])+' ('+holdoutcell+')')
        print('meandummy: '+str(torch.mean(torch.abs(curdelta-10)).item()))

            
            
            
        
            
        
        
    
    
  

