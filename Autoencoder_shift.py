
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
from matplotlib import pyplot as plt




#Assumption: at least 3 cell types
learning_rate = 0.00005
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inp = sc.read("input/nault.h5ad")
numvars = len(inp.var_names)

ctkey = 'cell_type'
ptkey = 'condition_key'
ptctrlkey = 'control'

holdoutcell = 'Macrophage'

holdoutidx = (inp.obs[ctkey]==holdoutcell)
train = inp[~holdoutidx].copy()
valid = inp[holdoutidx].copy()

celltypes= np.unique(train.obs[ctkey])
pttypes = np.unique(train.obs[ptkey])
ptctrlidx = [i for i in range(len(pttypes)) if pttypes[i] == ptctrlkey][0]


nct = len(celltypes)
npt = len(pttypes)
noctrlptslst = list(range(npt))
noctrlptslst.pop(ptctrlidx)

traindata = []

for ptt in pttypes:
    curlst = []
    for ctt in celltypes:
        adat = inp[(inp.obs[ptkey]==ptt) & (inp.obs[ctkey]==ctt)].X.copy()
        try:
            adat = adat.todense()
        except:
            pass
        adat = torch.from_numpy(adat.astype(np.float32)).to(device)
        
        curlst.append(adat)
    traindata.append(curlst)
    
celltypesval = np.unique(valid.obs[ctkey])
pttypesval = np.unique(valid.obs[ptkey])

nctval = len(celltypesval)
nptval = len(pttypesval)

valdata = []

for ptt in pttypesval:
    curlst = []
    for ctt in celltypesval:
        adat = inp[(inp.obs[ptkey]==ptt) & (inp.obs[ctkey]==ctt)].X.copy()
        try:
            adat = adat.todense()
        except:
            pass
        adat = torch.from_numpy(adat.astype(np.float32)).to(device)
        
        curlst.append(adat)
    valdata.append(curlst)


goodo=True
counter=0
batsize = 16
valbatsize = 64
latent_dim = inp.shape[1]//2

class Lin(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.nonlinear = nn.LeakyReLU(0.1)
        

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x
    
class LinOutput(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.nonlinear = nn.ReLU()
        

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x
    
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = Lin(numvars, latent_dim)
        self.l2 = Lin(latent_dim, latent_dim)
        self.l3 = Lin(latent_dim, latent_dim)
        self.l4 = LinOutput(latent_dim, numvars)

        

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x
    
    def encode(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x
    
    def decode(self, x):
        x = self.l3(x)
        x = self.l4(x)

        return x
    
net = Net()
net.to(device)
neteval = Net()
neteval.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
running_loss = 0
running_loss_recon = 0

recondata = [valdata[0]]
reconcts = len(recondata[0])

for pt2 in noctrlptslst:
    curlst = []
    for ct2 in range(reconcts):
        curlst.append(torch.clone(recondata[0][ct2]))
    recondata.append(curlst)
                
            

            

    


while goodo:
    counter += 1
    if counter % 10000 == 0:
        with torch.no_grad():
            print(counter)
            print('latent loss:')
            print(running_loss)
            print('recon loss:')
            print(running_loss_recon)
            running_loss = 0
            running_loss_recon = 0
            pt1 = ptctrlidx
            lossall = 0
            preddegs = []
            realdegs = []
            labslst = []
            state_dict = net.state_dict()
            neteval.load_state_dict(state_dict)
            neteval.eval()
            for ct2 in range(reconcts):
                for pt2 in noctrlptslst:
                    for s in range(recondata[pt2][ct2].shape[0]):
                        ct1 = torch.randperm(nct)[0]
                        

                        c2p1 = recondata[pt1][ct2]
                        ridx = s
                        c2p1 = c2p1[ridx]
                        
                        c2p1_lat = neteval.encode(c2p1)
                        
                        dummy = torch.ones_like(c2p1_lat)
                        dummy.to(device)
                        
                        c2p2_lat_pred = c2p1_lat+dummy
                        c2p2_pred = neteval.decode(c2p2_lat_pred)
                        
                        c2p2 = valdata[pt2][ct2]
                        ridx = np.random.randint(c2p2.shape[0])
                        c2p2 = c2p2[ridx]
                        
                        recondata[pt2][ct2][s] = torch.clone(c2p2_pred)

                        loss = torch.mean((c2p2 - c2p2_pred)**2).item()
                        lossall+=loss
                        
            print('Val: '+str(lossall))

            pert_dan = valdata[pt2][ct2].cpu().detach().numpy()
            cont_dan = recondata[0][0].cpu().detach().numpy()
            pred_dan = recondata[pt2][ct2].cpu().detach().numpy()
            
            pred_pert = anndata.AnnData(X=pred_dan)
            sc.pp.normalize_total(pred_pert)
            pred_pert.obs['condition_key'] = 'predicted'
            
            
            
            true_pert = anndata.AnnData(X=pert_dan)
            sc.pp.normalize_total(true_pert)
            true_pert.obs['condition_key'] = 'perturbed'
            
            control = anndata.AnnData(X=cont_dan)
            sc.pp.normalize_total(control)
            control.obs['condition_key'] = 'control'
            
            control.obs_names = control.obs_names+'-1'
            true_pert.obs_names = true_pert.obs_names+'-2'
            
                        
            allcon = ad.concat([pred_pert,true_pert,control])
            control.X[0,(control.X.var(axis=0)==0)] += np.amin(control.X[np.nonzero(control.X)])
            pred_pert.X[0,(pred_pert.X.var(axis=0)==0)] += np.amin(pred_pert.X[np.nonzero(pred_pert.X)])
            true_pert.X[0,(true_pert.X.var(axis=0)==0)] += np.amin(true_pert.X[np.nonzero(true_pert.X)])
            
            sc.pp.neighbors(allcon, n_neighbors=15)
            sc.tl.umap(allcon, min_dist=0.1, random_state=42)
            with plt.rc_context({'axes.spines.right': False, 'axes.spines.top': False}):
                # Generate the PCA plot
                ax = sc.pl.umap(allcon, color=["condition_key"], groups=["control", "perturbed","predicted"], show=False)
                #ax = sc.pl.umap(curdata, color=["condition_key"], groups=["perturbed_decoded","perturbed"], show=False)
                
                
        
    
                
                # Show the plot
                plt.show()
            
            
            

            
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
            

            print('myDEGs: '+str(np.intersect1d(degsGT,degsP).shape[0]))





            
    
    c1p1_lst = []
    c1p2_lst = []
    for i in range(batsize):
        cts = torch.randperm(nct)[:2]
        pt1 = 0
        pt2 = 1
        ct1 = cts[0]
        
        c1p1 = traindata[pt1][ct1]
        ridx = np.random.randint(c1p1.shape[0])
        c1p1 = c1p1[ridx]
        
        c1p2_sample_lst = []
        for i in range(10):
            c1p2 = traindata[pt2][ct1]
            ridx = np.random.randint(c1p2.shape[0])
            c1p2 = c1p2[ridx]
            c1p2_sample_lst.append(c1p2)
        c1p2 = torch.stack(c1p2_sample_lst)
        c1p1_lst.append(c1p1.unsqueeze(0))
        c1p2_lst.append(c1p2)
        
        
    c1p1_lst = torch.stack(c1p1_lst)
    c1p2_lst = torch.stack(c1p2_lst)
    optimizer.zero_grad()
    c1p1_lat = net.encode(c1p1_lst)
    c1p2_lat = net.encode(c1p2_lst)
    
    dummy = torch.ones_like(c1p1_lat)
    dummy.to(device)
    
    
    with torch.no_grad():
        diff = (((c1p2_lat - c1p1_lat)-dummy)**2).mean(axis=0).mean(axis=-1)
        closest_idx = torch.argmin(diff)
        rand_idx = np.random.randint(diff.shape[0])
    
    c1p1_decod = net.decode(c1p1_lat)
    c1p2_decod = net.decode(c1p2_lat[:,rand_idx,:])


    
    

    lossrecon = criterion(c1p1_decod,c1p1_lst) + criterion(c1p2_decod,c1p2_lst[:,rand_idx,:])
    losslatent = criterion((c1p2_lat[:,closest_idx:closest_idx+1,:]-c1p1_lat),dummy) 
    
    loss = lossrecon+losslatent

    
    running_loss+=losslatent.item()
    running_loss_recon+=lossrecon.item()
    loss.backward()
    optimizer.step()



    

