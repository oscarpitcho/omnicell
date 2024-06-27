import os
os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/l/ltri2014/jasonleq/'
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import scanpy as sc
import anndata as ad
import scipy
from scipy.sparse import issparse
import anndata
import pandas as pd  
import pickle
from torchdyn.core import NeuralODE


MAX_P_VAL = 0.05
P_VAL_ITERS = 10000
REPLICATES = 10

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
learning_rate = 0.00005
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datasetso = sys.argv[1]
#datasetso = 'kang'

inp = sc.read(f'input/{datasetso}.h5ad')
sc.pp.normalize_total(inp,target_sum=10000)
numvars = len(inp.var_names)

ctkey = 'cell_type'
ptkey = 'condition_key'
ptctrlkey = 'control'

holdoutcell = sys.argv[2]
#holdoutcell = 'NK'

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
batchsize = 16
valbatsize = 64
latent_dim = 2500

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
    
class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.ReLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)
    
class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        return model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
    
class OT_loss(nn.Module):
    _valid = 'emd sinkhorn sinkhorn_knopp_unbalanced'.split()

    def __init__(self, which='emd', use_cuda=True):
        if which not in self._valid:
            raise ValueError(f'{which} not known ({self._valid})')
        elif which == 'emd':
            self.fn = lambda m, n, M: ot.emd(m, n, M)
        elif which == 'sinkhorn':
            self.fn = lambda m, n, M : ot.sinkhorn(m, n, M, 2.0)
        elif which == 'sinkhorn_knopp_unbalanced':
            self.fn = lambda m, n, M : ot.unbalanced.sinkhorn_knopp_unbalanced(m, n, M, 1.0, 1.0)
        else:
            pass
        self.use_cuda=use_cuda

    def __call__(self, source, target, use_cuda=None):
        if use_cuda is None:
            use_cuda = self.use_cuda
        mu = torch.from_numpy(ot.unif(source.size()[0]))
        nu = torch.from_numpy(ot.unif(target.size()[0]))
        M = torch.cdist(source, target)**2
        pi = self.fn(mu, nu, M.detach().cpu())
        if type(pi) is np.ndarray:
            pi = torch.tensor(pi)
        elif type(pi) is torch.Tensor:
            pi = pi.clone().detach()
        pi = pi.cuda() if use_cuda else pi
        M = M.to(pi.device)
        loss = torch.sum(pi * M)
        return loss

    
net = Net()
flow_model = MLP(dim=numvars, out_dim=numvars, time_varying=True, w=256)
flow_model.to(device)
net.to(device)
neteval = Net()
neteval.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(net.parameters()) + list(flow_model.parameters()), lr=learning_rate)
running_loss = 0
running_loss_recon = 0
running_loss_delta = 0
running_loss_delta_counter = 0

recondata = [valdata[0]]
reconcts = len(recondata[0])

for pt2 in noctrlptslst:
    curlst = []
    for ct2 in range(reconcts):
        curlst.append(torch.clone(recondata[0][ct2]))
    recondata.append(curlst)
                
            

criterion=OT_loss()           

    


while goodo:
    counter += 1
    if counter % 2000000 == 0:
        goodo = False
    if counter % 250000 == 0:
        optimizer.zero_grad()
        print(counter)
        print('latent loss:')
        print(running_loss)
        print('recon loss:')
        print(running_loss_recon)
        print('abs diff with delta')
        print(running_loss_delta/running_loss_delta_counter)
        diffdelta_filename = f'diffdelta_autoencoder_{counter}_{holdoutcell}.pkl'
            
        with open(f'{diffdelta_filename}', 'wb') as f:
            pickle.dump(np.array([running_loss_delta/running_loss_delta_counter]), f)
                
        running_loss_delta = 0
        running_loss_delta_counter = 0
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
                    c2p2 = valdata[pt2][ct2]
                    ridx = s
                    c2p1 = c2p1[ridx]
                        
                    c2p1_lat = neteval.encode(c2p1)
                        
                    #use flow matching to match distributions
                        
                    t = torch.rand(batchsize, 1)
                    x0 = c2p1
                    x1 = c2p2
                    
                    a, b = pot.unif(x0.size()[0]), pot.unif(x1.size()[0])
                    M = torch.cdist(x0, x1) ** 2
                    M = M / M.max()
                    pi = pot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(), 0.01, reg_m=10)
                    p = pi.flatten()
                    p = p / p.sum()
                    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batchsize)
                    i, j = np.divmod(choices, pi.shape[1])
    
                    mu_t = x0[i] * (1 - t) + x1[j] * t
                    sigma_t = sigma_min
                    x = mu_t + sigma_t * normal_sample(batchsize, numvars).float()
                    ut = x1[j] - x0[i]
                    vt = model(torch.cat([x, t], dim=-1))
                    loss1 = torch.mean((vt - ut) ** 2)
                    
                    with torch.no_grad():
                        node = NeuralODE(torch_wrapper(flow_model), solver="dopri5", sensitivity="adjoint", atol=1e-1, rtol=1e-1)
                        traj = (node.trajectory(x0,
                                                     t_span=torch.linspace(0, 1, 100),).detach().cpu())
                    c2p2_lat_pred = traj[-1:, :, :]
                        
                    c2p2_pred = neteval.decode(c2p2_lat_pred)
                        
                    recondata[pt2][ct2][s] = torch.clone(c2p2_pred)

                    loss2 = criterion(c2p2, c2p2_pred)
                    loss = loss1 + loss2
                    loss.backward()
                    optimizer.step()
                        
            #print('Val: '+str(loss))

            pert_dan = valdata[pt2][ct2].cpu().detach().numpy()
            cont_dan = recondata[0][0].cpu().detach().numpy()
            pred_dan = recondata[pt2][ct2].cpu().detach().numpy()
            
            pred_pert = anndata.AnnData(X=pred_dan)
            sc.pp.normalize_total(pred_pert,target_sum=10000)
            pred_pert.obs['condition_key'] = 'predicted'
            
            
            
            true_pert = anndata.AnnData(X=pert_dan)
            true_pert.obs['condition_key'] = 'perturbed'
            
            control = anndata.AnnData(X=cont_dan)
            control.obs['condition_key'] = 'control'
            
            control.obs_names = control.obs_names+'-1'
            true_pert.obs_names = true_pert.obs_names+'-2'
            
                        
            allcon = ad.concat([pred_pert,true_pert,control])
            control.X[0,(control.X.var(axis=0)==0)] += np.amin(control.X[np.nonzero(control.X)])
            pred_pert.X[0,(pred_pert.X.var(axis=0)==0)] += np.amin(pred_pert.X[np.nonzero(pred_pert.X)])
            true_pert.X[0,(true_pert.X.var(axis=0)==0)] += np.amin(true_pert.X[np.nonzero(true_pert.X)])
            
            sc.pp.neighbors(allcon, n_neighbors=15)
            sc.tl.umap(allcon, min_dist=0.1, random_state=42)
            
            
            umap_filename = f'UMAP_autoencoder_{counter}_{holdoutcell}.pkl'
            
            with open(f'{umap_filename}', 'wb') as f:
                pickle.dump(allcon.obsm['X_umap'], f)

            

            
            
            

            
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
            
            pred_file_name = f'autoencoder__autoencoder_{counter}_{holdoutcell}.npz' 
            r2_mse_filename = f'r2_and_mse_autoencoder_{counter}_{holdoutcell}.pkl'
            c_r_filename = f'c_r_results_autoencoder_{counter}_{holdoutcell}.pkl'
            DEGs_overlap_filename = f'DEGs_overlaps_autoencoder_{counter}_{holdoutcell}.pkl'
            
            
            
            print(pred_file_name)
            
             
            
            
            #control_subsamples_dict = {} 
            #pred_pert_subsamples_dict = {} 
            #true_pert_subsamples_dict = {}
            #subsample_size = int(len(true_pert.obs.index)*0.95)
            #for i in range(REPLICATES):
            #    control_subsamples_dict[i] = control[np.random.choice(control.obs.index, size=subsample_size, replace=False)]
            #    pred_pert_subsamples_dict[i] = pred_pert[np.random.choice(pred_pert.obs.index, size=subsample_size, replace=False)]
            #    true_pert_subsamples_dict[i] = true_pert[np.random.choice(true_pert.obs.index, size=subsample_size, replace=False)]
        
            true_DEGs_df = get_DEGs(control, true_pert)
            pred_DEGs_df = get_DEGs(control, pred_pert)
        
            r2_and_mse = get_eval(true_pert, pred_pert, true_DEGs_df, [100,50,20], 0.05)
            c_r_results = {p: get_DEG_Coverage_Recall(true_DEGs_df, pred_DEGs_df, p) for p in [x/P_VAL_ITERS for x in range(1,int(P_VAL_ITERS*MAX_P_VAL))]}
            DEGs_overlaps = get_DEGs_overlaps(true_DEGs_df, pred_DEGs_df, [100,50,20], 0.05)
        
            try:
                with open(f'{r2_mse_filename}', 'wb') as f:
                    pickle.dump(r2_and_mse, f)
        
                with open(f'{c_r_filename}', 'wb') as f:
                    pickle.dump(c_r_results, f)
        
                with open(f'{DEGs_overlap_filename}', 'wb') as f:
                    pickle.dump(DEGs_overlaps, f)

            except Exception as error:
                print('An error occured:', error)





            
    
    c1p1_lst = []
    c1p2_lst = []
    for i in range(batchsize):
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
    with torch.no_grad():
        losslatent_delta = torch.mean(torch.abs(c1p2_lat[:,closest_idx:closest_idx+1,:]-c1p1_lat-1))
    
    loss = lossrecon+losslatent

    
    running_loss+=losslatent.item()
    running_loss_recon+=lossrecon.item()
    running_loss_delta+=losslatent_delta.item()
    running_loss_delta_counter+=1
    loss.backward()
    optimizer.step()



    