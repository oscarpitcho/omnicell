from os import listdir
import scipy
from scipy.sparse import issparse
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import json
import pickle 
import argparse
from scipy.stats import pearsonr

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
    sc.tl.rank_genes_groups(
        temp_concat, 'batch', method='wilcoxon', 
        groups = ['1'], ref = '0', rankby_abs = True, tie_correct=True
    )

    rankings = temp_concat.uns['rank_genes_groups']
    result_df = pd.DataFrame({'scores': rankings['scores']['1'],
                     'pvals_adj': rankings['pvals_adj']['1'],
                     'lfc': rankings['logfoldchanges']['1']},
                    index = rankings['names']['1'])
    return result_df

def get_eval(ctrl_adata, true_adata, pred_adata, DEGs, DEG_vals, pval_threshold, lfc_threshold):
        
    results_dict =  {}
    
    ctrl_mean = to_dense(ctrl_adata.X).mean(axis = 0)

    true_mean = to_dense(true_adata.X).mean(axis = 0)
    true_var = to_dense(true_adata.X).var(axis = 0)
    
    pred_mean = to_dense(pred_adata.X).mean(axis = 0)
    pred_var = to_dense(pred_adata.X).var(axis = 0)
    
    true_corr_mtx = np.corrcoef(to_dense(true_adata.X), rowvar=False).flatten()
    true_cov_mtx = np.cov(to_dense(true_adata.X), rowvar=False).flatten()
        
    pred_corr_mtx = np.corrcoef(to_dense(pred_adata.X), rowvar=False).flatten()
    pred_cov_mtx = np.cov(to_dense(pred_adata.X), rowvar=False).flatten()

    true_sub_diff = true_mean - ctrl_mean
    pred_sub_diff = pred_mean - ctrl_mean

    true_diff = true_mean/ctrl_mean
    pred_diff = pred_mean/ctrl_mean

    true_diff_mask = (np.isnan(true_diff) | np.isinf(true_diff))
    pred_diff_mask = (np.isnan(pred_diff) | np.isinf(pred_diff))
    
    common_mask = true_diff_mask | pred_diff_mask
    true_fold_diff = np.ma.array(true_diff, mask=common_mask).compressed()
    pred_fold_diff = np.ma.array(pred_diff, mask=common_mask).compressed()

    results_dict['all_genes_mean_sub_diff_R'] = pearsonr(true_sub_diff, pred_sub_diff)[0]
    results_dict['all_genes_mean_sub_diff_R2'] = pearsonr(true_sub_diff, pred_sub_diff)[0]**2
    results_dict['all_genes_mean_sub_diff_MSE'] = (np.square(true_sub_diff - pred_sub_diff)).mean(axis=0)

    results_dict['all_genes_mean_fold_diff_R'] = pearsonr(true_fold_diff, pred_fold_diff)[0]
    results_dict['all_genes_mean_fold_diff_R2'] = pearsonr(true_fold_diff, pred_fold_diff)[0]**2
    results_dict['all_genes_mean_fold_diff_MSE'] = (np.square(true_fold_diff - pred_fold_diff)).mean(axis=0)
    
    results_dict['all_genes_mean_R'] = pearsonr(true_mean, pred_mean)[0]
    results_dict['all_genes_mean_R2'] = pearsonr(true_mean, pred_mean)[0]**2
    results_dict['all_genes_mean_MSE'] = (np.square(true_mean - pred_mean)).mean(axis=0)

    results_dict['all_genes_var_R'] = pearsonr(true_var, pred_var)[0]
    results_dict['all_genes_var_R2'] = pearsonr(true_var, pred_var)[0]**2
    results_dict['all_genes_var_MSE'] = (np.square(true_var - pred_var)).mean(axis=0)
   
    corr_nas = np.logical_or(np.isnan(true_corr_mtx), np.isnan(pred_corr_mtx))
    cov_nas = np.logical_or(np.isnan(true_cov_mtx), np.isnan(pred_cov_mtx))

    results_dict['all_genes_corr_mtx_R'] = pearsonr(true_corr_mtx[~corr_nas], pred_corr_mtx[~corr_nas])[0]
    results_dict['all_genes_corr_mtx_R2'] = pearsonr(true_corr_mtx[~corr_nas], pred_corr_mtx[~corr_nas])[0]**2
    results_dict['all_genes_corr_mtx_MSE'] = (np.square(true_corr_mtx[~corr_nas] - pred_corr_mtx[~corr_nas])).mean(axis=0)

    results_dict['all_genes_cov_mtx_R'] = pearsonr(true_cov_mtx[~cov_nas], pred_cov_mtx[~cov_nas])[0]
    results_dict['all_genes_cov_mtx_R2'] = pearsonr(true_cov_mtx[~cov_nas], pred_cov_mtx[~cov_nas])[0]**2
    results_dict['all_genes_cov_mtx_MSE'] = (np.square(true_cov_mtx[~cov_nas] - pred_cov_mtx[~cov_nas])).mean(axis=0)

    if lfc_threshold:   
        significant_DEGs = DEGs[(DEGs['pvals_adj'] < pval_threshold) & (abs(DEGs) > lfc_threshold)]
    else:
        significant_DEGs = DEGs[DEGs['pvals_adj'] < pval_threshold]
    num_DEGs = len(significant_DEGs)
    DEG_vals.insert(0, num_DEGs)
    
    for val in DEG_vals:
        if ((val > num_DEGs) or (val == 0)):
            results_dict[f'Top_{val}_DEGs_sub_diff_mean_R'] = None
            results_dict[f'Top_{val}_DEGs_sub_diff_mean_R2'] = None
            results_dict[f'Top_{val}_DEGs_sub_diff_mean_MSE'] = None
            
            results_dict[f'Top_{val}_DEGs_fold_diff_mean_R'] = None
            results_dict[f'Top_{val}_DEGs_fold_diff_mean_R2'] = None
            results_dict[f'Top_{val}_DEGs_fold_diff_mean_MSE'] = None
            
            results_dict[f'Top_{val}_DEGs_mean_R'] = None
            results_dict[f'Top_{val}_DEGs_mean_R2'] = None
            results_dict[f'Top_{val}_DEGs_mean_MSE'] = None

            results_dict[f'Top_{val}_DEGs_var_R'] = None
            results_dict[f'Top_{val}_DEGs_var_R2'] = None
            results_dict[f'Top_{val}_DEGs_var_MSE'] = None
            
            results_dict[f'Top_{val}_DEGs_corr_mtx_R'] = None
            results_dict[f'Top_{val}_DEGs_corr_mtx_R2'] = None
            results_dict[f'Top_{val}_DEGs_corr_mtx_MSE'] = None
            
            results_dict[f'Top_{val}_DEGs_cov_mtx_R'] = None
            results_dict[f'Top_{val}_DEGs_cov_mtx_R2'] = None
            results_dict[f'Top_{val}_DEGs_cov_mtx_MSE'] = None
        
        else:
            top_DEGs = significant_DEGs[0:val].index

            ctrl_mean = to_dense(ctrl_adata[:,top_DEGs].X).mean(axis = 0)
            
            true_mean = to_dense(true_adata[:,top_DEGs].X).mean(axis = 0)
            true_var = to_dense(true_adata[:,top_DEGs].X).var(axis = 0)
            true_corr_mtx = np.corrcoef(to_dense(true_adata[:,top_DEGs].X), rowvar=False).flatten()
            true_cov_mtx = np.cov(to_dense(true_adata[:,top_DEGs].X), rowvar=False).flatten()

            pred_mean = to_dense(pred_adata[:,top_DEGs].X).mean(axis = 0)
            pred_var = to_dense(pred_adata[:,top_DEGs].X).var(axis = 0)
            pred_corr_mtx = np.corrcoef(to_dense(pred_adata[:,top_DEGs].X), rowvar=False).flatten()
            pred_cov_mtx = np.cov(to_dense(pred_adata[:,top_DEGs].X), rowvar=False).flatten()

            true_sub_diff = true_mean - ctrl_mean
            pred_sub_diff = pred_mean - ctrl_mean
        
            true_diff = true_mean/ctrl_mean
            pred_diff = pred_mean/ctrl_mean
        
            true_diff_mask = (np.isnan(true_diff) | np.isinf(true_diff))
            pred_diff_mask = (np.isnan(pred_diff) | np.isinf(pred_diff))
            
            common_mask = true_diff_mask | pred_diff_mask
            true_fold_diff = np.ma.array(true_diff, mask=common_mask).compressed()
            pred_fold_diff = np.ma.array(pred_diff, mask=common_mask).compressed()

            results_dict[f'Top_{val}_DEGs_sub_diff_R'] = pearsonr(true_sub_diff, pred_sub_diff)[0]
            results_dict[f'Top_{val}_DEGs_sub_diff_R2'] = pearsonr(true_sub_diff, pred_sub_diff)[0]**2
            results_dict[f'Top_{val}_DEGs_sub_diff_MSE'] = (np.square(true_sub_diff - pred_sub_diff)).mean(axis=0)
        
            results_dict[f'Top_{val}_DEGs_fold_diff_R'] = pearsonr(true_fold_diff, pred_fold_diff)[0]
            results_dict[f'Top_{val}_DEGs_fold_diff_R2'] = pearsonr(true_fold_diff, pred_fold_diff)[0]**2
            results_dict[f'Top_{val}_DEGs_fold_diff_MSE'] = (np.square(true_fold_diff - pred_fold_diff)).mean(axis=0)
    
            results_dict[f'Top_{val}_DEGs_mean_R'] = pearsonr(true_mean, pred_mean)[0]
            results_dict[f'Top_{val}_DEGs_mean_R2'] = pearsonr(true_mean, pred_mean)[0]**2
            results_dict[f'Top_{val}_DEGs_mean_MSE'] = (np.square(true_mean - pred_mean)).mean(axis=0)

            results_dict[f'Top_{val}_DEGs_var_R'] = pearsonr(true_var, pred_var)[0]
            results_dict[f'Top_{val}_DEGs_var_R2'] = pearsonr(true_var, pred_var)[0]**2
            results_dict[f'Top_{val}_DEGs_var_MSE'] = (np.square(true_var - pred_var)).mean(axis=0)
            
            corr_nas = np.logical_or(np.isnan(true_corr_mtx), np.isnan(pred_corr_mtx))
            cov_nas = np.logical_or(np.isnan(true_cov_mtx), np.isnan(pred_cov_mtx))

            results_dict[f'Top_{val}_DEGs_corr_mtx_R'] = pearsonr(true_corr_mtx[~corr_nas], pred_corr_mtx[~corr_nas])[0]
            results_dict[f'Top_{val}_DEGs_corr_mtx_R2'] = pearsonr(true_corr_mtx[~corr_nas], pred_corr_mtx[~corr_nas])[0]**2
            results_dict[f'Top_{val}_DEGs_corr_mtx_MSE'] = (np.square(true_corr_mtx[~corr_nas] - pred_corr_mtx[~corr_nas])).mean(axis=0)

            results_dict[f'Top_{val}_DEGs_cov_mtx_R'] = pearsonr(true_cov_mtx[~cov_nas], pred_cov_mtx[~cov_nas])[0]
            results_dict[f'Top_{val}_DEGs_cov_mtx_R2'] = pearsonr(true_cov_mtx[~cov_nas], pred_cov_mtx[~cov_nas])[0]**2
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

def get_DEGs_overlaps(true_DEGs, pred_DEGs, DEG_vals, pval_threshold, lfc_threshold = None):
    if lfc_threshold:
        significant_true_DEGs = true_DEGs[(true_DEGs['pvals_adj'] < pval_threshold) & (abs(true_DEGs['lfc']) > lfc_threshold)]
        significant_pred_DEGs = pred_DEGs[(pred_DEGs['pvals_adj'] < pval_threshold) & (abs(pred_DEGs['lfc']) > lfc_threshold)]
    else:
        significant_true_DEGs = true_DEGs[true_DEGs['pvals_adj'] < pval_threshold]
        significant_pred_DEGs = pred_DEGs[pred_DEGs['pvals_adj'] < pval_threshold]

    true_DEGs_for_comparison = [get_DEG_with_direction(gene,score) for gene, score in zip(significant_true_DEGs.index, significant_true_DEGs['scores'])]   
    pred_DEGs_for_comparison = [get_DEG_with_direction(gene,score) for gene, score in zip(significant_pred_DEGs.index, significant_pred_DEGs['scores'])]
    
    num_DEGs = len(significant_true_DEGs)
    DEG_vals.insert(0, num_DEGs)
    
    results = {}
    for val in DEG_vals:
        if val > num_DEGs:
            results[f'Overlap_in_top_{val}_DEGs'] = None
        else:
            results[f'Overlap_in_top_{val}_DEGs'] = len(set(true_DEGs_for_comparison[0:val]).intersection(set(pred_DEGs_for_comparison[0:val])))

    intersection = len(set(true_DEGs_for_comparison).intersection(set(pred_DEGs_for_comparison)))
    union = len(set(true_DEGs_for_comparison).union(set(pred_DEGs_for_comparison)))
    if union > 0:
        results['Jaccard'] = intersection/union
    else:
        results['Jaccard'] = None
    
    return results

parser = argparse.ArgumentParser(description='Script for analyzing results')

parser.add_argument('-d', '--dataset', type=str, help='name of dataset being analyzed', required=True)
parser.add_argument('-l', '--location_file', type=str, help='path to json file containing file locations', required=True)
parser.add_argument('-r', '--round', action='store_true', help='Rounds values <=0.5 to 0 in addition to the clip')
parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite pre-existing result files')

args = parser.parse_args()

with open(args.location_file) as json_file:
    eval_locs = json.load(json_file)

if args.dataset in eval_locs:
    pass
else:
    print(f'Dataset passed "{args.dataset}" is not valid, please choose one of the following: {eval_locs.keys()}')

path_to_results = eval_locs[args.dataset]['results_location']
#result_dirs = [f'{path_to_results}/{x}' for x in listdir(path_to_results)]
result_dirs = [f'{path_to_results[x]}/{y}' for x in path_to_results for y in listdir(path_to_results[x])]
raw_data = anndata.read_h5ad(eval_locs[args.dataset]['data_location'])

MAX_P_VAL = 0.05
P_VAL_ITERS = 10000
REPLICATES = 10

for directory in result_dirs:
    #Load in config file
    with open(f'{directory}/config.json') as json_file:
            config = json.load(json_file)

    #Iterate through all perturbation combinations
    for holdout_pert, holdout_cell in zip(config["holdout_perts"], config["holdout_cells"]):
        
        pred_file_name = f'pred_{holdout_pert}_{holdout_cell}.npz' 
        r2_mse_filename = f'r2_and_mse_{holdout_pert}_{holdout_cell}.pkl' if not args.round else f'r2_and_mse_rounded_{holdout_pert}_{holdout_cell}.pkl'
        c_r_filename = f'c_r_results_{holdout_pert}_{holdout_cell}.pkl' if not args.round else f'c_r_results_rounded_{holdout_pert}_{holdout_cell}.pkl'
        DEGs_overlap_filename = f'DEGs_overlaps_{holdout_pert}_{holdout_cell}.pkl' if not args.round else f'DEGs_overlaps_rounded_{holdout_pert}_{holdout_cell}.pkl'
        
        results_exist = ((r2_mse_filename in listdir(directory)) & (c_r_filename in listdir(directory)) & (DEGs_overlap_filename in listdir(directory)))

        print(pred_file_name)
        print(results_exist)
        
        if ((pred_file_name in listdir(directory)) & ((not results_exist) | args.overwrite)):
         
            model_output = np.load(f'{directory}/{pred_file_name}')
        
            # Create anndata objects from all the predictions
            pred_pert = anndata.AnnData(X=model_output['pred_pert'].clip(min=0))
            if args.round:
                pred_pert.X[pred_pert.X <= 0.5] = 0
            pred_pert.var_names = raw_data.var_names
            
            true_pert = anndata.AnnData(X=model_output['true_pert'])
            true_pert.var_names = raw_data.var_names
        
            control = anndata.AnnData(X=model_output['control'])
            control.var_names = raw_data.var_names

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
            # print(pred_pert)
            # print(pred_DEGs_df)
    
            r2_and_mse = get_eval(true_pert, pred_pert, true_DEGs_df, [100,50,20], 0.05)
            # print(r2_and_mse)
            c_r_results = {p: get_DEG_Coverage_Recall(true_DEGs_df, pred_DEGs_df, p) for p in [x/P_VAL_ITERS for x in range(1,int(P_VAL_ITERS*MAX_P_VAL))]}
            # print(c_r_results)
            DEGs_overlaps = get_DEGs_overlaps(true_DEGs_df, pred_DEGs_df, [100,50,20], 0.05)
            # print(DEGs_overlaps)

            try:
                with open(f'{directory}/{r2_mse_filename}', 'wb') as f:
                    pickle.dump(r2_and_mse, f)
    
                with open(f'{directory}/{c_r_filename}', 'wb') as f:
                    pickle.dump(c_r_results, f)
        
                with open(f'{directory}/{DEGs_overlap_filename}', 'wb') as f:
                    pickle.dump(DEGs_overlaps, f)
            except Exception as error:
                print('An error occured:', error)