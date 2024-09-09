import argparse
import scanpy as sc
import yaml
import json
from pathlib import Path
import sys
from scanpy import AnnData as AnnData
from os import listdir
from omnicell.evaluation.utils import get_DEGs, get_eval, get_DEG_Coverage_Recall, get_DEGs_overlaps, c_r_filename, DEGs_overlap_filename, r2_mse_filename
from omnicell.data.utils import prediction_filename
from omnicell.config.config import Config
from omnicell.processing.utils import to_dense
from statistics import mean
from scipy.sparse import issparse   
import scipy.sparse as sparse
from utils.encoder import NumpyTypeEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import yaml
import logging

logger = logging.getLogger(__name__)

random.seed(42)


def generate_evaluation(dir, args):
    
    with open(f'{dir}/config.yaml') as f:
        config = yaml.load(f, yaml.SafeLoader)
    

    #We save the evaluation config in the run directory
    with open(f'{dir}/eval_config.yaml', 'w+') as f:
        yaml.dump(args.__dict__, f, indent=2)

    config = Config(config)


    config = config.add_eval_config(args.__dict__)

    #TODO: Do we want to restructure the code to make this config handling centralized? --> Yes probably because this will get very brittle very quickly
    #TODO: Core issue --> The code is dependent on the underlying config structure, which is not good.
    eval_targets = config.get_eval_targets()
    raw_data = sc.read_h5ad(config.get_data_path())
    
    for (cell, pert) in eval_targets:
        pred_file_fn = prediction_filename(pert, cell)

        pred_fn = f"{prediction_filename(pert, cell)}-preds.npz"
        true_fn = f"{prediction_filename(pert, cell)}-ground_truth.npz"
        control_fn = f"{prediction_filename(pert, cell)}-control.npz"

        r2_and_mse_fn = r2_mse_filename(pert, cell)
        c_r_fn= c_r_filename(pert, cell)
        DEGs_overlap_fn = DEGs_overlap_filename(pert, cell)
        
        results_exist = ((r2_and_mse_fn in listdir(dir)) & (c_r_fn in listdir(dir)) & (DEGs_overlap_fn in listdir(dir)))

        preds_exist = pred_fn in listdir(dir) and true_fn in listdir(dir) and control_fn in listdir(dir)

        if not preds_exist:
            logger.warning(f"Predictions for {pert} and {cell} do not exist in {dir}")

            raise FileNotFoundError(f"Predictions for {pert} and {cell} do not exist in {dir}")

        if (not results_exist | args.overwrite):

            logger.info(f"Generating evaluations for {pert} and {cell}")

            pred_pert = sparse.load_npz(f'{dir}/{pred_fn}')
            true_pert = sparse.load_npz(f'{dir}/{true_fn}')
            control = sparse.load_npz(f'{dir}/{control_fn}')
            
        


            print(f"Pred pert is sparce matrix: {issparse(pred_pert)}")
            print(f"Pred pert matrix type: {type(pred_pert)}")
            
            #We need to convert the sparse matrices to dense matrices
            pred_pert = to_dense(pred_pert)
            true_pert = to_dense(true_pert)
            control = to_dense(control)

            print(f"Type of pred_pert: {type(pred_pert)}")


            pred_pert = sc.AnnData(X=pred_pert.clip(min=0))

            if args.round:
                pred_pert.X[pred_pert.X <= 0.5] = 0
            pred_pert.var_names = raw_data.var_names
            
            true_pert = sc.AnnData(X=true_pert)
            true_pert.var_names = raw_data.var_names
        
            control = sc.AnnData(X=control)
            control.var_names = raw_data.var_names

            true_DEGs_df = get_DEGs(control, true_pert)
            pred_DEGs_df = get_DEGs(control, pred_pert)

    
            r2_and_mse = get_eval(control, true_pert, pred_pert, true_DEGs_df, [100,50,20], args.pval_threshold, args.log_fold_change_threshold)
            c_r_results = {p: get_DEG_Coverage_Recall(true_DEGs_df, pred_DEGs_df, p) for p in [x/args.pval_iters for x in range(1,int(args.pval_iters*args.max_p_val))]}
            DEGs_overlaps = get_DEGs_overlaps(true_DEGs_df, pred_DEGs_df, [100,50,20], args.pval_threshold, args.log_fold_change_threshold)

            with open(f'{dir}/{r2_and_mse_fn}', 'w+') as f:
                json.dump(r2_and_mse, f, indent=2, cls=NumpyTypeEncoder)

            with open(f'{dir}/{c_r_fn}', 'w+') as f:
                json.dump(c_r_results, f, indent=2, cls=NumpyTypeEncoder)
            
            with open(f'{dir}/{DEGs_overlap_fn}', 'w+') as f:
                json.dump(DEGs_overlaps, f, indent=2, cls=NumpyTypeEncoder)



#Evaluation takes a target model and runs evals on it 
def average_shared_keys(dict_list):
    if not dict_list:
        return {}
    

    # Find the intersection of all keys
    common_keys = set.intersection(*map(set, dict_list))

    #Remove keys which are None or empty
    valid_keys = []
    for key in common_keys:
        values = [d[key] for d in dict_list]

        #keeping only the keys which are shared and never None
        if all([v is not None for v in values]):
            valid_keys.append(key)
    
    # Calculate averages for common keys
    result = {}
    for key in valid_keys:
        values = [d[key] for d in dict_list]  # Removed the unnecessary check
        result[key] = sum(values) / len(values)
    
    return result


def average_fold(fold_dir):
    config = Config(yaml.load(open(f'{fold_dir}/config.yaml'), Loader=yaml.SafeLoader))

    eval_targets = config.get_eval_targets()

    degs_dicts  = []
    r2_mse_dicts = []
    c_r_dicts = []

    for (cell, pert) in eval_targets:
        with open(f'{fold_dir}/{DEGs_overlap_filename(pert, cell)}', 'rb') as f:
            DEGs_overlaps = json.load(f)
            degs_dicts.append(DEGs_overlaps)
        
        with open(f'{fold_dir}/{r2_mse_filename(pert, cell)}', 'rb') as f:
            r2_mse = json.load(f)
            r2_mse_dicts.append(r2_mse)
        
        """        with open(f'{fold_dir}/{c_r_filename(pert, cell)}', 'rb') as f:
            c_r = pickle.load(f)"""



    avg_DEGs_overlaps = average_shared_keys(degs_dicts)
    avg_r2_mse = average_shared_keys(r2_mse_dicts)
    #avg_c_r = average_shared_keys(c_r_dicts)

    with open(f'{fold_dir}/avg_DEGs_overlaps.json', 'w+') as f:
        json.dump(avg_DEGs_overlaps, f, indent=2, cls=NumpyTypeEncoder)
    
    with open(f'{fold_dir}/avg_r2_mse.json', 'w+') as f:
        json.dump(avg_r2_mse, f, indent=2, cls=NumpyTypeEncoder)
    
    """with open(f'{fold_dir}/avg_c_r.pkl', 'wb') as f:
        pickle.dump(avg_c_r, f)"""



def average_run(run_dir):
    """Assumes we have already run average for each fold in the run directory and aggregates the results"""

    folds = [x for x in run_dir.iterdir() if x.is_dir()]

    degs_dicts  = []
    r2_mse_dicts = []
    for fold in folds:
        with open(f'{fold}/avg_DEGs_overlaps.json', 'rb') as f:
            DEGs_overlaps = json.load(f)
            degs_dicts.append(DEGs_overlaps)
        
        with open(f'{fold}/avg_r2_mse.json', 'rb') as f:
            r2_mse = json.load(f)
            r2_mse_dicts.append(r2_mse)

    avg_DEGs_overlaps = average_shared_keys(degs_dicts)
    avg_r2_mse = average_shared_keys(r2_mse_dicts)

    with open(f'{run_dir}/avg_DEGs_overlaps.json', 'w+') as f:
        json.dump(avg_DEGs_overlaps, f, indent=2, cls=NumpyTypeEncoder)
    
    with open(f'{run_dir}/avg_r2_mse.json', 'w+') as f:
        json.dump(avg_r2_mse, f, indent=2, cls=NumpyTypeEncoder) 

    #Do they have a nested structure? Nope

    #We want to average the results of the folds


def main(*args):

    logger.info("Starting evaluation script")


    parser = argparse.ArgumentParser(description='Analysis settings.')

    parser.add_argument('--model_name', type=str, default='', help='Path to yaml config file of the model.')
    parser.add_argument('--task_name', type=str, default='', help='Path to yaml config file of the task.')
    parser.add_argument('-r', '--round', action='store_true', help='Rounds values <=0.5 to 0 in addition to the clip')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite pre-existing result files')
    parser.add_argument('-pval', '--pval_threshold', type=float, default=0.05, help='Sets maximum adjusted p value for a gene to be called as a DEG')
    parser.add_argument('-lfc', '--log_fold_change_threshold', type=float, default=None, help='Sets minimum absolute log fold change for a gene to be called as a DEG')
    parser.add_argument('--replicates', type=int, default=10, help='Number of replicates to use for p value calculation')
    parser.add_argument('--pval_iters', type=int, default=10000, help='Number of iterations to use for p value calculation')
    parser.add_argument('--max_p_val', type=float, default=0.05, help='Maximum p value to use for p value calculation')


    args = parser.parse_args()

    root_dir = Path(f"./results/{args.model_name}/{args.task_name}").resolve()

    logger.info(f"Generating evaluations for model {args.model_name} and task {args.task_name}")


    #Get all subdirectories of the model and task, each dir is a run and each run might have several folds

    run_dirs = [x for x in root_dir.iterdir() if x.is_dir()]


    for rd in run_dirs:
        folds = [x for x in rd.iterdir() if x.is_dir()]
        for fold in folds:
            generate_evaluation(fold, args)
            average_fold(fold)

        average_run(rd)
        



            #We want to generate average stats per fold and per run
        
        
    

if __name__ == '__main__':
    main()