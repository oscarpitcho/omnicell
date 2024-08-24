import argparse
import scanpy as sc
import yaml
from pathlib import Path
import sys
from scanpy import AnnData as AnnData
from os import listdir
from omnicell.evaluation.utils import get_DEGs, get_eval, get_DEG_Coverage_Recall, get_DEGs_overlaps, c_r_filename, DEGs_overlap_filename, r2_mse_filename
from omnicell.data.utils import prediction_filename

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import yaml
import pickle

random.seed(42)


def generate_evaluation(dir, args):
    
    with open(f'{dir}/config.yaml') as f:
        config = yaml.load(f, yaml.SafeLoader)
    
    #We save the evaluation config in the run directory
    with open(f'{dir}/eval_config.yaml', 'w+') as f:
        yaml.dump(args.__dict__, f, indent=2)

    #TODO: Do we want to restructure the code to make this config handling centralized? --> Yes probably because this will get very brittle very quickly
    #TODO: Core issue --> The code is dependent on the underlying config structure, which is not good.
    eval_targets = config["task_config"]["datasplit"]["evals"]["evaluation_targets"]
    raw_data = sc.read_h5ad(config["task_config"]["data"]["path"])
    
    for (cell, pert) in eval_targets:
        pred_file_fn = prediction_filename(pert, cell)
        r2_and_mse_fn = r2_mse_filename(pert, cell)
        c_r_fn= c_r_filename(pert, cell)
        DEGs_overlap_fn = DEGs_overlap_filename(pert, cell)
        
        results_exist = ((r2_and_mse_fn in listdir(dir)) & (c_r_fn in listdir(dir)) & (DEGs_overlap_fn in listdir(dir)))

        print(results_exist)

        if ((pred_file_fn in listdir(dir)) & ((not results_exist) | args.overwrite)):
        
            model_output = np.load(f'{dir}/{pred_file_fn}', allow_pickle=True)




            pred_pert = sc.AnnData(X=model_output['pred_pert'].clip(min=0))
            if args.round:
                pred_pert.X[pred_pert.X <= 0.5] = 0
            pred_pert.var_names = raw_data.var_names
            
            true_pert = sc.AnnData(X=model_output['true_pert'])
            true_pert.var_names = raw_data.var_names
        
            control = sc.AnnData(X=model_output['control'])
            control.var_names = raw_data.var_names

            true_DEGs_df = get_DEGs(control, true_pert)
            pred_DEGs_df = get_DEGs(control, pred_pert)

    
            r2_and_mse = get_eval(control, true_pert, pred_pert, true_DEGs_df, [100,50,20], args.pval_threshold, args.log_fold_change_threshold)
            c_r_results = {p: get_DEG_Coverage_Recall(true_DEGs_df, pred_DEGs_df, p) for p in [x/args.pval_iters for x in range(1,int(args.pval_iters*args.max_p_val))]}
            DEGs_overlaps = get_DEGs_overlaps(true_DEGs_df, pred_DEGs_df, [100,50,20], args.pval_threshold, args.log_fold_change_threshold)

            try:
                with open(f'{dir}/{r2_mse_filename}', 'wb') as f:
                    pickle.dump(r2_and_mse, f)
    
                with open(f'{dir}/{c_r_filename}', 'wb') as f:
                    pickle.dump(c_r_results, f)
        
                with open(f'{dir}/{DEGs_overlap_filename}', 'wb') as f:
                    pickle.dump(DEGs_overlaps, f)
            except Exception as error:
                print('An error occured:', error)

#Evaluation takes a target model and runs evals on it 


def main(*args):


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

    root_dir = Path(f"./results/plots/{args.model_name}/{args.task_name}").resolve()


    #Get all subdirectories of the model and task, each dir is a run and each run might have several folds

    run_dirs = [x for x in root_dir.iterdir() if x.is_dir()]

    

    for rd in run_dirs:
        folds = [x for x in rd.iterdir() if x.is_dir()]
        for fold in folds:
            generate_evaluation(fold, args)
        
        
    

if __name__ == '__main__':
    main()