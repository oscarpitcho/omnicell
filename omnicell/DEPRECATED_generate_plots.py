import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import yaml
import pickle
import pandas as pd
from omnicell.evaluation.utils import r2_mse_filename, c_r_filename, DEGs_overlap_filename
from omnicell.data.utils import prediction_filename
import hashlib
import yaml
import json
from pathlib import Path
import argparse

def rearrange_results_df(df, metric = 'all'):
    # Separate lists for each category
    mean_cols_R2 = [name for name in df.index if name.endswith('mean_R2')]
    mean_cols_MSE = [name for name in df.index if name.endswith('mean_MSE')]
    
    var_cols_R2 = [name for name in df.index if name.endswith('var_R2')]
    var_cols_MSE = [name for name in df.index if name.endswith('var_MSE')]
    
    corr_cols_R2 = [name for name in df.index if name.endswith('corr_mtx_R2')]
    corr_cols_MSE = [name for name in df.index if name.endswith('corr_mtx_MSE')]
    
    cov_cols_R2 = [name for name in df.index if name.endswith('cov_mtx_R2')]
    cov_cols_MSE = [name for name in df.index if name.endswith('cov_mtx_MSE')]

    # Combine sorted lists with desired order
    if metric == 'all':
        new_order = mean_cols_R2 + mean_cols_MSE + var_cols_R2 + var_cols_MSE + corr_cols_R2 + corr_cols_MSE + cov_cols_R2 + cov_cols_MSE
    elif metric == 'MSE':
        new_order = mean_cols_MSE + var_cols_MSE + corr_cols_MSE + cov_cols_MSE
    elif metric == 'R2':
        new_order = mean_cols_R2 + var_cols_R2 + corr_cols_R2 + cov_cols_R2
    else:
        print(f'METRIC: {metric} IS NOT AN OPTION')
    # Reorder the dataframe based on the new order
    df = df.loc[new_order]

    return(df)

def are_identical(vals):
    return all(i == vals[0] for i in vals)

def df_to_heatmap(df, title, dims, path):
    fig, ax = plt.subplots(figsize=dims)    
    df.columns = [x.replace(' ', '\n') for x in df.columns]
    sns.heatmap(df, annot = df, linewidths=.5, ax=ax, cmap='Purples', fmt='0.4f', annot_kws={"size": 6}, cbar=False,
               xticklabels=True, yticklabels=True)
    
    ax.set_title(title, size=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=6)
    ax.xaxis.set_ticks_position("top")

    fig.savefig(f'{path}/{title}.png', dpi=300)

def plot_coverage_and_recall(coverage_results, recall_results, dims):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=dims)

    # Plotting Coverage on the first subplot
    ax1.plot(coverage_results, label=coverage_results.columns)
    # Plotting Recall on the second subplot
    ax2.plot(recall_results, label=recall_results.columns)
    
    # Adding titles and labels for the Coverage plot
    ax1.set_title("Coverage by Model")
    ax1.set_xlabel("p-value threshold")
    ax1.set_ylabel("Coverage")
    #ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adding titles and labels for the Recall plot
    ax2.set_title("Recall by Model")
    ax2.set_xlabel("p-value threshold")
    ax2.set_ylabel("Recall")
    ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


"""
Something along those lines will be useful later down the line, i.e. find shared evals across complex runs and compare them
"""
def get_variable_and_fixed_params(all_results, all_results_concat_dict):
    all_config_params = {}
    for key, val in all_results.items():
        for config_key, config_val in val['config'].items():
            if config_key in all_config_params:
                all_config_params[config_key].append(config_val)
            else:
                all_config_params[config_key] = [config_val]
    
    fixed_params = []
    variable_params = []
    for key, val in all_config_params.items():
        if are_identical(val) or len(val) < len(all_results_concat_dict):
            fixed_params.append(key)
        else:
            print(f'Config parameter "{key}" varies with values:\n {val}')
            variable_params.append(key)
    print(f'\nThe following parameters are fixed for all runs:\n {fixed_params}')

    return fixed_params, variable_params


#TODO: No support for folds yet --> Honestly this all might be fore efficient with a notebook
# What are the issues, saving all that shit consistently is a pain, selecting the right stuff etc. but at least we can deploy
# Several folds, several targets in each fold 
def main(*args):


    parser = argparse.ArgumentParser(description='Analysis settings.')

    #Add argument which is a list of paths to the dirs containgin the results

    parser.add_argument('--results_dirs', type=str, nargs='+', help='List of paths to the directories containing the results')
    parser.add_argument('--target_pert', type=str, default='', help='Target perturbation to compare to')
    parser.add_argument('--target_cell', type=str, default='', help='Name of the target perturbation')

    args = parser.parse_args()



    results_dir = [Path(x) for x in args.results_dirs]

    cell = args.target_cell
    pert = args.target_pert

    results_configs = []

    for rd in results_dir:
        with open(rd/'config.yaml', 'r') as f:
            results_configs.append(yaml.load(f, Loader=yaml.FullLoader))
        

    task_name = results_configs[0]['task_config']['name']

    hash = hashlib.sha256(json.dumps(results_configs).encode()).hexdigest()
    
    results_names = [c['model_config']['name'] for c in results_configs]

    combined_name = ' - '.join(results_names)
    save_path = Path(f"./results/{task_name}/{combined_name}/{hash}").resolve()

    if not save_path.exists():
        save_path.mkdir(parents=True)

    #Saing the configs of all the runs that contributed to these plots
    with open(f"{save_path}/all_configs.yaml", 'w+') as f:
        yaml.dump(results_configs, f, indent=2, default_flow_style=False)

    #Should it always be the same task


    #Get all subdirectories of the model and task, each dir is a run and each run might have several folds

    r2_and_mse_results_summary = None
    deg_summary = None
    for i, rd in enumerate(results_dir):
        result_name = results_names[i]

        
        
        print(pert)
        # if ((len(config['holdout_perts']) == 1) and (len(config['holdout_cells']) == 1)):                     
        with open(f'{rd}/{r2_mse_filename(pert, cell)}', 'rb') as f:
            r2_and_mse = pickle.load(f)
            
        with open(f'{rd}/{c_r_filename(pert, cell)}', 'rb') as f:
            c_r_results = pickle.load(f)
            
        with open(f'{rd}/{DEGs_overlap_filename(pert, cell)}', 'rb') as f:
            DEGs_overlaps = pickle.load(f)

        print(DEGs_overlaps)
        print(pd.DataFrame.from_dict(DEGs_overlaps, orient='index'))

        df_deg = pd.DataFrame.from_dict(DEGs_overlaps, orient='index')
        df_r2_mse = pd.DataFrame.from_dict(r2_and_mse, orient='index')
        df_deg.columns = [result_name]
        df_r2_mse.columns = [result_name]



        if r2_and_mse_results_summary is None:
            r2_and_mse_results_summary = df_r2_mse

        else:
            r2_and_mse_results_summary = pd.concat([r2_and_mse_results_summary, df_r2_mse], axis=1)

        if deg_summary is None:
            deg_summary = df_deg
        else:
            deg_summary = pd.concat([deg_summary, df_deg], axis=1)
      
        
    #Create a dataframe with all the results

    df_to_heatmap(rearrange_results_df(r2_and_mse_results_summary, metric = 'R2'), f'r2 for {pert} on {cell} - task: {task_name}', (10,14), save_path)
    df_to_heatmap(rearrange_results_df(r2_and_mse_results_summary, metric = 'MSE'), f'MSE for {pert} on {cell} - task: {task_name}', (10,14), save_path)
    df_to_heatmap(deg_summary, f'DEGs for {pert} on {cell} - task: {task_name}', (10,14), save_path)


if __name__ == '__main__':

    main()