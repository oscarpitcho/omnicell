import argparse
import scanpy as sc
import yaml
from pathlib import Path
import sys
import os
import hashlib
import json
import numpy as np
import random

random.seed(42)

def main(*args):

    parser = argparse.ArgumentParser(description='Analysis settings.')

    parser.add_argument('--task', type=str, default='', help='Path to yaml config file of the task.')
    parser.add_argument('--model', type=str, default='', help='Path to yaml config file of the model.')


    args = parser.parse_args()


    model_path = Path(args.model).resolve()
    task_path = Path(args.task).resolve()



    config_model = yaml.load(open(model_path), Loader=yaml.UnsafeLoader).to_dict()
    config_task = yaml.load(open(task_path), Loader=yaml.UnsafeLoader).to_dict()

    #Store the config and the paths to the config to make reproducibility easier. 
    config = {'args': args.__dict__, 'model_config': config_model, 'task_config': config_task}

    adata = sc.read_h5ad(config_task['dataset'])

    pert_types = adata.obs[config_task['pert_col']].unique()
    cell_types = adata.obs[config_task['cell_col']].unique()


    #Random splitting of perturbations
    if config_task.get('pert_random_holdout', None) is not None:
        pert_holdout_fraction = pert_holdout_fraction

        pert_holdout = np.random.choice(pert_types, int(pert_holdout_fraction * len(pert_types)), replace=False)

    #TODO: Implement if a single holdout pert
    """    elif config_task.get('pert_holdout', None) is not None:
        pert_holdout = pert_holdout"""
    
    #We need to load the data at this level and pass it on to the model 

    #Load the data according to the config: 

    hash_dir = hashlib.sha256(json.dumps(config).encode()).hexdigest()
    
    #We should pass this to the model to load checkpoints (eventually)
    save_path = Path(f"./results/{hash_dir}").resolve()


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}/config.json", 'w') as f:
        yaml.dump(config, f, indent=2)
        

    #For data dep models we have access to datashapes and we can pass them when instantiating the model

    #REGISTER YOUR MODELS HERE
    model = None
    if args.model == 'nearest_cell_type':
        from models.nearest_cell_type import NearestNeighborPredictor
        model = NearestNeighborPredictor(config_model)    

    elif args.model == 'transformer':
        #from cellot.models.cfm import train
        raise NotImplementedError()

    elif args.model == 'vae':
        raise NotImplementedError()
    
    else:
        raise ValueError('Unknown model name')
    

    model.train(training_data)
    
    #Some task parsing


    #So what kind of tasks do we have?
    #Several folds, 


    #Now we do the prediction tasks on the trained model
    #We can predict across cells or across perturbations --> This should be defined in the task config

    


if __name__ == '__main__':
    main()