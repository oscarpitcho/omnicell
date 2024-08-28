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
from omnicell.data.splitter import Splitter
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT
from omnicell.data.utils import get_pert_cell_data, get_cell_ctrl_data, prediction_filename
from omnicell.config.config import Config


random.seed(42)

def main(*args):

    parser = argparse.ArgumentParser(description='Analysis settings.')

    parser.add_argument('--task_config', type=str, default='', help='Path to yaml config file of the task.')
    parser.add_argument('--model_config', type=str, default='', help='Path to yaml config file of the model.')


    args = parser.parse_args()


    model_path = Path(args.model_config).resolve()
    task_path = Path(args.task_config).resolve()



    config_model = yaml.load(open(model_path), Loader=yaml.UnsafeLoader)
    config_task = yaml.load(open(task_path), Loader=yaml.UnsafeLoader)

    config = Config.empty().add_model_config(config_model).add_task_config(config_task).add_train_args(args.__dict__)

    #Store the config and the paths to the config to make reproducibility easier. 


    #This is part of the processing, should put it someplace else
    adata = sc.read_h5ad(config.get_data_path())

    #Making it faster for testing


    #Standardizing column names and key values
    adata.obs[PERT_KEY] = adata.obs[config.get_pert_key()]
    adata.obs[CELL_KEY] = adata.obs[config.get_cell_key()]
    adata.obs[PERT_KEY] = adata.obs[PERT_KEY].cat.rename_categories({config.get_control_pert() : CONTROL_PERT})
    adata = adata[:100000]


    model = None
    model_name = config.get_model_name()
    task_name = config.get_task_name()

            
    hash_dir = hashlib.sha256(json.dumps(config.to_dict()).encode()).hexdigest()
    
    save_path = Path(f"./results/{model_name}/{task_name}/{hash_dir}").resolve()


    #Saving run config
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}/config.yaml", 'w+') as f:
        yaml.dump(config.to_dict(), f, indent=2, default_flow_style=False)


    #Register your models here
    if model_name == 'nearest_neighbor':
        from omnicell.models.nearest_cell_type.NearestNeighborPredictor import NearestNeighborPredictor

        model = NearestNeighborPredictor(config_model)
        print(f"Model {model}")

    elif model_name == 'transformer':
        #from cellot.models.cfm import train
        raise NotImplementedError()

    elif model_name == 'vae':
        from omnicell.models.VAE.vae import VAEPredictor

        model = VAEPredictor(config_model)
    
    else:
        raise ValueError('Unknown model name')
    

    #Every fold corresponds to a training
    #We need to split the data according to the task config
    splitter = Splitter(config_task)
    folds = splitter.split(adata)

        

    #TODO: When generating a random fold we should save the config of the fold
    #Overwrite the config with the fold details so that we can reproduce the fold easily
    #What will happen when we have a pretrained model? All this logic will no longer be adequate
    for i, (adata_train, adata_eval, ho_perts, ho_cells, eval_targets) in enumerate(folds):
        fold_save = save_path / f"fold_{i}"

        if not os.path.exists(fold_save):
            os.makedirs(fold_save)


        #TODO: We will need to see how we handle the checkpointing logic with folds and stuff
        model.train(adata_train)

        #TODO: When random folds and whatnots are implemented modify the config to reflect the concrete fold and save that 

        with open(fold_save / f"config.yaml", 'w+') as f:
            yaml.dump(config.to_dict(), f, indent=2)


        #If we have random splitting we need to save the holdout perts and cells as these will not be the same for each fold
        with open(fold_save / f"holdout_perts.json", 'w+') as f:
            json.dump(ho_perts, f)

        with open(fold_save / f"holdout_cells.json", 'w+') as f:
            json.dump(ho_cells, f)


        for cell, pert in eval_targets:
            print(f"Making predictions for perturbation {pert} on cell type {cell}")


            #NOTE : These are taken on the entire data
            adata_ground_truth = get_pert_cell_data(adata, pert, cell)
            adata_control = get_cell_ctrl_data(adata, cell)
            preds = model.make_predict(adata_control, pert, cell)

            print(f"Type of predictions {type(preds)}")
            print(f"Type of adata_ground_truth.X {type(adata_ground_truth.X)}")
            print(f"Type of adata_control.X {type(adata_control.X)}")


            print(preds.shape)  
            print(adata_ground_truth.X.shape)
            print(adata_control.X.shape)

            np.savez(
                    f"{fold_save}/{prediction_filename(pert, cell)}",
                    pred_pert=preds, 
                    true_pert=adata_ground_truth.X.toarray(), 
                    control=adata_control.X.toarray())



if __name__ == '__main__':
    main()