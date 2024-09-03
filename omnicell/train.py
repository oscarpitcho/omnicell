import argparse
import scanpy as sc
import yaml
from pathlib import Path
import sys
import os
import hashlib
import json
import logging
import numpy as np
import random
from omnicell.data.splitter import Splitter
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT
from omnicell.data.utils import get_pert_cell_data, get_cell_ctrl_data, prediction_filename
from omnicell.data.preprocessing import preprocess
from omnicell.config.config import Config
import torch


logger = logging.getLogger(__name__)


random.seed(42)

def main(*args):


    parser = argparse.ArgumentParser(description='Analysis settings.')

    parser.add_argument('--task_config', type=str, default='', help='Path to yaml config file of the task.')
    parser.add_argument('--model_config', type=str, default='', help='Path to yaml config file of the model.')
    parser.add_argument('-l', '--log', dest='log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode, datasetsize will be capped at 10000')
    parser.add_argument('--slurm_id', type=int, default=1, help='Slurm id for the job')
    parser.add_argument('--loglevel', type=int, default=logging.INFO, help='Log level for the application')

    args = parser.parse_args()


 

    model_path = Path(args.model_config).resolve()
    task_path = Path(args.task_config).resolve()

    config_model = yaml.load(open(model_path), Loader=yaml.UnsafeLoader)
    config_task = yaml.load(open(task_path), Loader=yaml.UnsafeLoader)

    config = Config.empty().add_model_config(config_model).add_task_config(config_task).add_train_args(args.__dict__)

    logging.basicConfig(filename= f'output_{args.slurm_id}_{config.get_model_name()}.log', filemode= 'w', level=args.loglevel, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Application started")





    #Store the config and the paths to the config to make reproducibility easier. 


    #This is part of the processing, should put it someplace else
    adata = sc.read_h5ad(config.get_data_path())
    adata = preprocess(adata, config)

    model = None
    model_name = config.get_model_name()
    task_name = config.get_task_name()

            
    hash_dir = hashlib.sha256(json.dumps(config.to_dict()).encode()).hexdigest()
    
    save_path = Path(f"./results/{model_name}/{task_name}/{hash_dir}").resolve()

    logger.info(f"Config parsed, model name: {model_name}, task name: {task_name}")
    logger.info(f"Saving results to {save_path}")



    #Saving run config
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}/config.yaml", 'w+') as f:
        yaml.dump(config.to_dict(), f, indent=2, default_flow_style=False)

    #Register your models here
    #TODO: Change for prefix checking

    input_dim = adata.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pert_ids = adata.obs[PERT_KEY].unique()

    logger.info(f"Data loaded, # of cells: {adata.shape[0]}, # of features: {input_dim} # of perts: {len(pert_ids)}")
    logger.info(f"Running experiment on {device}")

    
    if model_name == 'nearest-neighbor':
        from omnicell.omnicell.models.nearest_neighbor.predictor import NearestNeighborPredictor
        logger.info("Nearest Neighbor model selected")
        model = NearestNeighborPredictor(config_model)

    elif model_name == 'transformer':
        #from cellot.models.cfm import train
        raise NotImplementedError()

    elif model_name == 'vae':
        from omnicell.models.VAE.predictor import VAEPredictor
        logger.info("VAE model selected")
        model = VAEPredictor(config_model, input_dim, device, pert_ids)
    
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
            logger.debug(f"Making predictions for {cell} and {pert}")

            #NOTE : These are taken on the entire data
            adata_ground_truth = get_pert_cell_data(adata, pert, cell)
            adata_control = get_cell_ctrl_data(adata, cell)
            preds = model.make_predict(adata_control, pert, cell)


            np.savez(
                    f"{fold_save}/{prediction_filename(pert, cell)}",
                    pred_pert=preds, 
                    true_pert=adata_ground_truth.X.toarray(), 
                    control=adata_control.X.toarray())



if __name__ == '__main__':
    main()