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


    print("Running main")
    parser = argparse.ArgumentParser(description='Analysis settings.')

    parser.add_argument('--task_config', type=str, default='', help='Path to yaml config file of the task.')
    parser.add_argument('--model_config', type=str, default='', help='Path to yaml config file of the model.')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode, datasetsize will be capped at 10000')
    parser.add_argument('--slurm_id', type=int, default=1, help='Slurm id for the job')
    parser.add_argument('-l', '--log', dest='loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level (default: %(default)s)", default='WARNING')

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


    input_dim = adata.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pert_ids = adata.obs[PERT_KEY].unique()

    logger.info(f"Data loaded, # of cells: {adata.shape[0]}, # of features: {input_dim} # of perts: {len(pert_ids)}")
    logger.info(f"Running experiment on {device}")


    #Register your models here
    #TODO: Change for prefix checking
    
    if model_name == 'nearest-neighbor':
        from omnicell.models.nearest_neighbor.predictor import NearestNeighborPredictor
        logger.info("Nearest Neighbor model selected")
        model = NearestNeighborPredictor(config_model)

    elif model_name == 'transformer':
        #from cellot.models.cfm import train
        raise NotImplementedError()

    elif model_name == 'vae':
        from omnicell.models.VAE.predictor import VAEPredictor
        logger.info("VAE model selected")
        model = VAEPredictor(config_model, input_dim, device, pert_ids)
    
    elif model_name == 'test':
        from omnicell.models.test.predictor import TestPredictor
        logger.info("Test model selected")
        model = TestPredictor(adata)
        

    
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

        logger.info(f"Running fold {i}")
        if not os.path.exists(fold_save):
            os.makedirs(fold_save)


        #TODO: We will need to see how we handle the checkpointing logic with folds and stuff
        model.train(adata_train)

        logger.info(f"Training completed for fold {i}")
        #TODO: When random folds and whatnots are implemented modify the config to reflect the concrete fold and save that 

        with open(fold_save / f"config.yaml", 'w+') as f:
            yaml.dump(config.to_dict(), f, indent=2)


        #If we have random splitting we need to save the holdout perts and cells as these will not be the same for each fold
        with open(fold_save / f"holdout_perts.json", 'w+') as f:
            json.dump(ho_perts, f)

        with open(fold_save / f"holdout_cells.json", 'w+') as f:
            json.dump(ho_cells, f)


        logger.info(f"Running evaluation for fold {i}")
        for cell, pert in eval_targets:

            logger.debug(f"Making predictions for {cell} and {pert}")

            #NOTE : These are taken on the entire data
            adata_ground_truth = get_pert_cell_data(adata, pert, cell)
            adata_ctrl_pert = get_cell_ctrl_data(adata, cell)

            logger.debug(f"Ground truth data loaded for {cell} and {pert} - # of ctrl cells {len(adata_ground_truth)}, # of ground truth cells {len(adata_ctrl_pert)}")


            control_sample_size = config.get_control_size()

            if control_sample_size > len(adata_ctrl_pert):
                logger.warning(f"Control size {config.get_control_size()} is larger than the number of control cells {len(adata_ctrl_pert)}, setting control size to the number of control cells")
                control_sample_size = len(adata_ctrl_pert)
                

            pushfwd_sample_size = config.get_test_size()
            if pushfwd_sample_size > len(adata_ground_truth):
                logger.warning(f"Test size {config.get_test_size()} is larger than the number of ground truth cells {len(adata_ground_truth)}, setting test size to the number of ground truth cells")
                pushfwd_sample_size = len(adata_ground_truth)

            adata_control = sc.pp.subsample(adata_ctrl_pert, n_obs=control_sample_size, copy=True)
            adata_pushfwd = sc.pp.subsample(adata_ctrl_pert, n_obs=pushfwd_sample_size, copy=True)

            preds = model.make_predict(adata_pushfwd, pert, cell)

            preds = preds.toarray() if not isinstance(preds, np.ndarray) else preds
            control  = adata_control.X.toarray() if not isinstance(adata_control.X, np.ndarray) else adata_control.X
            ground_truth = adata_ground_truth.X.toarray() if not isinstance(adata_ground_truth.X, np.ndarray) else adata_ground_truth.X





            np.savez(
                    f"{fold_save}/{prediction_filename(pert, cell)}",
                    pred_pert=preds, 
                    true_pert=ground_truth, 
                    control=control)



if __name__ == '__main__':
    main()