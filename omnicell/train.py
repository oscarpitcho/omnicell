import argparse
import scanpy as sc
import yaml
from pathlib import Path
import sys
import os
import hashlib
import json
import scipy
import logging
import numpy as np
import random
from omnicell.data.splitter import Splitter
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT
from omnicell.data.utils import get_pert_cell_data, get_cell_ctrl_data, prediction_filename
from omnicell.processing.utils import to_dense, to_coo
from omnicell.data.preprocessing import preprocess
from omnicell.config.config import Config
from omnicell.data.loader import Loader
import time
import datetime
import torch


logger = logging.getLogger(__name__)


random.seed(42)

def main(*args):


    print("Running main")
    parser = argparse.ArgumentParser(description='Analysis settings.')

    parser.add_argument('--data_config', type=str, default=None, help='Path to yaml config file of the task.')
    parser.add_argument('--model_config', type=str, default=None, help='Path to yaml config file of the model.')
    parser.add_argument('--eval_config', type=str, default=None, help='Path to yaml config file of the evaluation, if none provided the model will only be trained.')
    parser.add_argument('--test_mode', action='store_true', default=False, help='Run in test mode, datasetsize will be capped at 10000')
    parser.add_argument('--slurm_id', type=int, default=1, help='Slurm id for the job, useful for arrays')
    parser.add_argument('-l', '--log', dest='loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level (default: %(default)s)", default='INFO')

    args = parser.parse_args()

    now = datetime.datetime.now()

    now = now.strftime("%Y-%m-%d_%H:%M:%S")


    data_catalogue = json.load(open('data_catalogue.json'))

 

    model_config_path = Path(args.model_config).resolve()
    data_config_path = Path(args.data_config).resolve()
    evaluation_config_path = Path(args.eval_config).resolve() if not args.eval_config == None 

    config_model = yaml.load(open(model_config_path), Loader=yaml.UnsafeLoader)
    config_data = yaml.load(open(data_config_path), Loader=yaml.UnsafeLoader)

    config_evals = yaml.load(open(evaluation_config_path), Loader=yaml.UnsafeLoader) if not evaluation_config_path == None else None


    

    config = Config.empty()
    config = config.add_model_config(config_model) #There will always be a model config
    config = config.add_data_config(config_data) #There will always be a training config
    config = config.add_eval_config(config_evals) if not config_evals == None else config #There might not be an eval config

    


    logging.basicConfig(filename= f'output_{args.slurm_id}_{config.get_model_name()}_{config.get_task_name()}.log', filemode= 'w', level=args.loglevel, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Application started")


    #TODO: Hash task + Model config to see if this model has already been saved and trained. 


    #Store the config and the paths to the config to make reproducibility easier. 


    logger.info(f"Loading data from {config.get_data_path()}")
    adata = sc.read_h5ad(config.get_data_path())
    adata = preprocess(adata, config)

    logger.info(f"Data loaded from {config.get_data_path()}")

    logger.debug(f"Cell types: {adata.obs[CELL_KEY].unique()}")
    logger.debug(f"Perturbations: {adata.obs[PERT_KEY].unique()}")

    model = None
    model_name = config.get_model_name()
    task_name = config.get_task_name()

            
    #TODO: Check if model has been trained before and load it if it is the case, careful with timestamp causing failed equalities.
    hash_dir = hashlib.sha256(json.dumps(config.to_dict()).encode()).hexdigest()
    hash_dir = hash_dir[:4]
    
    save_path = Path(f"./results/{model_name}/{task_name}/{now}-{hash_dir}").resolve()

    logger.info(f"Config parsed, model name: {model_name}, task name: {task_name}")
    logger.info(f"Saving results to {save_path}")



    #Saving run config
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}/config.yaml", 'w+') as f:
        yaml.dump(config.to_dict(), f, indent=2, default_flow_style=False)


    loader = Loader(config, data_catalogue)

    input_dim = adata.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pert_ids = adata.obs[PERT_KEY].unique()

    logger.info(f"Data loaded, # of cells: {adata.shape[0]}, # of features: {input_dim} # of perts: {len(pert_ids)}")
    logger.info(f"Running experiment on {device}")


    #Register your models here
    #TODO: Change for prefix checking
    
    if 'nearest-neighbor' in model_name:

        if config.get_mode() == 'iid':
            raise ValueError("Nearest Neighbor model does not support iid mode")
        
        from omnicell.models.nearest_neighbor.predictor import NearestNeighborPredictor
        logger.info("Nearest Neighbor model selected")
        model = NearestNeighborPredictor(config_model)

    elif 'llm' in model_name:
        from omnicell.models.llm.llm_predictor import LLMPredictor
        logger.info("Transformer model selected")
        model = LLMPredictor(config_model, input_dim, device, pert_ids)
        

    elif 'vae' in model_name:
        from omnicell.models.VAE.predictor import VAEPredictor
        logger.info("VAE model selected")
        model = VAEPredictor(config_model, input_dim, device, pert_ids)

    elif 'scVIDR' in model_name:
        from omnicell.models.VAE.scVIDR_predictor import ScVIDRPredictor
        logger.info("scVIDR model selected")
        model = ScVIDRPredictor(config_model, input_dim, device, pert_ids)
    
    elif model_name == 'test':
        from omnicell.models.dummy_predictor.predictor import TestPredictor
        logger.info("Test model selected")
        model = TestPredictor(adata)
        

    
    else:
        raise ValueError('Unknown model name')
    

    #Every fold corresponds to a training
    #We need to split the data according to the task config



    #This can be adata | torch.DataLoader --> Model should check appropriate type    
    train_data = loader.get_training_data()    

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    #TODO: We need to save the model, and the config with it.
    #We should save the configs with it but only the ones that are relevant for the model, i.e. training and model config
    model.train(train_data)

    logger.info(f"Training completed")

    with open(save_path / f"config.yaml", 'w+') as f:
        yaml.dump(config.to_dict(), f, indent=2)


    #If it is None we are just running a training job
    if args.eval_config is not None:
        for cell_id, pert_id, ctrl_data, gt_data in loader.get_eval_data():

            logger.debug(f"Making predictions for {cell_id} and {pert_id}")


            preds = model.make_predict(ctrl_data, pert_id, cell_id)

            preds = to_coo(preds)
            control  = to_coo(ctrl_data.X)
            ground_truth = to_coo(gt_data.X)


            #TODO: We only need to save one control file per cell, if we have several perts we can reuse the same control file

            scipy.sparse.save_npz(f"{save_path}/{prediction_filename(pert_id, cell_id)}-preds", preds)
            scipy.sparse.save_npz(f"{save_path}/{prediction_filename(pert_id, cell_id)}-control", control)
            scipy.sparse.save_npz(f"{save_path}/{prediction_filename(pert_id, cell_id)}-ground_truth", ground_truth)




if __name__ == '__main__':
    main()