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
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT, DATA_CATALOGUE_PATH
from omnicell.data.utils import prediction_filename
from omnicell.processing.utils import to_dense, to_coo
from omnicell.config.config import Config
from omnicell.data.loader import DataLoader
from omnicell.data.catalogue import Catalogue
import time
import datetime
import torch


logger = logging.getLogger(__name__)


random.seed(42)

def get_model(model_name, config_model, loader):
    adata, pert_rep_map = loader.get_training_data()

    if pert_rep_map is not None:
        pert_keys = list(pert_rep_map.keys())
        pert_rep = np.array([pert_rep_map[k] for k in pert_keys])
        pert_map = {k: i for i, k in enumerate(pert_keys)}

    input_dim = adata.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pert_ids = adata.obs[PERT_KEY].unique()

    logger.info(f"Data loaded, # of cells: {adata.shape[0]}, # of features: {input_dim} # of perts: {len(pert_ids)}")
    logger.info(f"Running experiment on {device}")

    logger.debug(f"Training data loaded, perts are: {adata.obs[PERT_KEY].unique()}")

    if 'nearest-neighbor-generic' in model_name:
        from omnicell.models.nearest_neighbor.predictor import NearestNeighborPredictor
        logger.info("Nearest Neighbor model selected")
        model = NearestNeighborPredictor(config_model, pert_rep=pert_rep, pert_map=pert_map)

    elif 'nearest-neighbor-gene-dist' in model_name:
        from omnicell.models.nearest_neighbor.gene_distance import NearestNeighborPredictor
        logger.info("Nearest Neighbor Gene Distance model selected")
        model = NearestNeighborPredictor(config_model)

    elif 'flow' in model_name:
        from omnicell.models.flows.flow_predictor import FlowPredictor
        logger.info("Flow model selected")
        model = FlowPredictor(config_model, input_dim, pert_rep, pert_map)

    elif 'llm' in model_name:
        from omnicell.models.llm.llm_predictor import LLMPredictor
        logger.info("Transformer model selected")
        model = LLMPredictor(config_model, input_dim, device, pert_ids)
        
    elif 'vae' in model_name:
        from omnicell.models.VAE.vae import VAE
        logger.info("VAE model selected")
        model = VAE(config_model, input_dim, device, pert_ids)

    elif 'scVIDR' in model_name:
        from omnicell.models.VAE.scVIDR_predictor import ScVIDRPredictor
        logger.info("scVIDR model selected")
        model = ScVIDRPredictor(config_model, input_dim, device, pert_ids)

    elif "test" in model_name:
        from omnicell.models.dummy_predictor.predictor import TestPredictor
        logger.info("Test model selected")
        adata_cheat = loader.get_complete_training_dataset()
        model = TestPredictor(adata_cheat)
        
    else:
        raise ValueError('Unknown model name')
    
    return model, adata

def main(*args):
    print("Running main")
    parser = argparse.ArgumentParser(description='Analysis settings.')

    parser.add_argument('--datasplit_config', type=str, default=None, help='Path to yaml config of the datasplit.')
    parser.add_argument('--etl_config', type=str, default=None, help='Path to yaml config file of the etl process.')
    parser.add_argument('--model_config', type=str, default=None, help='Path to yaml config file of the model.')
    parser.add_argument('--eval_config', type=str, default=None, help='Path to yaml config file of the evaluations, if none provided the model will only be trained.')
    parser.add_argument('--test_mode', action='store_true', default=False, help='Run in test mode, datasetsize will be capped at 10000')
    parser.add_argument('--slurm_id', type=int, default=1, help='Slurm id for the job, useful for arrays')
    parser.add_argument('-l', '--log', dest='loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level (default: %(default)s)", default='INFO')

    args = parser.parse_args()

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H:%M:%S")

    config = Config.from_yamls(args.etl_config, args.datasplit_config, args.model_config)

    logging.basicConfig(
        filename=f'output_{args.slurm_id}_{config.get_model_name()}_{config.get_datasplit_config_name()}.log', 
        filemode= 'w', level=args.loglevel, format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Application started")

    model_savepath = config.get_train_path()

    catalogue = Catalogue(DATA_CATALOGUE_PATH)
    loader = DataLoader(config, catalogue)
    

    model, adata = get_model(config.get_model_name(), config.model_config, loader)


    if hasattr(model, 'save') and hasattr(model, 'load'):
        # Path depends on hash of config
        if os.path.exists(model_savepath):
            logger.info(f"Model already trained, loading model from {model_savepath}")
            model.load(model_savepath)
            logger.info("Model loaded")
        else:
            logger.info("Model not trained, training model")
            model.train(adata)
            logger.info("Training completed")
            logger.info(f"Saving model to {model_savepath}")
            os.makedirs(model_savepath, exist_ok=True)

            model.save(model_savepath)
            
            #We only save the training config (ETL + Split + Model)
            logger.info(f"Saving Training config to {model_savepath}")
            with open(f"{model_savepath}/training_config.yaml", 'w+') as f:
                yaml.dump(config.get_training_config().to_dict(), f, indent=2, default_flow_style=False)
    else:
        logger.info("Model does not support saving/loading, training from scratch")
        model.train(adata)
        logger.info("Training completed")    

    # if model has encode function then encode the full adata and save in the model dir
    if hasattr(model, 'encode'):
        logger.info("Encoding full dataset")
        adata = loader.get_complete_training_dataset()
        embedded_data = model.encode(adata)
        np.save(f"{model_savepath}/embedded_data.npy", embedded_data)

    #It is not none --> We are going to evaluate
    if args.eval_config is not None and hasattr(model, 'make_predict'):
        logger.info("Running evaluation")
        
        results_path = config.get_eval_path()
        logger.info(f"Will save results to {results_path}")

        #Saving run config
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        with open(f"{results_path}/config.yaml", 'w+') as f:
            yaml.dump(config.to_dict(), f, indent=2, default_flow_style=False)

        for cell_id, pert_id, ctrl_data, gt_data in loader.get_eval_data():
            logger.debug(f"Making predictions for cell: {cell_id}, pert: {pert_id}")
            preds = model.make_predict(ctrl_data, pert_id, cell_id)
            preds = to_coo(preds)
            control  = to_coo(ctrl_data.X)
            ground_truth = to_coo(gt_data.X)

            #TODO: We only need to save one control file per cell, if we have several perts we can reuse the same control file
            scipy.sparse.save_npz(f"{results_path}/{prediction_filename(pert_id, cell_id)}-preds", preds)
            scipy.sparse.save_npz(f"{results_path}/{prediction_filename(pert_id, cell_id)}-control", control)
            scipy.sparse.save_npz(f"{results_path}/{prediction_filename(pert_id, cell_id)}-ground_truth", ground_truth)


if __name__ == '__main__':
    main()