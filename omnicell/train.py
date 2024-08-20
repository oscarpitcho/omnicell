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
from omnicell.datasplits.Splitter import Splitter
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT


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

    #Store the config and the paths to the config to make reproducibility easier. 
    config = {'args': args.__dict__, 'model_config': config_model, 'task_config': config_task}


    #This is part of the processing, should put it someplace else
    adata = sc.read_h5ad(config_task['data']['path'])

    #Making it faster for testing
    adata = adata[:10000]
    print(adata.obs.columns)


    #Standardizing column names and key values
    adata.obs[PERT_KEY] = adata.obs[config_task['data']['pert_key']]
    adata.obs[CELL_KEY] = adata.obs[config_task['data']['cell_key']]
    adata.obs[PERT_KEY] = adata.obs[PERT_KEY].cat.rename_categories({config_task['data']['control'] : CONTROL_PERT})
    print(adata.obs[CELL_KEY].unique())
    adata = adata[:10000]

    print(adata.obs[PERT_KEY].unique())

    print(adata.obs.columns)
    """ pert_types = adata.obs[config_task['pert_col']].unique()
    cell_types = adata.obs[config_task['cell_col']].unique()


    #Random splitting of perturbations
    if config_task.get('pert_random_holdout', None) is not None:
        pert_holdout_fraction = pert_holdout_fraction

        pert_holdout = np.random.choice(pert_types, int(pert_holdout_fraction * len(pert_types)), replace=False)
     #TODO: Implement if a single holdout pert
    elif config_task.get('pert_holdout', None) is not None:
        pert_holdout = pert_holdout"""
    
    #We need to load the data at this level and pass it on to the model 

    #Load the data according to the config: 








    #For data dep models we have access to datashapes and we can pass them when instantiating the model

    #REGISTER YOUR MODELS HERE
    model = None
    model_name = config_model['name']

            
    hash_dir = hashlib.sha256(json.dumps(config).encode()).hexdigest()
        
    #We should pass this to the model to load checkpoints (eventually)
    save_path = Path(f"./results/{model_name}/{hash_dir}").resolve()


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}/config.yaml", 'w+') as f:
        yaml.dump(config, f, indent=2)

    if model_name == 'nearest_neighbor':
        from omnicell.models.nearest_cell_type.NearestNeighborPredictor import NearestNeighborPredictor

        model = NearestNeighborPredictor(config_model)
        print(f"Model {model}")

    elif model_name == 'transformer':
        #from cellot.models.cfm import train
        raise NotImplementedError()

    elif model_name == 'vae':
        raise NotImplementedError()
    
    else:
        raise ValueError('Unknown model name')
    



    #Every fold corresponds to a training
    #We need to split the data according to the task config
    splitter = Splitter(config_task)
    folds = splitter.split(adata)
        

    #What will happen when we have a pretrained model? All this logic will no longer be adequate
    for i, (adata_train, adata_eval, holdout_perts, holdout_cells) in enumerate(folds):
        fold_save = save_path / f"fold_{i}"

        if not os.path.exists(fold_save):
            os.makedirs(fold_save)



        #TODO: We will need to see how we handle the checkpointing logic with folds and stuff
        model.train(adata_train)


        #If we have random splitting we need to save the holdout perts and cells as these will not be the same for each fold
        with open(fold_save / f"holdout_perts.json", 'w+') as f:
            json.dump(holdout_perts, f)

        with open(fold_save / f"holdout_cells.json", 'w+') as f:
            json.dump(holdout_cells, f)



        #Should all this logic be put in the splitter idk
        #Each instance in this loop will define a task --> We need preds, ground truth and control
        #Making preds across perts
        across_perts_save = fold_save / "across_perts"
        if not os.path.exists(across_perts_save):
            os.makedirs(across_perts_save)
        for pert in holdout_perts:
            print(f"Making predictions for perturbation {pert}")
            #We need some ground truth data to save

            #Problem is that it would be easier to let the model to all that shit but then we are not sure what was the 

            adata_ground_truth = adata_eval[(adata_eval.obs[PERT_KEY] == pert) & (~adata_eval.obs[CELL_KEY].isin(holdout_cells))]


            #TODO: Across genes the control data is also in the training data, should we exclude some to have it separate?
            adata_control = adata_train[adata_train.obs[PERT_KEY] == CONTROL_PERT]            
            
            adata_predictions = model.predict_across_pert(pert)

            np.savez(
                    f"{across_perts_save}/pred_{pert}.npz", 
                    pred_pert=adata_predictions, 
                    true_pert=adata_ground_truth.X, 
                    control=adata_control.X,
                )
            

        across_cells_save = fold_save / "across_cells"
        if not os.path.exists(across_cells_save):
            os.makedirs(across_cells_save)


        #Nothing happens here if we do not have holdout cells
        for ho_cell in holdout_cells:

            across_cells_targets = config_task['datasplit']['task']['across_cells_targets']

            for target in across_cells_targets:
                adata_control = adata_eval[(adata_eval.obs[CELL_KEY] == ho_cell) & (adata_eval.obs[PERT_KEY] == CONTROL_PERT)]
                adata_ground_truth = adata_eval[(adata_eval.obs[CELL_KEY] == ho_cell) & (adata_eval.obs[PERT_KEY] == target)]

                #We pass data on the unseen cell type to the model
                adata_predictions = model.predict_across_cell(adata_control, target)
                np.savez(
                    f"{across_cells_save}/pred_{ho_cell}_{target}.npz", 
                    pred_pert=adata_predictions, 
                    true_pert=adata_ground_truth.X, 
                    control=adata_control.X,
                )
            #So here is the thing, we can evaluate the result on the holdout cell type on many different perturbation types and these can be in or out of distribution


            




        """ #Making preds across cells and perts
        for target in targets:
            for cell in holdout_cells:
        """

    
    #Some task parsing


    #So what kind of tasks do we have?
    #Several folds, 


    #Now we do the prediction tasks on the trained model
    #We can predict across cells or across perturbations --> This should be defined in the task config

    


if __name__ == '__main__':
    main()