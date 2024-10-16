
from typing import List, Tuple, Optional
import logging

#TODO: - Configs just for trainings, configs for evaluations, 

#TODO: Introduce config validation

logger = logging.getLogger(__name__)
class Config:
    """Immutable class representing a config file for a model or task"""
    VALID_KEYS = ['model_config', 'eval_config', 'datasplit_config', 'etl_config']

    # So what is the config object, what does it do? - Some top level dictionary that contains all subjacent configs
    def __init__(self, config):
        #self.task_config = config['task'].copy() if 'task' in config else None
        self.model_config = config['model_config'].copy() if 'model_config' in config else None
        self.eval_config = config['eval_config'].copy() if 'eval_config' in config else None
        self.datasplit_config = config['datasplit_config'].copy() if 'datasplit_config' in config else None
        self.etl_config = config['etl_config'].copy() if 'etl_config' in config else None

        #Log if any non valid keys
        for key in config.keys():
            if key not in self.VALID_KEYS:
                logger.warning(f"Key {key} is not a valid key in the config file, will be discarded")

    @classmethod
    def empty(cls):
        return Config({})
    
    def copy(self):
        config = {}
        config = self.to_dict()
        return Config(config)

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()
    
    #We completely control the serialization of the config object in this class 
    def to_dict(self):
        config = {}
        if self.model_config is not None:
            config['model_config'] = self.model_config.copy()
        if self.eval_config is not None:
            config['eval_config'] = self.eval_config.copy()
        if self.datasplit_config is not None:
            config['datasplit_config'] = self.datasplit_config.copy()
        if self.etl_config is not None:
            config['etl_config'] = self.etl_config.copy()
        return config
    
    #Bunch of getters and setters to no longer be fixed by the config file structure in the code
    def add_datasplit_config(self, datasplit_config)-> 'Config':
        config = self.copy()
        config.datasplit_config = datasplit_config
        return config
    
    def add_model_config(self, model_config) -> 'Config':
        config = self.copy()
        config.model_config = model_config
        return config
    
    def add_eval_config(self, eval_config)-> 'Config':
        config = self.copy()
        config.eval_config = eval_config
        return config
    
    def add_etl_config(self, etl_config)-> 'Config':
        config = self.copy()
        config.etl_config = etl_config
        return config
    
    """
    Returns a config with only the portions of the config that are relevant for training: etl, datasplit and model
    """
    def get_training_config(self) -> 'Config':
        datasplit_config = self.datasplit_config.copy()
        model_config = self.model_config.copy()
        return Config({'datasplit_config': datasplit_config, 'model_config': model_config, 'etl_config': self.etl_config})
    
    # GETTERS FOR MODEL
    def get_model_name(self)-> str:
        return self.model_config['name']
    
    def get_model_config(self):
        return self.model_config.copy()


    # GETTERS FOR TRAINING
    def get_training_dataset_name(self)-> str:
        return self.etl_config['dataset']
    
    def get_datasplit_config_name(self)-> str:
        return self.datasplit_config['name']
    
    """Returns mode of data split, iid or ood"""
    def get_mode(self) -> str:
        return self.datasplit_config['mode']
    
    def get_heldout_cells(self) -> List[str]:
        """Returns the heldout cells for the training data, returns an empty list if no cells are held out"""
        return self.datasplit_config.get('holdout_cells', [])
    
    def get_heldout_perts(self) -> List[str]:
        """Returns the heldout perturbations for the training data, returns an empty list if no perturbations are held out"""
        return self.datasplit_config.get('holdout_perts', [])
    
    def get_apply_normalization(self) -> bool:
        return self.etl_config['count_norm']
    
    def get_apply_log1p(self) -> bool:
        return self.etl_config['log1p']
   
    def get_cell_embedding_name(self) -> Optional[str]:
        return self.etl_config.get('cell_embedding', None)
    
    def get_pert_embedding_name(self) -> Optional[str]:
        return self.etl_config.get('pert_embedding', None) 
    
    # GETTERS FOR EVAL
    def get_eval_config_name(self)-> str:
        return self.eval_config['name']
    
    def get_eval_dataset_name(self)-> str:
        return self.eval_config['dataset']
    
    def get_eval_targets(self)-> List[Tuple[str, str]]:
        """Returns a list of [Cell, Pert] tuples that are used for evaluation"""
        return self.eval_config['evaluation_targets']
    

    
    

    
    