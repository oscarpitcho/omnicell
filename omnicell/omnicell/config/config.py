
from typing import List, Tuple

#TODO: - Configs just for trainings, configs for evaluations, 

#TODO: Introduce config validation
class Config:
    """Immutable class representing a config file for a model or task"""

    #So what is the config object, what does it do? - Some top level dictionary that contains all subjacent configs
    def __init__(self, config):
        #self.task_config = config['task'].copy() if 'task' in config else None
        self.model_config = config['model'].copy() if 'model' in config else None
        self.eval_config = config['eval'].copy() if 'eval' in config else None
        self.train_args = config['train_args'].copy() if 'train_args' in config else None
        self.timestamp = config['timestamp'] if 'timestamp' in config else None
        self.data_config = config['data_config'].copy() if 'data_config' in config else None

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
        if self.task_config is not None:
            config['task'] = self.task_config.copy()
        if self.model_config is not None:
            config['model'] = self.model_config.copy()
        if self.eval_config is not None:
            config['eval'] = self.eval_config.copy()
        if self.train_args is not None:
            config['train_args'] = self.train_args.copy()
        if self.timestamp is not None:
            config['timestamp'] = self.timestamp
        if self.data_config is not None:
            config['data_config'] = self.data_config.copy()
        return config
    
    #Bunch of getters and setters to no longer be fixed by the config file structure in the code
    def add_task_config(self, task_config)-> 'Config':
        config = self.copy()
        config.task_config = task_config
        return config
    
    def add_model_config(self, model_config) -> 'Config':
        config = self.copy()
        config.model_config = model_config
        return config
    
    def add_data_config(self, data_config)-> 'Config':
        config = self.copy()
        config.data_config = data_config
        return config
    
    """
    Returns a config with only the portions of the config that are relevant for training
    """
    def get_training_config(self) -> 'Config':
        data_config = self.data_config.copy()
        model_config = self.model_config.copy()
        return Config({'data_config': data_config, 'model': model_config})

    def add_eval_config(self, eval_config)-> 'Config':
        config = self.copy()
        config.eval_config = eval_config
        return config

    def add_train_args(self, train_args_dict)-> 'Config':
        config = self.copy()
        config.train_args = train_args_dict
        return config
    
    def get_dataset_name(self)-> str:
        return self.data_config['name']
    
    def get_data_loader_config(self)-> str:
        return self.data_config['loader']
    
    def get_pert_key(self)-> str:
        return self.task_config['data']['pert_key']

    def get_cell_key(self)-> str:
        return self.task_config['data']['cell_key']

    def get_control_pert(self)-> str:
        return self.task_config['data']['control']
    

    def get_var_names_key(self)-> str:
        return self.task_config['data'].get('var_names_key', None)


    def get_test_mode(self)-> bool:
        return self.train_args['test_mode']

    def set_target_evaluations(self, eval_targets: List[Tuple[str, str]])-> 'Config':
        config = self.copy()

        config.task_config['datasplit']['evals']['evaluation_targets'] = eval_targets
        return config

    def get_eval_targets(self)-> 'Config':
        """Returns a list of [Cell, Pert] tuples that are used for evaluation"""
        return self.task_config['datasplit']['evals']['evaluation_targets']
    

    def get_task_name(self)-> str:
        return self.task_config['name']
    
    def get_model_name(self)-> str:
        return self.model_config['name']

    def get_model_config(self):
        return self.model_config.copy()
    
    def get_var_gene_key(self):
        return self.task_config['data'].get(['gene_key'], None)
    

    def get_mode(self):
        return self.task_config['datasplit']['mode']
    

    def get_heldout_cells(self):
        """Returns the heldout cells for the training data, returns an empty list if no cells are held out"""
        return self.task_config['datasplit']['training'].get('holdout_cells', [])
    
    def get_heldout_perts(self):
        """Returns the heldout perturbations for the training data, returns an empty list if no perturbations are held out"""
        return self.task_config['datasplit']['training'].get('holdout_perts', [])
    

    def get_apply_normalization(self):
        return self.task_config['data']['apply_norm']
    
    def get_apply_log1p(self):
        return self.task_config['data']['apply_log1p']
    
    def set_heldout_cells(self, heldout_cells: List[str]):
        config = self.copy()
        config.task_config['datasplit']['training']['holdout_cells'] = heldout_cells
        return config
    
    def set_heldout_perts(self, heldout_perts: List[str]):
        config = self.copy()
        config.task_config['datasplit']['training']['holdout_perts'] = heldout_perts
        return config
    
    def add_timestamp(self, timestamp: str):
        config = self.copy()
        config.timestamp = timestamp
        return config

    def get_test_size(self):
        return self.task_config['datasplit']['test_size']
    
    def get_control_size(self):
        return self.task_config['datasplit']['control_size']

    def get_data_path(self):
        if self.task_config is None:
            raise ValueError("No data config found in the config file")
        return self.task_config['data']['path']

    