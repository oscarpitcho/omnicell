
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import yaml
import json
import hashlib

from copy import deepcopy

#TODO: - Configs just for trainings, configs for evaluations, 

#TODO: Introduce config validation

logger = logging.getLogger(__name__)
class Config:
    """Immutable class representing a config file for a model or task"""
    VALID_KEYS = ['model_config', 'eval_config', 'datasplit_config', 'etl_config']

    # So what is the config object, what does it do? - Some top level dictionary that contains all subjacent configs
    def __init__(self, model_config, etl_config, datasplit_config, eval_config=None):
        self.model_config = model_config
        self.datasplit_config = datasplit_config
        self.etl_config = etl_config
        self.eval_config = eval_config

        if self.has_local_cell_embedding:
            cell_model_config_path = Path(self.etl_config["cell_embedding_model"]).resolve()
            cell_etl_config_path = Path(self.etl_config["cell_embedding_etl"]).resolve()
            cell_model_config = yaml.load(open(cell_model_config_path), Loader=yaml.UnsafeLoader)
            cell_etl_config = yaml.load(open(cell_etl_config_path), Loader=yaml.UnsafeLoader)
            self.local_cell_embedding_config = Config(cell_model_config, cell_etl_config, self.datasplit_config)
        else:
            self.local_cell_embedding_config = None

    @staticmethod
    def from_yamls(model_config_path: str, etl_config_path: str, datasplit_config_path: str, eval_config_path: Optional[str] = None) -> 'Config':
        model_config_path = Path(model_config_path).resolve()
        etl_config_path = Path(etl_config_path).resolve()
        datasplit_config_path = Path(datasplit_config_path).resolve()
        eval_config_path = Path(eval_config_path).resolve() if eval_config_path is not None else None

        model_config = yaml.load(open(model_config_path), Loader=yaml.UnsafeLoader)
        etl_config = yaml.load(open(etl_config_path), Loader=yaml.UnsafeLoader)
        datasplit_config = yaml.load(open(datasplit_config_path), Loader=yaml.UnsafeLoader)
        eval_config = yaml.load(open(eval_config_path), Loader=yaml.UnsafeLoader) if eval_config_path is not None else None
        return Config(model_config, etl_config, datasplit_config, eval_config)
    
    def get_train_path(self):
        cell_emb = self.get_cell_embedding_name()
        cell_emb_path = f"{cell_emb}/" if cell_emb is not None else ""
        return Path(f"./models/{self.get_datasplit_config_name()}/{cell_emb_path}{self.get_model_name()}/{self.get_train_hash()}").resolve()
    
    def get_eval_path(self):
        cell_emb = self.get_cell_embedding_name()
        cell_emb_path = f"{cell_emb}/" if cell_emb is not None else ""
        return Path(f"./results/{self.get_datasplit_config_name()}/{cell_emb_path}{self.get_model_name()}/{self.get_train_hash()}/{self.get_eval_hash()}").resolve()

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()
    
    #We completely control the serialization of the config object in this class 
    def to_dict(self):
        serialized = deepcopy(vars(self))
        serialized['local_cell_embedding_config'] = serialized['local_cell_embedding_config'].to_dict() if serialized['local_cell_embedding_config'] is not None else None
        return serialized
    
    def get_train_hash(self):
        train_hash = hashlib.sha256(json.dumps(self.get_training_config().to_dict()).encode()).hexdigest()
        train_hash = train_hash[:8]
        return train_hash
    
    def get_eval_hash(self):
        eval_hash = hashlib.sha256(json.dumps(self.eval_config).encode()).hexdigest()
        eval_hash = eval_hash[:8]
        return eval_hash
    
    """
    Returns a config with only the portions of the config that are relevant for training: etl, datasplit and model
    """
    def get_training_config(self) -> 'Config':
        return deepcopy(Config(self.model_config, self.etl_config, self.datasplit_config))
    
    # GETTERS FOR MODEL
    def get_model_name(self)-> str:
        return self.model_config['name']
    
    def get_model_config(self):
        return deepcopy(self.model_config)

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
        cell_emb_name = self.etl_config.get('cell_embedding', None)
        if self.has_local_cell_embedding:
            cell_emb_name = self.local_cell_embedding_config.get_train_hash()
        return cell_emb_name
    
    @property
    def has_local_cell_embedding(self) -> bool:
        cell_emb_name = self.etl_config.get('cell_embedding', None)
        return cell_emb_name == 'local'
    
    def get_local_cell_embedding_path(self) -> Optional[str]:
        if not self.has_local_cell_embedding:
            return None
        
        return Path(f"{self.local_cell_embedding_config.get_train_path()}/embedded_data.npy").resolve()
    
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
    

    
    

    
    