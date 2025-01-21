from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import yaml
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the model."""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ModelConfig':
        with open(path) as f:
            data = yaml.safe_load(f)
            name = data.pop('name')
            return cls(name=name, parameters=data)

@dataclass
class ETLConfig:
    """Configuration for ETL process."""
    name: str
    count_norm: bool = False
    log1p: bool = False
    drop_unmatched_perts: bool = False
    metric_space: Optional[str] = None
    HVG: bool = False
    gene_embedding: Optional[str] = None
    pert_embedding: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> 'ETLConfig':
        with open(path) as f:
            return cls(**yaml.safe_load(f))

@dataclass
class DatasplitConfig:
    """Configuration for data splitting."""
    dataset: str
    name: str
    mode: str
    holdout_cells: List[str] = field(default_factory=list)
    holdout_perts: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> 'DatasplitConfig':
        with open(path) as f:
            return cls(**yaml.safe_load(f))

@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    name: str
    dataset: str
    evaluation_targets: List[Tuple[str, str]]

    @classmethod
    def from_yaml(cls, path: str) -> 'EvalConfig':
        with open(path) as f:
            return cls(**yaml.safe_load(f))

@dataclass
class Config:
    """Configuration for a model or task."""
    model_config: ModelConfig
    etl_config: ETLConfig
    datasplit_config: DatasplitConfig
    eval_config: Optional[EvalConfig] = None

    @classmethod
    def from_yamls(cls, 
                  model_yaml: str,
                  etl_yaml: str, 
                  datasplit_yaml: str,
                  eval_yaml: Optional[str] = None) -> 'Config':
        """Load config from separate YAML files."""
        config = cls(
            model_config=ModelConfig.from_yaml(model_yaml),
            etl_config=ETLConfig.from_yaml(etl_yaml),
            datasplit_config=DatasplitConfig.from_yaml(datasplit_yaml),
            eval_config=EvalConfig.from_yaml(eval_yaml) if eval_yaml else None
        )
        logger.info(f"Loaded config: {config}")
        return config

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create config from a dictionary."""
        return cls(
            model_config=ModelConfig(**config_dict['model_config']),
            etl_config=ETLConfig(**config_dict['etl_config']),
            datasplit_config=DatasplitConfig(**config_dict['datasplit_config']),
            eval_config=EvalConfig(**config_dict['eval_config']) if 'eval_config' in config_dict else None
        )

    def get_train_path(self) -> Path:
        """Get path for training artifacts."""
        return Path(
            f"./models/{self.datasplit_config.dataset}"
            f"/{self.etl_config.name}"
            f"/{self.model_config.name}"
            f"/{self.datasplit_config.name}"
            f"/{self.get_train_hash()}"
        ).resolve()

    def get_train_hash(self) -> str:
        """Get hash of training configuration."""
        config_dict = {
            'model_config': self.model_config.__dict__,
            'etl_config': self.etl_config.__dict__,
            'datasplit_config': self.datasplit_config.__dict__
        }
        return hashlib.sha256(json.dumps(config_dict).encode()).hexdigest()[:8]

    # [Rest of the methods remain the same, but use the new structured config objects]
    def get_train_path(self) -> Path:
        """Get the path for training artifacts."""
        datasplit_prefix = self._get_datasplit_prefix()
        datasplit_prefix_path = f"{datasplit_prefix}/" if datasplit_prefix else ""
        
        return Path(
            f"./models/{self.get_training_dataset_name()}"
            f"/{self.get_etl_config_name()}"
            f"/{self.get_model_name()}"
            f"/{datasplit_prefix_path}{self.get_datasplit_config_name()}"
            f"/{self.get_train_hash()}"
        ).resolve()

    def get_eval_path(self) -> Path:
        """Get the path for evaluation artifacts."""
        datasplit_prefix = self._get_datasplit_prefix()
        datasplit_prefix_path = f"{datasplit_prefix}/" if datasplit_prefix else ""
        
        train_and_eval_hash = hashlib.sha256(
            f"{self.get_train_hash()}/{self.get_eval_hash()}".encode()
        ).hexdigest()[:8]
        
        return Path(
            f"./results/{self.get_training_dataset_name()}"
            f"/{self.get_etl_config_name()}"
            f"/{self.get_model_name()}"
            f"/{datasplit_prefix_path}{self.get_datasplit_config_name()}"
            f"/{train_and_eval_hash}"
        ).resolve()

    def _get_datasplit_prefix(self) -> Optional[str]:
        """Helper method to get datasplit prefix."""
        datasplit_name_parts = self.get_datasplit_config_name().split('-')
        return '-'.join(datasplit_name_parts[0:-1]) if len(datasplit_name_parts) > 1 else None

    def get_train_hash(self) -> str:
        """Get hash of training configuration."""
        return hashlib.sha256(
            json.dumps(self.get_training_config().to_dict()).encode()
        ).hexdigest()[:8]
    
    def get_eval_hash(self) -> str:
        """Get hash of evaluation configuration."""
        if not self.eval_config:
            raise ValueError("No evaluation config present")
        return hashlib.sha256(
            json.dumps(self.eval_config.to_dict()).encode()
        ).hexdigest()[:8]

    def get_training_config(self) -> 'Config':
        """Get a new Config instance with only training-related configurations."""
        return Config(
            model_config=self.model_config,
            etl_config=self.etl_config,
            datasplit_config=self.datasplit_config
        )

    # Model config getters
    def get_model_name(self) -> str:
        return self.model_config.name

    def get_model_config(self) -> dict:
        return self.model_config.to_dict()

    # Training config getters
    def get_training_dataset_name(self) -> str:
        return self.datasplit_config.dataset

    def get_datasplit_config_name(self) -> str:
        return self.datasplit_config.name

    def get_mode(self) -> str:
        return self.datasplit_config.mode

    def get_heldout_cells(self) -> List[str]:
        return self.datasplit_config.holdout_cells

    def get_heldout_perts(self) -> List[str]:
        return self.datasplit_config.holdout_perts

    # ETL config getters
    def get_drop_unmatched_perts(self) -> bool:
        return self.etl_config.drop_unmatched_perts

    def get_etl_config_name(self) -> str:
        return self.etl_config.name

    def get_apply_normalization(self) -> bool:
        return self.etl_config.count_norm

    def get_apply_log1p(self) -> bool:
        return self.etl_config.log1p

    def get_metric_space(self) -> Optional[str]:
        return self.etl_config.metric_space

    def get_HVG(self) -> bool:
        return self.etl_config.HVG

    def get_gene_embedding_name(self) -> Optional[str]:
        return self.etl_config.gene_embedding

    def get_pert_embedding_name(self) -> Optional[str]:
        return self.etl_config.pert_embedding

    @property
    def has_local_cell_embedding(self) -> bool:
        return self.etl_config.cell_embedding_type == 'local'

    # Eval config getters
    def get_eval_config_name(self) -> str:
        if not self.eval_config:
            raise ValueError("No evaluation config present")
        return self.eval_config.name

    def get_eval_dataset_name(self) -> str:
        if not self.eval_config:
            raise ValueError("No evaluation config present")
        return self.eval_config.dataset

    def get_eval_targets(self) -> List[Tuple[str, str]]:
        if not self.eval_config:
            raise ValueError("No evaluation config present")
        return self.eval_config.evaluation_targets