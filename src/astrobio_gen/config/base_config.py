"""
Base Configuration System for Astrobio-Gen
==========================================

Production-ready configuration management with Hydra integration.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from omegaconf import OmegaConf


@dataclass
class AstroBioConfig:
    """Main configuration class for Astrobio-Gen"""

    # Model configuration
    model_name: str = "enhanced_datacube"
    model_type: str = "enhanced_datacube_unet"
    model_scaling: str = "efficient"

    # Training configuration
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Hardware configuration
    gpus: int = 1
    num_workers: int = 4
    mixed_precision: bool = True
    gradient_checkpointing: bool = False

    # Model features
    use_attention: bool = True
    use_transformer: bool = False
    use_physics_constraints: bool = True
    physics_weight: float = 0.2
    use_separable_conv: bool = True

    # Data configuration
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    output_dir: str = "outputs"

    # Experiment tracking
    experiment_name: str = "astrobio_experiment"
    use_wandb: bool = False
    use_mlflow: bool = False

    # Advanced features
    uncertainty_quantification: bool = False
    multimodal_integration: bool = False
    causal_inference: bool = False
    meta_learning: bool = False

    # Quality assurance
    zero_error_tolerance: bool = True
    real_data_only: bool = True
    production_ready: bool = True


def load_config(config_path: Optional[str] = None) -> AstroBioConfig:
    """Load configuration from file or return default"""

    if config_path is None:
        return AstroBioConfig()

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load based on file extension
    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config_dict = json.load(f)
    else:
        # Try OmegaConf for Hydra configs
        config_dict = OmegaConf.load(config_path)
        config_dict = OmegaConf.to_container(config_dict, resolve=True)

    # Create config object
    return AstroBioConfig(**config_dict)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration as dictionary"""

    return {
        "defaults": [
            "_self_",
            "model: enhanced_datacube",
            "trainer: gpu_light",
            "data: cube_dm",
            "logger: wandb",
            "callbacks: default",
            "hydra: default",
        ],
        "model": {
            "name": "enhanced_datacube",
            "type": "enhanced_datacube_unet",
            "n_input_vars": 5,
            "n_output_vars": 5,
            "base_features": 64,
            "depth": 4,
            "use_attention": True,
            "use_transformer": False,
            "use_physics_constraints": True,
            "physics_weight": 0.2,
            "use_separable_conv": True,
            "use_mixed_precision": True,
            "model_scaling": "efficient",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
        },
        "trainer": {
            "max_epochs": 100,
            "accelerator": "auto",
            "devices": 1,
            "precision": "16-mixed",
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 1,
            "val_check_interval": 1.0,
            "log_every_n_steps": 50,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "enable_model_summary": True,
        },
        "data": {
            "name": "cube_dm",
            "data_dir": "data",
            "cache_dir": "data/cache",
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
        },
        "logger": {
            "wandb": {
                "project": "astrobio-gen",
                "name": "${model.name}_${now:%Y%m%d_%H%M%S}",
                "tags": ["production", "world-class"],
                "notes": "Astrobio-Gen world-class training run",
            }
        },
        "callbacks": {
            "model_checkpoint": {
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 3,
                "save_last": True,
                "filename": "epoch_{epoch:03d}_val_loss_{val_loss:.4f}",
                "auto_insert_metric_name": False,
            },
            "early_stopping": {
                "monitor": "val_loss",
                "mode": "min",
                "patience": 10,
                "min_delta": 1e-4,
            },
            "learning_rate_monitor": {"logging_interval": "step"},
            "rich_progress_bar": {"leave": True},
        },
        "hydra": {
            "version_base": "1.3",
            "run": {"dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"},
            "sweep": {
                "dir": "multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}",
                "subdir": "${hydra.job.num}",
            },
        },
        "experiment": {
            "name": "astrobio_baseline",
            "description": "Baseline astrobiology experiment with enhanced features",
            "tags": ["baseline", "enhanced", "production"],
            "seed": 42,
            "deterministic": False,
        },
    }
