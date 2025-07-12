#!/usr/bin/env python3
"""
Surrogate Model Module
=====================

Unified surrogate modeling interface for astrobiology research.
Supports multiple operational modes and model architectures.

Author: AI Assistant
Date: 2025
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type
import torch
import torch.nn as nn

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Configure logging
logger = logging.getLogger(__name__)

# Available models
try:
    from models.surrogate_transformer import SurrogateTransformer
    from models.datacube_unet import CubeUNet
    from models.graph_vae import GVAE
    from models.fusion_transformer import FusionModel
    from models.spectral_surrogate import SpectralSurrogate
    _MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some models not available: {e}")
    _MODELS_AVAILABLE = False

# Available data modules
try:
    from datamodules.cube_dm import CubeDM
    _DATAMODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some data modules not available: {e}")
    _DATAMODULES_AVAILABLE = False

# Mode configuration
MODE_CONFIGS = {
    'scalar': {
        'model_class': 'SurrogateTransformer',
        'mode': 'scalar',
        'description': 'Fast habitability scoring',
        'outputs': ['habitability', 'surface_temp', 'atmospheric_pressure'],
        'inference_time_target': 0.4,  # seconds
        'memory_requirement': '2GB',
        'enabled': True
    },
    'datacube': {
        'model_class': 'CubeUNet',
        'mode': 'datacube',
        'description': 'Full 3D climate fields',
        'outputs': ['temperature_field', 'humidity_field', 'pressure_field'],
        'inference_time_target': 5.0,  # seconds
        'memory_requirement': '8GB',
        'enabled': True  # Enable datacube mode
    },
    'joint': {
        'model_class': 'SurrogateTransformer',
        'mode': 'joint',
        'description': 'Multi-planet-type modeling',
        'outputs': ['planet_type', 'habitability', 'spectral_features'],
        'inference_time_target': 1.0,  # seconds
        'memory_requirement': '4GB',
        'enabled': False  # Disable until trained
    },
    'spectral': {
        'model_class': 'SpectralSurrogate',
        'mode': 'spectral',
        'description': 'High-resolution spectrum synthesis',
        'outputs': ['spectrum'],
        'inference_time_target': 2.0,  # seconds
        'memory_requirement': '6GB',
        'enabled': False  # Disable until trained
    }
}

# Feature flags
FEATURE_FLAGS = {
    'enable_datacube': True,
    'enable_physics_constraints': True,
    'enable_uncertainty_quantification': True,
    'enable_validation': True,
    'enable_monitoring': True,
    'enable_gpu_acceleration': torch.cuda.is_available(),
    'enable_mixed_precision': True,
    'enable_distributed_training': False,
    'enable_model_checkpointing': True,
    'enable_logging': True
}

def get_surrogate_model(
    mode: str = "scalar",
    args: Optional[Dict[str, Any]] = None,
    **kwargs
) -> nn.Module:
    """
    Get surrogate model based on mode and feature flags
    
    Args:
        mode: Model mode ('scalar', 'datacube', 'joint', 'spectral')
        args: Additional arguments for model initialization
        **kwargs: Additional keyword arguments
        
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If mode is not supported or disabled
        ImportError: If required model is not available
    """
    if not _MODELS_AVAILABLE:
        raise ImportError("Surrogate models are not available. Please install required dependencies.")
    
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}. Available modes: {list(MODE_CONFIGS.keys())}")
    
    config = MODE_CONFIGS[mode]
    
    # Check if mode is enabled
    if not config['enabled']:
        raise ValueError(f"Mode '{mode}' is currently disabled. Check configuration.")
    
    # Check specific feature flags
    if mode == 'datacube' and not FEATURE_FLAGS['enable_datacube']:
        raise ValueError("Datacube mode is disabled. Set enable_datacube=True to enable.")
    
    # Initialize arguments
    model_args = args or {}
    model_args.update(kwargs)
    
    # Select model class
    if mode == 'scalar':
        model = SurrogateTransformer(mode='scalar', **model_args)
    elif mode == 'datacube':
        model = CubeUNet(**model_args)
    elif mode == 'joint':
        model = SurrogateTransformer(mode='joint', **model_args)
    elif mode == 'spectral':
        model = SpectralSurrogate(**model_args)
    else:
        raise ValueError(f"Mode '{mode}' not implemented")
    
    logger.info(f"Initialized {config['model_class']} in {mode} mode")
    return model

def get_data_module(
    mode: str = "scalar",
    args: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Optional[Any]:
    """
    Get data module based on mode
    
    Args:
        mode: Model mode
        args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Data module instance or None if not available
    """
    if not _DATAMODULES_AVAILABLE:
        logger.warning("Data modules are not available")
        return None
    
    if mode == 'datacube':
        return CubeDM(**(args or {}), **kwargs)
    else:
        logger.warning(f"No specific data module for mode '{mode}', using default")
        return None

def get_available_modes() -> Dict[str, Dict[str, Any]]:
    """
    Get available modes and their configurations
    
    Returns:
        Dictionary of mode configurations
    """
    return {
        mode: config for mode, config in MODE_CONFIGS.items()
        if config['enabled']
    }

def check_mode_requirements(mode: str) -> Dict[str, bool]:
    """
    Check if requirements for a specific mode are met
    
    Args:
        mode: Mode to check
        
    Returns:
        Dictionary of requirement checks
    """
    if mode not in MODE_CONFIGS:
        return {'valid_mode': False}
    
    config = MODE_CONFIGS[mode]
    checks = {
        'valid_mode': True,
        'enabled': config['enabled'],
        'models_available': _MODELS_AVAILABLE,
        'datamodules_available': _DATAMODULES_AVAILABLE,
        'gpu_available': torch.cuda.is_available(),
        'memory_adequate': True  # Would need actual memory check
    }
    
    # Mode-specific checks
    if mode == 'datacube':
        checks['datacube_enabled'] = FEATURE_FLAGS['enable_datacube']
    
    return checks

def configure_mode(mode: str, enabled: bool = True) -> None:
    """
    Configure mode availability
    
    Args:
        mode: Mode to configure
        enabled: Whether to enable the mode
    """
    if mode in MODE_CONFIGS:
        MODE_CONFIGS[mode]['enabled'] = enabled
        logger.info(f"Mode '{mode}' {'enabled' if enabled else 'disabled'}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

def set_feature_flag(flag: str, value: bool) -> None:
    """
    Set feature flag value
    
    Args:
        flag: Feature flag name
        value: Flag value
    """
    if flag in FEATURE_FLAGS:
        FEATURE_FLAGS[flag] = value
        logger.info(f"Feature flag '{flag}' set to {value}")
    else:
        raise ValueError(f"Unknown feature flag: {flag}")

def get_system_info() -> Dict[str, Any]:
    """
    Get system information and capabilities
    
    Returns:
        System information dictionary
    """
    info = {
        'modes_available': list(MODE_CONFIGS.keys()),
        'modes_enabled': [mode for mode, config in MODE_CONFIGS.items() if config['enabled']],
        'feature_flags': FEATURE_FLAGS.copy(),
        'system': {
            'models_available': _MODELS_AVAILABLE,
            'datamodules_available': _DATAMODULES_AVAILABLE,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }
    
    return info

# Convenience exports
__all__ = [
    'get_surrogate_model',
    'get_data_module',
    'get_available_modes',
    'check_mode_requirements',
    'configure_mode',
    'set_feature_flag',
    'get_system_info',
    'MODE_CONFIGS',
    'FEATURE_FLAGS'
]

# Log initialization
logger.info("Surrogate module initialized")
logger.info(f"Available modes: {list(get_available_modes().keys())}")
logger.info(f"Datacube mode: {'enabled' if FEATURE_FLAGS['enable_datacube'] else 'disabled'}")

# Example usage demonstration
if __name__ == "__main__":
    print("=== Surrogate Module Demo ===")
    print(f"Available modes: {list(get_available_modes().keys())}")
    print(f"System info: {get_system_info()}")
    
    # Test mode selection
    for mode in ['scalar', 'datacube']:
        if MODE_CONFIGS[mode]['enabled']:
            try:
                model = get_surrogate_model(mode)
                print(f"✓ {mode} mode: {type(model).__name__}")
            except Exception as e:
                print(f"✗ {mode} mode: {e}")
        else:
            print(f"- {mode} mode: disabled") 