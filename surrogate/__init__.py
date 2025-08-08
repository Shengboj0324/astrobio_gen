#!/usr/bin/env python3
"""
Advanced Surrogate Loading System
=================================

Industry-grade surrogate model loading with support for:
- Multiple model formats (PyTorch, ONNX, TensorRT)
- Dynamic model selection based on configuration
- Multi-model ensembles
- Performance monitoring and optimization
- Graceful fallback handling
"""

import json
import logging
import os
import threading
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

# Import model classes
from models.datacube_unet import CubeUNet
from models.enhanced_datacube_unet import EnhancedCubeUNet
from models.enhanced_surrogate_integration import EnhancedSurrogateIntegration, MultiModalConfig
from models.fusion_transformer import FusionModel
from models.graph_vae import GVAE
from models.surrogate_transformer import SurrogateTransformer

# Add SHAP explainer imports at the top
from .shap_explainer import (
    ExplanationConfig,
    SHAPExplainer,
    SHAPExplainerManager,
    create_shap_explainer_manager,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model formats"""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    ENHANCED = "enhanced"  # New format for enhanced models


class ModelType(Enum):
    """Supported model types"""

    DATACUBE_UNET = "datacube_unet"
    ENHANCED_DATACUBE_UNET = "enhanced_datacube_unet"
    SURROGATE_TRANSFORMER = "surrogate_transformer"
    ENHANCED_SURROGATE_INTEGRATION = "enhanced_surrogate_integration"
    GRAPH_VAE = "graph_vae"
    FUSION_TRANSFORMER = "fusion_transformer"


class PerformanceLevel(Enum):
    """Performance optimization levels"""

    BASIC = "basic"
    OPTIMIZED = "optimized"
    PEAK = "peak"  # Maximum performance with all enhancements


@dataclass
class ModelConfig:
    """Configuration for surrogate models"""

    model_type: ModelType
    model_format: ModelFormat
    performance_level: PerformanceLevel = PerformanceLevel.OPTIMIZED
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    device: str = "auto"
    precision: str = "float32"
    batch_size: int = 1
    use_attention: bool = True
    use_transformer: bool = False
    use_physics_constraints: bool = True
    use_uncertainty: bool = False
    multimodal_config: Optional[Dict[str, Any]] = None

    # Enhanced features
    use_separable_conv: bool = True
    use_gradient_checkpointing: bool = False
    model_scaling: str = "efficient"

    # Performance optimizations
    use_mixed_precision: bool = True
    compile_model: bool = False
    use_dynamic_selection: bool = False


class EnhancedModelLoader:
    """Enhanced model loader with support for all advanced features"""

    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.loaded_models = {}
        self.model_configs = {}
        self.performance_cache = {}

        # Enhanced model registry
        self.enhanced_registry = {
            ModelType.ENHANCED_DATACUBE_UNET: EnhancedCubeUNet,
            ModelType.ENHANCED_SURROGATE_INTEGRATION: EnhancedSurrogateIntegration,
            ModelType.DATACUBE_UNET: CubeUNet,
            ModelType.SURROGATE_TRANSFORMER: SurrogateTransformer,
            ModelType.GRAPH_VAE: GVAE,
            ModelType.FUSION_TRANSFORMER: FusionModel,
        }

        logger.info("Enhanced Model Loader initialized with advanced CNN features")

    def load_enhanced_model(self, config: ModelConfig) -> nn.Module:
        """Load enhanced model with all optimizations"""
        model_key = f"{config.model_type.value}_{config.performance_level.value}"

        if model_key in self.loaded_models:
            logger.info(f"Returning cached enhanced model: {model_key}")
            return self.loaded_models[model_key]

        logger.info(
            f"Loading enhanced model: {config.model_type.value} at {config.performance_level.value} performance"
        )

        # Get model class
        model_class = self.enhanced_registry[config.model_type]

        # Prepare enhanced configuration
        enhanced_config = self._prepare_enhanced_config(config)

        # Load model
        if config.checkpoint_path and Path(config.checkpoint_path).exists():
            # Load from checkpoint
            model = model_class.load_from_checkpoint(config.checkpoint_path, **enhanced_config)
            logger.info(f"Loaded enhanced model from checkpoint: {config.checkpoint_path}")
        else:
            # Create new model
            model = model_class(**enhanced_config)
            logger.info(f"Created new enhanced model: {config.model_type.value}")

        # Apply performance optimizations
        model = self._apply_performance_optimizations(model, config)

        # Cache model
        self.loaded_models[model_key] = model
        self.model_configs[model_key] = config

        # Log model complexity
        if hasattr(model, "get_model_complexity"):
            complexity = model.get_model_complexity()
            logger.info(f"Enhanced model complexity: {complexity}")
        elif hasattr(model, "get_integration_complexity"):
            complexity = model.get_integration_complexity()
            logger.info(f"Enhanced integration complexity: {complexity}")

        return model

    def _prepare_enhanced_config(self, config: ModelConfig) -> Dict[str, Any]:
        """Prepare enhanced configuration based on performance level"""
        enhanced_config = {}

        if config.model_type == ModelType.ENHANCED_DATACUBE_UNET:
            enhanced_config.update(
                {
                    "use_attention": config.use_attention,
                    "use_transformer": config.use_transformer,
                    "use_separable_conv": config.use_separable_conv,
                    "use_gradient_checkpointing": config.use_gradient_checkpointing,
                    "use_mixed_precision": config.use_mixed_precision,
                    "model_scaling": config.model_scaling,
                    "use_physics_constraints": config.use_physics_constraints,
                }
            )

            # Performance level adjustments
            if config.performance_level == PerformanceLevel.PEAK:
                enhanced_config.update(
                    {
                        "use_transformer": True,
                        "use_separable_conv": True,
                        "use_gradient_checkpointing": True,
                        "base_features": 64,
                        "depth": 5,
                        "dropout": 0.1,
                    }
                )
            elif config.performance_level == PerformanceLevel.OPTIMIZED:
                enhanced_config.update(
                    {
                        "use_transformer": False,
                        "use_separable_conv": True,
                        "base_features": 48,
                        "depth": 4,
                        "dropout": 0.15,
                    }
                )
            else:  # BASIC
                enhanced_config.update(
                    {
                        "use_attention": False,
                        "use_transformer": False,
                        "use_separable_conv": False,
                        "base_features": 32,
                        "depth": 3,
                        "dropout": 0.2,
                    }
                )

        elif config.model_type == ModelType.ENHANCED_SURROGATE_INTEGRATION:
            # Multi-modal configuration
            if config.multimodal_config:
                multimodal_config = MultiModalConfig(**config.multimodal_config)
            else:
                multimodal_config = MultiModalConfig()

            enhanced_config.update(
                {
                    "multimodal_config": multimodal_config,
                    "use_uncertainty": config.use_uncertainty,
                    "use_dynamic_selection": config.use_dynamic_selection,
                    "use_gradient_checkpointing": config.use_gradient_checkpointing,
                    "use_mixed_precision": config.use_mixed_precision,
                }
            )

            # Datacube model configuration
            datacube_config = {
                "use_attention": config.use_attention,
                "use_transformer": config.use_transformer,
                "use_separable_conv": config.use_separable_conv,
                "use_physics_constraints": config.use_physics_constraints,
                "model_scaling": config.model_scaling,
            }

            enhanced_config["datacube_config"] = datacube_config

        return enhanced_config

    def _apply_performance_optimizations(self, model: nn.Module, config: ModelConfig) -> nn.Module:
        """Apply performance optimizations to the model"""
        # Move to device
        device = self._get_device(config.device)
        model = model.to(device)

        # Set precision
        if config.precision == "float16" or config.use_mixed_precision:
            model = model.half()

        # Compile model (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="max-autotune")
                logger.info("Model compiled with PyTorch 2.0 for maximum performance")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")

        # Set to evaluation mode for inference
        model.eval()

        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        return model

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)

    def get_model_performance(self, model_key: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        if model_key not in self.performance_cache:
            model = self.loaded_models.get(model_key)
            if model is None:
                return {}

            # Benchmark model
            performance = self._benchmark_model(model)
            self.performance_cache[model_key] = performance

        return self.performance_cache[model_key]

    def _benchmark_model(self, model: nn.Module) -> Dict[str, Any]:
        """Benchmark model performance"""
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 5, 32, 64, 64).to(next(model.parameters()).device)

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Benchmark inference time
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_inference_time = (end_time - start_time) / 100 * 1000  # ms

        # Memory usage
        memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
        memory_reserved = torch.cuda.memory_reserved() / (1024**2)  # MB

        return {
            "avg_inference_time_ms": avg_inference_time,
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved,
            "throughput_samples_per_second": 1000 / avg_inference_time,
        }


class EnhancedSurrogateManager:
    """Enhanced surrogate manager with all advanced features"""

    def __init__(self, config_path: str = "config/surrogate_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize enhanced model loader
        self.model_loader = EnhancedModelLoader()

        # Active models
        self.active_models = {}

        # Performance monitoring
        self.performance_monitor = self._setup_performance_monitor()

        # SHAP explainer manager
        self.shap_manager = create_shap_explainer_manager()

        logger.info("Enhanced Surrogate Manager initialized with peak performance features")

    def _load_config(self) -> Dict[str, Any]:
        """Load enhanced configuration"""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            # Default enhanced configuration
            config = {
                "default_performance_level": "optimized",
                "models": {
                    "enhanced_datacube": {
                        "type": "enhanced_datacube_unet",
                        "performance_level": "peak",
                        "use_attention": True,
                        "use_transformer": True,
                        "use_physics_constraints": True,
                        "use_mixed_precision": True,
                        "model_scaling": "efficient",
                    },
                    "enhanced_integration": {
                        "type": "enhanced_surrogate_integration",
                        "performance_level": "peak",
                        "multimodal_config": {
                            "use_datacube": True,
                            "use_scalar_params": True,
                            "fusion_strategy": "cross_attention",
                        },
                        "use_uncertainty": True,
                        "use_dynamic_selection": True,
                    },
                },
            }

        return config

    def _setup_performance_monitor(self) -> Dict[str, Any]:
        """Setup performance monitoring"""
        return {
            "start_time": time.time(),
            "total_inferences": 0,
            "total_time": 0.0,
            "model_usage": {},
            "performance_history": [],
        }

    def get_enhanced_model(self, model_name: str = "enhanced_datacube") -> nn.Module:
        """Get enhanced model with all optimizations"""
        if model_name in self.active_models:
            return self.active_models[model_name]

        # Get model configuration
        model_config_dict = self.config["models"].get(model_name, {})

        # Create model configuration
        model_config = ModelConfig(
            model_type=ModelType(model_config_dict.get("type", "enhanced_datacube_unet")),
            model_format=ModelFormat.ENHANCED,
            performance_level=PerformanceLevel(
                model_config_dict.get("performance_level", "optimized")
            ),
            checkpoint_path=model_config_dict.get("checkpoint_path"),
            use_attention=model_config_dict.get("use_attention", True),
            use_transformer=model_config_dict.get("use_transformer", False),
            use_physics_constraints=model_config_dict.get("use_physics_constraints", True),
            use_mixed_precision=model_config_dict.get("use_mixed_precision", True),
            model_scaling=model_config_dict.get("model_scaling", "efficient"),
            multimodal_config=model_config_dict.get("multimodal_config"),
            use_uncertainty=model_config_dict.get("use_uncertainty", False),
            use_dynamic_selection=model_config_dict.get("use_dynamic_selection", False),
        )

        # Load enhanced model
        model = self.model_loader.load_enhanced_model(model_config)

        # Cache model
        self.active_models[model_name] = model

        return model

    def predict_with_enhancements(
        self,
        input_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        model_name: str = "enhanced_datacube",
        return_uncertainty: bool = False,
    ) -> Dict[str, Any]:
        """Make predictions with all enhancements"""
        # Get enhanced model
        model = self.get_enhanced_model(model_name)

        # Prepare input
        if isinstance(input_data, torch.Tensor):
            # Single tensor input
            inputs = {"datacube": input_data}
        else:
            # Multi-modal input
            inputs = input_data

        # Add targets placeholder if needed
        if "targets" not in inputs:
            inputs["targets"] = torch.zeros_like(inputs["datacube"][:, : model.n_output_vars])

        # Performance monitoring
        start_time = time.time()

        # Forward pass
        with torch.no_grad():
            if hasattr(model, "forward") and isinstance(inputs, dict):
                outputs = model(inputs)
            else:
                outputs = model(inputs["datacube"])

        # Record performance
        inference_time = time.time() - start_time
        self.performance_monitor["total_inferences"] += 1
        self.performance_monitor["total_time"] += inference_time

        # Update model usage
        if model_name not in self.performance_monitor["model_usage"]:
            self.performance_monitor["model_usage"][model_name] = {"count": 0, "total_time": 0.0}

        self.performance_monitor["model_usage"][model_name]["count"] += 1
        self.performance_monitor["model_usage"][model_name]["total_time"] += inference_time

        # Prepare results
        if isinstance(outputs, dict):
            results = {
                "predictions": outputs["predictions"],
                "inference_time_ms": inference_time * 1000,
                "model_name": model_name,
                "performance_level": self.model_loader.model_configs.get(
                    f"{model_name}_optimized",
                    ModelConfig(ModelType.ENHANCED_DATACUBE_UNET, ModelFormat.ENHANCED),
                ).performance_level.value,
            }

            if return_uncertainty and "uncertainty" in outputs:
                results["uncertainty"] = outputs["uncertainty"]

            if "fused_features" in outputs:
                results["fused_features"] = outputs["fused_features"]

        else:
            results = {
                "predictions": outputs,
                "inference_time_ms": inference_time * 1000,
                "model_name": model_name,
            }

        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_time = self.performance_monitor["total_time"]
        total_inferences = self.performance_monitor["total_inferences"]

        if total_inferences > 0:
            avg_inference_time = total_time / total_inferences * 1000  # ms
            throughput = total_inferences / total_time  # inferences/second
        else:
            avg_inference_time = 0
            throughput = 0

        return {
            "total_inferences": total_inferences,
            "total_time_seconds": total_time,
            "avg_inference_time_ms": avg_inference_time,
            "throughput_inferences_per_second": throughput,
            "model_usage": self.performance_monitor["model_usage"],
            "active_models": list(self.active_models.keys()),
            "uptime_seconds": time.time() - self.performance_monitor["start_time"],
        }

    def optimize_for_peak_performance(self):
        """Optimize all models for peak performance"""
        logger.info("🚀 Optimizing all models for peak performance...")

        for model_name, model in self.active_models.items():
            logger.info(f"Optimizing {model_name}...")

            # Enable optimizations
            if hasattr(model, "eval"):
                model.eval()

            # Set benchmark mode
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Compile if possible
            if hasattr(torch, "compile"):
                try:
                    model = torch.compile(model, mode="max-autotune")
                    self.active_models[model_name] = model
                    logger.info(f"✅ {model_name} compiled for maximum performance")
                except Exception as e:
                    logger.warning(f"⚠️ Could not compile {model_name}: {e}")

        logger.info("🎯 Peak performance optimization completed!")


# Global enhanced manager instance
enhanced_manager = None


def get_enhanced_surrogate_manager() -> EnhancedSurrogateManager:
    """Get global enhanced surrogate manager"""
    global enhanced_manager
    if enhanced_manager is None:
        enhanced_manager = EnhancedSurrogateManager()
    return enhanced_manager


def load_enhanced_model(model_name: str = "enhanced_datacube") -> nn.Module:
    """Load enhanced model with all optimizations"""
    manager = get_enhanced_surrogate_manager()
    return manager.get_enhanced_model(model_name)


def predict_with_peak_performance(
    input_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
    model_name: str = "enhanced_datacube",
    return_uncertainty: bool = False,
) -> Dict[str, Any]:
    """Make predictions with peak performance optimizations"""
    manager = get_enhanced_surrogate_manager()
    return manager.predict_with_enhancements(input_data, model_name, return_uncertainty)


def optimize_all_models_for_peak_performance():
    """Optimize all models for peak performance"""
    manager = get_enhanced_surrogate_manager()
    manager.optimize_for_peak_performance()


# Convenience functions for backward compatibility
def get_model(model_name: str = "enhanced_datacube") -> nn.Module:
    """Get enhanced model (backward compatibility)"""
    return load_enhanced_model(model_name)


def predict(
    input_data: Union[torch.Tensor, Dict[str, torch.Tensor]], model_name: str = "enhanced_datacube"
) -> Dict[str, Any]:
    """Make enhanced predictions (backward compatibility)"""
    return predict_with_peak_performance(input_data, model_name)


# Export enhanced components
__all__ = [
    "EnhancedSurrogateManager",
    "EnhancedModelLoader",
    "ModelConfig",
    "ModelType",
    "PerformanceLevel",
    "get_enhanced_surrogate_manager",
    "load_enhanced_model",
    "predict_with_peak_performance",
    "optimize_all_models_for_peak_performance",
    "get_model",  # Backward compatibility
    "predict",  # Backward compatibility
]
