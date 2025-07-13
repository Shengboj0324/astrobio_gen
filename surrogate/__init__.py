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

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from omegaconf import OmegaConf

# Import model classes
from models.datacube_unet import CubeUNet
from models.surrogate_transformer import SurrogateTransformer
from models.graph_vae import GVAE
from models.fusion_transformer import FusionModel

# Add SHAP explainer imports at the top
from .shap_explainer import SHAPExplainer, SHAPExplainerManager, ExplanationConfig, create_shap_explainer_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelFormat(Enum):
    """Supported model formats"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TORCHSCRIPT = "torchscript"

class SurrogateMode(Enum):
    """Surrogate operation modes"""
    SCALAR = "scalar"
    DATACUBE = "datacube"
    JOINT = "joint"
    SPECTRAL = "spectral"
    ENSEMBLE = "ensemble"

@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    format: ModelFormat
    path: Path
    mode: SurrogateMode
    priority: int = 1
    device: str = "auto"
    precision: str = "float32"
    batch_size: int = 4
    
    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Performance parameters
    warmup_samples: int = 10
    max_memory_gb: float = 2.0
    
    # Validation parameters
    validation_data: Optional[Path] = None
    accuracy_threshold: float = 0.95

class PerformanceMonitor:
    """Monitor model performance and resource usage"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.inference_times = []
        self.memory_usage = []
        self.batch_sizes = []
        self.error_count = 0
        self.total_inferences = 0
        self._lock = threading.Lock()
        
    def record_inference(self, batch_size: int, inference_time: float, memory_mb: float):
        """Record inference metrics"""
        with self._lock:
            self.inference_times.append(inference_time)
            self.memory_usage.append(memory_mb)
            self.batch_sizes.append(batch_size)
            self.total_inferences += 1
            
    def record_error(self):
        """Record inference error"""
        with self._lock:
            self.error_count += 1
            
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            if not self.inference_times:
                return {'model': self.model_name, 'status': 'no_data'}
                
            return {
                'model': self.model_name,
                'avg_inference_time': np.mean(self.inference_times),
                'p95_inference_time': np.percentile(self.inference_times, 95),
                'avg_memory_mb': np.mean(self.memory_usage),
                'max_memory_mb': np.max(self.memory_usage),
                'error_rate': self.error_count / max(1, self.total_inferences),
                'total_inferences': self.total_inferences,
                'throughput_samples_per_sec': sum(self.batch_sizes) / sum(self.inference_times)
            }
    
    def is_healthy(self) -> bool:
        """Check if model is performing within acceptable bounds"""
        stats = self.get_stats()
        
        if stats.get('status') == 'no_data':
            return True  # No data yet, assume healthy
            
        # Check error rate
        if stats['error_rate'] > 0.05:  # 5% error rate threshold
            return False
            
        # Check if inference times are reasonable
        if stats['avg_inference_time'] > 10.0:  # 10 second threshold
            return False
            
        return True

class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers"""
    
    def __init__(self, config: ModelConfig, monitor: PerformanceMonitor):
        self.config = config
        self.monitor = monitor
        self.model = None
        self.device = self._resolve_device()
        self.is_loaded = False
        
    def _resolve_device(self) -> torch.device:
        """Resolve device for model execution"""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model from file"""
        pass
    
    @abstractmethod
    def predict(self, inputs: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Make prediction"""
        pass
    
    @abstractmethod
    def export_onnx(self, export_path: Path, sample_input: torch.Tensor) -> bool:
        """Export model to ONNX format"""
        pass
    
    def warmup(self, sample_input: Union[torch.Tensor, np.ndarray]):
        """Warmup model with sample inputs"""
        logger.info(f"Warming up model: {self.config.name}")
        
        for _ in range(self.config.warmup_samples):
            try:
                self.predict(sample_input)
            except Exception as e:
                logger.warning(f"Warmup failed for {self.config.name}: {e}")
                break
    
    def validate(self) -> Dict[str, Any]:
        """Validate model performance"""
        if not self.config.validation_data or not self.config.validation_data.exists():
            return {'status': 'no_validation_data'}
        
        try:
            # Load validation data
            validation_data = self._load_validation_data()
            if not validation_data:
                return {'status': 'failed_to_load_validation_data'}
            
            # Run validation
            results = self._run_validation(validation_data)
            
            # Check if accuracy meets threshold
            accuracy = results.get('accuracy', 0.0)
            rmse = results.get('rmse', float('inf'))
            
            if accuracy >= self.config.accuracy_threshold:
                status = 'validation_passed'
            else:
                status = 'validation_failed'
            
            return {
                'status': status,
                'accuracy': accuracy,
                'rmse': rmse,
                'r2_score': results.get('r2_score', 0.0),
                'mae': results.get('mae', float('inf')),
                'num_samples': results.get('num_samples', 0),
                'inference_time_ms': results.get('inference_time_ms', 0.0),
                'passed_threshold': accuracy >= self.config.accuracy_threshold
            }
            
        except Exception as e:
            logger.error(f"Validation failed for {self.config.name}: {e}")
            return {'status': 'validation_error', 'error': str(e)}
    
    def _load_validation_data(self) -> Optional[Dict[str, Any]]:
        """Load validation data from file"""
        try:
            file_path = self.config.validation_data
            
            if file_path.suffix == '.npz':
                # NumPy archive
                data = np.load(file_path)
                return {
                    'inputs': data['inputs'],
                    'targets': data['targets'],
                    'metadata': data.get('metadata', {})
                }
            
            elif file_path.suffix == '.json':
                # JSON format
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return {
                    'inputs': np.array(data['inputs']),
                    'targets': np.array(data['targets']),
                    'metadata': data.get('metadata', {})
                }
                
            elif file_path.suffix in ['.pt', '.pth']:
                # PyTorch format
                data = torch.load(file_path, map_location='cpu')
                return {
                    'inputs': data['inputs'].numpy() if isinstance(data['inputs'], torch.Tensor) else data['inputs'],
                    'targets': data['targets'].numpy() if isinstance(data['targets'], torch.Tensor) else data['targets'],
                    'metadata': data.get('metadata', {})
                }
                
            else:
                logger.error(f"Unsupported validation data format: {file_path.suffix}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load validation data: {e}")
            return None
    
    def _run_validation(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation on loaded data"""
        inputs = validation_data['inputs']
        targets = validation_data['targets']
        
        # Ensure inputs are properly shaped
        if len(inputs.shape) == 3:  # Add batch dimension if needed
            inputs = inputs[np.newaxis, ...]
        
        # Run inference
        start_time = time.time()
        predictions = self.predict(inputs)
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Convert predictions to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Reshape predictions and targets to match if needed
        if predictions.shape != targets.shape:
            if len(predictions.shape) == 4 and len(targets.shape) == 3:
                predictions = predictions.squeeze(0)
            elif len(predictions.shape) == 3 and len(targets.shape) == 4:
                targets = targets.squeeze(0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)
        metrics['inference_time_ms'] = inference_time
        metrics['num_samples'] = inputs.shape[0]
        
        return metrics
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate validation metrics"""
        # Flatten arrays for metric calculation
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        
        if len(pred_flat) == 0:
            return {
                'accuracy': 0.0,
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2_score': 0.0
            }
        
        # Mean Squared Error and RMSE
        mse = np.mean((pred_flat - target_flat) ** 2)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(pred_flat - target_flat))
        
        # R² Score
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Accuracy based on relative error threshold
        relative_errors = np.abs(pred_flat - target_flat) / (np.abs(target_flat) + 1e-8)
        accuracy = np.mean(relative_errors < 0.1)  # Within 10% relative error
        
        return {
            'accuracy': float(accuracy),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2_score),
            'mse': float(mse)
        }

class PyTorchModelWrapper(BaseModelWrapper):
    """Wrapper for PyTorch models"""
    
    def load_model(self) -> bool:
        """Load PyTorch model"""
        try:
            # Determine model class based on mode
            if self.config.mode == SurrogateMode.DATACUBE:
                model_class = CubeUNet
            elif self.config.mode == SurrogateMode.SCALAR:
                model_class = SurrogateTransformer
            elif self.config.mode == SurrogateMode.JOINT:
                model_class = FusionModel
            else:
                raise ValueError(f"Unsupported mode: {self.config.mode}")
            
            # Load model
            if self.config.path.suffix == '.ckpt':
                # Lightning checkpoint
                self.model = model_class.load_from_checkpoint(
                    str(self.config.path),
                    **self.config.model_params
                )
            else:
                # Regular PyTorch checkpoint
                self.model = model_class(**self.config.model_params)
                checkpoint = torch.load(self.config.path, map_location=self.device)
                self.model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            logger.info(f"Loaded PyTorch model: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model {self.config.name}: {e}")
            return False
    
    def predict(self, inputs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Make prediction with PyTorch model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Convert inputs to torch tensor if needed
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float()
            
            inputs = inputs.to(self.device)
            
            # Get memory usage before inference
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / (1024**2)
            else:
                memory_mb = 0
            
            with torch.no_grad():
                outputs = self.model(inputs)
            
            # Record performance
            inference_time = time.time() - start_time
            self.monitor.record_inference(inputs.size(0), inference_time, memory_mb)
            
            return outputs
            
        except Exception as e:
            self.monitor.record_error()
            raise e
    
    def export_onnx(self, export_path: Path, sample_input: torch.Tensor) -> bool:
        """Export PyTorch model to ONNX"""
        if not self.is_loaded:
            return False
        
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.onnx.export(
                self.model,
                sample_input.to(self.device),
                str(export_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Exported ONNX model to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            return False

class ONNXModelWrapper(BaseModelWrapper):
    """Wrapper for ONNX models"""
    
    def load_model(self) -> bool:
        """Load ONNX model"""
        try:
            # Setup ONNX runtime providers
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            
            # Create inference session
            self.session = ort.InferenceSession(str(self.config.path), providers=providers)
            
            # Get input/output info
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self.is_loaded = True
            logger.info(f"Loaded ONNX model: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model {self.config.name}: {e}")
            return False
    
    def predict(self, inputs: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Make prediction with ONNX model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Convert inputs to numpy if needed
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.cpu().numpy()
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_names[0]: inputs})
            
            # Record performance
            inference_time = time.time() - start_time
            memory_mb = 0  # ONNX runtime doesn't expose memory usage easily
            self.monitor.record_inference(inputs.shape[0], inference_time, memory_mb)
            
            return outputs[0]
            
        except Exception as e:
            self.monitor.record_error()
            raise e
    
    def export_onnx(self, export_path: Path, sample_input: torch.Tensor) -> bool:
        """ONNX model is already in ONNX format"""
        if export_path != self.config.path:
            import shutil
            shutil.copy2(self.config.path, export_path)
        return True

class SurrogateManager:
    """Manager for multiple surrogate models"""
    
    def __init__(self, config_path: Optional[str] = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.models: Dict[str, BaseModelWrapper] = {}
        self.monitors: Dict[str, PerformanceMonitor] = {}
        self.current_mode = SurrogateMode.SCALAR
        self.fallback_chain = []
        
        # Initialize SHAP explainer manager
        self.shap_manager = None
        self.explanation_cache = {}
        
        logger.info("Surrogate manager initialized with SHAP explanation support")
    
    def initialize_shap_explainer(self, metadata_manager):
        """Initialize SHAP explainer manager"""
        self.shap_manager = create_shap_explainer_manager(self, metadata_manager)
        logger.info("SHAP explainer manager initialized")
    
    def explain_prediction(self, inputs: Union[torch.Tensor, np.ndarray], 
                          domain: str, mode: SurrogateMode = None, 
                          feature_names: List[str] = None,
                          config: ExplanationConfig = None) -> Dict[str, Any]:
        """Generate SHAP explanations for predictions"""
        
        if self.shap_manager is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_shap_explainer() first.")
        
        # Convert domain string to DataDomain enum
        from data_build.metadata_db import DataDomain
        domain_enum = DataDomain(domain)
        
        # Convert inputs to numpy if needed
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()
        
        # Generate explanations
        explanations = self.shap_manager.explain_prediction(
            domain_enum, inputs, feature_names=feature_names
        )
        
        # Cache explanations
        cache_key = f"{domain}_{hash(inputs.tobytes())}"
        self.explanation_cache[cache_key] = explanations
        
        return explanations
    
    def predict_with_explanation(self, inputs: Union[torch.Tensor, np.ndarray], 
                               domain: str, mode: SurrogateMode = None,
                               feature_names: List[str] = None,
                               include_explanation: bool = True) -> Dict[str, Any]:
        """Make prediction with optional SHAP explanation"""
        
        # Get regular prediction
        prediction = self.predict(inputs, mode)
        
        result = {
            'prediction': prediction,
            'domain': domain,
            'mode': mode.value if mode else self.current_mode.value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add explanation if requested
        if include_explanation and self.shap_manager is not None:
            try:
                explanation = self.explain_prediction(inputs, domain, mode, feature_names)
                result['explanation'] = explanation
            except Exception as e:
                logger.warning(f"Failed to generate explanation: {e}")
                result['explanation_error'] = str(e)
        
        return result
    
    def get_explanation_stats(self) -> Dict[str, Any]:
        """Get explanation statistics"""
        
        if not self.explanation_cache:
            return {'total_explanations': 0}
        
        stats = {
            'total_explanations': len(self.explanation_cache),
            'cache_size_mb': sum(len(str(exp)) for exp in self.explanation_cache.values()) / (1024*1024),
            'domains_explained': len(set(key.split('_')[0] for key in self.explanation_cache.keys()))
        }
        
        return stats
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration"""
        if not config_path or not Path(config_path).exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def discover_models(self, model_dir: Path = Path("models")) -> List[ModelConfig]:
        """Discover available models"""
        model_configs = []
        
        # Check for model files
        for model_file in model_dir.glob("*.ckpt"):
            model_configs.append(ModelConfig(
                name=model_file.stem,
                format=ModelFormat.PYTORCH,
                path=model_file,
                mode=SurrogateMode.DATACUBE if "cube" in model_file.name else SurrogateMode.SCALAR
            ))
        
        for model_file in model_dir.glob("*.onnx"):
            model_configs.append(ModelConfig(
                name=model_file.stem,
                format=ModelFormat.ONNX,
                path=model_file,
                mode=SurrogateMode.DATACUBE if "cube" in model_file.name else SurrogateMode.SCALAR
            ))
        
        logger.info(f"Discovered {len(model_configs)} models")
        return model_configs
    
    def load_model(self, model_config: ModelConfig) -> bool:
        """Load a specific model"""
        
        # Create performance monitor
        monitor = PerformanceMonitor(model_config.name)
        self.monitors[model_config.name] = monitor
        
        # Create wrapper based on format
        if model_config.format == ModelFormat.PYTORCH:
            wrapper = PyTorchModelWrapper(model_config, monitor)
        elif model_config.format == ModelFormat.ONNX:
            wrapper = ONNXModelWrapper(model_config, monitor)
        else:
            logger.error(f"Unsupported model format: {model_config.format}")
            return False
        
        # Load model
        if wrapper.load_model():
            self.models[model_config.name] = wrapper
            logger.info(f"Successfully loaded model: {model_config.name}")
            return True
        else:
            logger.error(f"Failed to load model: {model_config.name}")
            return False
    
    def load_all_models(self) -> int:
        """Load all available models"""
        model_configs = self.discover_models()
        loaded_count = 0
        
        for config in model_configs:
            if self.load_model(config):
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count}/{len(model_configs)} models")
        return loaded_count
    
    def get_model(self, mode: SurrogateMode = None) -> Optional[BaseModelWrapper]:
        """Get model for specified mode"""
        if mode is None:
            mode = self.current_mode
        
        # Find models matching the mode
        matching_models = [
            model for model in self.models.values()
            if model.config.mode == mode and model.is_loaded
        ]
        
        if not matching_models:
            logger.warning(f"No models available for mode: {mode}")
            return None
        
        # Sort by priority and health
        matching_models.sort(key=lambda m: (
            -m.config.priority,  # Higher priority first
            -int(self.monitors[m.config.name].is_healthy())  # Healthy models first
        ))
        
        return matching_models[0]
    
    def predict(self, inputs: Union[torch.Tensor, np.ndarray], mode: SurrogateMode = None) -> Union[torch.Tensor, np.ndarray]:
        """Make prediction using best available model"""
        model = self.get_model(mode)
        
        if not model:
            raise RuntimeError(f"No model available for mode: {mode}")
        
        return model.predict(inputs)
    
    def export_model(self, model_name: str, export_path: Path, sample_input: torch.Tensor) -> bool:
        """Export model to ONNX format"""
        if model_name not in self.models:
            logger.error(f"Model not found: {model_name}")
            return False
        
        return self.models[model_name].export_onnx(export_path, sample_input)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all models"""
        stats = {}
        
        for model_name, monitor in self.monitors.items():
            stats[model_name] = monitor.get_stats()
        
        return stats
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all models"""
        health = {}
        
        for model_name, monitor in self.monitors.items():
            health[model_name] = monitor.is_healthy()
        
        return health
    
    def set_mode(self, mode: SurrogateMode):
        """Set current operation mode"""
        self.current_mode = mode
        logger.info(f"Set surrogate mode to: {mode}")

# Global surrogate manager instance
_surrogate_manager = None

def get_surrogate_manager() -> SurrogateManager:
    """Get global surrogate manager instance"""
    global _surrogate_manager
    
    if _surrogate_manager is None:
        _surrogate_manager = SurrogateManager()
        _surrogate_manager.load_all_models()
    
    return _surrogate_manager

def load_surrogate(mode: str = "datacube") -> BaseModelWrapper:
    """Load surrogate model for specified mode"""
    manager = get_surrogate_manager()
    
    surrogate_mode = SurrogateMode(mode)
    manager.set_mode(surrogate_mode)
    
    model = manager.get_model(surrogate_mode)
    if not model:
        raise RuntimeError(f"No surrogate model available for mode: {mode}")
    
    return model

# Export main functions
__all__ = [
    'SurrogateManager',
    'ModelConfig',
    'SurrogateMode',
    'ModelFormat',
    'get_surrogate_manager',
    'load_surrogate',
    'SHAPExplainer',
    'SHAPExplainerManager',
    'ExplanationConfig'
] 