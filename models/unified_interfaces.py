#!/usr/bin/env python3
"""
Unified Interfaces for All Neural Network Components
===================================================

Standard interfaces and protocols for all rebuilt neural network components:
- Common base classes and protocols
- Standardized input/output formats
- Unified configuration system
- Consistent error handling
- Proper logging and monitoring
- Device and memory management utilities

Version: 2.0.0 (Production Ready)
Compatible with: All rebuilt components
"""

import abc
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Protocol

import torch
import torch.nn as nn
import pytorch_lightning as pl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of model types"""
    CNN = "cnn"
    GRAPH_VAE = "graph_vae"
    LLM = "llm"
    GALACTIC_NETWORK = "galactic_network"
    MULTIMODAL = "multimodal"


class DataModality(Enum):
    """Enumeration of data modalities"""
    DATACUBE = "datacube"
    GRAPH = "graph"
    TEXT = "text"
    SPECTRAL = "spectral"
    OBSERVATORY = "observatory"


@dataclass
class ModelMetadata:
    """Metadata for model identification and versioning"""
    name: str
    version: str
    model_type: ModelType
    supported_modalities: List[DataModality]
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    device: str = "auto"
    precision: str = "float32"
    memory_requirements_mb: int = 1000


@dataclass
class ValidationResult:
    """Result of input/output validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class TensorValidator:
    """Unified tensor validation utilities"""
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[torch.dtype] = None,
        expected_device: Optional[torch.device] = None,
        name: str = "tensor"
    ) -> ValidationResult:
        """Validate tensor properties"""
        
        errors = []
        warnings = []
        
        # Check if it's actually a tensor
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"{name} must be a torch.Tensor, got {type(tensor)}")
            return ValidationResult(False, errors, warnings, {})
        
        # Check shape
        if expected_shape is not None:
            if len(expected_shape) != tensor.dim():
                errors.append(f"{name} expected {len(expected_shape)}D, got {tensor.dim()}D")
            else:
                for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
                    if expected != -1 and actual != expected:
                        errors.append(f"{name} dimension {i} expected {expected}, got {actual}")
        
        # Check dtype
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            warnings.append(f"{name} dtype {tensor.dtype} != expected {expected_dtype}")
        
        # Check device
        if expected_device is not None and tensor.device != expected_device:
            warnings.append(f"{name} device {tensor.device} != expected {expected_device}")
        
        # Check for NaN or Inf
        if torch.isnan(tensor).any():
            errors.append(f"{name} contains NaN values")
        
        if torch.isinf(tensor).any():
            errors.append(f"{name} contains Inf values")
        
        metadata = {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "device": tensor.device,
            "requires_grad": tensor.requires_grad,
            "memory_mb": tensor.numel() * tensor.element_size() / 1024**2
        }
        
        return ValidationResult(len(errors) == 0, errors, warnings, metadata)
    
    @staticmethod
    def ensure_compatible_devices(*tensors: torch.Tensor) -> List[torch.Tensor]:
        """Ensure all tensors are on the same device"""
        if not tensors:
            return []
        
        target_device = tensors[0].device
        return [t.to(target_device) if t.device != target_device else t for t in tensors]
    
    @staticmethod
    def ensure_compatible_dtypes(*tensors: torch.Tensor, target_dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
        """Ensure all tensors have compatible dtypes"""
        return [t.to(target_dtype) if t.dtype != target_dtype else t for t in tensors]


class BaseNeuralNetworkProtocol(Protocol):
    """Protocol defining the interface for all neural network components"""
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        ...
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata"""
        ...
    
    def validate_input(self, batch: Dict[str, torch.Tensor]) -> ValidationResult:
        """Validate input batch"""
        ...
    
    def validate_output(self, output: Dict[str, torch.Tensor]) -> ValidationResult:
        """Validate output"""
        ...


class BaseNeuralNetwork(pl.LightningModule, abc.ABC):
    """
    Abstract base class for all neural network components
    
    Provides:
    - Common interface implementation
    - Standardized validation
    - Unified error handling
    - Consistent logging
    - Memory management utilities
    """
    
    def __init__(self, metadata: ModelMetadata):
        super().__init__()
        self.metadata = metadata
        self.validator = TensorValidator()
        
        # Setup logging
        self.logger_instance = logging.getLogger(f"{self.__class__.__name__}")
        
    @abc.abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model - must be implemented by subclasses"""
        pass
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata"""
        return self.metadata
    
    def validate_input(self, batch: Dict[str, torch.Tensor]) -> ValidationResult:
        """Validate input batch"""
        errors = []
        warnings = []
        metadata = {}
        
        # Check if batch is a dictionary
        if not isinstance(batch, dict):
            errors.append(f"Batch must be a dictionary, got {type(batch)}")
            return ValidationResult(False, errors, warnings, metadata)
        
        # Check if batch is empty
        if not batch:
            errors.append("Batch is empty")
            return ValidationResult(False, errors, warnings, metadata)
        
        # Validate each tensor in batch
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                result = self.validator.validate_tensor(tensor, name=f"batch['{key}']")
                errors.extend(result.errors)
                warnings.extend(result.warnings)
                metadata[key] = result.metadata
            else:
                warnings.append(f"batch['{key}'] is not a tensor: {type(tensor)}")
        
        return ValidationResult(len(errors) == 0, errors, warnings, metadata)
    
    def validate_output(self, output: Dict[str, torch.Tensor]) -> ValidationResult:
        """Validate output"""
        errors = []
        warnings = []
        metadata = {}
        
        # Check if output is a dictionary
        if not isinstance(output, dict):
            errors.append(f"Output must be a dictionary, got {type(output)}")
            return ValidationResult(False, errors, warnings, metadata)
        
        # Validate each tensor in output
        for key, tensor in output.items():
            if isinstance(tensor, torch.Tensor):
                result = self.validator.validate_tensor(tensor, name=f"output['{key}']")
                errors.extend(result.errors)
                warnings.extend(result.warnings)
                metadata[key] = result.metadata
        
        return ValidationResult(len(errors) == 0, errors, warnings, metadata)
    
    def safe_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Safe forward pass with validation and error handling"""
        
        # Validate input
        input_validation = self.validate_input(batch)
        if not input_validation.is_valid:
            raise ValueError(f"Input validation failed: {input_validation.errors}")
        
        # Log warnings
        for warning in input_validation.warnings:
            self.logger_instance.warning(warning)
        
        try:
            # Forward pass
            output = self.forward(batch)
            
            # Validate output
            output_validation = self.validate_output(output)
            if not output_validation.is_valid:
                self.logger_instance.error(f"Output validation failed: {output_validation.errors}")
                # Don't raise error for output validation, just log
            
            # Log output warnings
            for warning in output_validation.warnings:
                self.logger_instance.warning(warning)
            
            return output
            
        except Exception as e:
            self.logger_instance.error(f"Forward pass failed: {e}")
            raise
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = {}
        
        # Model parameters memory
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2
        stats["parameters_mb"] = param_memory
        
        # GPU memory if available
        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        
        return stats
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def to_device(self, device: Union[str, torch.device]):
        """Move model to device with proper error handling"""
        try:
            if isinstance(device, str):
                device = torch.device(device)
            
            self.to(device)
            self.metadata.device = str(device)
            self.logger_instance.info(f"Model moved to device: {device}")
            
        except Exception as e:
            self.logger_instance.error(f"Failed to move model to device {device}: {e}")
            raise


class ModelRegistry:
    """Registry for managing all neural network models"""
    
    def __init__(self):
        self._models: Dict[str, BaseNeuralNetwork] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
    
    def register_model(self, name: str, model: BaseNeuralNetwork):
        """Register a model in the registry"""
        if name in self._models:
            logger.warning(f"Model '{name}' already registered, overwriting")
        
        self._models[name] = model
        self._metadata[name] = model.get_metadata()
        
        logger.info(f"Registered model: {name} ({model.metadata.model_type.value})")
    
    def get_model(self, name: str) -> Optional[BaseNeuralNetwork]:
        """Get a model from the registry"""
        return self._models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self._models.keys())
    
    def get_models_by_type(self, model_type: ModelType) -> List[Tuple[str, BaseNeuralNetwork]]:
        """Get all models of a specific type"""
        return [
            (name, model) for name, model in self._models.items()
            if model.metadata.model_type == model_type
        ]
    
    def get_models_by_modality(self, modality: DataModality) -> List[Tuple[str, BaseNeuralNetwork]]:
        """Get all models that support a specific modality"""
        return [
            (name, model) for name, model in self._models.items()
            if modality in model.metadata.supported_modalities
        ]
    
    def validate_all_models(self) -> Dict[str, ValidationResult]:
        """Validate all registered models"""
        results = {}
        
        for name, model in self._models.items():
            try:
                # Create dummy batch for validation
                dummy_batch = self._create_dummy_batch(model.metadata)
                result = model.validate_input(dummy_batch)
                results[name] = result
            except Exception as e:
                results[name] = ValidationResult(
                    False, 
                    [f"Validation failed: {e}"], 
                    [], 
                    {}
                )
        
        return results
    
    def _create_dummy_batch(self, metadata: ModelMetadata) -> Dict[str, torch.Tensor]:
        """Create dummy batch for testing"""
        batch = {}
        
        for modality in metadata.supported_modalities:
            if modality == DataModality.DATACUBE:
                batch['datacube'] = torch.randn(2, 5, 8, 16, 16)
            elif modality == DataModality.GRAPH:
                # This would need proper graph data structure
                batch['graph_data'] = torch.randn(2, 64)
            elif modality == DataModality.TEXT:
                batch['text_input'] = torch.randint(0, 1000, (2, 50))
            elif modality == DataModality.SPECTRAL:
                batch['spectral_data'] = torch.randn(2, 1000)
            elif modality == DataModality.OBSERVATORY:
                batch['observatory_data'] = torch.randn(2, 64)
        
        return batch


# Global model registry instance
model_registry = ModelRegistry()


# Utility functions
def register_model(name: str, model: BaseNeuralNetwork):
    """Register a model globally"""
    model_registry.register_model(name, model)


def get_model(name: str) -> Optional[BaseNeuralNetwork]:
    """Get a model from global registry"""
    return model_registry.get_model(name)


def list_models() -> List[str]:
    """List all registered models"""
    return model_registry.list_models()


def validate_all_models() -> Dict[str, ValidationResult]:
    """Validate all registered models"""
    return model_registry.validate_all_models()
