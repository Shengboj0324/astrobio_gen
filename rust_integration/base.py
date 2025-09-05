"""
Base class for Rust accelerators with fallback mechanisms
=========================================================

This module provides the base class for all Rust accelerators, implementing
comprehensive error handling, fallback mechanisms, and performance monitoring.
"""

import logging
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RustAcceleratorBase(ABC):
    """
    Base class for Rust accelerators with automatic fallback
    
    This class provides:
    - Automatic detection of Rust availability
    - Graceful fallback to Python implementations
    - Performance monitoring and benchmarking
    - Comprehensive error handling
    - Logging and diagnostics
    """
    
    def __init__(self, enable_fallback: bool = True, log_performance: bool = False):
        """
        Initialize the Rust accelerator base
        
        Args:
            enable_fallback: Whether to fall back to Python if Rust fails
            log_performance: Whether to log performance metrics
        """
        self.enable_fallback = enable_fallback
        self.log_performance = log_performance
        self.performance_stats = {
            'rust_calls': 0,
            'python_calls': 0,
            'rust_total_time': 0.0,
            'python_total_time': 0.0,
            'rust_errors': 0,
            'fallback_triggers': 0
        }
        
        # Check Rust availability
        self.rust_available = self._check_rust_availability()
        
        if self.rust_available:
            logger.debug(f"✅ {self.__class__.__name__}: Rust acceleration enabled")
        else:
            logger.debug(f"📋 {self.__class__.__name__}: Using Python fallback")
    
    def _check_rust_availability(self) -> bool:
        """Check if Rust extensions are available and working"""
        try:
            import astrobio_rust
            # Test basic functionality
            _ = astrobio_rust.get_version()
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"Rust extensions available but not working: {e}")
            return False
    
    def _call_with_fallback(
        self,
        rust_func: callable,
        python_func: callable,
        *args,
        operation_name: str = "operation",
        **kwargs
    ) -> Any:
        """
        Call Rust function with automatic fallback to Python
        
        Args:
            rust_func: Rust implementation function
            python_func: Python fallback function
            *args: Arguments to pass to functions
            operation_name: Name of operation for logging
            **kwargs: Keyword arguments to pass to functions
            
        Returns:
            Result from either Rust or Python implementation
        """
        
        # Try Rust implementation first
        if self.rust_available:
            try:
                start_time = time.time()
                result = rust_func(*args, **kwargs)
                end_time = time.time()
                
                # Update performance stats
                self.performance_stats['rust_calls'] += 1
                self.performance_stats['rust_total_time'] += (end_time - start_time)
                
                if self.log_performance:
                    logger.debug(f"🦀 {operation_name}: Rust execution time: {end_time - start_time:.4f}s")
                
                return result
                
            except Exception as e:
                self.performance_stats['rust_errors'] += 1
                
                if self.enable_fallback:
                    logger.warning(f"⚠️  Rust {operation_name} failed: {e}")
                    logger.info(f"📋 Falling back to Python implementation")
                    self.performance_stats['fallback_triggers'] += 1
                else:
                    logger.error(f"❌ Rust {operation_name} failed: {e}")
                    raise
        
        # Use Python fallback
        if self.enable_fallback or not self.rust_available:
            try:
                start_time = time.time()
                result = python_func(*args, **kwargs)
                end_time = time.time()
                
                # Update performance stats
                self.performance_stats['python_calls'] += 1
                self.performance_stats['python_total_time'] += (end_time - start_time)
                
                if self.log_performance:
                    logger.debug(f"🐍 {operation_name}: Python execution time: {end_time - start_time:.4f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Python {operation_name} also failed: {e}")
                raise
        else:
            raise RuntimeError(f"Rust {operation_name} failed and fallback is disabled")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        
        # Calculate derived metrics
        if stats['rust_calls'] > 0:
            stats['rust_avg_time'] = stats['rust_total_time'] / stats['rust_calls']
        else:
            stats['rust_avg_time'] = 0.0
            
        if stats['python_calls'] > 0:
            stats['python_avg_time'] = stats['python_total_time'] / stats['python_calls']
        else:
            stats['python_avg_time'] = 0.0
        
        # Calculate speedup
        if stats['python_avg_time'] > 0 and stats['rust_avg_time'] > 0:
            stats['speedup_factor'] = stats['python_avg_time'] / stats['rust_avg_time']
        else:
            stats['speedup_factor'] = 1.0
        
        # Calculate success rates
        total_rust_attempts = stats['rust_calls'] + stats['rust_errors']
        if total_rust_attempts > 0:
            stats['rust_success_rate'] = stats['rust_calls'] / total_rust_attempts
        else:
            stats['rust_success_rate'] = 0.0
        
        stats['rust_available'] = self.rust_available
        stats['fallback_enabled'] = self.enable_fallback
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'rust_calls': 0,
            'python_calls': 0,
            'rust_total_time': 0.0,
            'python_total_time': 0.0,
            'rust_errors': 0,
            'fallback_triggers': 0
        }
    
    def print_performance_stats(self):
        """Print performance statistics"""
        stats = self.get_performance_stats()
        
        print(f"\n📊 {self.__class__.__name__} Performance Statistics")
        print("=" * 50)
        print(f"🦀 Rust calls: {stats['rust_calls']}")
        print(f"🐍 Python calls: {stats['python_calls']}")
        print(f"❌ Rust errors: {stats['rust_errors']}")
        print(f"🔄 Fallback triggers: {stats['fallback_triggers']}")
        
        if stats['rust_calls'] > 0:
            print(f"⚡ Rust avg time: {stats['rust_avg_time']:.4f}s")
        if stats['python_calls'] > 0:
            print(f"🐌 Python avg time: {stats['python_avg_time']:.4f}s")
        if stats['speedup_factor'] > 1.0:
            print(f"🚀 Speedup factor: {stats['speedup_factor']:.1f}x")
        
        print(f"✅ Rust success rate: {stats['rust_success_rate']:.1%}")
        print(f"🔧 Rust available: {stats['rust_available']}")
        print(f"🛡️  Fallback enabled: {stats['fallback_enabled']}")
    
    @abstractmethod
    def _validate_inputs(self, *args, **kwargs) -> None:
        """Validate inputs for the accelerator (to be implemented by subclasses)"""
        pass
    
    def _convert_to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to NumPy array if needed"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")
    
    def _convert_to_torch(self, array: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert NumPy array to PyTorch tensor"""
        tensor = torch.from_numpy(array)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
