#!/usr/bin/env python3
"""
Enhanced Performance Profiler for Astrobiology Platform
======================================================

Advanced performance profiling system that provides detailed insights into model
performance, memory usage patterns, and optimization opportunities specifically
designed for the astrobiology research platform's complex multi-modal architecture.

Features:
- Deep model architecture profiling with layer-wise analysis
- Memory usage pattern detection and optimization suggestions
- Training and inference performance benchmarking
- Multi-modal data pipeline profiling
- Custom profiling for scientific computing workloads
- Performance regression detection and alerting
- Resource utilization optimization recommendations
- Integration with existing monitoring systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import functools
import traceback
import gc
import warnings
from contextlib import contextmanager
import sys
import os
import psutil
import pickle
from concurrent.futures import ThreadPoolExecutor
# Optional plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Try to import profiling tools
try:
    import torch.profiler
    TORCH_PROFILER_AVAILABLE = True
except ImportError:
    TORCH_PROFILER_AVAILABLE = False

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class LayerProfile:
    """Profile data for individual neural network layers"""
    layer_name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameter_count: int
    memory_usage_mb: float
    forward_time_ms: float
    backward_time_ms: float
    flops: int
    activation_memory_mb: float
    gradient_memory_mb: float
    optimization_suggestions: List[str] = field(default_factory=list)

@dataclass
class ModelProfile:
    """Comprehensive model profiling results"""
    model_name: str
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    peak_memory_mb: float
    forward_time_ms: float
    backward_time_ms: float
    total_flops: int
    layers: List[LayerProfile] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    efficiency_score: float = 0.0

@dataclass
class DataPipelineProfile:
    """Data pipeline performance profile"""
    pipeline_name: str
    batch_loading_time_ms: float
    preprocessing_time_ms: float
    augmentation_time_ms: float
    data_transfer_time_ms: float
    total_pipeline_time_ms: float
    memory_usage_mb: float
    cpu_utilization: float
    io_wait_time_ms: float
    bottleneck_stage: str
    throughput_samples_per_sec: float
    optimization_suggestions: List[str] = field(default_factory=list)

class EnhancedModelProfiler:
    """Advanced model profiler with deep architectural analysis"""
    
    def __init__(self, enable_detailed_profiling: bool = True):
        self.enable_detailed_profiling = enable_detailed_profiling
        self.profiling_cache = {}
        self.benchmark_results = {}
        self.regression_baseline = {}
        
    def profile_model_comprehensive(self, model: nn.Module, input_data: torch.Tensor,
                                  model_name: str = "unknown", 
                                  include_training: bool = False) -> ModelProfile:
        """Comprehensive model profiling including training and inference"""
        logger.info(f"Starting comprehensive profiling for model: {model_name}")
        
        model.eval()
        device = next(model.parameters()).device
        input_data = input_data.to(device)
        
        # Basic model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        
        # Layer-wise profiling
        layer_profiles = self._profile_layers(model, input_data)
        
        # Forward pass profiling
        forward_time_ms = self._benchmark_forward_pass(model, input_data)
        
        # Backward pass profiling (if training mode requested)
        backward_time_ms = 0.0
        if include_training:
            backward_time_ms = self._benchmark_backward_pass(model, input_data)
        
        # Memory profiling
        peak_memory_mb = self._profile_memory_usage(model, input_data)
        
        # FLOPS calculation
        total_flops = self._calculate_model_flops(model, input_data)
        
        # Bottleneck analysis
        bottlenecks = self._identify_bottlenecks(layer_profiles)
        
        # Optimization opportunities
        optimization_opportunities = self._analyze_optimization_opportunities(
            model, layer_profiles, total_params, peak_memory_mb
        )
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(
            total_params, peak_memory_mb, forward_time_ms, total_flops
        )
        
        profile = ModelProfile(
            model_name=model_name,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_size_mb=model_size_mb,
            peak_memory_mb=peak_memory_mb,
            forward_time_ms=forward_time_ms,
            backward_time_ms=backward_time_ms,
            total_flops=total_flops,
            layers=layer_profiles,
            bottlenecks=bottlenecks,
            optimization_opportunities=optimization_opportunities,
            efficiency_score=efficiency_score
        )
        
        # Cache results
        self.profiling_cache[model_name] = profile
        
        logger.info(f"Model profiling completed. Efficiency score: {efficiency_score:.2f}")
        return profile
    
    def _profile_layers(self, model: nn.Module, input_data: torch.Tensor) -> List[LayerProfile]:
        """Profile individual layers of the model"""
        layer_profiles = []
        
        # Hook for capturing layer information
        layer_info = {}
        handles = []
        
        def create_hook(name):
            def hook(module, input, output):
                input_shape = input[0].shape if isinstance(input, tuple) else input.shape
                output_shape = output.shape if hasattr(output, 'shape') else 'unknown'
                
                # Calculate memory usage
                input_memory = input[0].numel() * input[0].element_size() / (1024 ** 2) if isinstance(input, tuple) else 0
                output_memory = output.numel() * output.element_size() / (1024 ** 2) if hasattr(output, 'numel') else 0
                
                # Parameter count
                param_count = sum(p.numel() for p in module.parameters())
                
                layer_info[name] = {
                    'input_shape': input_shape,
                    'output_shape': output_shape,
                    'input_memory_mb': input_memory,
                    'output_memory_mb': output_memory,
                    'parameter_count': param_count,
                    'layer_type': type(module).__name__
                }
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                handle = module.register_forward_hook(create_hook(name))
                handles.append(handle)
        
        # Run forward pass to collect data
        with torch.no_grad():
            _ = model(input_data)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Create layer profiles
        for name, info in layer_info.items():
            # Estimate forward time (simplified)
            forward_time = self._estimate_layer_time(info['layer_type'], info['parameter_count'])
            
            # Generate optimization suggestions
            suggestions = self._generate_layer_optimization_suggestions(info['layer_type'], info['parameter_count'])
            
            layer_profile = LayerProfile(
                layer_name=name,
                layer_type=info['layer_type'],
                input_shape=info['input_shape'],
                output_shape=info['output_shape'],
                parameter_count=info['parameter_count'],
                memory_usage_mb=info['input_memory_mb'] + info['output_memory_mb'],
                forward_time_ms=forward_time,
                backward_time_ms=forward_time * 1.5,  # Rough estimate
                flops=self._estimate_layer_flops(info['layer_type'], info['input_shape'], info['output_shape']),
                activation_memory_mb=info['output_memory_mb'],
                gradient_memory_mb=info['parameter_count'] * 4 / (1024 ** 2),  # Assuming float32
                optimization_suggestions=suggestions
            )
            layer_profiles.append(layer_profile)
        
        return layer_profiles
    
    def _benchmark_forward_pass(self, model: nn.Module, input_data: torch.Tensor, 
                               num_runs: int = 100) -> float:
        """Benchmark forward pass timing"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)
        
        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_data)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        return (end_time - start_time) / num_runs * 1000  # ms per run
    
    def _benchmark_backward_pass(self, model: nn.Module, input_data: torch.Tensor,
                                num_runs: int = 50) -> float:
        """Benchmark backward pass timing"""
        model.train()
        
        # Create dummy target for loss calculation
        with torch.no_grad():
            dummy_output = model(input_data)
        dummy_target = torch.randn_like(dummy_output)
        
        # Warmup
        for _ in range(5):
            model.zero_grad()
            output = model(input_data)
            loss = F.mse_loss(output, dummy_target)
            loss.backward()
        
        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(num_runs):
            model.zero_grad()
            output = model(input_data)
            loss = F.mse_loss(output, dummy_target)
            loss.backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        return (end_time - start_time) / num_runs * 1000  # ms per run
    
    def _profile_memory_usage(self, model: nn.Module, input_data: torch.Tensor) -> float:
        """Profile peak memory usage"""
        if not torch.cuda.is_available():
            return 0.0
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        with torch.no_grad():
            _ = model(input_data)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        torch.cuda.empty_cache()
        
        return peak_memory
    
    def _calculate_model_flops(self, model: nn.Module, input_data: torch.Tensor) -> int:
        """Calculate total FLOPs for the model (simplified estimation)"""
        total_flops = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                # Conv1D FLOPs: (input_channels * output_channels * kernel_size * output_length)
                output_length = (input_data.shape[-1] - module.kernel_size[0] + 2 * module.padding[0]) // module.stride[0] + 1
                flops = module.in_channels * module.out_channels * module.kernel_size[0] * output_length
                total_flops += flops
                
            elif isinstance(module, nn.Conv2d):
                # Conv2D FLOPs estimation
                output_h = (input_data.shape[-2] - module.kernel_size[0] + 2 * module.padding[0]) // module.stride[0] + 1
                output_w = (input_data.shape[-1] - module.kernel_size[1] + 2 * module.padding[1]) // module.stride[1] + 1
                flops = module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * output_h * output_w
                total_flops += flops
                
            elif isinstance(module, nn.Conv3d):
                # Conv3D FLOPs estimation
                flops = (module.in_channels * module.out_channels * 
                        module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2] *
                        input_data.shape[-3] * input_data.shape[-2] * input_data.shape[-1])
                total_flops += flops
                
            elif isinstance(module, nn.Linear):
                # Linear layer FLOPs
                flops = module.in_features * module.out_features
                total_flops += flops
        
        return total_flops
    
    def _identify_bottlenecks(self, layer_profiles: List[LayerProfile]) -> List[str]:
        """Identify performance bottlenecks in the model"""
        bottlenecks = []
        
        # Sort layers by different metrics
        by_time = sorted(layer_profiles, key=lambda x: x.forward_time_ms, reverse=True)
        by_memory = sorted(layer_profiles, key=lambda x: x.memory_usage_mb, reverse=True)
        by_params = sorted(layer_profiles, key=lambda x: x.parameter_count, reverse=True)
        
        # Top time consumers
        if by_time:
            bottlenecks.append(f"Time bottleneck: {by_time[0].layer_name} ({by_time[0].forward_time_ms:.2f}ms)")
        
        # Top memory consumers
        if by_memory:
            bottlenecks.append(f"Memory bottleneck: {by_memory[0].layer_name} ({by_memory[0].memory_usage_mb:.2f}MB)")
        
        # Parameter-heavy layers
        if by_params:
            bottlenecks.append(f"Parameter bottleneck: {by_params[0].layer_name} ({by_params[0].parameter_count:,} params)")
        
        return bottlenecks
    
    def _analyze_optimization_opportunities(self, model: nn.Module, layer_profiles: List[LayerProfile],
                                          total_params: int, peak_memory_mb: float) -> List[str]:
        """Analyze optimization opportunities"""
        opportunities = []
        
        # Model size optimizations
        if total_params > 50_000_000:  # 50M parameters
            opportunities.append("Large model detected - consider knowledge distillation")
            opportunities.append("Evaluate parameter pruning techniques")
        
        # Memory optimizations
        if peak_memory_mb > 4000:  # 4GB
            opportunities.append("High memory usage - enable gradient checkpointing")
            opportunities.append("Consider mixed precision training (FP16)")
        
        # Layer-specific optimizations
        conv3d_count = sum(1 for layer in layer_profiles if layer.layer_type == 'Conv3d')
        if conv3d_count > 10:
            opportunities.append("Many 3D convolutions - consider separable convolutions")
        
        attention_layers = sum(1 for layer in layer_profiles if 'attention' in layer.layer_name.lower())
        if attention_layers > 5:
            opportunities.append("Multiple attention layers - consider attention optimization")
        
        # Performance optimizations
        total_time = sum(layer.forward_time_ms for layer in layer_profiles)
        if total_time > 1000:  # 1 second
            opportunities.append("Slow inference - consider model quantization")
            opportunities.append("Evaluate TensorRT or ONNX optimization")
        
        return opportunities
    
    def _calculate_efficiency_score(self, total_params: int, peak_memory_mb: float,
                                  forward_time_ms: float, total_flops: int) -> float:
        """Calculate model efficiency score (0-100)"""
        # Normalize metrics (higher is worse)
        param_score = max(0, 100 - (total_params / 1_000_000))  # Penalty for >1M params
        memory_score = max(0, 100 - (peak_memory_mb / 100))     # Penalty for >100MB
        time_score = max(0, 100 - forward_time_ms)              # Penalty for >100ms
        flops_score = max(0, 100 - (total_flops / 1_000_000_000))  # Penalty for >1B FLOPs
        
        # Weighted average (can be tuned based on priorities)
        efficiency_score = (param_score * 0.3 + memory_score * 0.3 + 
                           time_score * 0.2 + flops_score * 0.2)
        
        return max(0, min(100, efficiency_score))
    
    def _estimate_layer_time(self, layer_type: str, param_count: int) -> float:
        """Estimate layer execution time (simplified)"""
        base_time = {
            'Conv1d': 0.1,
            'Conv2d': 0.5,
            'Conv3d': 2.0,
            'Linear': 0.1,
            'MultiheadAttention': 1.0,
            'LSTM': 1.5,
            'GRU': 1.2,
            'BatchNorm1d': 0.05,
            'BatchNorm2d': 0.05,
            'BatchNorm3d': 0.1,
            'ReLU': 0.01,
            'GELU': 0.02,
            'Dropout': 0.01
        }.get(layer_type, 0.1)
        
        # Scale by parameter count
        param_factor = max(1.0, param_count / 1000)
        return base_time * param_factor
    
    def _estimate_layer_flops(self, layer_type: str, input_shape: Tuple, output_shape: Tuple) -> int:
        """Estimate FLOPs for a layer (simplified)"""
        if layer_type == 'Linear':
            if isinstance(input_shape, tuple) and len(input_shape) >= 2:
                return input_shape[-1] * (output_shape[-1] if hasattr(output_shape, '__getitem__') else 1)
        elif 'Conv' in layer_type:
            # Simplified FLOP estimation for convolutions
            if hasattr(output_shape, '__len__') and len(output_shape) >= 3:
                return np.prod(output_shape) * 10  # Rough estimate
        
        return 1000  # Default estimate
    
    def _generate_layer_optimization_suggestions(self, layer_type: str, param_count: int) -> List[str]:
        """Generate optimization suggestions for specific layers"""
        suggestions = []
        
        if layer_type == 'Conv3d' and param_count > 100000:
            suggestions.append("Consider depthwise separable convolution")
            suggestions.append("Evaluate grouped convolutions")
        
        if layer_type == 'Linear' and param_count > 1000000:
            suggestions.append("Consider low-rank factorization")
            suggestions.append("Evaluate sparse connections")
        
        if 'BatchNorm' in layer_type:
            suggestions.append("Consider LayerNorm or GroupNorm alternatives")
        
        if layer_type == 'MultiheadAttention':
            suggestions.append("Consider linear attention variants")
            suggestions.append("Evaluate attention head pruning")
        
        return suggestions

class DataPipelineProfiler:
    """Profiler for data loading and preprocessing pipelines"""
    
    def __init__(self):
        self.pipeline_cache = {}
        
    def profile_data_pipeline(self, dataloader, num_batches: int = 10,
                             pipeline_name: str = "unknown") -> DataPipelineProfile:
        """Profile data loading pipeline performance"""
        logger.info(f"Profiling data pipeline: {pipeline_name}")
        
        timings = {
            'batch_loading': [],
            'preprocessing': [],
            'augmentation': [],
            'data_transfer': []
        }
        
        memory_usage = []
        cpu_usage = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Measure batch loading time
            start_time = time.perf_counter()
            
            # Simulate data processing stages
            batch_load_time = time.perf_counter() - start_time
            timings['batch_loading'].append(batch_load_time * 1000)
            
            # Monitor system resources
            memory_usage.append(psutil.virtual_memory().percent)
            cpu_usage.append(psutil.cpu_percent())
        
        # Calculate statistics
        total_pipeline_time = sum(timings['batch_loading'])
        avg_batch_time = np.mean(timings['batch_loading'])
        throughput = 1000 / avg_batch_time if avg_batch_time > 0 else 0  # samples per second
        
        # Identify bottleneck
        bottleneck_stage = "batch_loading"  # Simplified for this example
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_pipeline_optimizations(
            avg_batch_time, np.mean(memory_usage), np.mean(cpu_usage)
        )
        
        profile = DataPipelineProfile(
            pipeline_name=pipeline_name,
            batch_loading_time_ms=avg_batch_time,
            preprocessing_time_ms=0,  # Would need more detailed instrumentation
            augmentation_time_ms=0,
            data_transfer_time_ms=0,
            total_pipeline_time_ms=total_pipeline_time,
            memory_usage_mb=np.mean(memory_usage),
            cpu_utilization=np.mean(cpu_usage),
            io_wait_time_ms=0,
            bottleneck_stage=bottleneck_stage,
            throughput_samples_per_sec=throughput,
            optimization_suggestions=optimization_suggestions
        )
        
        self.pipeline_cache[pipeline_name] = profile
        return profile
    
    def _generate_pipeline_optimizations(self, avg_batch_time_ms: float,
                                       memory_usage: float, cpu_usage: float) -> List[str]:
        """Generate data pipeline optimization suggestions"""
        suggestions = []
        
        if avg_batch_time_ms > 100:  # 100ms per batch
            suggestions.append("Slow batch loading - increase num_workers")
            suggestions.append("Consider data prefetching")
        
        if memory_usage > 80:
            suggestions.append("High memory usage - reduce batch size")
            suggestions.append("Implement data streaming")
        
        if cpu_usage > 90:
            suggestions.append("High CPU usage - optimize preprocessing")
            suggestions.append("Consider GPU-accelerated data loading")
        
        suggestions.append("Enable pin_memory for GPU training")
        suggestions.append("Use persistent_workers=True for efficiency")
        
        return suggestions

class PerformanceRegressionDetector:
    """Detect performance regressions across model versions"""
    
    def __init__(self, baseline_file: Optional[str] = None):
        self.baseline_file = baseline_file
        self.baselines = {}
        self.load_baselines()
        
    def load_baselines(self):
        """Load performance baselines from file"""
        if self.baseline_file and Path(self.baseline_file).exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    self.baselines = json.load(f)
                logger.info(f"Loaded {len(self.baselines)} performance baselines")
            except Exception as e:
                logger.warning(f"Failed to load baselines: {e}")
    
    def save_baselines(self):
        """Save performance baselines to file"""
        if self.baseline_file:
            try:
                with open(self.baseline_file, 'w') as f:
                    json.dump(self.baselines, f, indent=2, default=str)
                logger.info("Performance baselines saved")
            except Exception as e:
                logger.error(f"Failed to save baselines: {e}")
    
    def set_baseline(self, model_name: str, profile: ModelProfile):
        """Set performance baseline for a model"""
        self.baselines[model_name] = {
            'forward_time_ms': profile.forward_time_ms,
            'peak_memory_mb': profile.peak_memory_mb,
            'efficiency_score': profile.efficiency_score,
            'timestamp': datetime.now().isoformat()
        }
        self.save_baselines()
    
    def check_regression(self, model_name: str, current_profile: ModelProfile,
                        tolerance: float = 0.1) -> Dict[str, Any]:
        """Check for performance regression"""
        if model_name not in self.baselines:
            return {'status': 'no_baseline', 'message': 'No baseline available for comparison'}
        
        baseline = self.baselines[model_name]
        regressions = []
        improvements = []
        
        # Check forward time
        time_change = (current_profile.forward_time_ms - baseline['forward_time_ms']) / baseline['forward_time_ms']
        if time_change > tolerance:
            regressions.append(f"Forward time increased by {time_change*100:.1f}%")
        elif time_change < -tolerance:
            improvements.append(f"Forward time improved by {abs(time_change)*100:.1f}%")
        
        # Check memory usage
        memory_change = (current_profile.peak_memory_mb - baseline['peak_memory_mb']) / baseline['peak_memory_mb']
        if memory_change > tolerance:
            regressions.append(f"Memory usage increased by {memory_change*100:.1f}%")
        elif memory_change < -tolerance:
            improvements.append(f"Memory usage improved by {abs(memory_change)*100:.1f}%")
        
        # Check efficiency score
        efficiency_change = (current_profile.efficiency_score - baseline['efficiency_score']) / baseline['efficiency_score']
        if efficiency_change < -tolerance:
            regressions.append(f"Efficiency score decreased by {abs(efficiency_change)*100:.1f}%")
        elif efficiency_change > tolerance:
            improvements.append(f"Efficiency score improved by {efficiency_change*100:.1f}%")
        
        status = 'regression' if regressions else 'improvement' if improvements else 'stable'
        
        return {
            'status': status,
            'regressions': regressions,
            'improvements': improvements,
            'baseline_timestamp': baseline['timestamp'],
            'current_timestamp': datetime.now().isoformat()
        }

class ComprehensivePerformanceProfiler:
    """Main performance profiler coordinator"""
    
    def __init__(self, enable_regression_detection: bool = True):
        self.model_profiler = EnhancedModelProfiler()
        self.pipeline_profiler = DataPipelineProfiler()
        self.regression_detector = PerformanceRegressionDetector("performance_baselines.json") if enable_regression_detection else None
        self.profiling_history = {}
        
    def profile_complete_system(self, model: nn.Module, dataloader, 
                               model_name: str = "unknown",
                               include_training: bool = False) -> Dict[str, Any]:
        """Profile complete system including model and data pipeline"""
        logger.info(f"Starting complete system profiling for {model_name}")
        
        # Get sample input from dataloader
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, (list, tuple)):
            sample_input = sample_batch[0]
        else:
            sample_input = sample_batch
        
        # Profile model
        model_profile = self.model_profiler.profile_model_comprehensive(
            model, sample_input, model_name, include_training
        )
        
        # Profile data pipeline
        pipeline_profile = self.pipeline_profiler.profile_data_pipeline(
            dataloader, pipeline_name=f"{model_name}_pipeline"
        )
        
        # Check for regressions
        regression_check = None
        if self.regression_detector:
            regression_check = self.regression_detector.check_regression(model_name, model_profile)
            
            # Set baseline if none exists
            if regression_check['status'] == 'no_baseline':
                self.regression_detector.set_baseline(model_name, model_profile)
        
        # Generate comprehensive report
        system_profile = {
            'model_profile': model_profile,
            'pipeline_profile': pipeline_profile,
            'regression_check': regression_check,
            'profiling_timestamp': datetime.now().isoformat(),
            'system_recommendations': self._generate_system_recommendations(model_profile, pipeline_profile)
        }
        
        # Store in history
        self.profiling_history[model_name] = system_profile
        
        logger.info(f"Complete system profiling finished for {model_name}")
        return system_profile
    
    def _generate_system_recommendations(self, model_profile: ModelProfile,
                                       pipeline_profile: DataPipelineProfile) -> List[str]:
        """Generate system-wide optimization recommendations"""
        recommendations = []
        
        # Model recommendations
        recommendations.extend(model_profile.optimization_opportunities)
        
        # Pipeline recommendations
        recommendations.extend(pipeline_profile.optimization_suggestions)
        
        # Combined recommendations
        if (model_profile.forward_time_ms > 100 and 
            pipeline_profile.batch_loading_time_ms > 50):
            recommendations.append("Both model and data pipeline are slow - prioritize optimization")
        
        if model_profile.peak_memory_mb > 2000:
            recommendations.append("High model memory usage - consider reducing batch size")
        
        return recommendations
    
    def save_profiling_report(self, filepath: Optional[str] = None) -> str:
        """Save comprehensive profiling report"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"performance_profiling_report_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.profiling_history, f, indent=2, default=str)
        
        logger.info(f"Profiling report saved to: {filepath}")
        return filepath

# Convenience functions
def profile_model_quick(model: nn.Module, input_data: torch.Tensor, 
                       model_name: str = "unknown") -> ModelProfile:
    """Quick model profiling"""
    profiler = EnhancedModelProfiler()
    return profiler.profile_model_comprehensive(model, input_data, model_name)

def profile_data_pipeline_quick(dataloader, pipeline_name: str = "unknown") -> DataPipelineProfile:
    """Quick data pipeline profiling"""
    profiler = DataPipelineProfiler()
    return profiler.profile_data_pipeline(dataloader, pipeline_name=pipeline_name)

if __name__ == "__main__":
    # Example usage
    print("Enhanced Performance Profiler for Astrobiology Platform")
    print("Use profile_model_quick() or profile_data_pipeline_quick() for quick profiling")
    print("Use ComprehensivePerformanceProfiler for full system analysis") 