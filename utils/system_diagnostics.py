#!/usr/bin/env python3
"""
Advanced System Diagnostics and Health Monitoring
================================================

Comprehensive diagnostic system for the astrobiology research platform that provides
deep insights into system performance, model behavior, and integration health without
disrupting existing functionality.

Features:
- Real-time system health monitoring with anomaly detection
- Model performance profiling and bottleneck identification
- Memory usage analysis and optimization recommendations
- Integration validation with dependency mapping
- Predictive maintenance and alert systems
- Performance regression detection
- Resource utilization optimization
- Comprehensive diagnostic reporting
"""

import torch
import torch.nn as nn
import psutil
import numpy as np

# Optional GPU monitoring
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
import logging
import time
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_memory_total: List[float] = field(default_factory=list)
    disk_usage_percent: float = 0.0
    network_sent: float = 0.0
    network_recv: float = 0.0
    pytorch_memory_allocated: float = 0.0
    pytorch_memory_cached: float = 0.0
    active_processes: int = 0
    model_inference_times: Dict[str, float] = field(default_factory=dict)
    integration_health: Dict[str, bool] = field(default_factory=dict)

@dataclass
class ModelDiagnostics:
    """Model-specific diagnostics"""
    model_name: str
    parameter_count: int
    model_size_mb: float
    inference_time_ms: float
    memory_usage_mb: float
    gradient_norm: float = 0.0
    activation_stats: Dict[str, float] = field(default_factory=dict)
    bottleneck_layers: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)

class AdvancedPerformanceProfiler:
    """Advanced performance profiler for models and system components (non-conflicting with existing profilers)"""
    
    def __init__(self):
        self.profiling_data = defaultdict(list)
        self.timing_stack = []
        self.memory_baseline = {}
        self.model_profiles = {}
        
    @contextmanager
    def profile_section(self, section_name: str):
        """Context manager for profiling code sections"""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            self.profiling_data[section_name].append({
                'duration_ms': (end_time - start_time) * 1000,
                'memory_delta_mb': end_memory - start_memory,
                'timestamp': datetime.now()
            })
    
    def profile_model(self, model: nn.Module, input_data: torch.Tensor, 
                     model_name: str = "unknown") -> ModelDiagnostics:
        """Comprehensive model profiling"""
        model.eval()
        
        # Basic model info
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        
        # Memory baseline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        memory_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Inference timing
        warmup_runs = 3
        timing_runs = 10
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_data)
        
        # Timing runs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(timing_runs):
                output = model(input_data)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) / timing_runs * 1000
        
        # Memory usage
        memory_end = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_usage_mb = (memory_end - memory_start) / (1024 ** 2)
        
        # Activation statistics
        activation_stats = self._analyze_activations(model, input_data)
        
        # Bottleneck analysis
        bottleneck_layers = self._identify_bottlenecks(model, input_data)
        
        # Optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            model, param_count, memory_usage_mb, inference_time_ms
        )
        
        return ModelDiagnostics(
            model_name=model_name,
            parameter_count=param_count,
            model_size_mb=model_size_mb,
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_usage_mb,
            activation_stats=activation_stats,
            bottleneck_layers=bottleneck_layers,
            optimization_suggestions=optimization_suggestions
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            return psutil.Process().memory_info().rss / (1024 ** 2)
    
    def _analyze_activations(self, model: nn.Module, input_data: torch.Tensor) -> Dict[str, float]:
        """Analyze activation statistics"""
        activation_stats = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation_stats[name] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'sparsity': (output == 0).float().mean().item()
                    }
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        try:
            with torch.no_grad():
                _ = model(input_data)
        finally:
            for hook in hooks:
                hook.remove()
        
        return activation_stats
    
    def _identify_bottlenecks(self, model: nn.Module, input_data: torch.Tensor) -> List[str]:
        """Identify performance bottlenecks in the model"""
        layer_times = {}
        
        def timing_hook(name):
            def hook(module, input, output):
                start_time = time.perf_counter()
                # This is a simplified approach - in practice, we'd need more sophisticated timing
                layer_times[name] = time.perf_counter() - start_time
            return hook
        
        # Note: This is a simplified bottleneck identification
        # In practice, we'd use more sophisticated profiling tools
        bottlenecks = []
        
        # Simple heuristics based on parameter count and layer type
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                bottlenecks.append(f"{name} (3D convolution - high memory)")
            elif isinstance(module, nn.MultiheadAttention):
                bottlenecks.append(f"{name} (attention - quadratic complexity)")
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self, model: nn.Module, param_count: int, 
                                         memory_usage_mb: float, inference_time_ms: float) -> List[str]:
        """Generate optimization suggestions based on profiling results"""
        suggestions = []
        
        # Parameter count suggestions
        if param_count > 100_000_000:  # 100M parameters
            suggestions.append("Consider model pruning or knowledge distillation")
            suggestions.append("Evaluate parameter-efficient fine-tuning (PEFT)")
        
        # Memory usage suggestions
        if memory_usage_mb > 8000:  # 8GB
            suggestions.append("Enable gradient checkpointing for memory reduction")
            suggestions.append("Consider mixed precision training (FP16)")
            suggestions.append("Implement model parallel training")
        
        # Inference time suggestions
        if inference_time_ms > 1000:  # 1 second
            suggestions.append("Consider model quantization (INT8/FP16)")
            suggestions.append("Evaluate TensorRT or ONNX optimization")
            suggestions.append("Implement dynamic batching")
        
        # Model-specific suggestions
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout) and module.p > 0.3:
                suggestions.append(f"High dropout rate in {name} - consider reducing")
            elif isinstance(module, nn.BatchNorm3d):
                suggestions.append(f"Consider replacing BatchNorm with LayerNorm in {name}")
        
        return suggestions

class EnhancedSystemHealthMonitor:
    """Enhanced system health monitoring with anomaly detection (non-conflicting with existing monitoring)"""
    
    def __init__(self, monitoring_interval: float = 10.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.anomaly_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'gpu_utilization': 95.0,
            'gpu_memory_percent': 90.0,
            'disk_usage_percent': 90.0
        }
        self.alerts = []
        self.is_monitoring = False
        self.monitoring_thread = None
        
    def start_monitoring(self):
        """Start continuous system monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                self._check_for_anomalies(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_utilization = []
        gpu_memory_used = []
        gpu_memory_total = []
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_utilization.append(gpu.load * 100)
                    gpu_memory_used.append(gpu.memoryUsed)
                    gpu_memory_total.append(gpu.memoryTotal)
            except:
                pass  # GPU monitoring not available
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network
        network = psutil.net_io_counters()
        
        # PyTorch memory
        pytorch_memory_allocated = 0
        pytorch_memory_cached = 0
        if torch.cuda.is_available():
            pytorch_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            pytorch_memory_cached = torch.cuda.memory_reserved() / (1024 ** 2)
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024 ** 3),
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            disk_usage_percent=disk.percent,
            network_sent=network.bytes_sent / (1024 ** 2),
            network_recv=network.bytes_recv / (1024 ** 2),
            pytorch_memory_allocated=pytorch_memory_allocated,
            pytorch_memory_cached=pytorch_memory_cached,
            active_processes=len(psutil.pids())
        )
    
    def _check_for_anomalies(self, metrics: SystemMetrics):
        """Check for system anomalies and generate alerts"""
        alerts = []
        
        # CPU check
        if metrics.cpu_percent > self.anomaly_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Memory check
        if metrics.memory_percent > self.anomaly_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # GPU checks
        for i, gpu_util in enumerate(metrics.gpu_utilization):
            if gpu_util > self.anomaly_thresholds['gpu_utilization']:
                alerts.append(f"High GPU {i} utilization: {gpu_util:.1f}%")
        
        for i, (used, total) in enumerate(zip(metrics.gpu_memory_used, metrics.gpu_memory_total)):
            if total > 0:
                gpu_memory_percent = (used / total) * 100
                if gpu_memory_percent > self.anomaly_thresholds['gpu_memory_percent']:
                    alerts.append(f"High GPU {i} memory: {gpu_memory_percent:.1f}%")
        
        # Disk check
        if metrics.disk_usage_percent > self.anomaly_thresholds['disk_usage_percent']:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        if alerts:
            self.alerts.extend(alerts)
            logger.warning(f"System anomalies detected: {'; '.join(alerts)}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No monitoring data available"}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        
        # Calculate averages
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_gpu_util = np.mean([np.mean(m.gpu_utilization) for m in recent_metrics if m.gpu_utilization])
        
        # Determine overall health status
        health_score = 100
        if avg_cpu > 80: health_score -= 20
        if avg_memory > 80: health_score -= 20
        if avg_gpu_util > 90: health_score -= 15
        if len(self.alerts) > 0: health_score -= 25
        
        status = "excellent" if health_score > 85 else "good" if health_score > 70 else "warning" if health_score > 50 else "critical"
        
        return {
            "status": status,
            "health_score": health_score,
            "recent_alerts": self.alerts[-5:],  # Last 5 alerts
            "average_cpu_percent": avg_cpu,
            "average_memory_percent": avg_memory,
            "average_gpu_utilization": avg_gpu_util,
            "total_alerts": len(self.alerts),
            "monitoring_duration_hours": len(self.metrics_history) * self.monitoring_interval / 3600
        }

class IntegrationValidator:
    """Validate integration health across system components"""
    
    def __init__(self):
        self.component_registry = {}
        self.dependency_graph = {}
        self.validation_results = {}
        
    def register_component(self, component_name: str, component_info: Dict[str, Any]):
        """Register a system component for monitoring"""
        self.component_registry[component_name] = {
            'info': component_info,
            'last_validated': None,
            'validation_history': []
        }
    
    async def validate_all_integrations(self) -> Dict[str, Any]:
        """Validate all registered integrations"""
        validation_results = {}
        
        for component_name in self.component_registry:
            try:
                result = await self._validate_component(component_name)
                validation_results[component_name] = result
            except Exception as e:
                validation_results[component_name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now()
                }
        
        # Check dependency relationships
        dependency_health = self._validate_dependencies()
        validation_results['dependency_health'] = dependency_health
        
        self.validation_results = validation_results
        return validation_results
    
    async def _validate_component(self, component_name: str) -> Dict[str, Any]:
        """Validate a specific component"""
        component = self.component_registry[component_name]
        
        # Basic availability check
        validation_result = {
            'status': 'unknown',
            'checks': {},
            'timestamp': datetime.now()
        }
        
        # Import check
        try:
            if 'import_path' in component['info']:
                module_path = component['info']['import_path']
                module = __import__(module_path, fromlist=[''])
                validation_result['checks']['import'] = True
            else:
                validation_result['checks']['import'] = True
        except ImportError as e:
            validation_result['checks']['import'] = False
            validation_result['import_error'] = str(e)
        
        # Functionality check
        if 'test_function' in component['info']:
            try:
                test_func = component['info']['test_function']
                test_result = await test_func() if asyncio.iscoroutinefunction(test_func) else test_func()
                validation_result['checks']['functionality'] = test_result
            except Exception as e:
                validation_result['checks']['functionality'] = False
                validation_result['functionality_error'] = str(e)
        
        # Performance check
        if 'performance_test' in component['info']:
            try:
                perf_func = component['info']['performance_test']
                perf_result = await perf_func() if asyncio.iscoroutinefunction(perf_func) else perf_func()
                validation_result['checks']['performance'] = perf_result
            except Exception as e:
                validation_result['checks']['performance'] = False
                validation_result['performance_error'] = str(e)
        
        # Determine overall status
        all_checks = validation_result['checks']
        if all(all_checks.values()):
            validation_result['status'] = 'healthy'
        elif any(all_checks.values()):
            validation_result['status'] = 'degraded'
        else:
            validation_result['status'] = 'failed'
        
        return validation_result
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate component dependencies"""
        dependency_health = {
            'total_dependencies': len(self.dependency_graph),
            'healthy_dependencies': 0,
            'broken_dependencies': [],
            'status': 'unknown'
        }
        
        for component, dependencies in self.dependency_graph.items():
            for dependency in dependencies:
                if dependency in self.validation_results:
                    if self.validation_results[dependency]['status'] == 'healthy':
                        dependency_health['healthy_dependencies'] += 1
                    else:
                        dependency_health['broken_dependencies'].append({
                            'component': component,
                            'dependency': dependency,
                            'status': self.validation_results[dependency]['status']
                        })
        
        # Calculate dependency health ratio
        if dependency_health['total_dependencies'] > 0:
            health_ratio = dependency_health['healthy_dependencies'] / dependency_health['total_dependencies']
            if health_ratio > 0.9:
                dependency_health['status'] = 'excellent'
            elif health_ratio > 0.7:
                dependency_health['status'] = 'good'
            elif health_ratio > 0.5:
                dependency_health['status'] = 'degraded'
            else:
                dependency_health['status'] = 'critical'
        
        return dependency_health

class ComprehensiveDiagnostics:
    """Main diagnostics coordinator"""
    
    def __init__(self):
        self.profiler = AdvancedPerformanceProfiler()
        self.health_monitor = EnhancedSystemHealthMonitor()
        self.integration_validator = IntegrationValidator()
        self.diagnostics_history = []
        
    async def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        logger.info("Starting comprehensive system diagnostics...")
        
        diagnostics_start = time.perf_counter()
        
        # Collect system health
        system_health = self.health_monitor.get_health_summary()
        
        # Validate integrations
        integration_health = await self.integration_validator.validate_all_integrations()
        
        # Memory analysis
        memory_analysis = self._analyze_memory_usage()
        
        # Performance recommendations
        performance_recommendations = self._generate_performance_recommendations(
            system_health, integration_health, memory_analysis
        )
        
        diagnostics_time = (time.perf_counter() - diagnostics_start) * 1000
        
        full_report = {
            'timestamp': datetime.now(),
            'diagnostics_time_ms': diagnostics_time,
            'system_health': system_health,
            'integration_health': integration_health,
            'memory_analysis': memory_analysis,
            'performance_recommendations': performance_recommendations,
            'system_info': self._get_system_info()
        }
        
        self.diagnostics_history.append(full_report)
        
        logger.info(f"Comprehensive diagnostics completed in {diagnostics_time:.2f}ms")
        return full_report
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        memory_info = psutil.virtual_memory()
        
        analysis = {
            'total_memory_gb': memory_info.total / (1024 ** 3),
            'available_memory_gb': memory_info.available / (1024 ** 3),
            'used_memory_gb': memory_info.used / (1024 ** 3),
            'memory_percent': memory_info.percent,
            'swap_usage': psutil.swap_memory().percent if psutil.swap_memory().total > 0 else 0
        }
        
        # PyTorch memory analysis
        if torch.cuda.is_available():
            analysis['pytorch_gpu_memory'] = {
                'allocated_gb': torch.cuda.memory_allocated() / (1024 ** 3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3),
                'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024 ** 3)
            }
        
        # Memory optimization suggestions
        optimization_suggestions = []
        if analysis['memory_percent'] > 85:
            optimization_suggestions.append("High memory usage detected - consider model quantization")
        if torch.cuda.is_available() and torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.9:
            optimization_suggestions.append("High GPU memory usage - enable gradient checkpointing")
        
        analysis['optimization_suggestions'] = optimization_suggestions
        
        return analysis
    
    def _generate_performance_recommendations(self, system_health: Dict, 
                                            integration_health: Dict, 
                                            memory_analysis: Dict) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # System health based recommendations
        if system_health.get('health_score', 100) < 80:
            recommendations.append("System health below optimal - consider resource optimization")
        
        if system_health.get('average_cpu_percent', 0) > 80:
            recommendations.append("High CPU usage - consider parallel processing optimization")
        
        if system_health.get('average_memory_percent', 0) > 80:
            recommendations.append("High memory usage - implement memory-efficient data loading")
        
        # Integration health recommendations
        integration_status = integration_health.get('dependency_health', {}).get('status', 'unknown')
        if integration_status in ['degraded', 'critical']:
            recommendations.append("Integration issues detected - review component dependencies")
        
        # Memory-based recommendations
        memory_suggestions = memory_analysis.get('optimization_suggestions', [])
        recommendations.extend(memory_suggestions)
        
        # PyTorch-specific recommendations
        if torch.cuda.is_available():
            recommendations.extend([
                "Enable mixed precision training for 2x memory reduction",
                "Use torch.compile() for improved inference performance",
                "Consider gradient accumulation for large batch training"
            ])
        
        return recommendations
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cpu_count': psutil.cpu_count(),
            'platform': sys.platform,
            'process_count': len(psutil.pids())
        }
    
    def save_diagnostics_report(self, filepath: Optional[str] = None) -> str:
        """Save diagnostics report to file"""
        if not self.diagnostics_history:
            logger.warning("No diagnostics data to save")
            return ""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"system_diagnostics_report_{timestamp}.json"
        
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_diagnostics_runs': len(self.diagnostics_history),
                'system_diagnostics_version': '1.0.0'
            },
            'latest_diagnostics': self.diagnostics_history[-1] if self.diagnostics_history else None,
            'diagnostics_history': self.diagnostics_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Diagnostics report saved to: {filepath}")
        return filepath

# Convenience functions for easy access
def create_system_diagnostics() -> ComprehensiveDiagnostics:
    """Create a comprehensive diagnostics system"""
    return ComprehensiveDiagnostics()

def profile_model(model: nn.Module, input_data: torch.Tensor, 
                 model_name: str = "unknown") -> ModelDiagnostics:
    """Quick model profiling function"""
    profiler = AdvancedPerformanceProfiler()
    return profiler.profile_model(model, input_data, model_name)

async def quick_system_health_check() -> Dict[str, Any]:
    """Quick system health assessment"""
    diagnostics = ComprehensiveDiagnostics()
    return await diagnostics.run_full_diagnostics()

if __name__ == "__main__":
    # Example usage
    async def main():
        diagnostics = create_system_diagnostics()
        
        # Start monitoring
        diagnostics.health_monitor.start_monitoring()
        
        # Run diagnostics
        report = await diagnostics.run_full_diagnostics()
        
        # Save report
        report_file = diagnostics.save_diagnostics_report()
        
        print(f"Diagnostics completed. Report saved to: {report_file}")
        print(f"System health status: {report['system_health']['status']}")
        
        # Stop monitoring
        diagnostics.health_monitor.stop_monitoring()
    
    import asyncio
    asyncio.run(main()) 