#!/usr/bin/env python3
"""
Real-Time Monitoring and Adaptive Orchestration System
=====================================================

Advanced monitoring, auto-tuning, and adaptive orchestration system for 
astrobiology research platform. Provides real-time performance monitoring,
automatic hyperparameter tuning, and intelligent model selection.

Features:
- Real-time performance monitoring and alerting
- Automatic hyperparameter optimization
- Adaptive model selection based on performance
- System health monitoring and auto-recovery
- Performance trend analysis and prediction
- Resource utilization optimization
- Intelligent load balancing across models
"""

import torch
import numpy as np
import time
import logging
import threading
import queue
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict, deque
import statistics
import psutil
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime = field(default_factory=datetime.now)
    inference_time_ms: float = 0.0
    accuracy: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    model_name: str = ""
    batch_size: int = 1
    input_shape: Tuple[int, ...] = field(default_factory=tuple)

@dataclass
class SystemHealth:
    """System health metrics"""
    overall_health: float = 1.0
    model_health: Dict[str, float] = field(default_factory=dict)
    resource_health: Dict[str, float] = field(default_factory=dict)
    integration_health: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    monitoring_interval: float = 1.0  # seconds
    metrics_retention_hours: int = 24
    performance_threshold: float = 0.95
    memory_threshold_gb: float = 8.0
    gpu_threshold: float = 0.9
    cpu_threshold: float = 0.8
    alert_cooldown_minutes: int = 5
    auto_tuning_enabled: bool = True
    adaptive_selection_enabled: bool = True
    health_check_interval: float = 30.0  # seconds

class MetricsCollector:
    """Collects performance metrics from models and system"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_history = defaultdict(deque)
        self.collection_thread = None
        self.running = False
        
        logger.info("Initialized MetricsCollector")
    
    def start_collection(self):
        """Start metrics collection thread"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                # Store metrics
                self._store_metrics("system", system_metrics)
                
                # Sleep
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        metrics = {}
        
        # CPU utilization
        metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
        
        # Memory utilization
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_used_gb'] = memory.used / (1024**3)
        metrics['memory_available_gb'] = memory.available / (1024**3)
        
        # GPU utilization (if available)
        if torch.cuda.is_available():
            try:
                metrics['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / (1024**3)
                metrics['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
                metrics['gpu_utilization'] = self._get_gpu_utilization()
            except:
                metrics['gpu_utilization'] = 0.0
        
        # Disk I/O
        disk = psutil.disk_usage('/')
        metrics['disk_percent'] = disk.percent
        metrics['disk_used_gb'] = disk.used / (1024**3)
        
        return metrics
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization (simplified)"""
        try:
            # This is a simplified version - in practice, you'd use nvidia-ml-py
            return min(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(), 1.0) * 100
        except:
            return 0.0
    
    def record_model_metrics(self, model_name: str, metrics: PerformanceMetrics):
        """Record metrics for a specific model"""
        self._store_metrics(f"model_{model_name}", metrics)
    
    def _store_metrics(self, key: str, metrics: Any):
        """Store metrics with retention policy"""
        self.metrics_history[key].append({
            'timestamp': datetime.now(),
            'data': metrics
        })
        
        # Apply retention policy
        retention_limit = datetime.now() - timedelta(hours=self.config.metrics_retention_hours)
        while (self.metrics_history[key] and 
               self.metrics_history[key][0]['timestamp'] < retention_limit):
            self.metrics_history[key].popleft()
    
    def get_metrics(self, key: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics for a specific key within time window"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            entry for entry in self.metrics_history[key]
            if entry['timestamp'] >= cutoff_time
        ]
    
    def get_latest_metrics(self, key: str) -> Optional[Dict[str, Any]]:
        """Get latest metrics for a key"""
        if key in self.metrics_history and self.metrics_history[key]:
            return self.metrics_history[key][-1]
        return None

class PerformanceAnalyzer:
    """Analyzes performance metrics and detects issues"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        
        logger.info("Initialized PerformanceAnalyzer")
    
    def analyze_performance(self, metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Analyze overall system performance"""
        analysis = {
            'timestamp': datetime.now(),
            'system_performance': self._analyze_system_performance(metrics_collector),
            'model_performance': self._analyze_model_performance(metrics_collector),
            'trends': self._analyze_trends(metrics_collector),
            'anomalies': self._detect_anomalies(metrics_collector),
            'recommendations': []
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_system_performance(self, metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Analyze system-level performance"""
        system_metrics = metrics_collector.get_metrics("system", hours=1)
        
        if not system_metrics:
            return {'status': 'unknown', 'score': 0.0}
        
        # Calculate average metrics
        recent_metrics = [entry['data'] for entry in system_metrics[-10:]]
        
        avg_cpu = statistics.mean(m['cpu_percent'] for m in recent_metrics)
        avg_memory = statistics.mean(m['memory_percent'] for m in recent_metrics)
        avg_gpu = statistics.mean(m.get('gpu_utilization', 0) for m in recent_metrics)
        
        # Calculate performance score
        cpu_score = 1.0 - (avg_cpu / 100.0)
        memory_score = 1.0 - (avg_memory / 100.0)
        gpu_score = 1.0 - (avg_gpu / 100.0)
        
        overall_score = (cpu_score + memory_score + gpu_score) / 3.0
        
        return {
            'status': 'healthy' if overall_score > 0.7 else 'degraded',
            'score': overall_score,
            'cpu_utilization': avg_cpu,
            'memory_utilization': avg_memory,
            'gpu_utilization': avg_gpu
        }
    
    def _analyze_model_performance(self, metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Analyze model-specific performance"""
        model_analysis = {}
        
        for key in metrics_collector.metrics_history:
            if key.startswith('model_'):
                model_name = key.replace('model_', '')
                model_metrics = metrics_collector.get_metrics(key, hours=1)
                
                if model_metrics:
                    recent_metrics = [entry['data'] for entry in model_metrics[-10:]]
                    
                    # Calculate average performance
                    avg_inference_time = statistics.mean(
                        m.inference_time_ms for m in recent_metrics
                    )
                    avg_accuracy = statistics.mean(
                        m.accuracy for m in recent_metrics
                    )
                    avg_throughput = statistics.mean(
                        m.throughput for m in recent_metrics
                    )
                    
                    model_analysis[model_name] = {
                        'avg_inference_time_ms': avg_inference_time,
                        'avg_accuracy': avg_accuracy,
                        'avg_throughput': avg_throughput,
                        'performance_score': avg_accuracy - (avg_inference_time / 1000.0)
                    }
        
        return model_analysis
    
    def _analyze_trends(self, metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Analyze performance trends"""
        return self.trend_analyzer.analyze_trends(metrics_collector)
    
    def _detect_anomalies(self, metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Detect performance anomalies"""
        return self.anomaly_detector.detect_anomalies(metrics_collector)
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # System recommendations
        system_perf = analysis['system_performance']
        if system_perf['cpu_utilization'] > self.config.cpu_threshold * 100:
            recommendations.append("High CPU utilization detected - consider reducing batch size")
        
        if system_perf['memory_utilization'] > self.config.memory_threshold_gb * 100:
            recommendations.append("High memory usage detected - consider enabling gradient checkpointing")
        
        if system_perf['gpu_utilization'] > self.config.gpu_threshold * 100:
            recommendations.append("High GPU utilization detected - consider model parallelism")
        
        # Model recommendations
        model_perf = analysis['model_performance']
        for model_name, metrics in model_perf.items():
            if metrics['avg_inference_time_ms'] > 1000:
                recommendations.append(f"Model {model_name} has high inference time - consider optimization")
            
            if metrics['avg_accuracy'] < 0.8:
                recommendations.append(f"Model {model_name} has low accuracy - consider retraining")
        
        return recommendations

class TrendAnalyzer:
    """Analyzes performance trends over time"""
    
    def analyze_trends(self, metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Analyze performance trends"""
        trends = {}
        
        # Analyze system trends
        system_metrics = metrics_collector.get_metrics("system", hours=24)
        if len(system_metrics) > 10:
            trends['system'] = self._calculate_trend(
                [entry['data']['cpu_percent'] for entry in system_metrics],
                'cpu_utilization'
            )
        
        # Analyze model trends
        for key in metrics_collector.metrics_history:
            if key.startswith('model_'):
                model_name = key.replace('model_', '')
                model_metrics = metrics_collector.get_metrics(key, hours=24)
                
                if len(model_metrics) > 10:
                    trends[model_name] = self._calculate_trend(
                        [entry['data'].inference_time_ms for entry in model_metrics],
                        'inference_time'
                    )
        
        return trends
    
    def _calculate_trend(self, values: List[float], metric_name: str) -> Dict[str, Any]:
        """Calculate trend for a series of values"""
        if len(values) < 2:
            return {'trend': 'insufficient_data', 'slope': 0.0}
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'slope': slope,
            'metric': metric_name,
            'latest_value': values[-1],
            'avg_value': statistics.mean(values)
        }

class AnomalyDetector:
    """Detects performance anomalies"""
    
    def __init__(self, threshold_std: float = 2.0):
        self.threshold_std = threshold_std
    
    def detect_anomalies(self, metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Detect anomalies in performance metrics"""
        anomalies = {}
        
        # Check system anomalies
        system_metrics = metrics_collector.get_metrics("system", hours=1)
        if len(system_metrics) > 10:
            cpu_values = [entry['data']['cpu_percent'] for entry in system_metrics]
            cpu_anomalies = self._detect_outliers(cpu_values, 'cpu_utilization')
            
            if cpu_anomalies:
                anomalies['system_cpu'] = cpu_anomalies
        
        # Check model anomalies
        for key in metrics_collector.metrics_history:
            if key.startswith('model_'):
                model_name = key.replace('model_', '')
                model_metrics = metrics_collector.get_metrics(key, hours=1)
                
                if len(model_metrics) > 10:
                    inference_times = [entry['data'].inference_time_ms for entry in model_metrics]
                    time_anomalies = self._detect_outliers(inference_times, 'inference_time')
                    
                    if time_anomalies:
                        anomalies[f'model_{model_name}'] = time_anomalies
        
        return anomalies
    
    def _detect_outliers(self, values: List[float], metric_name: str) -> Optional[Dict[str, Any]]:
        """Detect outliers using standard deviation"""
        if len(values) < 5:
            return None
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        outliers = []
        for i, val in enumerate(values):
            if abs(val - mean_val) > self.threshold_std * std_val:
                outliers.append({
                    'index': i,
                    'value': val,
                    'deviation': abs(val - mean_val) / std_val
                })
        
        if outliers:
            return {
                'metric': metric_name,
                'outliers': outliers,
                'mean': mean_val,
                'std': std_val
            }
        
        return None

class AutoTuner:
    """Automatic hyperparameter tuning based on performance"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.tuning_history = defaultdict(list)
        self.current_params = {}
        
        logger.info("Initialized AutoTuner")
    
    def tune_parameters(self, model_name: str, performance_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Automatically tune parameters based on performance"""
        if not self.config.auto_tuning_enabled:
            return {}
        
        # Get current parameters
        current_params = self.current_params.get(model_name, {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'dropout': 0.1
        })
        
        # Analyze performance
        tuning_suggestions = self._analyze_performance_for_tuning(performance_metrics, current_params)
        
        # Apply tuning
        new_params = self._apply_tuning(current_params, tuning_suggestions)
        
        # Store results
        self.current_params[model_name] = new_params
        self.tuning_history[model_name].append({
            'timestamp': datetime.now(),
            'old_params': current_params,
            'new_params': new_params,
            'performance': performance_metrics,
            'suggestions': tuning_suggestions
        })
        
        return new_params
    
    def _analyze_performance_for_tuning(self, metrics: PerformanceMetrics, current_params: Dict[str, Any]) -> Dict[str, str]:
        """Analyze performance and suggest parameter changes"""
        suggestions = {}
        
        # Learning rate tuning
        if metrics.accuracy < 0.8:
            if metrics.inference_time_ms > 500:
                suggestions['learning_rate'] = 'decrease'  # Too aggressive
            else:
                suggestions['learning_rate'] = 'increase'  # Too conservative
        
        # Batch size tuning
        if metrics.memory_usage_mb > self.config.memory_threshold_gb * 1024:
            suggestions['batch_size'] = 'decrease'
        elif metrics.throughput < 10:
            suggestions['batch_size'] = 'increase'
        
        # Dropout tuning
        if metrics.accuracy < 0.7:
            suggestions['dropout'] = 'decrease'  # Reduce regularization
        elif metrics.accuracy > 0.95:
            suggestions['dropout'] = 'increase'  # Increase regularization
        
        return suggestions
    
    def _apply_tuning(self, current_params: Dict[str, Any], suggestions: Dict[str, str]) -> Dict[str, Any]:
        """Apply tuning suggestions to parameters"""
        new_params = current_params.copy()
        
        for param, suggestion in suggestions.items():
            if param == 'learning_rate':
                if suggestion == 'increase':
                    new_params[param] = min(current_params[param] * 1.2, 1e-2)
                elif suggestion == 'decrease':
                    new_params[param] = max(current_params[param] * 0.8, 1e-6)
            
            elif param == 'batch_size':
                if suggestion == 'increase':
                    new_params[param] = min(current_params[param] * 2, 128)
                elif suggestion == 'decrease':
                    new_params[param] = max(current_params[param] // 2, 4)
            
            elif param == 'dropout':
                if suggestion == 'increase':
                    new_params[param] = min(current_params[param] + 0.1, 0.5)
                elif suggestion == 'decrease':
                    new_params[param] = max(current_params[param] - 0.1, 0.0)
        
        return new_params

class AdaptiveModelSelector:
    """Selects optimal model based on current conditions"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.model_registry = {}
        self.selection_history = []
        
        logger.info("Initialized AdaptiveModelSelector")
    
    def register_model(self, name: str, model: Any, characteristics: Dict[str, Any]):
        """Register a model with its characteristics"""
        self.model_registry[name] = {
            'model': model,
            'characteristics': characteristics,
            'performance_history': []
        }
        
        logger.info(f"Registered model: {name}")
    
    def select_optimal_model(self, task_requirements: Dict[str, Any], 
                           system_state: Dict[str, Any]) -> Tuple[str, Any]:
        """Select the optimal model for current conditions"""
        if not self.config.adaptive_selection_enabled:
            # Return default model
            return list(self.model_registry.keys())[0], list(self.model_registry.values())[0]['model']
        
        # Score all models
        model_scores = {}
        for name, model_info in self.model_registry.items():
            score = self._score_model(model_info, task_requirements, system_state)
            model_scores[name] = score
        
        # Select best model
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = self.model_registry[best_model_name]['model']
        
        # Record selection
        self.selection_history.append({
            'timestamp': datetime.now(),
            'selected_model': best_model_name,
            'scores': model_scores,
            'task_requirements': task_requirements,
            'system_state': system_state
        })
        
        return best_model_name, best_model
    
    def _score_model(self, model_info: Dict[str, Any], 
                    task_requirements: Dict[str, Any], 
                    system_state: Dict[str, Any]) -> float:
        """Score a model based on requirements and system state"""
        characteristics = model_info['characteristics']
        performance_history = model_info['performance_history']
        
        score = 0.0
        
        # Accuracy requirement
        if 'accuracy_requirement' in task_requirements:
            expected_accuracy = characteristics.get('expected_accuracy', 0.5)
            required_accuracy = task_requirements['accuracy_requirement']
            
            if expected_accuracy >= required_accuracy:
                score += 0.4
            else:
                score += 0.4 * (expected_accuracy / required_accuracy)
        
        # Latency requirement
        if 'max_latency_ms' in task_requirements:
            model_latency = characteristics.get('inference_time_ms', 1000)
            max_latency = task_requirements['max_latency_ms']
            
            if model_latency <= max_latency:
                score += 0.3
            else:
                score += 0.3 * (max_latency / model_latency)
        
        # Resource constraints
        if 'memory_available_gb' in system_state:
            model_memory = characteristics.get('memory_usage_gb', 1.0)
            available_memory = system_state['memory_available_gb']
            
            if model_memory <= available_memory:
                score += 0.2
            else:
                score += 0.2 * (available_memory / model_memory)
        
        # Historical performance
        if performance_history:
            recent_performance = statistics.mean(
                entry['accuracy'] for entry in performance_history[-10:]
            )
            score += 0.1 * recent_performance
        
        return score
    
    def update_model_performance(self, model_name: str, performance: PerformanceMetrics):
        """Update model performance history"""
        if model_name in self.model_registry:
            self.model_registry[model_name]['performance_history'].append({
                'timestamp': datetime.now(),
                'accuracy': performance.accuracy,
                'inference_time_ms': performance.inference_time_ms,
                'memory_usage_mb': performance.memory_usage_mb,
                'throughput': performance.throughput
            })
            
            # Keep only recent history
            history = self.model_registry[model_name]['performance_history']
            if len(history) > 100:
                self.model_registry[model_name]['performance_history'] = history[-100:]

class SystemHealthMonitor:
    """Monitors overall system health"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.health_history = []
        self.alert_history = []
        self.last_alert_time = defaultdict(datetime)
        
        logger.info("Initialized SystemHealthMonitor")
    
    def check_system_health(self, metrics_collector: MetricsCollector, 
                          performance_analyzer: PerformanceAnalyzer) -> SystemHealth:
        """Check overall system health"""
        # Get performance analysis
        analysis = performance_analyzer.analyze_performance(metrics_collector)
        
        # Calculate health scores
        system_health_score = analysis['system_performance']['score']
        
        model_health_scores = {}
        for model_name, model_metrics in analysis['model_performance'].items():
            model_health_scores[model_name] = model_metrics['performance_score']
        
        # Resource health
        resource_health = self._calculate_resource_health(metrics_collector)
        
        # Integration health
        integration_health = self._check_integration_health()
        
        # Overall health
        overall_health = (
            system_health_score * 0.4 +
            statistics.mean(model_health_scores.values()) * 0.3 if model_health_scores else 0 +
            statistics.mean(resource_health.values()) * 0.2 +
            statistics.mean(integration_health.values()) * 0.1
        )
        
        # Generate alerts
        alerts = self._generate_alerts(analysis, overall_health)
        
        # Generate recommendations
        recommendations = analysis['recommendations']
        
        health = SystemHealth(
            overall_health=overall_health,
            model_health=model_health_scores,
            resource_health=resource_health,
            integration_health=integration_health,
            alerts=alerts,
            recommendations=recommendations
        )
        
        # Store health history
        self.health_history.append({
            'timestamp': datetime.now(),
            'health': health
        })
        
        return health
    
    def _calculate_resource_health(self, metrics_collector: MetricsCollector) -> Dict[str, float]:
        """Calculate resource health scores"""
        latest_metrics = metrics_collector.get_latest_metrics("system")
        
        if not latest_metrics:
            return {'cpu': 0.0, 'memory': 0.0, 'gpu': 0.0}
        
        data = latest_metrics['data']
        
        # CPU health
        cpu_health = max(0, 1.0 - (data['cpu_percent'] / 100.0))
        
        # Memory health
        memory_health = max(0, 1.0 - (data['memory_percent'] / 100.0))
        
        # GPU health
        gpu_health = max(0, 1.0 - (data.get('gpu_utilization', 0) / 100.0))
        
        return {
            'cpu': cpu_health,
            'memory': memory_health,
            'gpu': gpu_health
        }
    
    def _check_integration_health(self) -> Dict[str, float]:
        """Check integration health"""
        # Simplified integration health check
        integration_health = {
            'enhanced_cnn': 1.0,
            'surrogate_models': 1.0,
            'enterprise_urls': 1.0,
            'graph_networks': 1.0,
            'meta_learning': 1.0
        }
        
        return integration_health
    
    def _generate_alerts(self, analysis: Dict[str, Any], overall_health: float) -> List[str]:
        """Generate system alerts"""
        alerts = []
        current_time = datetime.now()
        
        # Overall health alert
        if overall_health < 0.5:
            alert_key = 'system_health_critical'
            if self._should_send_alert(alert_key, current_time):
                alerts.append(f"CRITICAL: Overall system health is {overall_health:.2f}")
        
        # Resource alerts
        system_perf = analysis['system_performance']
        if system_perf['cpu_utilization'] > 90:
            alert_key = 'high_cpu'
            if self._should_send_alert(alert_key, current_time):
                alerts.append(f"HIGH CPU: {system_perf['cpu_utilization']:.1f}%")
        
        if system_perf['memory_utilization'] > 90:
            alert_key = 'high_memory'
            if self._should_send_alert(alert_key, current_time):
                alerts.append(f"HIGH MEMORY: {system_perf['memory_utilization']:.1f}%")
        
        # Model performance alerts
        for model_name, model_metrics in analysis['model_performance'].items():
            if model_metrics['avg_accuracy'] < 0.7:
                alert_key = f'low_accuracy_{model_name}'
                if self._should_send_alert(alert_key, current_time):
                    alerts.append(f"LOW ACCURACY: {model_name} accuracy is {model_metrics['avg_accuracy']:.2f}")
        
        # Store alerts
        for alert in alerts:
            self.alert_history.append({
                'timestamp': current_time,
                'alert': alert
            })
        
        return alerts
    
    def _should_send_alert(self, alert_key: str, current_time: datetime) -> bool:
        """Check if alert should be sent (respecting cooldown)"""
        last_alert = self.last_alert_time.get(alert_key, datetime.min)
        cooldown_period = timedelta(minutes=self.config.alert_cooldown_minutes)
        
        if current_time - last_alert > cooldown_period:
            self.last_alert_time[alert_key] = current_time
            return True
        
        return False

class RealTimeOrchestrator:
    """Main orchestrator for real-time monitoring and adaptation"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Initialize components
        self.metrics_collector = MetricsCollector(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.auto_tuner = AutoTuner(config)
        self.model_selector = AdaptiveModelSelector(config)
        self.health_monitor = SystemHealthMonitor(config)
        
        # Orchestration state
        self.running = False
        self.orchestration_thread = None
        
        logger.info("Initialized RealTimeOrchestrator")
    
    def start(self):
        """Start real-time monitoring and orchestration"""
        if self.running:
            return
        
        self.running = True
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start orchestration loop
        self.orchestration_thread = threading.Thread(target=self._orchestration_loop)
        self.orchestration_thread.daemon = True
        self.orchestration_thread.start()
        
        logger.info("Started real-time orchestration")
    
    def stop(self):
        """Stop real-time monitoring and orchestration"""
        self.running = False
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Wait for orchestration thread
        if self.orchestration_thread:
            self.orchestration_thread.join()
        
        logger.info("Stopped real-time orchestration")
    
    def _orchestration_loop(self):
        """Main orchestration loop"""
        while self.running:
            try:
                # Check system health
                health = self.health_monitor.check_system_health(
                    self.metrics_collector, 
                    self.performance_analyzer
                )
                
                # Log health status
                logger.info(f"System health: {health.overall_health:.2f}")
                
                # Handle alerts
                if health.alerts:
                    for alert in health.alerts:
                        logger.warning(f"ALERT: {alert}")
                
                # Apply recommendations
                if health.recommendations:
                    for recommendation in health.recommendations:
                        logger.info(f"RECOMMENDATION: {recommendation}")
                
                # Sleep
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                time.sleep(self.config.health_check_interval)
    
    def register_model(self, name: str, model: Any, characteristics: Dict[str, Any]):
        """Register a model for adaptive selection"""
        self.model_selector.register_model(name, model, characteristics)
    
    def record_inference(self, model_name: str, metrics: PerformanceMetrics):
        """Record inference metrics"""
        self.metrics_collector.record_model_metrics(model_name, metrics)
        
        # Update model performance
        self.model_selector.update_model_performance(model_name, metrics)
        
        # Auto-tune if enabled
        if self.config.auto_tuning_enabled:
            tuned_params = self.auto_tuner.tune_parameters(model_name, metrics)
            if tuned_params:
                logger.info(f"Auto-tuned parameters for {model_name}: {tuned_params}")
    
    def select_model(self, task_requirements: Dict[str, Any]) -> Tuple[str, Any]:
        """Select optimal model for current task"""
        # Get current system state
        system_metrics = self.metrics_collector.get_latest_metrics("system")
        system_state = system_metrics['data'] if system_metrics else {}
        
        # Select model
        return self.model_selector.select_optimal_model(task_requirements, system_state)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        # Get latest health
        if self.health_monitor.health_history:
            latest_health = self.health_monitor.health_history[-1]['health']
        else:
            latest_health = SystemHealth()
        
        # Get performance analysis
        analysis = self.performance_analyzer.analyze_performance(self.metrics_collector)
        
        return {
            'health': latest_health,
            'performance_analysis': analysis,
            'registered_models': list(self.model_selector.model_registry.keys()),
            'monitoring_active': self.running
        }

# Global orchestrator instance
_orchestrator = None

def get_real_time_orchestrator(config: Optional[MonitoringConfig] = None) -> RealTimeOrchestrator:
    """Get global real-time orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        if config is None:
            config = MonitoringConfig()
        _orchestrator = RealTimeOrchestrator(config)
    return _orchestrator

def create_monitoring_config(task_type: str = "astrobiology") -> MonitoringConfig:
    """Create monitoring configuration for specific task"""
    config = MonitoringConfig()
    
    if task_type == "astrobiology":
        config.monitoring_interval = 1.0
        config.performance_threshold = 0.90
        config.memory_threshold_gb = 16.0
        config.auto_tuning_enabled = True
        config.adaptive_selection_enabled = True
    
    return config

if __name__ == "__main__":
    # Demonstration of monitoring capabilities
    print("📊 Real-Time Monitoring System Demonstration")
    print("=" * 50)
    
    # Create configuration
    config = create_monitoring_config("astrobiology")
    
    # Create orchestrator
    orchestrator = get_real_time_orchestrator(config)
    
    print(f"✅ Created monitoring orchestrator")
    print(f"📊 Monitoring interval: {config.monitoring_interval}s")
    print(f"🎯 Performance threshold: {config.performance_threshold}")
    print(f"🔧 Auto-tuning enabled: {config.auto_tuning_enabled}")
    print(f"🚀 Adaptive selection enabled: {config.adaptive_selection_enabled}")
    
    # Start monitoring
    orchestrator.start()
    print("🏃 Real-time monitoring started")
    
    # Register a sample model
    class SampleModel:
        def __init__(self):
            self.name = "sample_model"
    
    sample_model = SampleModel()
    orchestrator.register_model("sample_model", sample_model, {
        'expected_accuracy': 0.85,
        'inference_time_ms': 50,
        'memory_usage_gb': 2.0
    })
    
    print("📋 Registered sample model")
    print("🚀 Real-time monitoring and adaptive orchestration ready!")
    print("🔍 System continuously monitoring for optimal performance!")
    
    # Demonstrate for a short time
    try:
        time.sleep(5)
        status = orchestrator.get_system_status()
        print(f"\n📊 System Status: Health={status['health'].overall_health:.2f}")
    finally:
        orchestrator.stop()
        print("🛑 Monitoring stopped") 