"""
Continuous Validation System for Astrobiology AI
===============================================

Implements continuous monitoring, regression detection, and automated testing
for rock-solid reliability in production environments.

Features:
- Real-time performance monitoring
- Automated regression detection
- Continuous integration testing
- Live dashboard metrics
- Alert system for critical issues
- Performance benchmarking
- Resource utilization tracking
"""

import os
import time
import json
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests/continuous_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    timestamp: str
    model_name: str
    loss_value: float
    gradient_norm: float
    memory_usage_mb: float
    forward_time_ms: float
    backward_time_ms: float
    parameter_count: int
    training_step: int

@dataclass
class SystemHealth:
    """Overall system health metrics"""
    timestamp: str
    total_models: int
    working_models: int
    success_rate: float
    average_loss: float
    memory_usage_gb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    disk_usage_gb: float

class ContinuousValidationSystem:
    """
    Continuous validation and monitoring system
    """
    
    def __init__(self, config_path: str = "config/master_training.yaml"):
        self.config_path = config_path
        self.metrics_history: List[PerformanceMetrics] = []
        self.health_history: List[SystemHealth] = []
        self.alerts: List[Dict[str, Any]] = []
        self.baseline_metrics: Optional[Dict[str, float]] = None
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Monitoring configuration
        self.config = {
            'monitoring_interval': 60,  # seconds
            'regression_threshold': 0.1,  # 10% performance degradation
            'memory_limit_gb': 16,
            'cpu_limit_percent': 90,
            'alert_cooldown': 300,  # 5 minutes
            'max_history_size': 1000,
            'performance_window': 10  # Number of recent measurements for trend analysis
        }
        
        logger.info("🔍 Continuous Validation System initialized")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("🚀 Continuous monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("⏹️ Continuous monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Run validation cycle
                self._run_validation_cycle()
                
                # Sleep until next cycle
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                logger.error(f"Monitoring cycle failed: {e}")
                self._create_alert("MONITORING_ERROR", f"Monitoring cycle failed: {e}")
                time.sleep(self.config['monitoring_interval'])
    
    def _run_validation_cycle(self):
        """Run a single validation cycle"""
        logger.info("🔄 Running validation cycle...")
        
        try:
            # Import and initialize system
            from train_sota_unified import SOTAUnifiedTrainer
            
            trainer = SOTAUnifiedTrainer(self.config_path)
            models = trainer.initialize_sota_models()
            trainer.initialize_sota_training()
            
            # Collect system health metrics
            system_health = self._collect_system_health(models)
            self.health_history.append(system_health)
            
            # Test each model
            for model_name, model in models.items():
                metrics = self._test_model_performance(model_name, model)
                if metrics:
                    self.metrics_history.append(metrics)
            
            # Check for regressions
            self._check_for_regressions()
            
            # Cleanup old history
            self._cleanup_history()
            
            # Save metrics
            self._save_metrics()
            
            logger.info(f"✅ Validation cycle completed - {len(models)} models tested")
            
        except Exception as e:
            logger.error(f"Validation cycle failed: {e}")
            self._create_alert("VALIDATION_FAILURE", f"Validation cycle failed: {e}")
    
    def _collect_system_health(self, models: Dict[str, nn.Module]) -> SystemHealth:
        """Collect overall system health metrics"""
        try:
            # Test unified training
            from train_sota_unified import SOTAUnifiedTrainer
            trainer = SOTAUnifiedTrainer(self.config_path)
            trainer.initialize_sota_training()
            
            dummy_batch = trainer._create_dummy_batch()
            
            if trainer.sota_orchestrator:
                losses = trainer.sota_orchestrator.unified_training_step(dummy_batch, 0)
                
                working_models = 0
                total_loss = 0.0
                
                for model_name, model_losses in losses.items():
                    if isinstance(model_losses, dict) and 'total_loss' in model_losses:
                        loss_val = model_losses['total_loss']
                        if not (torch.isnan(torch.tensor(loss_val)) or torch.isinf(torch.tensor(loss_val))):
                            working_models += 1
                            total_loss += loss_val
                
                success_rate = working_models / len(losses) if losses else 0.0
                average_loss = total_loss / working_models if working_models > 0 else 0.0
            else:
                working_models = 0
                success_rate = 0.0
                average_loss = 0.0
            
            # System resource metrics (simplified)
            try:
                import psutil
                memory_gb = psutil.virtual_memory().used / (1024**3)
                cpu_percent = psutil.cpu_percent()
                disk_gb = psutil.disk_usage('.').used / (1024**3)
            except ImportError:
                memory_gb = 0.0
                cpu_percent = 0.0
                disk_gb = 0.0
            
            return SystemHealth(
                timestamp=datetime.now().isoformat(),
                total_models=len(models),
                working_models=working_models,
                success_rate=success_rate,
                average_loss=average_loss,
                memory_usage_gb=memory_gb,
                cpu_usage_percent=cpu_percent,
                gpu_usage_percent=0.0,  # Simplified
                disk_usage_gb=disk_gb
            )
            
        except Exception as e:
            logger.error(f"Error collecting system health: {e}")
            return SystemHealth(
                timestamp=datetime.now().isoformat(),
                total_models=len(models),
                working_models=0,
                success_rate=0.0,
                average_loss=0.0,
                memory_usage_gb=0.0,
                cpu_usage_percent=0.0,
                gpu_usage_percent=0.0,
                disk_usage_gb=0.0
            )
    
    def _test_model_performance(self, model_name: str, model: nn.Module) -> Optional[PerformanceMetrics]:
        """Test individual model performance"""
        try:
            model.train()
            
            # Create test input
            if model_name == 'rebuilt_graph_vae':
                from torch_geometric.data import Data
                x = torch.randn(12, 16, requires_grad=True)
                edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
                batch = torch.zeros(12, dtype=torch.long)
                test_input = Data(x=x, edge_index=edge_index, batch=batch)
            elif model_name == 'rebuilt_datacube_cnn':
                test_input = torch.randn(1, 5, 4, 4, 8, 16, 16, requires_grad=True)
            elif model_name == 'rebuilt_llm_integration':
                test_input = {
                    'input_ids': torch.randint(0, 1000, (2, 32)),
                    'attention_mask': torch.ones(2, 32),
                    'labels': torch.randint(0, 1000, (2, 32))
                }
            elif model_name == 'diffusion_model':
                test_input = torch.randn(2, 3, 32, 32, requires_grad=True)
            elif model_name == 'spectral_surrogate':
                test_input = torch.randn(4, 4, requires_grad=True)
            elif model_name == 'surrogate_transformer':
                test_input = torch.randn(4, 128, requires_grad=True)
            elif model_name == 'advanced_graph_neural_network':
                test_input = torch.randn(10, 128, requires_grad=True)
            else:
                test_input = torch.randn(4, 128, requires_grad=True)
            
            # Forward pass timing
            start_time = time.time()
            if isinstance(test_input, dict):
                output = model(**test_input)
            else:
                output = model(test_input)
            forward_time = (time.time() - start_time) * 1000  # ms
            
            # Extract loss
            if isinstance(output, dict) and 'loss' in output:
                loss = output['loss']
                if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                    loss_value = loss.item()
                else:
                    return None  # Skip invalid loss
            else:
                return None  # Skip models without loss
            
            # Backward pass timing
            model.zero_grad()
            start_time = time.time()
            loss.backward()
            backward_time = (time.time() - start_time) * 1000  # ms
            
            # Calculate gradient norm
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            gradient_norm = total_norm ** (1. / 2)
            
            # Parameter count
            param_count = sum(p.numel() for p in model.parameters())
            
            # Memory usage (simplified)
            memory_mb = param_count * 4 / (1024 * 1024)  # Approximate
            
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                model_name=model_name,
                loss_value=loss_value,
                gradient_norm=gradient_norm,
                memory_usage_mb=memory_mb,
                forward_time_ms=forward_time,
                backward_time_ms=backward_time,
                parameter_count=param_count,
                training_step=0
            )
            
        except Exception as e:
            logger.warning(f"Error testing {model_name}: {e}")
            return None
    
    def _check_for_regressions(self):
        """Check for performance regressions"""
        if len(self.metrics_history) < self.config['performance_window']:
            return
        
        # Group metrics by model
        model_metrics = {}
        for metric in self.metrics_history[-self.config['performance_window']:]:
            if metric.model_name not in model_metrics:
                model_metrics[metric.model_name] = []
            model_metrics[metric.model_name].append(metric)
        
        # Check each model for regressions
        for model_name, metrics in model_metrics.items():
            if len(metrics) >= 2:
                recent_loss = metrics[-1].loss_value
                baseline_loss = metrics[0].loss_value
                
                if baseline_loss > 0:
                    regression = (recent_loss - baseline_loss) / baseline_loss
                    
                    if regression > self.config['regression_threshold']:
                        self._create_alert(
                            "PERFORMANCE_REGRESSION",
                            f"Model {model_name} shows {regression*100:.1f}% performance regression"
                        )
    
    def _create_alert(self, alert_type: str, message: str):
        """Create system alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': 'HIGH' if 'CRITICAL' in alert_type else 'MEDIUM'
        }
        
        self.alerts.append(alert)
        logger.warning(f"🚨 ALERT [{alert_type}]: {message}")
    
    def _cleanup_history(self):
        """Cleanup old history to prevent memory bloat"""
        max_size = self.config['max_history_size']
        
        if len(self.metrics_history) > max_size:
            self.metrics_history = self.metrics_history[-max_size:]
        
        if len(self.health_history) > max_size:
            self.health_history = self.health_history[-max_size:]
        
        if len(self.alerts) > max_size:
            self.alerts = self.alerts[-max_size:]
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            metrics_data = {
                'performance_metrics': [asdict(m) for m in self.metrics_history[-100:]],
                'health_metrics': [asdict(h) for h in self.health_history[-100:]],
                'alerts': self.alerts[-50:],
                'last_updated': datetime.now().isoformat()
            }
            
            with open('tests/continuous_metrics.json', 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        if not self.health_history:
            return {"error": "No health data available"}
        
        recent_health = self.health_history[-1]
        
        # Calculate trends
        if len(self.health_history) >= 2:
            prev_health = self.health_history[-2]
            success_rate_trend = recent_health.success_rate - prev_health.success_rate
            memory_trend = recent_health.memory_usage_gb - prev_health.memory_usage_gb
        else:
            success_rate_trend = 0.0
            memory_trend = 0.0
        
        # Count recent alerts
        recent_alerts = [a for a in self.alerts 
                        if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)]
        
        report = {
            'system_status': {
                'overall_health': 'EXCELLENT' if recent_health.success_rate > 0.9 else 
                                'GOOD' if recent_health.success_rate > 0.7 else 
                                'NEEDS_ATTENTION',
                'success_rate': recent_health.success_rate,
                'working_models': f"{recent_health.working_models}/{recent_health.total_models}",
                'average_loss': recent_health.average_loss,
                'memory_usage_gb': recent_health.memory_usage_gb
            },
            'trends': {
                'success_rate_trend': success_rate_trend,
                'memory_trend_gb': memory_trend,
                'trend_direction': 'IMPROVING' if success_rate_trend > 0 else 
                                 'STABLE' if success_rate_trend == 0 else 'DECLINING'
            },
            'alerts': {
                'recent_alerts_count': len(recent_alerts),
                'critical_alerts': [a for a in recent_alerts if a['severity'] == 'HIGH'],
                'last_alert': recent_alerts[-1] if recent_alerts else None
            },
            'recommendations': self._generate_recommendations(recent_health, recent_alerts)
        }
        
        return report
    
    def _generate_recommendations(self, health: SystemHealth, alerts: List[Dict]) -> List[str]:
        """Generate recommendations based on current health"""
        recommendations = []
        
        if health.success_rate < 0.8:
            recommendations.append("Investigate failing models and fix critical issues")
        
        if health.memory_usage_gb > self.config['memory_limit_gb'] * 0.8:
            recommendations.append("Monitor memory usage - approaching limit")
        
        if len(alerts) > 5:
            recommendations.append("High alert frequency - investigate system stability")
        
        if health.average_loss > 100:
            recommendations.append("High average loss - check model convergence")
        
        if not recommendations:
            recommendations.append("System operating normally - continue monitoring")
        
        return recommendations
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation suite"""
        logger.info("🧪 Running comprehensive validation suite...")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'model_tests': {},
            'integration_tests': {},
            'stress_tests': {},
            'overall_score': 0.0
        }
        
        try:
            # Run model tests
            from train_sota_unified import SOTAUnifiedTrainer
            
            trainer = SOTAUnifiedTrainer(self.config_path)
            models = trainer.initialize_sota_models()
            trainer.initialize_sota_training()
            
            # Test each model
            working_models = 0
            for model_name, model in models.items():
                test_result = self._comprehensive_model_test(model_name, model)
                validation_results['model_tests'][model_name] = test_result
                
                if test_result['status'] == 'PASS':
                    working_models += 1
            
            # Test integration
            integration_result = self._test_system_integration(trainer)
            validation_results['integration_tests'] = integration_result
            
            # Calculate overall score
            model_score = (working_models / len(models)) * 100 if models else 0
            integration_score = 100 if integration_result['status'] == 'PASS' else 0
            
            validation_results['overall_score'] = (model_score + integration_score) / 2
            
            logger.info(f"✅ Comprehensive validation completed - Score: {validation_results['overall_score']:.1f}%")
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _comprehensive_model_test(self, model_name: str, model: nn.Module) -> Dict[str, Any]:
        """Comprehensive test for individual model"""
        try:
            # Test forward pass, backward pass, and gradient flow
            model.train()
            
            # Create test input (same logic as before)
            test_input = self._create_test_input(model_name)
            
            # Forward pass
            if isinstance(test_input, dict):
                output = model(**test_input)
            else:
                output = model(test_input)
            
            # Check loss
            if isinstance(output, dict) and 'loss' in output:
                loss = output['loss']
                if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                    # Backward pass
                    model.zero_grad()
                    loss.backward()
                    
                    # Check gradients
                    healthy_grads = sum(1 for p in model.parameters() 
                                      if p.requires_grad and p.grad is not None and 
                                      not torch.isnan(p.grad).any() and p.grad.norm() > 1e-8)
                    
                    total_params = sum(1 for p in model.parameters() if p.requires_grad)
                    
                    if healthy_grads > total_params * 0.8:
                        return {
                            'status': 'PASS',
                            'loss': loss.item(),
                            'healthy_gradients': healthy_grads,
                            'total_parameters': total_params
                        }
            
            return {'status': 'FAIL', 'reason': 'Invalid loss or gradients'}
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    def _create_test_input(self, model_name: str):
        """Create appropriate test input for model"""
        if model_name == 'rebuilt_graph_vae':
            from torch_geometric.data import Data
            x = torch.randn(12, 16, requires_grad=True)
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            batch = torch.zeros(12, dtype=torch.long)
            return Data(x=x, edge_index=edge_index, batch=batch)
        elif model_name == 'rebuilt_datacube_cnn':
            return torch.randn(1, 5, 4, 4, 8, 16, 16, requires_grad=True)
        elif model_name == 'rebuilt_llm_integration':
            return {
                'input_ids': torch.randint(0, 1000, (2, 32)),
                'attention_mask': torch.ones(2, 32),
                'labels': torch.randint(0, 1000, (2, 32))
            }
        elif model_name == 'diffusion_model':
            return torch.randn(2, 3, 32, 32, requires_grad=True)
        elif model_name == 'spectral_surrogate':
            return torch.randn(4, 4, requires_grad=True)
        elif model_name == 'surrogate_transformer':
            return torch.randn(4, 128, requires_grad=True)
        elif model_name == 'advanced_graph_neural_network':
            return torch.randn(10, 128, requires_grad=True)
        else:
            return torch.randn(4, 128, requires_grad=True)
    
    def _test_system_integration(self, trainer) -> Dict[str, Any]:
        """Test system integration"""
        try:
            if trainer.sota_orchestrator:
                dummy_batch = trainer._create_dummy_batch()
                losses = trainer.sota_orchestrator.unified_training_step(dummy_batch, 0)
                
                working = sum(1 for model_losses in losses.values()
                            if isinstance(model_losses, dict) and 'total_loss' in model_losses and
                            not torch.isnan(torch.tensor(model_losses['total_loss'])))
                
                success_rate = working / len(losses) if losses else 0.0
                
                return {
                    'status': 'PASS' if success_rate > 0.7 else 'FAIL',
                    'working_models': working,
                    'total_models': len(losses),
                    'success_rate': success_rate
                }
            else:
                return {'status': 'FAIL', 'reason': 'No orchestrator available'}
                
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}


# Convenience functions
def start_continuous_monitoring():
    """Start continuous monitoring system"""
    validator = ContinuousValidationSystem()
    validator.start_monitoring()
    return validator

def run_validation_suite():
    """Run complete validation suite"""
    validator = ContinuousValidationSystem()
    return validator.run_comprehensive_validation()


if __name__ == "__main__":
    # Run validation when executed directly
    print("🚀 Starting Continuous Validation System")
    
    validator = ContinuousValidationSystem()
    report = validator.run_comprehensive_validation()
    
    print(f"✅ Validation completed!")
    print(f"Overall score: {report['overall_score']:.1f}%")
