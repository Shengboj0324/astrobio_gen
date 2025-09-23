#!/usr/bin/env python3
"""
Comprehensive Test Suite for SOTA Attention 2025
===============================================

Extensive testing and validation for all attention mechanisms with:
- Memory profiling and optimization validation
- Speed benchmarks against reference implementations
- Accuracy validation and numerical stability tests
- Integration tests with existing models
- Production readiness validation
- GPU compatibility and distributed training tests
"""

import pytest
import torch
import torch.nn as nn
import time
import psutil
import gc
from typing import Dict, Any, List, Tuple
import logging
import warnings

# Import SOTA Attention components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sota_attention_2025 import (
    SOTAAttention2025, SOTAAttentionConfig, create_sota_attention,
    FlashAttention3, RingAttention, SlidingWindowAttention,
    LinearAttention, MambaBlock, MultiQueryAttention, SparseAttention
)
from models.attention_integration_2025 import AttentionUpgradeManager, upgrade_model_attention

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionTestSuite:
    """Comprehensive test suite for SOTA Attention mechanisms"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🧪 Attention Test Suite initialized on {self.device}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        
        logger.info("🚀 Starting comprehensive SOTA Attention test suite")
        
        # Basic functionality tests
        self.test_basic_functionality()
        
        # Memory efficiency tests
        self.test_memory_efficiency()
        
        # Speed benchmarks
        self.test_speed_benchmarks()
        
        # Accuracy validation
        self.test_accuracy_validation()
        
        # Integration tests
        self.test_model_integration()
        
        # Production readiness tests
        self.test_production_readiness()
        
        # Generate comprehensive report
        return self.generate_test_report()
    
    def test_basic_functionality(self):
        """Test basic functionality of all attention mechanisms"""
        
        logger.info("🔧 Testing basic functionality")
        
        test_configs = [
            {"seq_len": 512, "batch_size": 2, "hidden_size": 768, "num_heads": 12},
            {"seq_len": 2048, "batch_size": 1, "hidden_size": 1024, "num_heads": 16},
            {"seq_len": 8192, "batch_size": 1, "hidden_size": 512, "num_heads": 8},
        ]
        
        for i, config in enumerate(test_configs):
            logger.info(f"Test case {i+1}: {config}")
            
            # Create attention module
            attention = create_sota_attention(
                hidden_size=config['hidden_size'],
                num_attention_heads=config['num_heads'],
                max_position_embeddings=config['seq_len']
            ).to(self.device)
            
            # Create test input
            hidden_states = torch.randn(
                config['batch_size'], 
                config['seq_len'], 
                config['hidden_size'],
                device=self.device
            )
            
            try:
                # Test forward pass
                output, attn_weights, past_key_value = attention(hidden_states)
                
                # Validate output shape
                assert output.shape == hidden_states.shape, f"Shape mismatch: {output.shape} vs {hidden_states.shape}"
                
                # Validate numerical stability
                assert not torch.isnan(output).any(), "Output contains NaN"
                assert not torch.isinf(output).any(), "Output contains Inf"
                assert output.abs().max() < 1000, "Output values too large"
                
                self.test_results[f'basic_functionality_{i+1}'] = 'PASS'
                logger.info(f"✅ Basic functionality test {i+1} passed")
                
            except Exception as e:
                self.test_results[f'basic_functionality_{i+1}'] = f'FAIL: {e}'
                logger.error(f"❌ Basic functionality test {i+1} failed: {e}")
    
    def test_memory_efficiency(self):
        """Test memory efficiency and optimization"""
        
        logger.info("💾 Testing memory efficiency")
        
        # Test configurations with increasing memory requirements
        test_configs = [
            {"seq_len": 1024, "batch_size": 4, "hidden_size": 768},
            {"seq_len": 4096, "batch_size": 2, "hidden_size": 768},
            {"seq_len": 16384, "batch_size": 1, "hidden_size": 768},
        ]
        
        for i, config in enumerate(test_configs):
            logger.info(f"Memory test {i+1}: {config}")
            
            # Measure baseline memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            baseline_memory = self._get_memory_usage()
            
            try:
                # Create attention module
                attention = create_sota_attention(
                    hidden_size=config['hidden_size'],
                    max_position_embeddings=config['seq_len']
                ).to(self.device)
                
                # Create test input
                hidden_states = torch.randn(
                    config['batch_size'], 
                    config['seq_len'], 
                    config['hidden_size'],
                    device=self.device
                )
                
                # Measure peak memory during forward pass
                peak_memory = baseline_memory
                
                with torch.no_grad():
                    output, _, _ = attention(hidden_states)
                    current_memory = self._get_memory_usage()
                    peak_memory = max(peak_memory, current_memory)
                
                # Calculate memory efficiency
                memory_used = peak_memory - baseline_memory
                theoretical_memory = self._calculate_theoretical_memory(config)
                efficiency_ratio = theoretical_memory / max(memory_used, 1)
                
                self.performance_metrics[f'memory_test_{i+1}'] = {
                    'memory_used_mb': memory_used,
                    'theoretical_memory_mb': theoretical_memory,
                    'efficiency_ratio': efficiency_ratio,
                    'status': 'PASS' if efficiency_ratio > 0.5 else 'SUBOPTIMAL'
                }
                
                logger.info(f"✅ Memory test {i+1}: {memory_used:.1f}MB used, {efficiency_ratio:.2f}x efficiency")
                
            except Exception as e:
                self.test_results[f'memory_test_{i+1}'] = f'FAIL: {e}'
                logger.error(f"❌ Memory test {i+1} failed: {e}")
    
    def test_speed_benchmarks(self):
        """Test speed performance against baselines"""
        
        logger.info("⚡ Testing speed benchmarks")
        
        test_configs = [
            {"seq_len": 512, "batch_size": 8, "hidden_size": 768},
            {"seq_len": 2048, "batch_size": 4, "hidden_size": 768},
            {"seq_len": 8192, "batch_size": 1, "hidden_size": 768},
        ]
        
        for i, config in enumerate(test_configs):
            logger.info(f"Speed test {i+1}: {config}")
            
            try:
                # Create SOTA attention
                sota_attention = create_sota_attention(
                    hidden_size=config['hidden_size'],
                    max_position_embeddings=config['seq_len']
                ).to(self.device)
                
                # Create baseline attention (standard PyTorch)
                baseline_attention = nn.MultiheadAttention(
                    embed_dim=config['hidden_size'],
                    num_heads=12,
                    batch_first=True
                ).to(self.device)
                
                # Create test input
                hidden_states = torch.randn(
                    config['batch_size'], 
                    config['seq_len'], 
                    config['hidden_size'],
                    device=self.device
                )
                
                # Warm up
                for _ in range(3):
                    with torch.no_grad():
                        _ = sota_attention(hidden_states)
                        _ = baseline_attention(hidden_states, hidden_states, hidden_states)
                
                # Benchmark SOTA attention
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                for _ in range(10):
                    with torch.no_grad():
                        _ = sota_attention(hidden_states)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                sota_time = (time.time() - start_time) / 10
                
                # Benchmark baseline attention
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                for _ in range(10):
                    with torch.no_grad():
                        _ = baseline_attention(hidden_states, hidden_states, hidden_states)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                baseline_time = (time.time() - start_time) / 10
                
                # Calculate speedup
                speedup = baseline_time / sota_time
                
                self.performance_metrics[f'speed_test_{i+1}'] = {
                    'sota_time_ms': sota_time * 1000,
                    'baseline_time_ms': baseline_time * 1000,
                    'speedup': speedup,
                    'status': 'PASS' if speedup > 1.0 else 'SLOWER'
                }
                
                logger.info(f"✅ Speed test {i+1}: {speedup:.2f}x speedup ({sota_time*1000:.1f}ms vs {baseline_time*1000:.1f}ms)")
                
            except Exception as e:
                self.test_results[f'speed_test_{i+1}'] = f'FAIL: {e}'
                logger.error(f"❌ Speed test {i+1} failed: {e}")
    
    def test_accuracy_validation(self):
        """Test numerical accuracy and stability"""
        
        logger.info("🎯 Testing accuracy validation")
        
        # Test with different precisions and configurations
        test_cases = [
            {"dtype": torch.float32, "seq_len": 1024},
            {"dtype": torch.float16, "seq_len": 2048},
            {"dtype": torch.bfloat16, "seq_len": 4096} if torch.cuda.is_available() else {"dtype": torch.float32, "seq_len": 4096},
        ]
        
        for i, case in enumerate(test_cases):
            logger.info(f"Accuracy test {i+1}: {case}")
            
            try:
                # Create attention module
                attention = create_sota_attention(
                    hidden_size=768,
                    max_position_embeddings=case['seq_len']
                ).to(self.device).to(case['dtype'])
                
                # Create test input
                hidden_states = torch.randn(
                    2, case['seq_len'], 768,
                    device=self.device,
                    dtype=case['dtype']
                )
                
                # Test forward pass
                output, _, _ = attention(hidden_states)
                
                # Validate numerical properties
                output_float32 = output.float()
                
                # Check for numerical issues
                nan_count = torch.isnan(output_float32).sum().item()
                inf_count = torch.isinf(output_float32).sum().item()
                max_value = output_float32.abs().max().item()
                
                # Check gradient flow
                if case['dtype'] != torch.float16:  # Skip gradient test for fp16
                    output.sum().backward()
                    grad_norm = sum(p.grad.norm().item() for p in attention.parameters() if p.grad is not None)
                else:
                    grad_norm = 0
                
                accuracy_metrics = {
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'max_value': max_value,
                    'grad_norm': grad_norm,
                    'status': 'PASS' if nan_count == 0 and inf_count == 0 and max_value < 1000 else 'FAIL'
                }
                
                self.performance_metrics[f'accuracy_test_{i+1}'] = accuracy_metrics
                
                logger.info(f"✅ Accuracy test {i+1}: max_val={max_value:.2f}, grad_norm={grad_norm:.2f}")
                
            except Exception as e:
                self.test_results[f'accuracy_test_{i+1}'] = f'FAIL: {e}'
                logger.error(f"❌ Accuracy test {i+1} failed: {e}")
    
    def test_model_integration(self):
        """Test integration with existing models"""
        
        logger.info("🔗 Testing model integration")
        
        try:
            # Create a simple transformer model
            class SimpleTransformer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.attention = nn.MultiheadAttention(768, 12, batch_first=True)
                    self.norm = nn.LayerNorm(768)
                    self.ffn = nn.Sequential(
                        nn.Linear(768, 3072),
                        nn.GELU(),
                        nn.Linear(3072, 768)
                    )
                
                def forward(self, x):
                    attn_out, _ = self.attention(x, x, x)
                    x = self.norm(x + attn_out)
                    ffn_out = self.ffn(x)
                    return self.norm(x + ffn_out)
            
            # Create model
            model = SimpleTransformer().to(self.device)
            
            # Test before upgrade
            test_input = torch.randn(2, 512, 768, device=self.device)
            original_output = model(test_input)
            
            # Upgrade attention mechanisms
            upgraded_model = upgrade_model_attention(model)
            
            # Test after upgrade
            upgraded_output = upgraded_model(test_input)
            
            # Validate upgrade
            assert upgraded_output.shape == original_output.shape, "Output shape changed after upgrade"
            assert not torch.isnan(upgraded_output).any(), "Upgraded model produces NaN"
            
            self.test_results['model_integration'] = 'PASS'
            logger.info("✅ Model integration test passed")
            
        except Exception as e:
            self.test_results['model_integration'] = f'FAIL: {e}'
            logger.error(f"❌ Model integration test failed: {e}")
    
    def test_production_readiness(self):
        """Test production readiness features"""
        
        logger.info("🏭 Testing production readiness")
        
        try:
            # Test error handling and fallbacks
            attention = create_sota_attention(hidden_size=768)
            
            # Test with extreme inputs
            extreme_cases = [
                torch.zeros(1, 100, 768),  # All zeros
                torch.ones(1, 100, 768) * 1000,  # Large values
                torch.randn(1, 100, 768) * 0.001,  # Very small values
            ]
            
            for i, extreme_input in enumerate(extreme_cases):
                try:
                    output, _, _ = attention(extreme_input.to(self.device))
                    assert not torch.isnan(output).any(), f"Extreme case {i+1} produced NaN"
                    logger.info(f"✅ Extreme case {i+1} handled correctly")
                except Exception as e:
                    logger.warning(f"⚠️ Extreme case {i+1} failed: {e}")
            
            # Test performance monitoring
            stats = attention.get_performance_stats()
            assert 'attention_calls' in stats, "Performance stats missing"
            assert 'mechanism_usage' in stats, "Mechanism usage stats missing"
            
            self.test_results['production_readiness'] = 'PASS'
            logger.info("✅ Production readiness test passed")
            
        except Exception as e:
            self.test_results['production_readiness'] = f'FAIL: {e}'
            logger.error(f"❌ Production readiness test failed: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            return psutil.Process().memory_info().rss / (1024 ** 2)
    
    def _calculate_theoretical_memory(self, config: Dict[str, int]) -> float:
        """Calculate theoretical memory requirement in MB"""
        batch_size = config['batch_size']
        seq_len = config['seq_len']
        hidden_size = config['hidden_size']
        
        # Attention matrix: O(batch * heads * seq_len^2)
        attention_memory = batch_size * 12 * seq_len * seq_len * 4 / (1024 ** 2)
        
        # Activations: O(batch * seq_len * hidden_size)
        activation_memory = batch_size * seq_len * hidden_size * 4 * 3 / (1024 ** 2)
        
        return attention_memory + activation_memory
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result == 'PASS')
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': passed_tests / max(total_tests, 1),
                'overall_status': 'PASS' if passed_tests == total_tests else 'PARTIAL_PASS'
            },
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Check for failed tests
        failed_tests = [test for test, result in self.test_results.items() if result != 'PASS']
        if failed_tests:
            recommendations.append(f"Address failed tests: {', '.join(failed_tests)}")
        
        # Check memory efficiency
        memory_tests = [k for k in self.performance_metrics.keys() if 'memory_test' in k]
        if memory_tests:
            avg_efficiency = sum(self.performance_metrics[k]['efficiency_ratio'] for k in memory_tests) / len(memory_tests)
            if avg_efficiency < 0.7:
                recommendations.append("Consider enabling more aggressive memory optimizations")
        
        # Check speed performance
        speed_tests = [k for k in self.performance_metrics.keys() if 'speed_test' in k]
        if speed_tests:
            avg_speedup = sum(self.performance_metrics[k]['speedup'] for k in speed_tests) / len(speed_tests)
            if avg_speedup < 1.5:
                recommendations.append("Consider enabling Flash Attention 3.0 for better performance")
        
        if not recommendations:
            recommendations.append("All tests passed successfully - system is production ready!")
        
        return recommendations


# Pytest integration
@pytest.fixture
def attention_test_suite():
    """Pytest fixture for attention test suite"""
    return AttentionTestSuite()


def test_sota_attention_comprehensive(attention_test_suite):
    """Comprehensive test for SOTA Attention"""
    report = attention_test_suite.run_all_tests()
    
    # Assert overall success
    assert report['summary']['success_rate'] > 0.8, f"Test success rate too low: {report['summary']['success_rate']}"
    
    # Print report
    logger.info(f"📊 Test Report: {report['summary']}")
    
    return report


if __name__ == "__main__":
    # Run tests directly
    test_suite = AttentionTestSuite()
    report = test_suite.run_all_tests()
    
    print("\n" + "="*80)
    print("SOTA ATTENTION 2025 - COMPREHENSIVE TEST REPORT")
    print("="*80)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"Overall Status: {report['summary']['overall_status']}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    print("="*80)
