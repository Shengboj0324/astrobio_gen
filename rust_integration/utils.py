"""
Utility functions for Rust integration
======================================

This module provides utility functions for monitoring, benchmarking,
and validating the Rust integration system.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_rust_status() -> Dict[str, Any]:
    """
    Get comprehensive status of Rust integration
    
    Returns:
        Dictionary containing status information
    """
    status = {
        'rust_available': False,
        'rust_version': None,
        'rust_error': None,
        'performance_info': {},
        'test_results': {}
    }
    
    try:
        import astrobio_rust
        status['rust_available'] = True
        status['rust_version'] = astrobio_rust.get_version()
        status['performance_info'] = astrobio_rust.get_performance_info()
        
        # Run basic functionality test
        test_data = np.random.randn(2, 4, 8, 2, 4, 8, 16).astype(np.float32)
        test_samples = [test_data[i] for i in range(test_data.shape[0])]
        
        start_time = time.time()
        inputs, targets = astrobio_rust.process_datacube_batch(
            test_samples, (0, 2, 1, 3, 4, 5, 6), 0.005
        )
        end_time = time.time()
        
        status['test_results'] = {
            'basic_test_passed': True,
            'test_execution_time': end_time - start_time,
            'input_shape': list(inputs.shape),
            'target_shape': list(targets.shape)
        }
        
    except ImportError as e:
        status['rust_error'] = f"Import error: {e}"
    except Exception as e:
        status['rust_error'] = f"Runtime error: {e}"
        status['test_results'] = {'basic_test_passed': False, 'error': str(e)}
    
    return status


def validate_rust_installation() -> Tuple[bool, List[str]]:
    """
    Validate Rust installation and functionality
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        import astrobio_rust
    except ImportError as e:
        errors.append(f"Cannot import astrobio_rust: {e}")
        return False, errors
    
    # Test version function
    try:
        version = astrobio_rust.get_version()
        if not isinstance(version, str) or not version:
            errors.append("get_version() returned invalid result")
    except Exception as e:
        errors.append(f"get_version() failed: {e}")
    
    # Test performance info function
    try:
        perf_info = astrobio_rust.get_performance_info()
        if not isinstance(perf_info, dict):
            errors.append("get_performance_info() returned invalid result")
    except Exception as e:
        errors.append(f"get_performance_info() failed: {e}")
    
    # Test datacube processing
    try:
        # Create small test data
        test_shape = (2, 4, 8, 2, 4, 8, 16)  # Small but realistic shape
        test_data = np.random.randn(*test_shape).astype(np.float32)
        test_samples = [test_data[i] for i in range(test_data.shape[0])]
        
        # Test process_datacube_batch
        inputs, targets = astrobio_rust.process_datacube_batch(
            test_samples, (0, 2, 1, 3, 4, 5, 6), 0.005
        )
        
        # Validate results
        if not isinstance(inputs, np.ndarray) or not isinstance(targets, np.ndarray):
            errors.append("process_datacube_batch() returned invalid types")
        
        if inputs.shape != targets.shape:
            errors.append("process_datacube_batch() returned mismatched shapes")
        
        expected_shape = (2, 8, 4, 2, 4, 8, 16)  # After transpose
        if inputs.shape != expected_shape:
            errors.append(f"process_datacube_batch() returned wrong shape: {inputs.shape} vs {expected_shape}")
        
    except Exception as e:
        errors.append(f"process_datacube_batch() test failed: {e}")
    
    # Test stack_and_transpose
    try:
        test_shape = (4, 8, 2, 4, 8, 16)
        test_data = np.random.randn(2, *test_shape).astype(np.float32)
        test_samples = [test_data[i] for i in range(test_data.shape[0])]
        
        result = astrobio_rust.stack_and_transpose(test_samples, (0, 2, 1, 3, 4, 5, 6))
        
        if not isinstance(result, np.ndarray):
            errors.append("stack_and_transpose() returned invalid type")
        
        expected_shape = (2, 8, 4, 2, 4, 8, 16)
        if result.shape != expected_shape:
            errors.append(f"stack_and_transpose() returned wrong shape: {result.shape} vs {expected_shape}")
        
    except Exception as e:
        errors.append(f"stack_and_transpose() test failed: {e}")
    
    # Test add_noise_and_convert
    try:
        test_array = np.random.randn(2, 4, 8, 2, 4, 8, 16).astype(np.float32)
        inputs, targets = astrobio_rust.add_noise_and_convert(test_array, 0.005)
        
        if not isinstance(inputs, np.ndarray) or not isinstance(targets, np.ndarray):
            errors.append("add_noise_and_convert() returned invalid types")
        
        if inputs.shape != targets.shape or inputs.shape != test_array.shape:
            errors.append("add_noise_and_convert() returned wrong shapes")
        
        # Check that targets have noise (should be different from inputs)
        if np.allclose(inputs, targets):
            errors.append("add_noise_and_convert() did not add noise")
        
    except Exception as e:
        errors.append(f"add_noise_and_convert() test failed: {e}")
    
    return len(errors) == 0, errors


def benchmark_performance(
    batch_sizes: List[int] = [1, 2, 4, 8],
    num_runs: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Benchmark Rust vs Python performance
    
    Args:
        batch_sizes: List of batch sizes to test
        num_runs: Number of runs per test
        verbose: Whether to print results
        
    Returns:
        Dictionary containing benchmark results
    """
    from .datacube_accelerator import DatacubeAccelerator
    
    results = {
        'batch_sizes': batch_sizes,
        'num_runs': num_runs,
        'rust_times': {},
        'python_times': {},
        'speedups': {},
        'rust_available': False
    }
    
    # Check if Rust is available
    try:
        import astrobio_rust
        results['rust_available'] = True
    except ImportError:
        if verbose:
            print("❌ Rust not available, skipping Rust benchmarks")
        return results
    
    accelerator = DatacubeAccelerator(enable_fallback=True, log_performance=False)
    
    for batch_size in batch_sizes:
        if verbose:
            print(f"\n🧪 Benchmarking batch size: {batch_size}")
        
        # Create test data
        sample_shape = (12, 24, 4, 20, 32, 64)  # Realistic but manageable size
        test_samples = []
        for _ in range(batch_size):
            sample = np.random.randn(*sample_shape).astype(np.float32)
            test_samples.append(sample)
        
        # Benchmark Rust implementation
        rust_times = []
        for run in range(num_runs):
            start_time = time.time()
            try:
                import astrobio_rust
                inputs, targets = astrobio_rust.process_datacube_batch(
                    test_samples, (0, 2, 1, 3, 4, 5, 6), 0.005
                )
                end_time = time.time()
                rust_times.append(end_time - start_time)
            except Exception as e:
                if verbose:
                    print(f"   ❌ Rust run {run+1} failed: {e}")
                break
        
        # Benchmark Python implementation
        python_times = []
        for run in range(num_runs):
            start_time = time.time()
            inputs, targets = accelerator._python_process_batch(
                test_samples, (0, 2, 1, 3, 4, 5, 6), 0.005
            )
            end_time = time.time()
            python_times.append(end_time - start_time)
        
        # Calculate statistics
        if rust_times and python_times:
            rust_avg = np.mean(rust_times)
            python_avg = np.mean(python_times)
            speedup = python_avg / rust_avg if rust_avg > 0 else 0
            
            results['rust_times'][batch_size] = {
                'times': rust_times,
                'mean': rust_avg,
                'std': np.std(rust_times)
            }
            results['python_times'][batch_size] = {
                'times': python_times,
                'mean': python_avg,
                'std': np.std(python_times)
            }
            results['speedups'][batch_size] = speedup
            
            if verbose:
                print(f"   🦀 Rust:   {rust_avg:.4f}s ± {np.std(rust_times):.4f}s")
                print(f"   🐍 Python: {python_avg:.4f}s ± {np.std(python_times):.4f}s")
                print(f"   🚀 Speedup: {speedup:.1f}x")
        
        elif python_times:
            python_avg = np.mean(python_times)
            results['python_times'][batch_size] = {
                'times': python_times,
                'mean': python_avg,
                'std': np.std(python_times)
            }
            
            if verbose:
                print(f"   🐍 Python: {python_avg:.4f}s ± {np.std(python_times):.4f}s")
                print(f"   ❌ Rust benchmark failed")
    
    if verbose:
        print(f"\n📊 Benchmark Summary:")
        if results['speedups']:
            avg_speedup = np.mean(list(results['speedups'].values()))
            print(f"   Average speedup: {avg_speedup:.1f}x")
        else:
            print(f"   No speedup data available")
    
    return results
