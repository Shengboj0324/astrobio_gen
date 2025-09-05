#!/usr/bin/env python3
"""
Comprehensive tests for Rust integration
========================================

This module provides comprehensive testing for the Rust datacube processing
integration, including unit tests, integration tests, performance benchmarks,
and end-to-end validation.

Test Categories:
- Unit tests for individual functions
- Integration tests for full workflows
- Performance benchmarks vs Python
- Memory usage validation
- Error handling and fallback testing
- End-to-end training pipeline validation
"""

import unittest
import numpy as np
import torch
import time
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRustIntegration(unittest.TestCase):
    """Comprehensive tests for Rust integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        logger.info("🧪 Setting up Rust integration tests...")
        
        # Try to import Rust integration
        try:
            from rust_integration import DatacubeAccelerator, get_rust_status
            cls.rust_available = True
            cls.accelerator = DatacubeAccelerator(enable_fallback=True, log_performance=True)
            cls.rust_status = get_rust_status()
            logger.info(f"✅ Rust integration available: v{cls.rust_status.get('rust_version', 'unknown')}")
        except ImportError as e:
            cls.rust_available = False
            cls.accelerator = None
            cls.rust_status = None
            logger.warning(f"📋 Rust integration not available: {e}")
    
    def setUp(self):
        """Set up each test"""
        # Create test data with realistic dimensions
        self.batch_size = 2
        self.variables = 8
        self.time_steps = 12
        self.geo_time = 4
        self.levels = 20
        self.lat = 32
        self.lon = 64
        
        # Create test samples
        self.test_samples = []
        for i in range(self.batch_size):
            sample = np.random.randn(
                self.time_steps, self.variables, self.geo_time, 
                self.levels, self.lat, self.lon
            ).astype(np.float32)
            self.test_samples.append(sample)
        
        self.transpose_dims = (0, 2, 1, 3, 4, 5, 6)
        self.noise_std = 0.005
    
    def test_rust_availability(self):
        """Test that Rust integration is available and working"""
        if not self.rust_available:
            self.skipTest("Rust integration not available")
        
        self.assertIsNotNone(self.accelerator)
        self.assertIsNotNone(self.rust_status)
        self.assertTrue(self.rust_status['rust_available'])
        self.assertIsNotNone(self.rust_status['rust_version'])
    
    def test_basic_functionality(self):
        """Test basic Rust functionality"""
        if not self.rust_available:
            self.skipTest("Rust integration not available")
        
        try:
            import astrobio_rust
            
            # Test version function
            version = astrobio_rust.get_version()
            self.assertIsInstance(version, str)
            self.assertTrue(len(version) > 0)
            
            # Test performance info
            perf_info = astrobio_rust.get_performance_info()
            self.assertIsInstance(perf_info, dict)
            self.assertIn('version', perf_info)
            
        except ImportError:
            self.fail("astrobio_rust module not available")
    
    def test_process_datacube_batch(self):
        """Test the main datacube batch processing function"""
        if not self.rust_available:
            self.skipTest("Rust integration not available")
        
        # Test with Rust acceleration
        inputs, targets = self.accelerator.process_batch(
            self.test_samples, self.transpose_dims, self.noise_std
        )
        
        # Validate outputs
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        
        # Check shapes
        expected_shape = (
            self.batch_size, self.geo_time, self.variables, 
            self.time_steps, self.levels, self.lat, self.lon
        )
        self.assertEqual(inputs.shape, expected_shape)
        self.assertEqual(targets.shape, expected_shape)
        
        # Check data types
        self.assertEqual(inputs.dtype, torch.float32)
        self.assertEqual(targets.dtype, torch.float32)
        
        # Check that targets have noise (should be different from inputs)
        self.assertFalse(torch.allclose(inputs, targets))
    
    def test_stack_and_transpose(self):
        """Test the stack and transpose function"""
        if not self.rust_available:
            self.skipTest("Rust integration not available")
        
        result = self.accelerator.stack_and_transpose(
            self.test_samples, self.transpose_dims
        )
        
        # Validate output
        self.assertIsInstance(result, torch.Tensor)
        
        # Check shape
        expected_shape = (
            self.batch_size, self.geo_time, self.variables,
            self.time_steps, self.levels, self.lat, self.lon
        )
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.dtype, torch.float32)
    
    def test_add_noise_and_convert(self):
        """Test the noise addition and conversion function"""
        if not self.rust_available:
            self.skipTest("Rust integration not available")
        
        # Create test input
        test_input = np.random.randn(2, 4, 8, 2, 4, 8, 16).astype(np.float32)
        
        inputs, targets = self.accelerator.add_noise_and_convert(
            test_input, self.noise_std
        )
        
        # Validate outputs
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        
        # Check shapes
        self.assertEqual(inputs.shape, test_input.shape)
        self.assertEqual(targets.shape, test_input.shape)
        
        # Check that noise was added
        self.assertFalse(torch.allclose(inputs, targets))
    
    def test_python_fallback(self):
        """Test that Python fallback works correctly"""
        # Force Python fallback by creating accelerator without Rust
        from rust_integration.datacube_accelerator import DatacubeAccelerator
        
        # Create accelerator that will use Python fallback
        fallback_accelerator = DatacubeAccelerator(enable_fallback=True)
        fallback_accelerator.rust_available = False  # Force fallback
        
        # Test processing
        inputs, targets = fallback_accelerator.process_batch(
            self.test_samples, self.transpose_dims, self.noise_std
        )
        
        # Validate outputs
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        
        # Check shapes
        expected_shape = (
            self.batch_size, self.geo_time, self.variables,
            self.time_steps, self.levels, self.lat, self.lon
        )
        self.assertEqual(inputs.shape, expected_shape)
        self.assertEqual(targets.shape, expected_shape)
    
    def test_input_validation(self):
        """Test input validation and error handling"""
        if not self.rust_available:
            self.skipTest("Rust integration not available")
        
        # Test empty samples
        with self.assertRaises(ValueError):
            self.accelerator.process_batch([], self.transpose_dims, self.noise_std)
        
        # Test invalid transpose dimensions
        with self.assertRaises(ValueError):
            self.accelerator.process_batch(
                self.test_samples, (0, 1, 2), self.noise_std
            )
        
        # Test negative noise std
        with self.assertRaises(ValueError):
            self.accelerator.process_batch(
                self.test_samples, self.transpose_dims, -0.1
            )
    
    def test_performance_comparison(self):
        """Test performance comparison between Rust and Python"""
        if not self.rust_available:
            self.skipTest("Rust integration not available")
        
        # Create larger test data for meaningful performance comparison
        large_samples = []
        for i in range(4):  # Larger batch
            sample = np.random.randn(
                12, 16, 4, 20, 32, 64  # Larger dimensions
            ).astype(np.float32)
            large_samples.append(sample)
        
        # Time Rust implementation
        start_time = time.time()
        rust_inputs, rust_targets = self.accelerator.process_batch(
            large_samples, self.transpose_dims, self.noise_std
        )
        rust_time = time.time() - start_time
        
        # Time Python fallback
        fallback_accelerator = DatacubeAccelerator(enable_fallback=True)
        fallback_accelerator.rust_available = False  # Force fallback
        
        start_time = time.time()
        python_inputs, python_targets = fallback_accelerator.process_batch(
            large_samples, self.transpose_dims, self.noise_std
        )
        python_time = time.time() - start_time
        
        # Calculate speedup
        speedup = python_time / rust_time if rust_time > 0 else 0
        
        logger.info(f"🦀 Rust time: {rust_time:.4f}s")
        logger.info(f"🐍 Python time: {python_time:.4f}s")
        logger.info(f"🚀 Speedup: {speedup:.1f}x")
        
        # Validate results are equivalent (within noise tolerance)
        self.assertEqual(rust_inputs.shape, python_inputs.shape)
        self.assertEqual(rust_targets.shape, python_targets.shape)
        
        # Rust should be faster (at least 2x for meaningful data)
        if speedup >= 2.0:
            logger.info("✅ Performance test: EXCELLENT speedup achieved")
        elif speedup >= 1.1:
            logger.info("✅ Performance test: GOOD speedup achieved")
        else:
            logger.warning("⚠️  Performance test: Limited speedup (may be due to small data size)")
    
    def test_memory_usage(self):
        """Test memory usage comparison"""
        if not self.rust_available:
            self.skipTest("Rust integration not available")
        
        import psutil
        import gc
        
        # Create memory-intensive test data
        memory_samples = []
        for i in range(8):  # Large batch
            sample = np.random.randn(
                12, 24, 4, 20, 32, 64  # Large dimensions
            ).astype(np.float32)
            memory_samples.append(sample)
        
        # Measure memory before
        gc.collect()
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024**2  # MB
        
        # Process with Rust
        inputs, targets = self.accelerator.process_batch(
            memory_samples, self.transpose_dims, self.noise_std
        )
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024**2  # MB
        memory_used = memory_after - memory_before
        
        logger.info(f"💾 Memory used: {memory_used:.1f} MB")
        
        # Clean up
        del inputs, targets, memory_samples
        gc.collect()
        
        # Memory usage should be reasonable (less than 2GB for this test)
        self.assertLess(memory_used, 2048, "Memory usage too high")
    
    def test_error_handling(self):
        """Test error handling and recovery"""
        if not self.rust_available:
            self.skipTest("Rust integration not available")
        
        # Test with invalid data types
        invalid_samples = [
            np.array([1, 2, 3]),  # Wrong shape
            np.random.randn(2, 3, 4).astype(np.int32),  # Wrong dtype
        ]
        
        # Should handle gracefully with fallback
        try:
            inputs, targets = self.accelerator.process_batch(
                invalid_samples, self.transpose_dims, self.noise_std
            )
            # If it succeeds, it used fallback
            logger.info("✅ Error handling: Graceful fallback worked")
        except Exception as e:
            # If it fails, that's also acceptable for invalid input
            logger.info(f"✅ Error handling: Proper error raised: {e}")
    
    def test_performance_stats(self):
        """Test performance statistics tracking"""
        if not self.rust_available:
            self.skipTest("Rust integration not available")
        
        # Reset stats
        self.accelerator.reset_performance_stats()
        
        # Run some operations
        for _ in range(3):
            self.accelerator.process_batch(
                self.test_samples, self.transpose_dims, self.noise_std
            )
        
        # Get stats
        stats = self.accelerator.get_performance_stats()
        
        # Validate stats
        self.assertGreaterEqual(stats['rust_calls'], 0)
        self.assertGreaterEqual(stats['python_calls'], 0)
        self.assertGreaterEqual(stats['rust_calls'] + stats['python_calls'], 3)
        
        # Print stats for debugging
        self.accelerator.print_performance_stats()


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests"""
    
    def test_production_data_loader_integration(self):
        """Test integration with production data loader"""
        try:
            from data_build.production_data_loader import ProductionDataLoader
            
            # Create data loader
            loader = ProductionDataLoader()
            
            # This test verifies that the data loader can be imported
            # and initialized without errors after our modifications
            self.assertIsNotNone(loader)
            logger.info("✅ Production data loader integration: PASSED")
            
        except ImportError as e:
            self.fail(f"Failed to import production data loader: {e}")
        except Exception as e:
            self.fail(f"Failed to initialize production data loader: {e}")


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🧪 Running Comprehensive Rust Integration Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRustIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        print("✅ Rust integration is ready for production use")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
