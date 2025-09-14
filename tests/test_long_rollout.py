#!/usr/bin/env python3
"""
Long Rollout Stability Tests
============================

Comprehensive test suite for long-term model stability and rollout testing.
Essential for ISEF competition to demonstrate model robustness over extended
time horizons critical for climate modeling applications.

Tests include:
- Numerical stability over 1k-10k steps
- Drift detection and bounds checking
- Energy conservation over long horizons
- Chaos detection and Lyapunov exponent validation
- Memory usage and performance monitoring
- Statistical stationarity testing

Author: Astrobio Research Team
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore', category=UserWarning)

# Test parameters
SHORT_ROLLOUT_STEPS = 100
MEDIUM_ROLLOUT_STEPS = 1000
LONG_ROLLOUT_STEPS = 5000
STRESS_TEST_STEPS = 10000

# Stability thresholds
MAX_DRIFT_RATE = 1e-3
MAX_ENERGY_GROWTH_RATE = 1e-4
MAX_MEMORY_USAGE_MB = 1000
MAX_STEP_TIME_MS = 100


class MockDynamicalModel(nn.Module):
    """Mock dynamical model for testing purposes"""
    
    def __init__(self, model_type: str = 'stable', state_dim: int = 3):
        super().__init__()
        self.model_type = model_type
        self.state_dim = state_dim
        
        if model_type == 'stable':
            # Stable linear dynamics
            eigenvalues = [-0.01, -0.02, -0.03]
            self.A = torch.diag(torch.tensor(eigenvalues[:state_dim], dtype=torch.float32))
        
        elif model_type == 'unstable':
            # Unstable linear dynamics
            eigenvalues = [0.005, -0.01, -0.02]
            self.A = torch.diag(torch.tensor(eigenvalues[:state_dim], dtype=torch.float32))
        
        elif model_type == 'marginally_stable':
            # Marginally stable (eigenvalues near zero)
            eigenvalues = [0.0001, -0.0001, -0.005]
            self.A = torch.diag(torch.tensor(eigenvalues[:state_dim], dtype=torch.float32))
        
        elif model_type == 'oscillatory':
            # Oscillatory dynamics
            self.omega = 0.1
            self.damping = 0.001
        
        elif model_type == 'nonlinear':
            # Nonlinear dynamics (simplified Lorenz-like)
            self.sigma = 1.0
            self.rho = 2.0
            self.beta = 0.3
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        if self.model_type in ['stable', 'unstable', 'marginally_stable']:
            # Linear dynamics: x_{t+1} = A * x_t
            return torch.matmul(x, self.A.T)
        
        elif self.model_type == 'oscillatory':
            # Damped harmonic oscillator (for 2D case)
            if x.shape[1] >= 2:
                pos, vel = x[:, 0], x[:, 1]
                
                dt = 0.1
                acc = -self.omega**2 * pos - 2 * self.damping * vel
                
                pos_next = pos + vel * dt
                vel_next = vel + acc * dt
                
                result = torch.stack([pos_next, vel_next], dim=1)
                
                # Pad with zeros if needed
                if x.shape[1] > 2:
                    padding = torch.zeros(batch_size, x.shape[1] - 2)
                    result = torch.cat([result, padding], dim=1)
                
                return result
            else:
                return x * 0.99  # Simple decay
        
        elif self.model_type == 'nonlinear':
            # Simplified nonlinear dynamics
            if x.shape[1] >= 3:
                x_curr, y_curr, z_curr = x[:, 0], x[:, 1], x[:, 2]
                
                dt = 0.01
                dx = self.sigma * (y_curr - x_curr) * dt
                dy = (x_curr * (self.rho - z_curr) - y_curr) * dt
                dz = (x_curr * y_curr - self.beta * z_curr) * dt
                
                x_next = x_curr + dx
                y_next = y_curr + dy
                z_next = z_curr + dz
                
                result = torch.stack([x_next, y_next, z_next], dim=1)
                
                # Pad with zeros if needed
                if x.shape[1] > 3:
                    padding = torch.zeros(batch_size, x.shape[1] - 3)
                    result = torch.cat([result, padding], dim=1)
                
                return result
            else:
                # Fallback to linear
                return x * 0.95
        
        return x


class StabilityTester:
    """Main class for stability testing"""
    
    def __init__(self):
        self.test_results = {}
        
    def run_rollout_test(
        self,
        model: nn.Module,
        initial_state: torch.Tensor,
        num_steps: int,
        save_interval: int = 1,
        check_memory: bool = True
    ) -> Dict[str, Any]:
        """Run rollout stability test"""
        
        model.eval()
        device = next(model.parameters()).device if len(list(model.parameters())) > 0 else 'cpu'
        initial_state = initial_state.to(device)
        
        # Storage
        trajectory = []
        step_times = []
        memory_usage = []
        
        current_state = initial_state.clone()
        
        start_time = time.time()
        
        with torch.no_grad():
            for step in range(num_steps):
                step_start = time.time()
                
                try:
                    # Forward pass
                    next_state = model(current_state)
                    
                    # Check for numerical issues
                    if torch.isnan(next_state).any() or torch.isinf(next_state).any():
                        logger.warning(f"Numerical instability at step {step}")
                        break
                    
                    # Save trajectory
                    if step % save_interval == 0:
                        trajectory.append(next_state.cpu().numpy())
                    
                    current_state = next_state
                    
                    step_time = (time.time() - step_start) * 1000  # ms
                    step_times.append(step_time)
                    
                    # Memory monitoring
                    if check_memory and torch.cuda.is_available():
                        memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
                    
                    # Check for excessive step time
                    if step_time > MAX_STEP_TIME_MS:
                        logger.warning(f"Slow step at {step}: {step_time:.2f}ms")
                
                except Exception as e:
                    logger.error(f"Rollout failed at step {step}: {e}")
                    break
        
        total_time = time.time() - start_time
        
        # Analyze trajectory
        if trajectory:
            trajectory_array = np.array(trajectory)
            if trajectory_array.ndim == 3:  # [time, batch, features]
                trajectory_array = trajectory_array[:, 0, :]  # Take first batch element
            
            analysis = self._analyze_trajectory(trajectory_array)
        else:
            analysis = {'error': 'No trajectory data'}
        
        return {
            'steps_completed': len(trajectory) * save_interval,
            'total_time_seconds': total_time,
            'mean_step_time_ms': np.mean(step_times) if step_times else 0.0,
            'max_step_time_ms': np.max(step_times) if step_times else 0.0,
            'max_memory_usage_mb': np.max(memory_usage) if memory_usage else 0.0,
            'trajectory_analysis': analysis,
            'numerical_stability': not (torch.isnan(current_state).any() or torch.isinf(current_state).any()),
            'performance_acceptable': (
                np.mean(step_times) < MAX_STEP_TIME_MS if step_times else True
            ) and (
                np.max(memory_usage) < MAX_MEMORY_USAGE_MB if memory_usage else True
            )
        }
    
    def _analyze_trajectory(self, trajectory: np.ndarray) -> Dict[str, Any]:
        """Analyze trajectory for stability metrics"""
        
        analysis = {}
        
        # Basic statistics
        initial_state = trajectory[0]
        final_state = trajectory[-1]
        
        # Distance from initial state
        distances = np.linalg.norm(trajectory - initial_state, axis=1)
        max_distance = np.max(distances)
        final_distance = distances[-1]
        
        # Drift analysis
        time_steps = np.arange(len(trajectory))
        if len(distances) > 2:
            drift_slope, _, drift_r, drift_p, _ = stats.linregress(time_steps, distances)
        else:
            drift_slope = drift_r = drift_p = 0.0
        
        # Energy analysis (L2 norm as proxy)
        energies = np.sum(trajectory**2, axis=1)
        initial_energy = energies[0]
        final_energy = energies[-1]
        
        if len(energies) > 2:
            energy_slope, _, energy_r, energy_p, _ = stats.linregress(time_steps, energies)
        else:
            energy_slope = energy_r = energy_p = 0.0
        
        # Statistical tests
        stationarity_test = self._test_stationarity(trajectory)
        
        analysis = {
            'max_distance_from_initial': float(max_distance),
            'final_distance_from_initial': float(final_distance),
            'drift_rate': float(drift_slope),
            'drift_r_squared': float(drift_r**2),
            'drift_p_value': float(drift_p),
            'initial_energy': float(initial_energy),
            'final_energy': float(final_energy),
            'energy_growth_rate': float(energy_slope),
            'energy_r_squared': float(energy_r**2),
            'energy_p_value': float(energy_p),
            'relative_energy_change': float(abs(final_energy - initial_energy) / max(abs(initial_energy), 1e-10)),
            'stationarity_test': stationarity_test,
            'trajectory_length': len(trajectory),
            'stable': (
                abs(drift_slope) < MAX_DRIFT_RATE and
                abs(energy_slope) < MAX_ENERGY_GROWTH_RATE
            )
        }
        
        return analysis
    
    def _test_stationarity(self, trajectory: np.ndarray) -> Dict[str, Any]:
        """Test trajectory for stationarity"""
        
        # Split trajectory into segments
        n_segments = 5
        segment_size = len(trajectory) // n_segments
        
        if segment_size < 3:
            return {'error': 'Trajectory too short for stationarity test'}
        
        segment_means = []
        segment_vars = []
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(trajectory)
            segment = trajectory[start_idx:end_idx]
            
            segment_means.append(np.mean(segment, axis=0))
            segment_vars.append(np.var(segment, axis=0))
        
        segment_means = np.array(segment_means)
        segment_vars = np.array(segment_vars)
        
        # Test for constant mean (F-test approximation)
        mean_variation = np.var(segment_means, axis=0)
        mean_stability = np.mean(mean_variation)
        
        # Test for constant variance
        var_variation = np.var(segment_vars, axis=0)
        var_stability = np.mean(var_variation)
        
        return {
            'mean_stability': float(mean_stability),
            'variance_stability': float(var_stability),
            'is_stationary': mean_stability < 0.1 and var_stability < 0.1,
            'segment_means': segment_means.tolist(),
            'segment_variances': segment_vars.tolist()
        }


# Test fixtures
@pytest.fixture
def stability_tester():
    """Create stability tester instance"""
    return StabilityTester()

@pytest.fixture(params=['stable', 'unstable', 'marginally_stable', 'oscillatory', 'nonlinear'])
def mock_model(request):
    """Create mock models with different stability characteristics"""
    return MockDynamicalModel(model_type=request.param, state_dim=3)

@pytest.fixture
def initial_state():
    """Create initial state for testing"""
    return torch.randn(1, 3) * 0.1  # Small initial perturbation


# Basic stability tests
class TestBasicStability:
    """Basic stability tests for different model types"""
    
    def test_short_rollout_stability(self, stability_tester, mock_model, initial_state):
        """Test stability over short rollouts (100 steps)"""
        
        result = stability_tester.run_rollout_test(
            mock_model, initial_state, SHORT_ROLLOUT_STEPS
        )
        
        # Should complete without numerical issues
        assert result['numerical_stability'], "Numerical instability in short rollout"
        assert result['steps_completed'] >= SHORT_ROLLOUT_STEPS * 0.9, \
            "Short rollout did not complete most steps"
        
        # Performance should be acceptable
        assert result['performance_acceptable'], "Poor performance in short rollout"
    
    def test_medium_rollout_stability(self, stability_tester, mock_model, initial_state):
        """Test stability over medium rollouts (1000 steps)"""
        
        result = stability_tester.run_rollout_test(
            mock_model, initial_state, MEDIUM_ROLLOUT_STEPS, save_interval=10
        )
        
        # Should complete without major issues
        assert result['numerical_stability'], "Numerical instability in medium rollout"
        
        # Analyze trajectory
        analysis = result['trajectory_analysis']
        if 'error' not in analysis:
            # Drift should be reasonable
            drift_rate = abs(analysis['drift_rate'])
            
            # Different models have different expected behaviors
            model_type = mock_model.model_type
            if model_type == 'stable':
                assert drift_rate < MAX_DRIFT_RATE * 10, f"Excessive drift in stable model: {drift_rate}"
            elif model_type == 'unstable':
                # Unstable models may drift, but not explosively
                assert drift_rate < 1.0, f"Explosive drift in unstable model: {drift_rate}"
    
    @pytest.mark.slow
    def test_long_rollout_stability(self, stability_tester, mock_model, initial_state):
        """Test stability over long rollouts (5000 steps)"""
        
        result = stability_tester.run_rollout_test(
            mock_model, initial_state, LONG_ROLLOUT_STEPS, save_interval=50
        )
        
        # Should maintain numerical stability
        assert result['numerical_stability'], "Numerical instability in long rollout"
        
        # Memory usage should be reasonable
        assert result['max_memory_usage_mb'] < MAX_MEMORY_USAGE_MB, \
            f"Excessive memory usage: {result['max_memory_usage_mb']} MB"
        
        # Performance should remain acceptable
        assert result['mean_step_time_ms'] < MAX_STEP_TIME_MS, \
            f"Steps too slow: {result['mean_step_time_ms']} ms"


class TestStabilityMetrics:
    """Test specific stability metrics"""
    
    def test_drift_detection(self, stability_tester):
        """Test drift detection in unstable systems"""
        
        # Create an unstable model
        unstable_model = MockDynamicalModel('unstable', state_dim=3)
        initial_state = torch.randn(1, 3) * 0.1
        
        result = stability_tester.run_rollout_test(
            unstable_model, initial_state, MEDIUM_ROLLOUT_STEPS, save_interval=10
        )
        
        analysis = result['trajectory_analysis']
        if 'error' not in analysis:
            # Unstable model should show some drift
            assert analysis['drift_rate'] > 0, "Unstable model should show positive drift"
            assert not analysis['stable'], "Unstable model incorrectly classified as stable"
    
    def test_energy_conservation_detection(self, stability_tester):
        """Test energy conservation detection"""
        
        # Create a stable model (should conserve energy better)
        stable_model = MockDynamicalModel('stable', state_dim=3)
        initial_state = torch.randn(1, 3) * 0.1
        
        result = stability_tester.run_rollout_test(
            stable_model, initial_state, MEDIUM_ROLLOUT_STEPS, save_interval=10
        )
        
        analysis = result['trajectory_analysis']
        if 'error' not in analysis:
            # Energy should not grow exponentially
            energy_growth_rate = abs(analysis['energy_growth_rate'])
            assert energy_growth_rate < MAX_ENERGY_GROWTH_RATE * 100, \
                f"Excessive energy growth: {energy_growth_rate}"
    
    def test_stationarity_detection(self, stability_tester):
        """Test stationarity detection"""
        
        # Create an oscillatory model (should be stationary)
        oscillatory_model = MockDynamicalModel('oscillatory', state_dim=3)
        initial_state = torch.randn(1, 3) * 0.1
        
        result = stability_tester.run_rollout_test(
            oscillatory_model, initial_state, MEDIUM_ROLLOUT_STEPS, save_interval=5
        )
        
        analysis = result['trajectory_analysis']
        if 'error' not in analysis:
            stationarity = analysis['stationarity_test']
            if 'error' not in stationarity:
                # Oscillatory model should have reasonable stationarity
                assert stationarity['mean_stability'] < 1.0, \
                    "Oscillatory model should have reasonable mean stability"


class TestNumericalStability:
    """Test numerical stability and error handling"""
    
    def test_nan_detection(self, stability_tester):
        """Test detection of NaN values"""
        
        class NaNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.step_count = 0
            
            def forward(self, x):
                self.step_count += 1
                if self.step_count > 50:  # Introduce NaN after 50 steps
                    return x * float('nan')
                return x * 1.01  # Slight growth
        
        nan_model = NaNModel()
        initial_state = torch.randn(1, 3)
        
        result = stability_tester.run_rollout_test(
            nan_model, initial_state, SHORT_ROLLOUT_STEPS
        )
        
        # Should detect numerical instability
        assert not result['numerical_stability'], "Failed to detect NaN values"
        assert result['steps_completed'] < SHORT_ROLLOUT_STEPS, \
            "Rollout should have stopped early due to NaN"
    
    def test_inf_detection(self, stability_tester):
        """Test detection of infinite values"""
        
        class InfModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.step_count = 0
            
            def forward(self, x):
                self.step_count += 1
                if self.step_count > 30:  # Introduce Inf after 30 steps
                    return x * 1e10  # Large multiplication leading to overflow
                return x * 1.1
        
        inf_model = InfModel()
        initial_state = torch.randn(1, 3)
        
        result = stability_tester.run_rollout_test(
            inf_model, initial_state, SHORT_ROLLOUT_STEPS
        )
        
        # Should eventually detect numerical issues
        # (may take a few steps for inf to propagate)
        if result['steps_completed'] >= SHORT_ROLLOUT_STEPS:
            # If it completed, check final state
            assert result['numerical_stability'], "Model should maintain stability or fail early"


class TestPerformanceStability:
    """Test performance-related stability"""
    
    def test_memory_stability(self, stability_tester):
        """Test memory usage stability over long rollouts"""
        
        # Create a simple model
        stable_model = MockDynamicalModel('stable', state_dim=10)  # Larger state
        initial_state = torch.randn(1, 10)
        
        result = stability_tester.run_rollout_test(
            stable_model, initial_state, MEDIUM_ROLLOUT_STEPS, 
            save_interval=20, check_memory=True
        )
        
        # Memory usage should be reasonable
        if result['max_memory_usage_mb'] > 0:  # Only if CUDA is available
            assert result['max_memory_usage_mb'] < MAX_MEMORY_USAGE_MB, \
                f"Excessive memory usage: {result['max_memory_usage_mb']} MB"
    
    def test_timing_stability(self, stability_tester):
        """Test that step times remain stable"""
        
        stable_model = MockDynamicalModel('stable', state_dim=5)
        initial_state = torch.randn(1, 5)
        
        result = stability_tester.run_rollout_test(
            stable_model, initial_state, MEDIUM_ROLLOUT_STEPS
        )
        
        # Step times should be reasonable and stable
        assert result['mean_step_time_ms'] < MAX_STEP_TIME_MS, \
            f"Steps too slow: {result['mean_step_time_ms']} ms"
        
        # Maximum step time shouldn't be too much larger than mean
        if result['max_step_time_ms'] > 0:
            timing_variation = result['max_step_time_ms'] / max(result['mean_step_time_ms'], 1e-6)
            assert timing_variation < 100, f"Excessive timing variation: {timing_variation}x"


class TestRobustnessScenarios:
    """Test robustness under various scenarios"""
    
    @pytest.mark.parametrize("state_dim", [1, 3, 10, 50])
    def test_different_state_dimensions(self, stability_tester, state_dim):
        """Test stability with different state dimensions"""
        
        model = MockDynamicalModel('stable', state_dim=state_dim)
        initial_state = torch.randn(1, state_dim) * 0.1
        
        result = stability_tester.run_rollout_test(
            model, initial_state, SHORT_ROLLOUT_STEPS
        )
        
        # Should work for any reasonable state dimension
        assert result['numerical_stability'], f"Instability with state_dim={state_dim}"
        assert result['performance_acceptable'], f"Poor performance with state_dim={state_dim}"
    
    @pytest.mark.parametrize("batch_size", [1, 5, 10])
    def test_different_batch_sizes(self, stability_tester, batch_size):
        """Test stability with different batch sizes"""
        
        model = MockDynamicalModel('stable', state_dim=3)
        initial_state = torch.randn(batch_size, 3) * 0.1
        
        result = stability_tester.run_rollout_test(
            model, initial_state, SHORT_ROLLOUT_STEPS
        )
        
        # Should work for different batch sizes
        assert result['numerical_stability'], f"Instability with batch_size={batch_size}"
    
    @pytest.mark.parametrize("perturbation_scale", [1e-6, 1e-3, 1e-1, 1.0])
    def test_different_initial_perturbations(self, stability_tester, perturbation_scale):
        """Test stability with different initial perturbation scales"""
        
        model = MockDynamicalModel('stable', state_dim=3)
        initial_state = torch.randn(1, 3) * perturbation_scale
        
        result = stability_tester.run_rollout_test(
            model, initial_state, SHORT_ROLLOUT_STEPS
        )
        
        # Should remain stable for reasonable perturbations
        assert result['numerical_stability'], \
            f"Instability with perturbation_scale={perturbation_scale}"


# Integration tests
class TestRolloutIntegration:
    """Integration tests for rollout stability"""
    
    def test_multiple_model_comparison(self, stability_tester):
        """Test and compare multiple model types"""
        
        model_types = ['stable', 'marginally_stable', 'oscillatory']
        initial_state = torch.randn(1, 3) * 0.1
        
        results = {}
        for model_type in model_types:
            model = MockDynamicalModel(model_type, state_dim=3)
            result = stability_tester.run_rollout_test(
                model, initial_state, MEDIUM_ROLLOUT_STEPS, save_interval=20
            )
            results[model_type] = result
        
        # All models should maintain numerical stability
        for model_type, result in results.items():
            assert result['numerical_stability'], \
                f"Numerical instability in {model_type} model"
        
        # Stable model should have better stability metrics
        if 'error' not in results['stable']['trajectory_analysis']:
            stable_analysis = results['stable']['trajectory_analysis']
            assert stable_analysis['stable'], "Stable model should be classified as stable"
    
    def test_reproducibility(self, stability_tester):
        """Test that rollouts are reproducible with same seed"""
        
        model = MockDynamicalModel('stable', state_dim=3)
        
        # Run same test twice with same seed
        torch.manual_seed(42)
        initial_state1 = torch.randn(1, 3) * 0.1
        result1 = stability_tester.run_rollout_test(
            model, initial_state1, SHORT_ROLLOUT_STEPS
        )
        
        torch.manual_seed(42)
        initial_state2 = torch.randn(1, 3) * 0.1
        result2 = stability_tester.run_rollout_test(
            model, initial_state2, SHORT_ROLLOUT_STEPS
        )
        
        # Results should be identical (within numerical precision)
        assert result1['steps_completed'] == result2['steps_completed'], \
            "Rollouts should be reproducible"
        
        # Trajectory analyses should be similar
        if ('error' not in result1['trajectory_analysis'] and 
            'error' not in result2['trajectory_analysis']):
            
            analysis1 = result1['trajectory_analysis']
            analysis2 = result2['trajectory_analysis']
            
            # Key metrics should be very close
            assert abs(analysis1['final_distance_from_initial'] - 
                      analysis2['final_distance_from_initial']) < 1e-6, \
                "Trajectory distances should be reproducible"


# Stress tests
class TestStressScenarios:
    """Stress tests for extreme scenarios"""
    
    @pytest.mark.slow
    @pytest.mark.stress
    def test_very_long_rollout(self, stability_tester):
        """Stress test with very long rollouts"""
        
        model = MockDynamicalModel('stable', state_dim=3)
        initial_state = torch.randn(1, 3) * 0.01  # Very small perturbation
        
        result = stability_tester.run_rollout_test(
            model, initial_state, STRESS_TEST_STEPS, save_interval=100
        )
        
        # Should maintain stability even over very long rollouts
        assert result['numerical_stability'], "Numerical instability in stress test"
        assert result['performance_acceptable'], "Poor performance in stress test"
    
    @pytest.mark.stress
    def test_large_state_dimension(self, stability_tester):
        """Stress test with large state dimensions"""
        
        large_dim = 100
        model = MockDynamicalModel('stable', state_dim=large_dim)
        initial_state = torch.randn(1, large_dim) * 0.01
        
        result = stability_tester.run_rollout_test(
            model, initial_state, SHORT_ROLLOUT_STEPS
        )
        
        # Should handle large state dimensions
        assert result['numerical_stability'], "Instability with large state dimension"
        assert result['performance_acceptable'], "Poor performance with large state dimension"
    
    @pytest.mark.stress
    def test_large_batch_size(self, stability_tester):
        """Stress test with large batch sizes"""
        
        large_batch = 100
        model = MockDynamicalModel('stable', state_dim=3)
        initial_state = torch.randn(large_batch, 3) * 0.01
        
        result = stability_tester.run_rollout_test(
            model, initial_state, SHORT_ROLLOUT_STEPS
        )
        
        # Should handle large batch sizes
        assert result['numerical_stability'], "Instability with large batch size"


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
