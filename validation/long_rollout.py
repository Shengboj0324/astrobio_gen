#!/usr/bin/env python3
"""
Long-Term Rollout Stability Testing Framework
============================================

Comprehensive testing framework for model stability over extended rollouts:
- 1k-10k step stability analysis with drift detection
- Energy conservation monitoring over long horizons
- Numerical stability and error accumulation analysis
- Chaos theory metrics (Lyapunov exponents, attractor analysis)
- Statistical stationarity testing
- Phase space trajectory analysis
- Divergence detection and early warning systems

Critical for ISEF competition to demonstrate model robustness and
long-term predictive capability essential for climate modeling.

Author: Astrobio Research Team
"""

import asyncio
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats, signal
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D

# Statistical analysis
from scipy.stats import kstest, jarque_bera, adfuller
from scipy.signal import periodogram, welch
from scipy.spatial.distance import pdist, squareform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class StabilityMetrics:
    """Comprehensive metrics for long-term stability analysis"""
    
    def __init__(self):
        self.metrics_cache = {}
    
    def calculate_drift_metrics(
        self,
        trajectory: np.ndarray,
        reference_state: Optional[np.ndarray] = None,
        time_steps: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate drift and divergence metrics
        
        Args:
            trajectory: State trajectory [time, features]
            reference_state: Reference state for comparison
            time_steps: Time step array
            
        Returns:
            Dictionary with drift metrics
        """
        trajectory = np.array(trajectory)
        
        if reference_state is None:
            reference_state = trajectory[0]  # Use initial state
        
        if time_steps is None:
            time_steps = np.arange(len(trajectory))
        
        # Calculate distance from reference state
        distances = np.linalg.norm(trajectory - reference_state, axis=1)
        
        # Linear drift rate
        if len(distances) > 1:
            drift_slope, drift_intercept, drift_r, drift_p, drift_std_err = stats.linregress(
                time_steps, distances
            )
        else:
            drift_slope = drift_intercept = drift_r = drift_p = drift_std_err = 0.0
        
        # Exponential growth detection
        log_distances = np.log(np.maximum(distances, 1e-10))
        exp_slope, exp_intercept, exp_r, exp_p, exp_std_err = stats.linregress(
            time_steps, log_distances
        )
        
        # Maximum and final distances
        max_distance = np.max(distances)
        final_distance = distances[-1]
        mean_distance = np.mean(distances)
        
        # Time to significant drift (distance > 2 * initial std)
        initial_std = np.std(trajectory[:min(10, len(trajectory))], axis=0).mean()
        drift_threshold = 2 * initial_std
        drift_indices = np.where(distances > drift_threshold)[0]
        time_to_drift = time_steps[drift_indices[0]] if len(drift_indices) > 0 else np.inf
        
        return {
            'linear_drift_rate': float(drift_slope),
            'linear_drift_r_squared': float(drift_r**2),
            'linear_drift_p_value': float(drift_p),
            'exponential_growth_rate': float(exp_slope),
            'exponential_growth_r_squared': float(exp_r**2),
            'exponential_growth_p_value': float(exp_p),
            'max_distance_from_reference': float(max_distance),
            'final_distance_from_reference': float(final_distance),
            'mean_distance_from_reference': float(mean_distance),
            'time_to_significant_drift': float(time_to_drift),
            'drift_threshold': float(drift_threshold),
            'distance_trajectory': distances.tolist()
        }
    
    def calculate_energy_conservation(
        self,
        trajectory: np.ndarray,
        energy_function: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Calculate energy conservation metrics
        
        Args:
            trajectory: State trajectory [time, features]
            energy_function: Function to calculate energy from state
            
        Returns:
            Dictionary with energy conservation metrics
        """
        trajectory = np.array(trajectory)
        
        if energy_function is None:
            # Default: use L2 norm as energy proxy
            energies = np.sum(trajectory**2, axis=1)
        else:
            energies = np.array([energy_function(state) for state in trajectory])
        
        # Energy statistics
        initial_energy = energies[0]
        final_energy = energies[-1]
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        # Energy drift
        time_steps = np.arange(len(energies))
        energy_slope, _, energy_r, energy_p, _ = stats.linregress(time_steps, energies)
        
        # Energy conservation violation
        energy_variation = std_energy / mean_energy if mean_energy != 0 else np.inf
        max_energy_change = np.max(np.abs(energies - initial_energy))
        
        # Relative energy error
        relative_energy_error = abs(final_energy - initial_energy) / abs(initial_energy) if initial_energy != 0 else np.inf
        
        return {
            'initial_energy': float(initial_energy),
            'final_energy': float(final_energy),
            'mean_energy': float(mean_energy),
            'energy_std': float(std_energy),
            'energy_drift_rate': float(energy_slope),
            'energy_drift_r_squared': float(energy_r**2),
            'energy_drift_p_value': float(energy_p),
            'energy_variation_coefficient': float(energy_variation),
            'max_energy_change': float(max_energy_change),
            'relative_energy_error': float(relative_energy_error),
            'energy_trajectory': energies.tolist()
        }
    
    def calculate_lyapunov_exponent(
        self,
        trajectory: np.ndarray,
        dt: float = 1.0,
        max_neighbors: int = 50
    ) -> Dict[str, float]:
        """
        Estimate largest Lyapunov exponent using Wolf's algorithm
        
        Args:
            trajectory: State trajectory [time, features]
            dt: Time step size
            max_neighbors: Maximum number of neighbors to consider
            
        Returns:
            Dictionary with Lyapunov exponent estimates
        """
        trajectory = np.array(trajectory)
        n_points, n_dims = trajectory.shape
        
        if n_points < 100:
            logger.warning("Trajectory too short for reliable Lyapunov exponent estimation")
            return {
                'largest_lyapunov_exponent': 0.0,
                'lyapunov_exponent_std': 0.0,
                'estimation_error': 'Trajectory too short'
            }
        
        try:
            # Calculate pairwise distances
            distances = pdist(trajectory)
            distance_matrix = squareform(distances)
            
            # Find nearest neighbors for each point
            lyapunov_estimates = []
            
            for i in range(n_points - 10):  # Leave some points for evolution
                # Find nearest neighbors
                neighbor_distances = distance_matrix[i]
                neighbor_indices = np.argsort(neighbor_distances)[1:max_neighbors+1]  # Exclude self
                
                # Calculate divergence rates
                for j in neighbor_indices:
                    if j >= n_points - 10:  # Ensure we can evolve
                        continue
                    
                    # Initial separation
                    initial_separation = distance_matrix[i, j]
                    
                    if initial_separation < 1e-10:  # Too close
                        continue
                    
                    # Evolution time (limited to avoid getting too far)
                    max_evolution_steps = min(10, n_points - max(i, j) - 1)
                    
                    for k in range(1, max_evolution_steps + 1):
                        # Separation after k steps
                        evolved_separation = np.linalg.norm(trajectory[i + k] - trajectory[j + k])
                        
                        if evolved_separation > 0 and initial_separation > 0:
                            # Local Lyapunov exponent
                            local_lyapunov = np.log(evolved_separation / initial_separation) / (k * dt)
                            
                            # Filter out extreme values
                            if abs(local_lyapunov) < 10:  # Reasonable bound
                                lyapunov_estimates.append(local_lyapunov)
            
            if len(lyapunov_estimates) > 0:
                # Remove outliers (beyond 3 standard deviations)
                lyapunov_array = np.array(lyapunov_estimates)
                mean_lyap = np.mean(lyapunov_array)
                std_lyap = np.std(lyapunov_array)
                
                # Filter outliers
                valid_estimates = lyapunov_array[
                    np.abs(lyapunov_array - mean_lyap) <= 3 * std_lyap
                ]
                
                if len(valid_estimates) > 0:
                    largest_lyapunov = np.mean(valid_estimates)
                    lyapunov_std = np.std(valid_estimates)
                else:
                    largest_lyapunov = mean_lyap
                    lyapunov_std = std_lyap
                
                # Interpretation
                if largest_lyapunov > 0.01:
                    stability_assessment = "chaotic"
                elif largest_lyapunov > -0.01:
                    stability_assessment = "marginally_stable"
                else:
                    stability_assessment = "stable"
                
            else:
                largest_lyapunov = 0.0
                lyapunov_std = 0.0
                stability_assessment = "unknown"
            
        except Exception as e:
            logger.error(f"Lyapunov exponent calculation failed: {e}")
            return {
                'largest_lyapunov_exponent': 0.0,
                'lyapunov_exponent_std': 0.0,
                'estimation_error': str(e)
            }
        
        return {
            'largest_lyapunov_exponent': float(largest_lyapunov),
            'lyapunov_exponent_std': float(lyapunov_std),
            'num_estimates': len(lyapunov_estimates) if 'lyapunov_estimates' in locals() else 0,
            'stability_assessment': stability_assessment,
            'estimation_method': 'wolf_algorithm'
        }
    
    def calculate_attractor_dimension(
        self,
        trajectory: np.ndarray,
        max_embedding_dim: int = 10
    ) -> Dict[str, float]:
        """
        Estimate attractor dimension using correlation dimension method
        
        Args:
            trajectory: State trajectory [time, features]
            max_embedding_dim: Maximum embedding dimension to test
            
        Returns:
            Dictionary with dimension estimates
        """
        trajectory = np.array(trajectory)
        
        if len(trajectory) < 1000:
            logger.warning("Trajectory too short for reliable dimension estimation")
            return {
                'correlation_dimension': 0.0,
                'dimension_error': 'Trajectory too short'
            }
        
        try:
            # Use the trajectory directly if multi-dimensional, or embed if 1D
            if trajectory.ndim == 1:
                # Time delay embedding for 1D time series
                embedding_dim = min(max_embedding_dim, 5)  # Reasonable default
                delay = 1
                embedded = self._time_delay_embedding(trajectory, embedding_dim, delay)
            else:
                embedded = trajectory
            
            # Calculate correlation dimension
            correlation_dim = self._correlation_dimension(embedded)
            
            # Estimate intrinsic dimension using PCA
            pca = PCA()
            pca.fit(embedded)
            
            # Find number of components needed for 95% variance
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            intrinsic_dim = np.argmax(cumsum_variance >= 0.95) + 1
            
            # Effective dimension (based on participation ratio)
            eigenvalues = pca.explained_variance_
            participation_ratio = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
            
        except Exception as e:
            logger.error(f"Attractor dimension calculation failed: {e}")
            return {
                'correlation_dimension': 0.0,
                'dimension_error': str(e)
            }
        
        return {
            'correlation_dimension': float(correlation_dim),
            'intrinsic_dimension': int(intrinsic_dim),
            'participation_ratio': float(participation_ratio),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'embedding_dimension': embedded.shape[1]
        }
    
    def _time_delay_embedding(
        self, 
        time_series: np.ndarray, 
        embedding_dim: int, 
        delay: int
    ) -> np.ndarray:
        """Create time delay embedding of 1D time series"""
        
        n_points = len(time_series) - (embedding_dim - 1) * delay
        embedded = np.zeros((n_points, embedding_dim))
        
        for i in range(embedding_dim):
            embedded[:, i] = time_series[i * delay:i * delay + n_points]
        
        return embedded
    
    def _correlation_dimension(self, trajectory: np.ndarray, n_scales: int = 20) -> float:
        """Estimate correlation dimension using Grassberger-Procaccia algorithm"""
        
        n_points = len(trajectory)
        
        # Sample points to make computation feasible
        if n_points > 2000:
            indices = np.random.choice(n_points, 2000, replace=False)
            sample_trajectory = trajectory[indices]
        else:
            sample_trajectory = trajectory
        
        n_sample = len(sample_trajectory)
        
        # Calculate pairwise distances
        distances = pdist(sample_trajectory)
        
        # Range of scales
        min_dist = np.min(distances[distances > 0])
        max_dist = np.max(distances)
        scales = np.logspace(np.log10(min_dist), np.log10(max_dist * 0.1), n_scales)
        
        # Calculate correlation sum for each scale
        correlation_sums = []
        
        for scale in scales:
            # Count pairs within scale
            count = np.sum(distances < scale)
            # Normalize by total number of pairs
            correlation_sum = count / (n_sample * (n_sample - 1) / 2)
            correlation_sums.append(correlation_sum)
        
        correlation_sums = np.array(correlation_sums)
        
        # Estimate dimension from slope of log(C(r)) vs log(r)
        # Use middle portion of the scaling region
        valid_indices = (correlation_sums > 0) & (correlation_sums < 0.9)
        
        if np.sum(valid_indices) > 3:
            log_scales = np.log(scales[valid_indices])
            log_correlation = np.log(correlation_sums[valid_indices])
            
            # Linear regression to find slope
            slope, _, r_value, _, _ = stats.linregress(log_scales, log_correlation)
            
            # The slope is the correlation dimension
            correlation_dim = slope
        else:
            correlation_dim = 0.0
        
        return correlation_dim
    
    def calculate_stationarity_metrics(
        self,
        trajectory: np.ndarray
    ) -> Dict[str, Any]:
        """
        Test for statistical stationarity of the trajectory
        
        Args:
            trajectory: State trajectory [time, features]
            
        Returns:
            Dictionary with stationarity test results
        """
        trajectory = np.array(trajectory)
        
        stationarity_results = {}
        
        # For each feature dimension
        for dim in range(trajectory.shape[1]):
            series = trajectory[:, dim]
            
            # Augmented Dickey-Fuller test
            try:
                adf_statistic, adf_p_value, adf_lags, adf_nobs, adf_critical_values, adf_icbest = adfuller(
                    series, autolag='AIC'
                )
                
                adf_result = {
                    'statistic': float(adf_statistic),
                    'p_value': float(adf_p_value),
                    'lags': int(adf_lags),
                    'critical_values': {k: float(v) for k, v in adf_critical_values.items()},
                    'is_stationary': adf_p_value < 0.05
                }
            except Exception as e:
                adf_result = {'error': str(e)}
            
            # Jarque-Bera test for normality
            try:
                jb_statistic, jb_p_value = jarque_bera(series)
                jb_result = {
                    'statistic': float(jb_statistic),
                    'p_value': float(jb_p_value),
                    'is_normal': jb_p_value > 0.05
                }
            except Exception as e:
                jb_result = {'error': str(e)}
            
            # Mean and variance stationarity (split into segments)
            n_segments = 5
            segment_size = len(series) // n_segments
            
            if segment_size > 10:
                segment_means = []
                segment_vars = []
                
                for i in range(n_segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(series)
                    segment = series[start_idx:end_idx]
                    
                    segment_means.append(np.mean(segment))
                    segment_vars.append(np.var(segment))
                
                # Test for constant mean and variance
                mean_f_stat, mean_p_value = stats.f_oneway(*[
                    series[i*segment_size:(i+1)*segment_size] 
                    for i in range(n_segments-1)
                ] + [series[(n_segments-1)*segment_size:]])
                
                variance_stability = {
                    'segment_means': segment_means,
                    'segment_variances': segment_vars,
                    'mean_f_statistic': float(mean_f_stat),
                    'mean_p_value': float(mean_p_value),
                    'mean_is_constant': mean_p_value > 0.05,
                    'variance_coefficient_of_variation': float(np.std(segment_vars) / np.mean(segment_vars)) if np.mean(segment_vars) > 0 else np.inf
                }
            else:
                variance_stability = {'error': 'Too few points for segmentation analysis'}
            
            stationarity_results[f'dimension_{dim}'] = {
                'augmented_dickey_fuller': adf_result,
                'jarque_bera': jb_result,
                'variance_stability': variance_stability
            }
        
        # Overall stationarity assessment
        overall_stationary = True
        overall_normal = True
        
        for dim_result in stationarity_results.values():
            if 'augmented_dickey_fuller' in dim_result:
                if not dim_result['augmented_dickey_fuller'].get('is_stationary', False):
                    overall_stationary = False
            
            if 'jarque_bera' in dim_result:
                if not dim_result['jarque_bera'].get('is_normal', False):
                    overall_normal = False
        
        stationarity_results['overall_assessment'] = {
            'is_stationary': overall_stationary,
            'is_normal': overall_normal,
            'num_dimensions': trajectory.shape[1]
        }
        
        return stationarity_results
    
    def calculate_spectral_analysis(
        self,
        trajectory: np.ndarray,
        dt: float = 1.0
    ) -> Dict[str, Any]:
        """
        Perform spectral analysis of the trajectory
        
        Args:
            trajectory: State trajectory [time, features]
            dt: Time step size
            
        Returns:
            Dictionary with spectral analysis results
        """
        trajectory = np.array(trajectory)
        
        spectral_results = {}
        
        # For each dimension
        for dim in range(trajectory.shape[1]):
            series = trajectory[:, dim]
            
            # Power spectral density using Welch's method
            try:
                frequencies, psd = welch(series, fs=1.0/dt, nperseg=min(256, len(series)//4))
                
                # Find dominant frequencies
                peak_indices = signal.find_peaks(psd, height=np.max(psd)*0.1)[0]
                dominant_frequencies = frequencies[peak_indices]
                dominant_powers = psd[peak_indices]
                
                # Sort by power
                sorted_indices = np.argsort(dominant_powers)[::-1]
                dominant_frequencies = dominant_frequencies[sorted_indices]
                dominant_powers = dominant_powers[sorted_indices]
                
                # Spectral characteristics
                total_power = np.trapz(psd, frequencies)
                peak_frequency = frequencies[np.argmax(psd)]
                spectral_centroid = np.trapz(frequencies * psd, frequencies) / total_power
                
                spectral_results[f'dimension_{dim}'] = {
                    'frequencies': frequencies.tolist(),
                    'power_spectral_density': psd.tolist(),
                    'dominant_frequencies': dominant_frequencies[:5].tolist(),  # Top 5
                    'dominant_powers': dominant_powers[:5].tolist(),
                    'total_power': float(total_power),
                    'peak_frequency': float(peak_frequency),
                    'spectral_centroid': float(spectral_centroid)
                }
                
            except Exception as e:
                spectral_results[f'dimension_{dim}'] = {'error': str(e)}
        
        return spectral_results


class LongRolloutTester:
    """Main class for long-term rollout stability testing"""
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        results_path: str = "results"
    ):
        self.device = device
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics calculator
        self.stability_metrics = StabilityMetrics()
        
        # Results storage
        self.rollout_results = {}
        
        logger.info(f"🔄 Long Rollout Tester initialized")
        logger.info(f"   Device: {device}")
        logger.info(f"   Results path: {results_path}")
    
    def create_test_model(self, model_type: str = 'stable') -> nn.Module:
        """
        Create test models with different stability characteristics
        
        Args:
            model_type: 'stable', 'unstable', 'chaotic', 'oscillatory'
            
        Returns:
            Test model
        """
        
        class TestDynamicalSystem(nn.Module):
            def __init__(self, system_type: str, state_dim: int = 3):
                super().__init__()
                self.system_type = system_type
                self.state_dim = state_dim
                
                if system_type == 'stable':
                    # Stable linear system
                    eigenvalues = [-0.1, -0.2, -0.3]
                    self.A = torch.diag(torch.tensor(eigenvalues, dtype=torch.float32))
                    
                elif system_type == 'unstable':
                    # Unstable linear system
                    eigenvalues = [0.05, -0.1, -0.2]
                    self.A = torch.diag(torch.tensor(eigenvalues, dtype=torch.float32))
                    
                elif system_type == 'chaotic':
                    # Lorenz-like chaotic system (simplified)
                    self.sigma = 10.0
                    self.rho = 28.0
                    self.beta = 8.0/3.0
                    
                elif system_type == 'oscillatory':
                    # Harmonic oscillator
                    self.omega = 0.1
                    self.damping = 0.01
            
            def forward(self, x):
                batch_size = x.shape[0]
                
                if self.system_type in ['stable', 'unstable']:
                    # Linear dynamics: x_{t+1} = A * x_t
                    return torch.matmul(x, self.A.T)
                
                elif self.system_type == 'chaotic':
                    # Lorenz system (simplified discrete version)
                    dt = 0.01
                    x_curr, y_curr, z_curr = x[:, 0], x[:, 1], x[:, 2]
                    
                    dx = self.sigma * (y_curr - x_curr) * dt
                    dy = (x_curr * (self.rho - z_curr) - y_curr) * dt
                    dz = (x_curr * y_curr - self.beta * z_curr) * dt
                    
                    x_next = x_curr + dx
                    y_next = y_curr + dy
                    z_next = z_curr + dz
                    
                    return torch.stack([x_next, y_next, z_next], dim=1)
                
                elif self.system_type == 'oscillatory':
                    # Damped harmonic oscillator
                    dt = 0.1
                    pos, vel = x[:, 0], x[:, 1]
                    
                    # Simple Euler integration
                    acc = -self.omega**2 * pos - 2 * self.damping * vel
                    
                    pos_next = pos + vel * dt
                    vel_next = vel + acc * dt
                    
                    # Add a third dimension for consistency
                    energy = 0.5 * (vel_next**2 + self.omega**2 * pos_next**2)
                    
                    return torch.stack([pos_next, vel_next, energy], dim=1)
        
        return TestDynamicalSystem(model_type)
    
    async def run_long_rollout_test(
        self,
        model: nn.Module,
        initial_state: torch.Tensor,
        num_steps: int = 5000,
        save_interval: int = 1,
        model_name: str = 'test_model'
    ) -> Dict[str, Any]:
        """
        Run long-term rollout stability test
        
        Args:
            model: Model to test
            initial_state: Initial state tensor [batch_size, state_dim]
            num_steps: Number of rollout steps
            save_interval: How often to save states
            model_name: Name for results storage
            
        Returns:
            Complete stability analysis results
        """
        logger.info(f"🔄 Running long rollout test: {model_name} ({num_steps} steps)")
        start_time = time.time()
        
        model.eval()
        model = model.to(self.device)
        initial_state = initial_state.to(self.device)
        
        # Storage for trajectory
        trajectory = []
        energies = []
        step_times = []
        
        current_state = initial_state.clone()
        
        # Rollout loop
        with torch.no_grad():
            for step in range(num_steps):
                step_start_time = time.time()
                
                try:
                    # Forward pass
                    next_state = model(current_state)
                    
                    # Check for numerical issues
                    if torch.isnan(next_state).any() or torch.isinf(next_state).any():
                        logger.warning(f"Numerical instability detected at step {step}")
                        break
                    
                    # Save trajectory
                    if step % save_interval == 0:
                        trajectory.append(next_state.cpu().numpy())
                        
                        # Calculate energy (L2 norm)
                        energy = torch.sum(next_state**2, dim=1).mean().item()
                        energies.append(energy)
                    
                    current_state = next_state
                    step_times.append(time.time() - step_start_time)
                    
                    # Progress logging
                    if step % (num_steps // 10) == 0 and step > 0:
                        logger.info(f"   Step {step}/{num_steps} completed")
                
                except Exception as e:
                    logger.error(f"Rollout failed at step {step}: {e}")
                    break
        
        rollout_time = time.time() - start_time
        
        if not trajectory:
            return {
                'model_name': model_name,
                'error': 'No trajectory data generated',
                'steps_completed': 0,
                'rollout_time': rollout_time
            }
        
        # Convert trajectory to numpy array
        trajectory_array = np.array(trajectory)
        if trajectory_array.ndim == 3:  # [time, batch, features]
            trajectory_array = trajectory_array[:, 0, :]  # Take first batch element
        
        logger.info(f"   Analyzing {len(trajectory)} trajectory points...")
        
        # Comprehensive stability analysis
        analysis_results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'rollout_parameters': {
                'num_steps_requested': num_steps,
                'steps_completed': len(trajectory),
                'save_interval': save_interval,
                'initial_state': initial_state.cpu().numpy().tolist(),
                'rollout_time_seconds': rollout_time,
                'mean_step_time_ms': float(np.mean(step_times) * 1000) if step_times else 0.0
            }
        }
        
        # 1. Drift analysis
        logger.info("   📈 Calculating drift metrics...")
        drift_metrics = self.stability_metrics.calculate_drift_metrics(trajectory_array)
        analysis_results['drift_analysis'] = drift_metrics
        
        # 2. Energy conservation
        logger.info("   ⚡ Analyzing energy conservation...")
        energy_metrics = self.stability_metrics.calculate_energy_conservation(trajectory_array)
        analysis_results['energy_conservation'] = energy_metrics
        
        # 3. Lyapunov exponent
        logger.info("   🌀 Estimating Lyapunov exponent...")
        lyapunov_metrics = self.stability_metrics.calculate_lyapunov_exponent(trajectory_array)
        analysis_results['lyapunov_analysis'] = lyapunov_metrics
        
        # 4. Attractor dimension
        logger.info("   📐 Estimating attractor dimension...")
        dimension_metrics = self.stability_metrics.calculate_attractor_dimension(trajectory_array)
        analysis_results['attractor_analysis'] = dimension_metrics
        
        # 5. Stationarity analysis
        logger.info("   📊 Testing stationarity...")
        stationarity_metrics = self.stability_metrics.calculate_stationarity_metrics(trajectory_array)
        analysis_results['stationarity_analysis'] = stationarity_metrics
        
        # 6. Spectral analysis
        logger.info("   🎵 Performing spectral analysis...")
        spectral_metrics = self.stability_metrics.calculate_spectral_analysis(trajectory_array)
        analysis_results['spectral_analysis'] = spectral_metrics
        
        # 7. Overall stability assessment
        stability_score = self._calculate_overall_stability_score(analysis_results)
        analysis_results['overall_stability_score'] = stability_score
        
        # Store results
        self.rollout_results[model_name] = analysis_results
        
        logger.info(f"✅ Long rollout test completed: {model_name}")
        logger.info(f"   Steps completed: {len(trajectory)}/{num_steps}")
        logger.info(f"   Stability score: {stability_score:.4f}")
        logger.info(f"   Analysis time: {rollout_time:.2f}s")
        
        return analysis_results
    
    def _calculate_overall_stability_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall stability score (0-1, higher is better)"""
        
        scores = []
        weights = []
        
        # Drift score (lower drift is better)
        if 'drift_analysis' in analysis_results:
            drift_data = analysis_results['drift_analysis']
            
            # Linear drift rate score
            drift_rate = abs(drift_data.get('linear_drift_rate', 0.0))
            drift_score = max(0, 1 - drift_rate * 1000)  # Scale appropriately
            scores.append(drift_score)
            weights.append(3.0)
            
            # Time to drift score
            time_to_drift = drift_data.get('time_to_significant_drift', np.inf)
            if np.isfinite(time_to_drift):
                drift_time_score = min(1.0, time_to_drift / 1000)  # Normalize by expected time
            else:
                drift_time_score = 1.0
            scores.append(drift_time_score)
            weights.append(2.0)
        
        # Energy conservation score
        if 'energy_conservation' in analysis_results:
            energy_data = analysis_results['energy_conservation']
            
            relative_error = energy_data.get('relative_energy_error', np.inf)
            if np.isfinite(relative_error):
                energy_score = max(0, 1 - relative_error * 10)
            else:
                energy_score = 0.0
            scores.append(energy_score)
            weights.append(2.5)
        
        # Lyapunov exponent score (negative is stable, positive is chaotic)
        if 'lyapunov_analysis' in analysis_results:
            lyapunov_data = analysis_results['lyapunov_analysis']
            lyapunov_exp = lyapunov_data.get('largest_lyapunov_exponent', 0.0)
            
            if lyapunov_exp < -0.01:
                lyapunov_score = 1.0  # Stable
            elif lyapunov_exp < 0.01:
                lyapunov_score = 0.7  # Marginally stable
            else:
                lyapunov_score = max(0, 0.5 - lyapunov_exp * 10)  # Chaotic
            
            scores.append(lyapunov_score)
            weights.append(2.0)
        
        # Stationarity score
        if 'stationarity_analysis' in analysis_results:
            stationarity_data = analysis_results['stationarity_analysis']
            overall_assessment = stationarity_data.get('overall_assessment', {})
            
            stationarity_score = 0.0
            if overall_assessment.get('is_stationary', False):
                stationarity_score += 0.7
            if overall_assessment.get('is_normal', False):
                stationarity_score += 0.3
            
            scores.append(stationarity_score)
            weights.append(1.5)
        
        # Calculate weighted average
        if scores:
            overall_score = np.average(scores, weights=weights)
        else:
            overall_score = 0.0
        
        return float(np.clip(overall_score, 0.0, 1.0))
    
    async def comprehensive_stability_test_suite(
        self,
        model_types: Optional[List[str]] = None,
        num_steps: int = 5000,
        num_initial_conditions: int = 5
    ) -> Dict[str, Any]:
        """
        Run comprehensive stability test suite across multiple models
        
        Args:
            model_types: List of model types to test
            num_steps: Number of rollout steps per test
            num_initial_conditions: Number of different initial conditions to test
            
        Returns:
            Complete test suite results
        """
        logger.info("🚀 Starting comprehensive stability test suite...")
        start_time = time.time()
        
        if model_types is None:
            model_types = ['stable', 'unstable', 'chaotic', 'oscillatory']
        
        suite_results = {
            'timestamp': datetime.now().isoformat(),
            'test_parameters': {
                'model_types': model_types,
                'num_steps': num_steps,
                'num_initial_conditions': num_initial_conditions
            },
            'individual_results': {},
            'comparative_analysis': {}
        }
        
        # Test each model type
        for model_type in model_types:
            logger.info(f"🔧 Testing {model_type} model...")
            
            # Create test model
            model = self.create_test_model(model_type)
            
            # Test with multiple initial conditions
            model_results = {}
            
            for ic_idx in range(num_initial_conditions):
                # Generate random initial condition
                torch.manual_seed(42 + ic_idx)  # For reproducibility
                initial_state = torch.randn(1, 3) * 0.5  # Small initial perturbation
                
                test_name = f"{model_type}_ic{ic_idx}"
                
                # Run rollout test
                result = await self.run_long_rollout_test(
                    model=model,
                    initial_state=initial_state,
                    num_steps=num_steps,
                    model_name=test_name
                )
                
                model_results[f'initial_condition_{ic_idx}'] = result
            
            suite_results['individual_results'][model_type] = model_results
        
        # Comparative analysis
        logger.info("🔍 Performing comparative analysis...")
        comparative_analysis = self._generate_stability_comparative_analysis(
            suite_results['individual_results']
        )
        suite_results['comparative_analysis'] = comparative_analysis
        
        total_time = time.time() - start_time
        suite_results['total_runtime_seconds'] = total_time
        
        # Save results
        await self._save_rollout_results(suite_results)
        
        logger.info(f"✅ Comprehensive stability test suite completed in {total_time:.2f}s")
        
        return suite_results
    
    def _generate_stability_comparative_analysis(
        self, 
        individual_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comparative analysis across all tested models"""
        
        comparative = {
            'stability_score_ranking': [],
            'lyapunov_exponent_comparison': {},
            'drift_rate_comparison': {},
            'energy_conservation_comparison': {},
            'model_type_performance': {}
        }
        
        # Collect metrics across all tests
        all_results = []
        for model_type, model_results in individual_results.items():
            for ic_name, result in model_results.items():
                if 'error' not in result:
                    all_results.append({
                        'model_type': model_type,
                        'initial_condition': ic_name,
                        'full_name': f"{model_type}_{ic_name}",
                        'stability_score': result.get('overall_stability_score', 0.0),
                        'lyapunov_exponent': result.get('lyapunov_analysis', {}).get('largest_lyapunov_exponent', 0.0),
                        'drift_rate': result.get('drift_analysis', {}).get('linear_drift_rate', 0.0),
                        'energy_error': result.get('energy_conservation', {}).get('relative_energy_error', np.inf)
                    })
        
        # Stability score ranking
        ranked_results = sorted(all_results, key=lambda x: x['stability_score'], reverse=True)
        comparative['stability_score_ranking'] = [
            (r['full_name'], r['stability_score']) for r in ranked_results
        ]
        
        # Lyapunov exponent comparison
        comparative['lyapunov_exponent_comparison'] = {
            r['full_name']: r['lyapunov_exponent'] for r in all_results
        }
        
        # Drift rate comparison
        comparative['drift_rate_comparison'] = {
            r['full_name']: r['drift_rate'] for r in all_results
        }
        
        # Energy conservation comparison
        comparative['energy_conservation_comparison'] = {
            r['full_name']: r['energy_error'] for r in all_results
            if np.isfinite(r['energy_error'])
        }
        
        # Model type performance summary
        for model_type in individual_results.keys():
            type_results = [r for r in all_results if r['model_type'] == model_type]
            
            if type_results:
                comparative['model_type_performance'][model_type] = {
                    'mean_stability_score': np.mean([r['stability_score'] for r in type_results]),
                    'std_stability_score': np.std([r['stability_score'] for r in type_results]),
                    'mean_lyapunov_exponent': np.mean([r['lyapunov_exponent'] for r in type_results]),
                    'num_tests': len(type_results)
                }
        
        return comparative
    
    async def _save_rollout_results(self, results: Dict[str, Any]) -> None:
        """Save rollout results and generate visualizations"""
        
        # Save main results as JSON
        results_file = self.results_path / "rollout.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary CSV
        csv_data = []
        for model_type, model_results in results['individual_results'].items():
            for ic_name, result in model_results.items():
                if 'error' not in result:
                    row = {
                        'model_type': model_type,
                        'initial_condition': ic_name,
                        'stability_score': result.get('overall_stability_score', 0.0),
                        'steps_completed': result.get('rollout_parameters', {}).get('steps_completed', 0),
                        'rollout_time': result.get('rollout_parameters', {}).get('rollout_time_seconds', 0.0),
                        'drift_rate': result.get('drift_analysis', {}).get('linear_drift_rate', 0.0),
                        'lyapunov_exponent': result.get('lyapunov_analysis', {}).get('largest_lyapunov_exponent', 0.0),
                        'energy_error': result.get('energy_conservation', {}).get('relative_energy_error', np.inf)
                    }
                    csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(self.results_path / "rollout.csv", index=False)
        
        # Generate figures
        await self._generate_rollout_figures(results)
        
        logger.info(f"📁 Rollout results saved to {self.results_path}")
    
    async def _generate_rollout_figures(self, results: Dict[str, Any]) -> None:
        """Generate rollout visualization figures"""
        
        plt.style.use('seaborn-v0_8')
        
        # Figure 1: Stability Scores Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Long-Term Rollout Stability Analysis', fontsize=16)
        
        individual_results = results['individual_results']
        comparative = results['comparative_analysis']
        
        # Stability scores by model type
        model_types = list(individual_results.keys())
        stability_scores = []
        
        for model_type in model_types:
            type_scores = []
            for ic_name, result in individual_results[model_type].items():
                if 'error' not in result:
                    type_scores.append(result.get('overall_stability_score', 0.0))
            stability_scores.append(type_scores)
        
        # Box plot of stability scores
        axes[0, 0].boxplot(stability_scores, labels=model_types)
        axes[0, 0].set_title('Stability Scores by Model Type')
        axes[0, 0].set_ylabel('Stability Score (0-1)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Lyapunov exponents
        lyapunov_data = comparative['lyapunov_exponent_comparison']
        if lyapunov_data:
            names = list(lyapunov_data.keys())
            values = list(lyapunov_data.values())
            
            colors = []
            for name in names:
                if 'stable' in name:
                    colors.append('green')
                elif 'unstable' in name:
                    colors.append('red')
                elif 'chaotic' in name:
                    colors.append('purple')
                elif 'oscillatory' in name:
                    colors.append('orange')
                else:
                    colors.append('gray')
            
            axes[0, 1].bar(range(len(names)), values, color=colors, alpha=0.7)
            axes[0, 1].set_title('Lyapunov Exponents')
            axes[0, 1].set_ylabel('Largest Lyapunov Exponent')
            axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].set_xticks(range(len(names)))
            axes[0, 1].set_xticklabels([n.replace('_ic', '_') for n in names], rotation=45, ha='right')
        
        # Drift rates
        drift_data = comparative['drift_rate_comparison']
        if drift_data:
            names = list(drift_data.keys())
            values = [abs(v) for v in drift_data.values()]
            
            axes[1, 0].bar(range(len(names)), values, alpha=0.7)
            axes[1, 0].set_title('Drift Rates (Absolute)')
            axes[1, 0].set_ylabel('|Linear Drift Rate|')
            axes[1, 0].set_yscale('log')
            axes[1, 0].set_xticks(range(len(names)))
            axes[1, 0].set_xticklabels([n.replace('_ic', '_') for n in names], rotation=45, ha='right')
        
        # Energy conservation
        energy_data = comparative['energy_conservation_comparison']
        if energy_data:
            names = list(energy_data.keys())
            values = list(energy_data.values())
            
            axes[1, 1].bar(range(len(names)), values, alpha=0.7)
            axes[1, 1].set_title('Energy Conservation Error')
            axes[1, 1].set_ylabel('Relative Energy Error')
            axes[1, 1].set_yscale('log')
            axes[1, 1].set_xticks(range(len(names)))
            axes[1, 1].set_xticklabels([n.replace('_ic', '_') for n in names], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.results_path / "fig_rollout.svg", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Model Type Performance Summary
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Type Performance Summary', fontsize=16)
        
        if 'model_type_performance' in comparative:
            performance_data = comparative['model_type_performance']
            
            model_types = list(performance_data.keys())
            mean_scores = [performance_data[mt]['mean_stability_score'] for mt in model_types]
            std_scores = [performance_data[mt]['std_stability_score'] for mt in model_types]
            mean_lyapunov = [performance_data[mt]['mean_lyapunov_exponent'] for mt in model_types]
            
            # Mean stability scores with error bars
            axes[0].bar(model_types, mean_scores, yerr=std_scores, capsize=5, alpha=0.7)
            axes[0].set_title('Mean Stability Scores by Model Type')
            axes[0].set_ylabel('Mean Stability Score')
            axes[0].set_ylim(0, 1)
            
            # Mean Lyapunov exponents
            colors = ['green', 'red', 'purple', 'orange'][:len(model_types)]
            bars = axes[1].bar(model_types, mean_lyapunov, color=colors, alpha=0.7)
            axes[1].set_title('Mean Lyapunov Exponents by Model Type')
            axes[1].set_ylabel('Mean Lyapunov Exponent')
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add text annotations
            for bar, value in zip(bars, mean_lyapunov):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01 * np.sign(height),
                           f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(self.results_path / "fig_model_performance.svg", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Rollout figures generated successfully")


# Example usage and testing
async def main():
    """Example usage of the long rollout testing framework"""
    
    # Initialize tester
    tester = LongRolloutTester()
    
    # Run comprehensive stability test suite
    results = await tester.comprehensive_stability_test_suite(
        model_types=['stable', 'unstable', 'chaotic', 'oscillatory'],
        num_steps=2000,  # Reduced for testing
        num_initial_conditions=3  # Reduced for testing
    )
    
    # Print summary
    print("\n🔄 Long Rollout Stability Test Results Summary:")
    print("=" * 60)
    
    comparative = results['comparative_analysis']
    
    # Print stability rankings
    print("🏆 Stability Score Rankings:")
    for i, (model_name, score) in enumerate(comparative['stability_score_ranking'][:10], 1):
        print(f"  {i}. {model_name}: {score:.4f}")
    
    # Print model type performance
    if 'model_type_performance' in comparative:
        print(f"\n📊 Model Type Performance:")
        for model_type, performance in comparative['model_type_performance'].items():
            mean_score = performance['mean_stability_score']
            mean_lyapunov = performance['mean_lyapunov_exponent']
            print(f"  {model_type}: Score={mean_score:.4f}, Lyapunov={mean_lyapunov:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
