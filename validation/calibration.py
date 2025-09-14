#!/usr/bin/env python3
"""
Comprehensive Model Calibration and Uncertainty Validation System
================================================================

Advanced calibration framework for uncertainty quantification validation:
- Reliability diagrams for probabilistic predictions
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
- Continuous Ranked Probability Score (CRPS) for distributional accuracy
- Coverage probability analysis for prediction intervals
- Sharpness vs reliability trade-off analysis
- Temperature scaling and Platt scaling recalibration methods

Essential for ISEF competition to demonstrate proper uncertainty quantification
and scientific rigor in probabilistic predictions.

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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistical analysis
from scipy.stats import kstest, anderson

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class CalibrationMetrics:
    """Comprehensive calibration metrics for uncertainty quantification"""
    
    def __init__(self):
        self.metrics_cache = {}
    
    def expected_calibration_error(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 15,
        strategy: str = 'uniform'
    ) -> Dict[str, float]:
        """
        Calculate Expected Calibration Error (ECE)
        
        Args:
            predictions: Model predictions [N]
            confidences: Confidence scores [N]
            targets: True targets [N] 
            n_bins: Number of bins for calibration
            strategy: Binning strategy ('uniform' or 'quantile')
            
        Returns:
            Dictionary with ECE metrics
        """
        # Convert to binary classification if needed
        if len(np.unique(targets)) > 2:
            # For regression, use prediction intervals
            return self._regression_ece(predictions, confidences, targets, n_bins, strategy)
        
        # Binary classification ECE
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        targets = np.array(targets)
        
        # Create bins
        if strategy == 'uniform':
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
        elif strategy == 'quantile':
            bin_boundaries = np.quantile(confidences, np.linspace(0, 1, n_bins + 1))
            bin_boundaries[0] = 0.0  # Ensure first bin starts at 0
            bin_boundaries[-1] = 1.0  # Ensure last bin ends at 1
        else:
            raise ValueError(f"Unknown binning strategy: {strategy}")
        
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0  # Maximum Calibration Error
        total_samples = len(predictions)
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = targets[in_bin].mean()
                
                # Average confidence in this bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Calibration error for this bin
                bin_calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                
                # Weight by proportion of samples in bin
                ece += bin_calibration_error * prop_in_bin
                mce = max(mce, bin_calibration_error)
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
                bin_counts.append(0)
        
        return {
            'ece': float(ece),
            'mce': float(mce),
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'bin_boundaries': bin_boundaries.tolist(),
            'n_bins': n_bins,
            'strategy': strategy,
            'total_samples': total_samples
        }
    
    def _regression_ece(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        n_bins: int,
        strategy: str
    ) -> Dict[str, float]:
        """Calculate ECE for regression tasks using prediction intervals"""
        
        # Calculate z-scores (normalized residuals)
        residuals = np.abs(targets - predictions)
        z_scores = residuals / (uncertainties + 1e-8)
        
        # Create bins based on predicted uncertainty
        if strategy == 'uniform':
            bin_boundaries = np.linspace(0, np.max(uncertainties), n_bins + 1)
        else:
            bin_boundaries = np.quantile(uncertainties, np.linspace(0, 1, n_bins + 1))
        
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        total_samples = len(predictions)
        
        bin_rmse = []
        bin_uncertainties = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # RMSE in this bin
                rmse_in_bin = np.sqrt(np.mean(residuals[in_bin]**2))
                
                # Average predicted uncertainty in this bin
                avg_uncertainty_in_bin = uncertainties[in_bin].mean()
                
                # Calibration error (difference between predicted and actual uncertainty)
                bin_calibration_error = abs(avg_uncertainty_in_bin - rmse_in_bin)
                
                ece += bin_calibration_error * prop_in_bin
                mce = max(mce, bin_calibration_error)
                
                bin_rmse.append(rmse_in_bin)
                bin_uncertainties.append(avg_uncertainty_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_rmse.append(0.0)
                bin_uncertainties.append(0.0)
                bin_counts.append(0)
        
        return {
            'ece': float(ece),
            'mce': float(mce),
            'bin_rmse': bin_rmse,
            'bin_uncertainties': bin_uncertainties,
            'bin_counts': bin_counts,
            'bin_boundaries': bin_boundaries.tolist(),
            'n_bins': n_bins,
            'strategy': strategy,
            'total_samples': total_samples
        }
    
    def continuous_ranked_probability_score(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        distribution: str = 'normal'
    ) -> Dict[str, float]:
        """
        Calculate Continuous Ranked Probability Score (CRPS)
        
        Args:
            predictions: Point predictions [N]
            uncertainties: Prediction uncertainties [N]
            targets: True targets [N]
            distribution: Assumed distribution ('normal', 'laplace', 'student_t')
            
        Returns:
            Dictionary with CRPS metrics
        """
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        targets = np.array(targets)
        
        if distribution == 'normal':
            # For normal distribution: CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
            # where z = (y - μ)/σ, Φ is CDF, φ is PDF
            z = (targets - predictions) / (uncertainties + 1e-8)
            
            # Standard normal CDF and PDF
            phi_z = stats.norm.cdf(z)
            pdf_z = stats.norm.pdf(z)
            
            crps_values = uncertainties * (
                z * (2 * phi_z - 1) + 
                2 * pdf_z - 
                1 / np.sqrt(np.pi)
            )
            
        elif distribution == 'laplace':
            # For Laplace distribution
            z = np.abs(targets - predictions) / (uncertainties + 1e-8)
            crps_values = uncertainties * (z - 0.5 * np.exp(-z) - 0.5)
            
        elif distribution == 'student_t':
            # Simplified CRPS for Student-t (approximation)
            # Use normal CRPS with scale adjustment
            z = (targets - predictions) / (uncertainties + 1e-8)
            phi_z = stats.norm.cdf(z)
            pdf_z = stats.norm.pdf(z)
            
            # Scale factor for Student-t (df=3 assumption)
            scale_factor = 1.2
            crps_values = scale_factor * uncertainties * (
                z * (2 * phi_z - 1) + 
                2 * pdf_z - 
                1 / np.sqrt(np.pi)
            )
            
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        mean_crps = np.mean(crps_values)
        
        # Calculate relative CRPS (compared to climatological forecast)
        climatological_variance = np.var(targets)
        climatological_crps = np.sqrt(climatological_variance) * (1 - 1/np.sqrt(np.pi))
        relative_crps = mean_crps / climatological_crps if climatological_crps > 0 else np.inf
        
        return {
            'crps': float(mean_crps),
            'crps_std': float(np.std(crps_values)),
            'relative_crps': float(relative_crps),
            'climatological_crps': float(climatological_crps),
            'crps_values': crps_values.tolist(),
            'distribution': distribution
        }
    
    def coverage_probability(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate coverage probabilities for prediction intervals
        
        Args:
            predictions: Point predictions [N]
            uncertainties: Prediction uncertainties (std dev) [N]
            targets: True targets [N]
            confidence_levels: List of confidence levels to evaluate
            
        Returns:
            Dictionary with coverage statistics
        """
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        targets = np.array(targets)
        
        coverage_results = {}
        
        for confidence_level in confidence_levels:
            # Calculate z-score for confidence level (assuming normal distribution)
            z_score = stats.norm.ppf(0.5 + confidence_level/2)
            
            # Prediction intervals
            lower_bound = predictions - z_score * uncertainties
            upper_bound = predictions + z_score * uncertainties
            
            # Check coverage
            within_interval = (targets >= lower_bound) & (targets <= upper_bound)
            empirical_coverage = np.mean(within_interval)
            
            # Coverage error
            coverage_error = abs(empirical_coverage - confidence_level)
            
            # Interval width statistics
            interval_widths = upper_bound - lower_bound
            mean_width = np.mean(interval_widths)
            median_width = np.median(interval_widths)
            
            coverage_results[f'{confidence_level:.0%}'] = {
                'expected_coverage': confidence_level,
                'empirical_coverage': float(empirical_coverage),
                'coverage_error': float(coverage_error),
                'mean_interval_width': float(mean_width),
                'median_interval_width': float(median_width),
                'z_score': float(z_score),
                'num_within_interval': int(within_interval.sum()),
                'total_samples': len(targets)
            }
        
        return coverage_results
    
    def sharpness_metrics(
        self,
        uncertainties: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate sharpness metrics for uncertainty estimates
        
        Args:
            uncertainties: Prediction uncertainties [N]
            targets: True targets [N] (for normalization)
            
        Returns:
            Dictionary with sharpness metrics
        """
        uncertainties = np.array(uncertainties)
        targets = np.array(targets)
        
        # Basic sharpness statistics
        mean_uncertainty = np.mean(uncertainties)
        median_uncertainty = np.median(uncertainties)
        std_uncertainty = np.std(uncertainties)
        
        # Normalized sharpness (relative to target variance)
        target_std = np.std(targets)
        normalized_sharpness = mean_uncertainty / target_std if target_std > 0 else np.inf
        
        # Uncertainty distribution statistics
        q25_uncertainty = np.percentile(uncertainties, 25)
        q75_uncertainty = np.percentile(uncertainties, 75)
        iqr_uncertainty = q75_uncertainty - q25_uncertainty
        
        # Coefficient of variation for uncertainties
        cv_uncertainty = std_uncertainty / mean_uncertainty if mean_uncertainty > 0 else np.inf
        
        return {
            'mean_uncertainty': float(mean_uncertainty),
            'median_uncertainty': float(median_uncertainty),
            'std_uncertainty': float(std_uncertainty),
            'normalized_sharpness': float(normalized_sharpness),
            'q25_uncertainty': float(q25_uncertainty),
            'q75_uncertainty': float(q75_uncertainty),
            'iqr_uncertainty': float(iqr_uncertainty),
            'cv_uncertainty': float(cv_uncertainty),
            'min_uncertainty': float(np.min(uncertainties)),
            'max_uncertainty': float(np.max(uncertainties))
        }
    
    def reliability_diagram_data(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 15
    ) -> Dict[str, Any]:
        """
        Generate data for reliability diagrams
        
        Args:
            predictions: Model predictions [N]
            confidences: Confidence scores [N]
            targets: True targets [N]
            n_bins: Number of bins
            
        Returns:
            Data for plotting reliability diagrams
        """
        # Calculate ECE with detailed bin information
        ece_results = self.expected_calibration_error(
            predictions, confidences, targets, n_bins
        )
        
        # Additional statistics for plotting
        bin_centers = []
        bin_boundaries = ece_results['bin_boundaries']
        
        for i in range(len(bin_boundaries) - 1):
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
        
        # Perfect calibration line
        perfect_calibration = bin_centers.copy()
        
        return {
            'bin_centers': bin_centers,
            'bin_accuracies': ece_results['bin_accuracies'],
            'bin_confidences': ece_results['bin_confidences'],
            'bin_counts': ece_results['bin_counts'],
            'perfect_calibration': perfect_calibration,
            'ece': ece_results['ece'],
            'mce': ece_results['mce'],
            'total_samples': ece_results['total_samples']
        }


class TemperatureScaling:
    """Temperature scaling for model calibration"""
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
    
    def fit(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        method: str = 'nll'
    ) -> float:
        """
        Fit temperature scaling parameter
        
        Args:
            logits: Model logits [N, num_classes] or [N] for regression
            targets: True targets [N]
            method: Optimization method ('nll' or 'ece')
            
        Returns:
            Optimal temperature
        """
        logits = np.array(logits)
        targets = np.array(targets)
        
        if method == 'nll':
            # Minimize negative log-likelihood
            def nll_loss(temperature):
                scaled_logits = logits / temperature
                if logits.ndim == 1:  # Regression case
                    # Assume Gaussian with unit variance
                    return np.mean(0.5 * (scaled_logits - targets)**2)
                else:  # Classification case
                    # Softmax cross-entropy
                    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
                    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                    return -np.mean(np.log(probs[np.arange(len(targets)), targets.astype(int)] + 1e-8))
            
            # Optimize temperature
            result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
            self.temperature = result.x
            
        elif method == 'ece':
            # Minimize Expected Calibration Error
            def ece_loss(temperature):
                scaled_logits = logits / temperature
                if logits.ndim == 1:  # Regression case
                    # Use scaled predictions as confidences
                    predictions = scaled_logits
                    confidences = 1.0 / (1.0 + np.abs(predictions - targets))
                else:  # Classification case
                    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
                    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                    predictions = np.argmax(probs, axis=1)
                    confidences = np.max(probs, axis=1)
                
                # Calculate ECE
                metrics = CalibrationMetrics()
                ece_result = metrics.expected_calibration_error(
                    predictions, confidences, targets
                )
                return ece_result['ece']
            
            # Optimize temperature
            result = minimize_scalar(ece_loss, bounds=(0.1, 10.0), method='bounded')
            self.temperature = result.x
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.is_fitted = True
        return self.temperature
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Model logits
            
        Returns:
            Temperature-scaled logits
        """
        if not self.is_fitted:
            raise ValueError("Temperature scaling must be fitted first")
        
        return logits / self.temperature


class CalibrationValidationFramework:
    """Comprehensive framework for model calibration validation"""
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        results_path: str = "results"
    ):
        self.device = device
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics_calculator = CalibrationMetrics()
        self.temperature_scaler = TemperatureScaling()
        
        # Results storage
        self.calibration_results = {}
        
        logger.info(f"🎯 Calibration Validation Framework initialized")
        logger.info(f"   Device: {device}")
        logger.info(f"   Results path: {results_path}")
    
    def generate_test_predictions(
        self,
        num_samples: int = 2000,
        task_type: str = 'regression',
        calibration_quality: str = 'good'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic predictions for calibration testing
        
        Args:
            num_samples: Number of samples
            task_type: 'regression' or 'classification'
            calibration_quality: 'good', 'overconfident', 'underconfident'
            
        Returns:
            Tuple of (predictions, uncertainties/confidences, targets)
        """
        logger.info(f"🔧 Generating test predictions: {num_samples} samples, {task_type}, {calibration_quality}")
        
        np.random.seed(42)  # For reproducibility
        
        if task_type == 'regression':
            # Generate true function
            x = np.linspace(-3, 3, num_samples)
            true_function = 2 * np.sin(x) + 0.5 * x**2 - 1
            
            # Add heteroscedastic noise
            noise_std = 0.3 + 0.2 * np.abs(x)  # Varying noise
            targets = true_function + np.random.normal(0, noise_std)
            
            # Generate predictions with some error
            prediction_noise = 0.1
            predictions = true_function + np.random.normal(0, prediction_noise, num_samples)
            
            # Generate uncertainties based on calibration quality
            true_uncertainties = noise_std + prediction_noise
            
            if calibration_quality == 'good':
                # Well-calibrated uncertainties
                uncertainties = true_uncertainties * (1 + np.random.normal(0, 0.1, num_samples))
            elif calibration_quality == 'overconfident':
                # Underestimated uncertainties (overconfident)
                uncertainties = true_uncertainties * 0.6
            elif calibration_quality == 'underconfident':
                # Overestimated uncertainties (underconfident)
                uncertainties = true_uncertainties * 1.8
            else:
                raise ValueError(f"Unknown calibration quality: {calibration_quality}")
            
            # Ensure positive uncertainties
            uncertainties = np.maximum(uncertainties, 0.01)
            
        elif task_type == 'classification':
            # Binary classification
            # Generate 2D features
            features = np.random.randn(num_samples, 2)
            
            # True decision boundary
            true_logits = 2 * features[:, 0] - features[:, 1] + 0.5 * features[:, 0] * features[:, 1]
            true_probs = 1 / (1 + np.exp(-true_logits))
            targets = np.random.binomial(1, true_probs)
            
            # Model predictions with some error
            prediction_logits = true_logits + np.random.normal(0, 0.5, num_samples)
            prediction_probs = 1 / (1 + np.exp(-prediction_logits))
            
            # Apply calibration quality
            if calibration_quality == 'good':
                # Well-calibrated
                confidences = prediction_probs
            elif calibration_quality == 'overconfident':
                # Push probabilities towards extremes
                confidences = np.where(prediction_probs > 0.5, 
                                     0.5 + 0.8 * (prediction_probs - 0.5),
                                     0.5 - 0.8 * (0.5 - prediction_probs))
            elif calibration_quality == 'underconfident':
                # Push probabilities towards center
                confidences = 0.5 + 0.3 * (prediction_probs - 0.5)
            else:
                raise ValueError(f"Unknown calibration quality: {calibration_quality}")
            
            predictions = (confidences > 0.5).astype(int)
            uncertainties = confidences  # Use confidences as uncertainties for classification
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        logger.info(f"✅ Test predictions generated: {predictions.shape}")
        
        return predictions, uncertainties, targets
    
    async def comprehensive_calibration_analysis(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        task_type: str = 'regression',
        model_name: str = 'test_model'
    ) -> Dict[str, Any]:
        """
        Run comprehensive calibration analysis
        
        Args:
            predictions: Model predictions
            uncertainties: Prediction uncertainties or confidences
            targets: True targets
            task_type: 'regression' or 'classification'
            model_name: Name for results storage
            
        Returns:
            Complete calibration analysis results
        """
        logger.info(f"🎯 Running comprehensive calibration analysis: {model_name}")
        start_time = time.time()
        
        analysis_results = {
            'model_name': model_name,
            'task_type': task_type,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(predictions)
        }
        
        # 1. Expected Calibration Error
        logger.info("  📊 Calculating Expected Calibration Error...")
        ece_results = self.metrics_calculator.expected_calibration_error(
            predictions, uncertainties, targets
        )
        analysis_results['ece'] = ece_results
        
        # 2. Continuous Ranked Probability Score (for regression)
        if task_type == 'regression':
            logger.info("  📈 Calculating CRPS...")
            crps_results = self.metrics_calculator.continuous_ranked_probability_score(
                predictions, uncertainties, targets
            )
            analysis_results['crps'] = crps_results
            
            # Coverage probability analysis
            logger.info("  🎯 Analyzing coverage probabilities...")
            coverage_results = self.metrics_calculator.coverage_probability(
                predictions, uncertainties, targets
            )
            analysis_results['coverage'] = coverage_results
            
            # Sharpness metrics
            logger.info("  🔪 Calculating sharpness metrics...")
            sharpness_results = self.metrics_calculator.sharpness_metrics(
                uncertainties, targets
            )
            analysis_results['sharpness'] = sharpness_results
        
        # 3. Reliability diagram data
        logger.info("  📉 Generating reliability diagram data...")
        reliability_data = self.metrics_calculator.reliability_diagram_data(
            predictions, uncertainties, targets
        )
        analysis_results['reliability_diagram'] = reliability_data
        
        # 4. Temperature scaling calibration
        logger.info("  🌡️ Applying temperature scaling...")
        try:
            if task_type == 'regression':
                # For regression, use predictions as logits
                optimal_temp = self.temperature_scaler.fit(predictions, targets, method='nll')
                calibrated_predictions = self.temperature_scaler.transform(predictions)
            else:
                # For classification, assume we have logits
                # Convert confidences back to logits (approximation)
                logits = np.log(uncertainties / (1 - uncertainties + 1e-8))
                optimal_temp = self.temperature_scaler.fit(logits, targets, method='ece')
                calibrated_logits = self.temperature_scaler.transform(logits)
                calibrated_predictions = 1 / (1 + np.exp(-calibrated_logits))
            
            # Recalculate metrics after temperature scaling
            calibrated_ece = self.metrics_calculator.expected_calibration_error(
                calibrated_predictions, calibrated_predictions, targets
            )
            
            analysis_results['temperature_scaling'] = {
                'optimal_temperature': float(optimal_temp),
                'original_ece': ece_results['ece'],
                'calibrated_ece': calibrated_ece['ece'],
                'improvement': float(ece_results['ece'] - calibrated_ece['ece'])
            }
            
        except Exception as e:
            logger.warning(f"  ⚠️ Temperature scaling failed: {e}")
            analysis_results['temperature_scaling'] = {'error': str(e)}
        
        # 5. Statistical tests for calibration
        logger.info("  📊 Performing statistical tests...")
        statistical_tests = await self._perform_calibration_tests(
            predictions, uncertainties, targets, task_type
        )
        analysis_results['statistical_tests'] = statistical_tests
        
        # 6. Overall calibration score
        calibration_score = self._calculate_overall_calibration_score(analysis_results)
        analysis_results['overall_calibration_score'] = calibration_score
        
        analysis_time = time.time() - start_time
        analysis_results['analysis_time_seconds'] = analysis_time
        
        # Store results
        self.calibration_results[model_name] = analysis_results
        
        logger.info(f"✅ Calibration analysis completed: {model_name}")
        logger.info(f"   ECE: {ece_results['ece']:.6f}")
        logger.info(f"   Overall score: {calibration_score:.4f}")
        logger.info(f"   Analysis time: {analysis_time:.2f}s")
        
        return analysis_results
    
    async def _perform_calibration_tests(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """Perform statistical tests for calibration quality"""
        
        tests = {}
        
        try:
            if task_type == 'regression':
                # Test if normalized residuals follow standard normal distribution
                residuals = targets - predictions
                normalized_residuals = residuals / (uncertainties + 1e-8)
                
                # Kolmogorov-Smirnov test against standard normal
                ks_statistic, ks_p_value = kstest(normalized_residuals, 'norm')
                
                # Anderson-Darling test
                ad_result = anderson(normalized_residuals, dist='norm')
                
                tests['kolmogorov_smirnov'] = {
                    'statistic': float(ks_statistic),
                    'p_value': float(ks_p_value),
                    'is_calibrated_at_0.05': ks_p_value > 0.05
                }
                
                tests['anderson_darling'] = {
                    'statistic': float(ad_result.statistic),
                    'critical_values': ad_result.critical_values.tolist(),
                    'significance_levels': ad_result.significance_level.tolist(),
                    'is_calibrated_at_5_percent': ad_result.statistic < ad_result.critical_values[2]
                }
                
                # Test coverage probabilities
                coverage_68_expected = 0.68
                coverage_95_expected = 0.95
                
                # 68% interval
                lower_68 = predictions - uncertainties
                upper_68 = predictions + uncertainties
                within_68 = ((targets >= lower_68) & (targets <= upper_68)).mean()
                
                # 95% interval
                lower_95 = predictions - 1.96 * uncertainties
                upper_95 = predictions + 1.96 * uncertainties
                within_95 = ((targets >= lower_95) & (targets <= upper_95)).mean()
                
                # Binomial tests for coverage
                from scipy.stats import binom_test
                
                n_samples = len(targets)
                coverage_68_p = binom_test(int(within_68 * n_samples), n_samples, coverage_68_expected)
                coverage_95_p = binom_test(int(within_95 * n_samples), n_samples, coverage_95_expected)
                
                tests['coverage_tests'] = {
                    '68_percent': {
                        'expected': coverage_68_expected,
                        'observed': float(within_68),
                        'p_value': float(coverage_68_p),
                        'is_calibrated': coverage_68_p > 0.05
                    },
                    '95_percent': {
                        'expected': coverage_95_expected,
                        'observed': float(within_95),
                        'p_value': float(coverage_95_p),
                        'is_calibrated': coverage_95_p > 0.05
                    }
                }
                
            else:  # Classification
                # Hosmer-Lemeshow test for binary classification
                # Group predictions into deciles and test if observed frequencies match expected
                
                # Sort by predicted probability
                sort_idx = np.argsort(uncertainties)
                sorted_probs = uncertainties[sort_idx]
                sorted_targets = targets[sort_idx]
                
                # Create 10 groups
                n_groups = 10
                group_size = len(sorted_probs) // n_groups
                
                hl_statistic = 0.0
                
                for i in range(n_groups):
                    start_idx = i * group_size
                    if i == n_groups - 1:  # Last group gets remaining samples
                        end_idx = len(sorted_probs)
                    else:
                        end_idx = (i + 1) * group_size
                    
                    group_probs = sorted_probs[start_idx:end_idx]
                    group_targets = sorted_targets[start_idx:end_idx]
                    
                    # Expected and observed positive cases
                    expected_pos = np.sum(group_probs)
                    observed_pos = np.sum(group_targets)
                    
                    # Expected and observed negative cases
                    expected_neg = len(group_probs) - expected_pos
                    observed_neg = len(group_targets) - observed_pos
                    
                    # Chi-square contribution
                    if expected_pos > 0:
                        hl_statistic += (observed_pos - expected_pos)**2 / expected_pos
                    if expected_neg > 0:
                        hl_statistic += (observed_neg - expected_neg)**2 / expected_neg
                
                # Chi-square test with df = n_groups - 2
                hl_p_value = 1 - stats.chi2.cdf(hl_statistic, n_groups - 2)
                
                tests['hosmer_lemeshow'] = {
                    'statistic': float(hl_statistic),
                    'p_value': float(hl_p_value),
                    'degrees_of_freedom': n_groups - 2,
                    'is_calibrated_at_0.05': hl_p_value > 0.05
                }
        
        except Exception as e:
            tests['error'] = str(e)
        
        return tests
    
    def _calculate_overall_calibration_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall calibration quality score (0-1, higher is better)"""
        
        scores = []
        weights = []
        
        # ECE score (lower is better, so invert)
        if 'ece' in analysis_results:
            ece = analysis_results['ece']['ece']
            ece_score = max(0, 1 - ece * 10)  # Scale ECE to 0-1 range
            scores.append(ece_score)
            weights.append(3.0)  # High weight for ECE
        
        # CRPS score (for regression)
        if 'crps' in analysis_results:
            relative_crps = analysis_results['crps']['relative_crps']
            crps_score = max(0, 1 - relative_crps)  # Relative CRPS should be < 1 for good models
            scores.append(crps_score)
            weights.append(2.0)
        
        # Coverage score
        if 'coverage' in analysis_results:
            coverage_errors = []
            for level, coverage_data in analysis_results['coverage'].items():
                error = coverage_data['coverage_error']
                coverage_errors.append(error)
            
            if coverage_errors:
                mean_coverage_error = np.mean(coverage_errors)
                coverage_score = max(0, 1 - mean_coverage_error * 10)
                scores.append(coverage_score)
                weights.append(2.0)
        
        # Statistical tests score
        if 'statistical_tests' in analysis_results:
            test_scores = []
            
            if 'kolmogorov_smirnov' in analysis_results['statistical_tests']:
                ks_calibrated = analysis_results['statistical_tests']['kolmogorov_smirnov']['is_calibrated_at_0.05']
                test_scores.append(1.0 if ks_calibrated else 0.0)
            
            if 'coverage_tests' in analysis_results['statistical_tests']:
                coverage_tests = analysis_results['statistical_tests']['coverage_tests']
                for level, test_data in coverage_tests.items():
                    test_scores.append(1.0 if test_data['is_calibrated'] else 0.0)
            
            if 'hosmer_lemeshow' in analysis_results['statistical_tests']:
                hl_calibrated = analysis_results['statistical_tests']['hosmer_lemeshow']['is_calibrated_at_0.05']
                test_scores.append(1.0 if hl_calibrated else 0.0)
            
            if test_scores:
                statistical_score = np.mean(test_scores)
                scores.append(statistical_score)
                weights.append(1.5)
        
        # Temperature scaling improvement
        if 'temperature_scaling' in analysis_results and 'improvement' in analysis_results['temperature_scaling']:
            improvement = analysis_results['temperature_scaling']['improvement']
            # Score based on how much temperature scaling improved calibration
            temp_score = min(1.0, max(0.0, 0.5 + improvement * 10))
            scores.append(temp_score)
            weights.append(1.0)
        
        # Calculate weighted average
        if scores:
            overall_score = np.average(scores, weights=weights)
        else:
            overall_score = 0.0
        
        return float(np.clip(overall_score, 0.0, 1.0))
    
    async def run_calibration_validation_suite(
        self,
        models_to_test: Optional[List[str]] = None,
        num_samples: int = 2000
    ) -> Dict[str, Any]:
        """
        Run comprehensive calibration validation suite
        
        Args:
            models_to_test: List of calibration qualities to test
            num_samples: Number of samples per test
            
        Returns:
            Complete validation results
        """
        logger.info("🚀 Starting calibration validation suite...")
        start_time = time.time()
        
        if models_to_test is None:
            models_to_test = ['good', 'overconfident', 'underconfident']
        
        suite_results = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': num_samples,
            'models_tested': models_to_test,
            'individual_results': {},
            'comparative_analysis': {}
        }
        
        # Test regression models
        logger.info("📈 Testing regression models...")
        for calibration_quality in models_to_test:
            model_name = f"regression_{calibration_quality}"
            
            # Generate test data
            predictions, uncertainties, targets = self.generate_test_predictions(
                num_samples=num_samples,
                task_type='regression',
                calibration_quality=calibration_quality
            )
            
            # Run analysis
            results = await self.comprehensive_calibration_analysis(
                predictions, uncertainties, targets,
                task_type='regression',
                model_name=model_name
            )
            
            suite_results['individual_results'][model_name] = results
        
        # Test classification models
        logger.info("📊 Testing classification models...")
        for calibration_quality in models_to_test:
            model_name = f"classification_{calibration_quality}"
            
            # Generate test data
            predictions, confidences, targets = self.generate_test_predictions(
                num_samples=num_samples,
                task_type='classification',
                calibration_quality=calibration_quality
            )
            
            # Run analysis
            results = await self.comprehensive_calibration_analysis(
                predictions, confidences, targets,
                task_type='classification',
                model_name=model_name
            )
            
            suite_results['individual_results'][model_name] = results
        
        # Comparative analysis
        logger.info("🔍 Performing comparative analysis...")
        comparative_analysis = self._generate_comparative_analysis(
            suite_results['individual_results']
        )
        suite_results['comparative_analysis'] = comparative_analysis
        
        total_time = time.time() - start_time
        suite_results['total_runtime_seconds'] = total_time
        
        # Save results
        await self._save_calibration_results(suite_results)
        
        logger.info(f"✅ Calibration validation suite completed in {total_time:.2f}s")
        
        return suite_results
    
    def _generate_comparative_analysis(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis across all tested models"""
        
        comparative = {
            'calibration_score_ranking': [],
            'ece_comparison': {},
            'best_calibrated_models': {},
            'calibration_method_effectiveness': {}
        }
        
        # Collect scores for ranking
        model_scores = {}
        ece_scores = {}
        
        for model_name, results in individual_results.items():
            score = results.get('overall_calibration_score', 0.0)
            model_scores[model_name] = score
            
            ece = results.get('ece', {}).get('ece', float('inf'))
            ece_scores[model_name] = ece
        
        # Rank by calibration score
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        comparative['calibration_score_ranking'] = ranked_models
        
        # ECE comparison
        comparative['ece_comparison'] = ece_scores
        
        # Find best models by task type
        regression_models = {k: v for k, v in model_scores.items() if 'regression' in k}
        classification_models = {k: v for k, v in model_scores.items() if 'classification' in k}
        
        if regression_models:
            best_regression = max(regression_models.items(), key=lambda x: x[1])
            comparative['best_calibrated_models']['regression'] = best_regression
        
        if classification_models:
            best_classification = max(classification_models.items(), key=lambda x: x[1])
            comparative['best_calibrated_models']['classification'] = best_classification
        
        # Temperature scaling effectiveness
        temp_improvements = {}
        for model_name, results in individual_results.items():
            if 'temperature_scaling' in results and 'improvement' in results['temperature_scaling']:
                improvement = results['temperature_scaling']['improvement']
                temp_improvements[model_name] = improvement
        
        comparative['calibration_method_effectiveness']['temperature_scaling'] = temp_improvements
        
        return comparative
    
    async def _save_calibration_results(self, results: Dict[str, Any]) -> None:
        """Save calibration results and generate visualizations"""
        
        # Save main results as JSON
        results_file = self.results_path / "calibration.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary CSV
        csv_data = []
        for model_name, model_results in results['individual_results'].items():
            row = {
                'model': model_name,
                'task_type': model_results.get('task_type', 'unknown'),
                'overall_score': model_results.get('overall_calibration_score', 0.0),
                'ece': model_results.get('ece', {}).get('ece', 0.0),
                'mce': model_results.get('ece', {}).get('mce', 0.0),
            }
            
            # Add CRPS for regression models
            if 'crps' in model_results:
                row['crps'] = model_results['crps']['crps']
                row['relative_crps'] = model_results['crps']['relative_crps']
            
            # Add coverage information
            if 'coverage' in model_results:
                for level, coverage_data in model_results['coverage'].items():
                    row[f'coverage_{level}'] = coverage_data['empirical_coverage']
                    row[f'coverage_error_{level}'] = coverage_data['coverage_error']
            
            csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(self.results_path / "calibration.csv", index=False)
        
        # Generate figures
        await self._generate_calibration_figures(results)
        
        logger.info(f"📁 Calibration results saved to {self.results_path}")
    
    async def _generate_calibration_figures(self, results: Dict[str, Any]) -> None:
        """Generate calibration visualization figures"""
        
        plt.style.use('seaborn-v0_8')
        
        individual_results = results['individual_results']
        
        # Figure 1: Reliability Diagrams
        n_models = len(individual_results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle('Reliability Diagrams - Model Calibration Assessment', fontsize=16)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, model_results) in enumerate(individual_results.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            if 'reliability_diagram' in model_results:
                rel_data = model_results['reliability_diagram']
                
                # Plot reliability curve
                ax.plot(rel_data['bin_confidences'], rel_data['bin_accuracies'], 
                       'o-', linewidth=2, markersize=6, label='Model')
                
                # Perfect calibration line
                ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, label='Perfect Calibration')
                
                # Add bin counts as bar heights (normalized)
                bin_counts = np.array(rel_data['bin_counts'])
                if bin_counts.max() > 0:
                    normalized_counts = bin_counts / bin_counts.max() * 0.1
                    bin_centers = rel_data['bin_centers']
                    ax.bar(bin_centers, normalized_counts, alpha=0.3, width=0.05)
                
                ax.set_xlabel('Confidence')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'{model_name}\nECE: {rel_data["ece"]:.4f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
        
        # Hide empty subplots
        for i in range(n_models, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_path / "fig_calibration.svg", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Calibration Scores Comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Calibration Performance Comparison', fontsize=16)
        
        model_names = list(individual_results.keys())
        calibration_scores = [individual_results[name].get('overall_calibration_score', 0.0) 
                            for name in model_names]
        ece_scores = [individual_results[name].get('ece', {}).get('ece', 0.0) 
                     for name in model_names]
        
        # Calibration scores
        bars = axes[0].bar(range(len(model_names)), calibration_scores)
        axes[0].set_title('Overall Calibration Scores')
        axes[0].set_ylabel('Calibration Score (0-1, higher is better)')
        axes[0].set_xticks(range(len(model_names)))
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].set_ylim(0, 1)
        
        # Color bars by quality
        for i, (bar, name) in enumerate(zip(bars, model_names)):
            if 'good' in name:
                bar.set_color('green')
            elif 'overconfident' in name:
                bar.set_color('red')
            elif 'underconfident' in name:
                bar.set_color('orange')
        
        # ECE scores
        axes[1].bar(range(len(model_names)), ece_scores, color='lightcoral')
        axes[1].set_title('Expected Calibration Error (ECE)')
        axes[1].set_ylabel('ECE (lower is better)')
        axes[1].set_xticks(range(len(model_names)))
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.results_path / "fig_calibration_comparison.svg", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Coverage Analysis (for regression models)
        regression_results = {k: v for k, v in individual_results.items() if 'regression' in k}
        
        if regression_results:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Coverage Probability Analysis (Regression Models)', fontsize=16)
            
            coverage_levels = ['68%', '95%']
            model_names = list(regression_results.keys())
            
            for i, level in enumerate(coverage_levels):
                expected = 0.68 if level == '68%' else 0.95
                empirical_coverages = []
                
                for model_name in model_names:
                    coverage_data = regression_results[model_name].get('coverage', {})
                    if level in coverage_data:
                        empirical_coverages.append(coverage_data[level]['empirical_coverage'])
                    else:
                        empirical_coverages.append(0.0)
                
                # Plot empirical vs expected coverage
                axes[i].bar(range(len(model_names)), empirical_coverages, alpha=0.7)
                axes[i].axhline(y=expected, color='red', linestyle='--', 
                              label=f'Expected ({expected:.0%})')
                axes[i].set_title(f'{level} Coverage Probability')
                axes[i].set_ylabel('Empirical Coverage')
                axes[i].set_xticks(range(len(model_names)))
                axes[i].set_xticklabels([name.replace('regression_', '') for name in model_names])
                axes[i].legend()
                axes[i].set_ylim(0, 1)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_path / "fig_coverage_analysis.svg", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("📊 Calibration figures generated successfully")


# Example usage and testing
async def main():
    """Example usage of the calibration validation framework"""
    
    # Initialize framework
    framework = CalibrationValidationFramework()
    
    # Run comprehensive calibration validation suite
    results = await framework.run_calibration_validation_suite(
        models_to_test=['good', 'overconfident', 'underconfident'],
        num_samples=1000
    )
    
    # Print summary
    print("\n🎯 Calibration Validation Results Summary:")
    print("=" * 60)
    
    comparative = results['comparative_analysis']
    
    # Print rankings
    print("🏆 Model Calibration Rankings:")
    for i, (model_name, score) in enumerate(comparative['calibration_score_ranking'], 1):
        print(f"  {i}. {model_name}: {score:.4f}")
    
    # Print ECE comparison
    print(f"\n📊 Expected Calibration Error (ECE):")
    for model_name, ece in comparative['ece_comparison'].items():
        print(f"  {model_name}: {ece:.6f}")
    
    # Print best models
    if 'best_calibrated_models' in comparative:
        print(f"\n🥇 Best Calibrated Models:")
        for task_type, (model_name, score) in comparative['best_calibrated_models'].items():
            print(f"  {task_type}: {model_name} ({score:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
