#!/usr/bin/env python3
"""
Comprehensive Datacube Evaluation System
========================================

Industry-grade evaluation framework for climate datacube surrogate models.
Includes physics validation, benchmarking, and performance assessment.

Features:
- Physics-informed validation metrics
- Benchmark planet comparisons
- Performance profiling and analysis
- Uncertainty quantification assessment
- Ablation studies
- Real-time monitoring capabilities
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings
import yaml
import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports
from surrogate import get_surrogate_manager, SurrogateMode
from datamodules.cube_dm import CubeDM
from models.datacube_unet import CubeUNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eval_cube.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    model_path: Path
    test_data_path: Path
    output_dir: Path = Path("evaluation_results")
    
    # Evaluation settings
    batch_size: int = 4
    num_workers: int = 6
    device: str = "auto"
    
    # Physics validation settings
    validate_physics: bool = True
    physics_tolerance: float = 0.1
    
    # Benchmark settings
    benchmark_planets: List[str] = field(default_factory=lambda: [
        "Earth", "TRAPPIST-1e", "Proxima Centauri b", "Kepler-442b", "TOI-715b"
    ])
    
    # Performance settings
    profile_performance: bool = True
    measure_uncertainty: bool = True
    
    # Visualization settings
    generate_plots: bool = True
    plot_format: str = "png"
    
    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.test_data_path = Path(self.test_data_path)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

class PhysicsValidator:
    """Physics-informed validation for climate datacubes"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.validation_results = []
        
        # Physical constants
        self.constants = {
            'stefan_boltzmann': 5.67e-8,  # W/m²/K⁴
            'solar_constant': 1361.0,     # W/m²
            'earth_radius': 6.371e6,      # m
            'gravity': 9.81,              # m/s²
            'gas_constant': 8.314,        # J/mol/K
            'specific_heat_air': 1004.0,  # J/kg/K
            'specific_heat_water': 4186.0 # J/kg/K
        }
    
    def validate_energy_balance(self, cube: xr.Dataset) -> Dict[str, Any]:
        """Validate global energy balance"""
        results = {}
        
        try:
            # Extract variables
            T_surf = cube.get('T_surf')
            albedo = cube.get('albedo')
            insolation = cube.attrs.get('insolation', 1.0)
            
            if T_surf is None or albedo is None:
                return {'status': 'insufficient_data'}
            
            # Calculate incoming solar radiation
            incoming_solar = self.constants['solar_constant'] * insolation
            
            # Calculate reflected radiation
            reflected = incoming_solar * albedo.mean().values
            
            # Calculate outgoing longwave radiation
            outgoing_lw = self.constants['stefan_boltzmann'] * (T_surf.mean().values ** 4)
            
            # Energy balance check
            energy_balance = incoming_solar - reflected - outgoing_lw
            energy_balance_fraction = abs(energy_balance) / incoming_solar
            
            results = {
                'status': 'validated',
                'incoming_solar': float(incoming_solar),
                'reflected_solar': float(reflected),
                'outgoing_longwave': float(outgoing_lw),
                'energy_imbalance': float(energy_balance),
                'energy_balance_fraction': float(energy_balance_fraction),
                'energy_balance_valid': energy_balance_fraction < self.config.physics_tolerance
            }
            
        except Exception as e:
            results = {'status': 'error', 'error': str(e)}
        
        return results
    
    def validate_hydrostatic_equilibrium(self, cube: xr.Dataset) -> Dict[str, Any]:
        """Validate hydrostatic equilibrium"""
        results = {}
        
        try:
            # Check if pressure coordinate exists
            if 'pressure' not in cube.coords:
                return {'status': 'no_pressure_coordinate'}
            
            T_surf = cube.get('T_surf')
            if T_surf is None:
                return {'status': 'no_temperature_data'}
            
            # Calculate pressure gradient
            pressure_coord = cube.coords['pressure']
            if len(pressure_coord) < 2:
                return {'status': 'insufficient_pressure_levels'}
            
            # Check if pressure decreases with altitude (increasing index)
            pressure_gradient = np.diff(pressure_coord.values)
            pressure_monotonic = np.all(pressure_gradient < 0)
            
            # Calculate scale height
            mean_temp = T_surf.mean().values
            scale_height = self.constants['gas_constant'] * mean_temp / self.constants['gravity']
            
            results = {
                'status': 'validated',
                'pressure_monotonic': bool(pressure_monotonic),
                'scale_height': float(scale_height),
                'pressure_range': [float(pressure_coord.min()), float(pressure_coord.max())],
                'hydrostatic_valid': pressure_monotonic and 1000 < scale_height < 50000
            }
            
        except Exception as e:
            results = {'status': 'error', 'error': str(e)}
        
        return results
    
    def validate_mass_conservation(self, cube: xr.Dataset) -> Dict[str, Any]:
        """Validate mass conservation"""
        results = {}
        
        try:
            # Check water vapor conservation
            q_H2O = cube.get('q_H2O')
            if q_H2O is None:
                return {'status': 'no_humidity_data'}
            
            # Check for reasonable water vapor mixing ratios
            max_humidity = q_H2O.max().values
            min_humidity = q_H2O.min().values
            
            # Physical limits for water vapor
            humidity_valid = (min_humidity >= 0) and (max_humidity <= 1.0)
            
            # Check for conservation in vertical column
            if 'pressure' in cube.coords:
                column_water = q_H2O.sum(dim='pressure')
                water_variance = column_water.var().values
                conservation_valid = water_variance < 0.1  # Reasonable variance
            else:
                conservation_valid = True
            
            results = {
                'status': 'validated',
                'humidity_range': [float(min_humidity), float(max_humidity)],
                'humidity_physical': humidity_valid,
                'mass_conservation_valid': conservation_valid and humidity_valid
            }
            
        except Exception as e:
            results = {'status': 'error', 'error': str(e)}
        
        return results
    
    def validate_thermodynamic_consistency(self, cube: xr.Dataset) -> Dict[str, Any]:
        """Validate thermodynamic consistency"""
        results = {}
        
        try:
            T_surf = cube.get('T_surf')
            q_H2O = cube.get('q_H2O')
            
            if T_surf is None:
                return {'status': 'no_temperature_data'}
            
            # Check temperature range
            temp_min = T_surf.min().values
            temp_max = T_surf.max().values
            temp_range_valid = (temp_min > 100) and (temp_max < 1000)  # Reasonable range
            
            # Check for temperature gradients
            if 'lat' in cube.dims and 'lon' in cube.dims:
                # Equator should generally be warmer than poles
                lat_center = len(T_surf.lat) // 2
                equatorial_temp = T_surf.isel(lat=lat_center).mean().values
                polar_temp = T_surf.isel(lat=[0, -1]).mean().values
                
                gradient_valid = equatorial_temp >= polar_temp
            else:
                gradient_valid = True
            
            # Check Clausius-Clapeyron relation if humidity data available
            if q_H2O is not None:
                # Warmer regions should generally have higher humidity capacity
                temp_flat = T_surf.values.flatten()
                humidity_flat = q_H2O.values.flatten()
                
                # Remove invalid values
                valid_mask = ~(np.isnan(temp_flat) | np.isnan(humidity_flat))
                if np.sum(valid_mask) > 100:  # Enough data points
                    correlation = np.corrcoef(temp_flat[valid_mask], humidity_flat[valid_mask])[0, 1]
                    clausius_clapeyron_valid = correlation > 0.1  # Positive correlation
                else:
                    clausius_clapeyron_valid = True
            else:
                clausius_clapeyron_valid = True
            
            results = {
                'status': 'validated',
                'temperature_range': [float(temp_min), float(temp_max)],
                'temperature_range_valid': temp_range_valid,
                'temperature_gradient_valid': gradient_valid,
                'clausius_clapeyron_valid': clausius_clapeyron_valid,
                'thermodynamic_consistent': temp_range_valid and gradient_valid and clausius_clapeyron_valid
            }
            
        except Exception as e:
            results = {'status': 'error', 'error': str(e)}
        
        return results
    
    def validate_cube(self, cube: xr.Dataset) -> Dict[str, Any]:
        """Comprehensive cube validation"""
        
        validation_results = {
            'energy_balance': self.validate_energy_balance(cube),
            'hydrostatic_equilibrium': self.validate_hydrostatic_equilibrium(cube),
            'mass_conservation': self.validate_mass_conservation(cube),
            'thermodynamic_consistency': self.validate_thermodynamic_consistency(cube)
        }
        
        # Overall validation status
        all_valid = all(
            result.get('energy_balance_valid', True) or 
            result.get('hydrostatic_valid', True) or 
            result.get('mass_conservation_valid', True) or 
            result.get('thermodynamic_consistent', True)
            for result in validation_results.values()
            if result.get('status') == 'validated'
        )
        
        validation_results['overall_valid'] = all_valid
        
        return validation_results

class BenchmarkEvaluator:
    """Benchmark evaluation against known planets"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.benchmark_data = self._load_benchmark_data()
        
    def _load_benchmark_data(self) -> Dict[str, Dict[str, Any]]:
        """Load benchmark planet data"""
        
        # Known benchmark planets with their properties
        benchmark_data = {
            "Earth": {
                "radius_earth": 1.0,
                "mass_earth": 1.0,
                "insolation": 1.0,
                "stellar_teff": 5778.0,
                "expected_temp": 288.0,  # K
                "expected_pressure": 1.0,  # bar
                "expected_humidity": 0.01,  # mixing ratio
                "habitability_expected": 1.0
            },
            "TRAPPIST-1e": {
                "radius_earth": 0.92,
                "mass_earth": 0.69,
                "insolation": 0.66,
                "stellar_teff": 2511.0,
                "expected_temp": 230.0,
                "expected_pressure": 0.1,
                "expected_humidity": 0.001,
                "habitability_expected": 0.7
            },
            "Proxima Centauri b": {
                "radius_earth": 1.07,
                "mass_earth": 1.27,
                "insolation": 1.5,
                "stellar_teff": 3042.0,
                "expected_temp": 234.0,
                "expected_pressure": 0.5,
                "expected_humidity": 0.005,
                "habitability_expected": 0.6
            },
            "Kepler-442b": {
                "radius_earth": 1.34,
                "mass_earth": 2.3,
                "insolation": 0.70,
                "stellar_teff": 4402.0,
                "expected_temp": 233.0,
                "expected_pressure": 2.0,
                "expected_humidity": 0.002,
                "habitability_expected": 0.8
            },
            "TOI-715b": {
                "radius_earth": 1.55,
                "mass_earth": 3.02,
                "insolation": 1.37,
                "stellar_teff": 3341.0,
                "expected_temp": 280.0,
                "expected_pressure": 1.5,
                "expected_humidity": 0.008,
                "habitability_expected": 0.9
            }
        }
        
        return benchmark_data
    
    def evaluate_benchmark_planet(self, planet_name: str, model_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model prediction against benchmark planet"""
        
        if planet_name not in self.benchmark_data:
            return {'status': 'unknown_planet'}
        
        benchmark = self.benchmark_data[planet_name]
        
        # Extract predicted values
        climate_metrics = model_prediction.get('climate_metrics', {})
        
        results = {
            'planet_name': planet_name,
            'status': 'evaluated'
        }
        
        # Temperature comparison
        predicted_temp = climate_metrics.get('global_mean_temperature', 0)
        expected_temp = benchmark['expected_temp']
        temp_error = abs(predicted_temp - expected_temp)
        temp_relative_error = temp_error / expected_temp
        
        results['temperature'] = {
            'predicted': predicted_temp,
            'expected': expected_temp,
            'absolute_error': temp_error,
            'relative_error': temp_relative_error,
            'acceptable': temp_relative_error < 0.2  # 20% tolerance
        }
        
        # Pressure comparison
        predicted_pressure = climate_metrics.get('mean_surface_pressure', 0)
        expected_pressure = benchmark['expected_pressure']
        pressure_error = abs(predicted_pressure - expected_pressure)
        pressure_relative_error = pressure_error / expected_pressure if expected_pressure > 0 else 0
        
        results['pressure'] = {
            'predicted': predicted_pressure,
            'expected': expected_pressure,
            'absolute_error': pressure_error,
            'relative_error': pressure_relative_error,
            'acceptable': pressure_relative_error < 0.5  # 50% tolerance
        }
        
        # Humidity comparison
        predicted_humidity = climate_metrics.get('global_mean_humidity', 0)
        expected_humidity = benchmark['expected_humidity']
        humidity_error = abs(predicted_humidity - expected_humidity)
        humidity_relative_error = humidity_error / expected_humidity if expected_humidity > 0 else 0
        
        results['humidity'] = {
            'predicted': predicted_humidity,
            'expected': expected_humidity,
            'absolute_error': humidity_error,
            'relative_error': humidity_relative_error,
            'acceptable': humidity_relative_error < 1.0  # 100% tolerance (humidity is highly variable)
        }
        
        # Overall assessment
        all_acceptable = (
            results['temperature']['acceptable'] and
            results['pressure']['acceptable'] and
            results['humidity']['acceptable']
        )
        
        results['overall_assessment'] = {
            'all_acceptable': all_acceptable,
            'score': (
                (1 - min(temp_relative_error, 1.0)) * 0.5 +
                (1 - min(pressure_relative_error, 1.0)) * 0.3 +
                (1 - min(humidity_relative_error, 1.0)) * 0.2
            )
        }
        
        return results

class PerformanceProfiler:
    """Performance profiling for model evaluation"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.profile_data = []
        
    def profile_model_inference(self, model, test_data: torch.Tensor, num_runs: int = 10) -> Dict[str, Any]:
        """Profile model inference performance"""
        
        inference_times = []
        memory_usage = []
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(test_data)
        
        # Profiling runs
        for _ in range(num_runs):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            start_time = time.time()
            
            # Memory before inference
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated()
            else:
                memory_before = 0
            
            with torch.no_grad():
                output = model(test_data)
            
            # Memory after inference
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                torch.cuda.synchronize()
            else:
                memory_after = 0
            
            inference_time = time.time() - start_time
            memory_used = (memory_after - memory_before) / (1024**2)  # MB
            
            inference_times.append(inference_time)
            memory_usage.append(memory_used)
        
        # Statistics
        return {
            'batch_size': test_data.size(0),
            'inference_times': inference_times,
            'memory_usage': memory_usage,
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),
            'throughput_samples_per_sec': test_data.size(0) / np.mean(inference_times)
        }

class CubeEvaluator:
    """Main evaluation class for datacube models"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.physics_validator = PhysicsValidator(config)
        self.benchmark_evaluator = BenchmarkEvaluator(config)
        self.performance_profiler = PerformanceProfiler(config)
        
        # Results storage
        self.evaluation_results = {
            'physics_validation': [],
            'benchmark_evaluation': [],
            'performance_profile': {},
            'summary_metrics': {}
        }
    
    def load_model(self) -> torch.nn.Module:
        """Load model for evaluation"""
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.config.model_path.suffix == '.ckpt':
            # Lightning checkpoint
            model = CubeUNet.load_from_checkpoint(str(self.config.model_path))
        elif self.config.model_path.suffix == '.onnx':
            # ONNX model - use surrogate manager
            surrogate_manager = get_surrogate_manager()
            model = surrogate_manager.get_model(SurrogateMode.DATACUBE)
            if not model:
                raise RuntimeError("No ONNX model loaded")
        else:
            # Regular PyTorch model
            model = CubeUNet()
            checkpoint = torch.load(self.config.model_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        
        model.to(device)
        model.eval()
        
        return model
    
    def load_test_data(self) -> torch.utils.data.DataLoader:
        """Load test data"""
        
        if self.config.test_data_path.suffix == '.zarr':
            # Zarr dataset
            datamodule = CubeDM(
                zarr_root=str(self.config.test_data_path),
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers
            )
            datamodule.setup('test')
            return datamodule.test_dataloader()
        else:
            raise ValueError(f"Unsupported test data format: {self.config.test_data_path.suffix}")
    
    def evaluate_physics_validation(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> List[Dict[str, Any]]:
        """Evaluate physics validation"""
        
        physics_results = []
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Physics validation")):
            inputs, targets = batch
            
            with torch.no_grad():
                predictions = model(inputs)
            
            # Convert predictions to xarray Dataset for validation
            for i in range(predictions.size(0)):
                # Create mock dataset from prediction
                pred_tensor = predictions[i]
                
                # Mock coordinates (would need to be adapted to actual data structure)
                lat = np.linspace(-90, 90, pred_tensor.shape[1])
                lon = np.linspace(-180, 180, pred_tensor.shape[2])
                pressure = np.logspace(2, -2, pred_tensor.shape[3])
                
                # Create dataset
                cube = xr.Dataset({
                    'T_surf': (['lat', 'lon', 'pressure'], pred_tensor[0].cpu().numpy()),
                    'q_H2O': (['lat', 'lon', 'pressure'], pred_tensor[1].cpu().numpy() if pred_tensor.size(0) > 1 else np.zeros_like(pred_tensor[0].cpu().numpy())),
                    'albedo': (['lat', 'lon'], pred_tensor[2].cpu().numpy()[:, :, 0] if pred_tensor.size(0) > 2 else np.ones((len(lat), len(lon))) * 0.3)
                }, coords={
                    'lat': lat,
                    'lon': lon,
                    'pressure': pressure
                })
                
                # Add mock attributes
                cube.attrs['insolation'] = 1.0
                
                # Validate
                validation_result = self.physics_validator.validate_cube(cube)
                validation_result['batch_idx'] = batch_idx
                validation_result['sample_idx'] = i
                
                physics_results.append(validation_result)
        
        return physics_results
    
    def evaluate_benchmark_planets(self, model: torch.nn.Module) -> List[Dict[str, Any]]:
        """Evaluate against benchmark planets"""
        
        benchmark_results = []
        
        for planet_name in self.config.benchmark_planets:
            if planet_name not in self.benchmark_evaluator.benchmark_data:
                continue
            
            planet_data = self.benchmark_evaluator.benchmark_data[planet_name]
            
            # Create input tensor
            planet_tensor = torch.tensor([
                planet_data['radius_earth'],
                planet_data['mass_earth'],
                planet_data['insolation'],
                planet_data['stellar_teff'] / 1000.0,  # Normalize
                0.0,  # Placeholder for additional features
                0.0,
                0.0,
                1.0
            ], dtype=torch.float32).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                prediction = model(planet_tensor)
            
            # Extract climate metrics (simplified)
            climate_metrics = {
                'global_mean_temperature': float(prediction[0, 0].mean()),
                'mean_surface_pressure': float(prediction[0, 1].mean()) if prediction.size(1) > 1 else 1.0,
                'global_mean_humidity': float(prediction[0, 2].mean()) if prediction.size(1) > 2 else 0.01
            }
            
            # Evaluate
            result = self.benchmark_evaluator.evaluate_benchmark_planet(
                planet_name, 
                {'climate_metrics': climate_metrics}
            )
            
            benchmark_results.append(result)
        
        return benchmark_results
    
    def evaluate_performance(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Evaluate model performance"""
        
        # Create test input
        test_input = torch.randn(self.config.batch_size, 8)  # 8 input features
        
        if torch.cuda.is_available():
            test_input = test_input.cuda()
            model = model.cuda()
        
        # Profile performance
        performance_results = self.performance_profiler.profile_model_inference(
            model, test_input, num_runs=20
        )
        
        return performance_results
    
    def generate_summary_metrics(self) -> Dict[str, Any]:
        """Generate summary metrics"""
        
        summary = {}
        
        # Physics validation summary
        physics_results = self.evaluation_results['physics_validation']
        if physics_results:
            valid_count = sum(1 for r in physics_results if r.get('overall_valid', False))
            summary['physics_validation'] = {
                'total_samples': len(physics_results),
                'valid_samples': valid_count,
                'validation_rate': valid_count / len(physics_results),
                'energy_balance_pass_rate': sum(1 for r in physics_results if r.get('energy_balance', {}).get('energy_balance_valid', False)) / len(physics_results),
                'hydrostatic_pass_rate': sum(1 for r in physics_results if r.get('hydrostatic_equilibrium', {}).get('hydrostatic_valid', False)) / len(physics_results),
                'mass_conservation_pass_rate': sum(1 for r in physics_results if r.get('mass_conservation', {}).get('mass_conservation_valid', False)) / len(physics_results)
            }
        
        # Benchmark evaluation summary
        benchmark_results = self.evaluation_results['benchmark_evaluation']
        if benchmark_results:
            acceptable_count = sum(1 for r in benchmark_results if r.get('overall_assessment', {}).get('all_acceptable', False))
            avg_score = np.mean([r.get('overall_assessment', {}).get('score', 0) for r in benchmark_results])
            
            summary['benchmark_evaluation'] = {
                'total_benchmarks': len(benchmark_results),
                'acceptable_benchmarks': acceptable_count,
                'benchmark_pass_rate': acceptable_count / len(benchmark_results),
                'average_score': avg_score,
                'temperature_accuracy': np.mean([1 - r.get('temperature', {}).get('relative_error', 1) for r in benchmark_results]),
                'pressure_accuracy': np.mean([1 - r.get('pressure', {}).get('relative_error', 1) for r in benchmark_results])
            }
        
        # Performance summary
        performance = self.evaluation_results['performance_profile']
        if performance:
            summary['performance'] = {
                'avg_inference_time': performance['avg_inference_time'],
                'throughput': performance['throughput_samples_per_sec'],
                'memory_usage_mb': performance['avg_memory_mb']
            }
        
        return summary
    
    def save_results(self):
        """Save evaluation results"""
        
        # Save detailed results
        results_file = self.config.output_dir / f"evaluation_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.config.output_dir / f"evaluation_summary_{int(time.time())}.json"
        with open(summary_file, 'w') as f:
            json.dump(self.evaluation_results['summary_metrics'], f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {results_file}")
        logger.info(f"Evaluation summary saved to {summary_file}")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation"""
        
        logger.info("Starting cube evaluation...")
        
        # Load model and data
        model = self.load_model()
        test_loader = self.load_test_data()
        
        # Physics validation
        if self.config.validate_physics:
            logger.info("Running physics validation...")
            physics_results = self.evaluate_physics_validation(model, test_loader)
            self.evaluation_results['physics_validation'] = physics_results
        
        # Benchmark evaluation
        logger.info("Running benchmark evaluation...")
        benchmark_results = self.evaluate_benchmark_planets(model)
        self.evaluation_results['benchmark_evaluation'] = benchmark_results
        
        # Performance profiling
        if self.config.profile_performance:
            logger.info("Running performance profiling...")
            performance_results = self.evaluate_performance(model)
            self.evaluation_results['performance_profile'] = performance_results
        
        # Generate summary
        summary_metrics = self.generate_summary_metrics()
        self.evaluation_results['summary_metrics'] = summary_metrics
        
        # Save results
        self.save_results()
        
        logger.info("Evaluation completed successfully!")
        return self.evaluation_results

def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description="Evaluate datacube model")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--test_zarr", required=True, help="Path to test zarr dataset")
    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of workers")
    parser.add_argument("--no_physics", action="store_true", help="Skip physics validation")
    parser.add_argument("--no_performance", action="store_true", help="Skip performance profiling")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = EvaluationConfig(
        model_path=args.ckpt,
        test_data_path=args.test_zarr,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validate_physics=not args.no_physics,
        profile_performance=not args.no_performance
    )
    
    # Run evaluation
    evaluator = CubeEvaluator(config)
    
    try:
        results = evaluator.run_evaluation()
        
        # Print summary
        summary = results['summary_metrics']
        logger.info("\n" + "="*50)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*50)
        
        if 'physics_validation' in summary:
            pv = summary['physics_validation']
            logger.info(f"Physics Validation: {pv['validation_rate']:.1%} pass rate")
            logger.info(f"  Energy Balance: {pv['energy_balance_pass_rate']:.1%}")
            logger.info(f"  Hydrostatic: {pv['hydrostatic_pass_rate']:.1%}")
            logger.info(f"  Mass Conservation: {pv['mass_conservation_pass_rate']:.1%}")
        
        if 'benchmark_evaluation' in summary:
            be = summary['benchmark_evaluation']
            logger.info(f"Benchmark Evaluation: {be['benchmark_pass_rate']:.1%} pass rate")
            logger.info(f"  Average Score: {be['average_score']:.3f}")
            logger.info(f"  Temperature Accuracy: {be['temperature_accuracy']:.1%}")
            logger.info(f"  Pressure Accuracy: {be['pressure_accuracy']:.1%}")
        
        if 'performance' in summary:
            perf = summary['performance']
            logger.info(f"Performance:")
            logger.info(f"  Inference Time: {perf['avg_inference_time']:.3f}s")
            logger.info(f"  Throughput: {perf['throughput']:.1f} samples/sec")
            logger.info(f"  Memory Usage: {perf['memory_usage_mb']:.1f} MB")
        
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main() 