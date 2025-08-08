"""
NASA-Level Validation Framework for Astrobiology Surrogate
========================================================

Comprehensive validation suite for ensuring model reliability:
- Benchmark planet validation
- Physics constraint verification
- Uncertainty calibration (CRPS, coverage)
- Performance benchmarking
- Cross-validation protocols
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Physics constants
STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4
EARTH_RADIUS = 6.371e6  # m
SOLAR_LUMINOSITY = 3.828e26  # W

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for validation metrics"""

    r2_score: float
    mae: float
    rmse: float
    mape: float  # Mean Absolute Percentage Error
    max_error: float
    bias: float

    # Uncertainty metrics
    crps: Optional[float] = None  # Continuous Ranked Probability Score
    coverage_68: Optional[float] = None  # 68% interval coverage
    coverage_95: Optional[float] = None  # 95% interval coverage

    # Physics constraints
    physics_violations: Optional[int] = None
    energy_balance_error: Optional[float] = None
    mass_conservation_error: Optional[float] = None


@dataclass
class BenchmarkPlanet:
    """Reference planet for validation"""

    name: str
    parameters: List[
        float
    ]  # [radius, mass, period, insolation, st_teff, st_logg, st_met, host_mass]
    expected_temperature: float
    expected_habitability: float
    temperature_tolerance: float
    source: str  # Literature reference
    notes: str = ""


class BenchmarkSuite:
    """Comprehensive validation suite for NASA-level testing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.benchmark_planets = self._load_benchmark_planets()
        self.validation_history = []

    def _load_benchmark_planets(self) -> List[BenchmarkPlanet]:
        """Load comprehensive benchmark planet dataset"""

        # Expanded benchmark set with literature values
        benchmarks = [
            BenchmarkPlanet(
                name="Earth",
                parameters=[1.0, 1.0, 365.25, 1.0, 5778, 4.44, 0.0, 1.0],
                expected_temperature=288.0,
                expected_habitability=1.0,
                temperature_tolerance=1.0,
                source="NASA Earth Fact Sheet",
                notes="Primary calibration target",
            ),
            BenchmarkPlanet(
                name="Mars",
                parameters=[0.53, 0.107, 687.0, 0.43, 5778, 4.44, 0.0, 1.0],
                expected_temperature=210.0,
                expected_habitability=0.1,
                temperature_tolerance=5.0,
                source="NASA Mars Fact Sheet",
                notes="Cold, thin atmosphere reference",
            ),
            BenchmarkPlanet(
                name="Venus",
                parameters=[0.95, 0.815, 224.7, 1.91, 5778, 4.44, 0.0, 1.0],
                expected_temperature=737.0,
                expected_habitability=0.0,
                temperature_tolerance=10.0,
                source="NASA Venus Fact Sheet",
                notes="Extreme greenhouse reference",
            ),
            BenchmarkPlanet(
                name="TRAPPIST-1e",
                parameters=[0.91, 0.77, 6.1, 0.66, 2559, 5.4, 0.04, 0.089],
                expected_temperature=251.0,
                expected_habitability=0.8,
                temperature_tolerance=5.0,
                source="Grimm et al. 2018, A&A",
                notes="M-dwarf habitable zone candidate",
            ),
            BenchmarkPlanet(
                name="Proxima Centauri b",
                parameters=[1.07, 1.17, 11.2, 1.5, 3042, 5.2, -0.29, 0.123],
                expected_temperature=234.0,
                expected_habitability=0.6,
                temperature_tolerance=10.0,
                source="Anglada-Escudé et al. 2016, Nature",
                notes="Nearest potentially habitable exoplanet",
            ),
            BenchmarkPlanet(
                name="Kepler-452b",
                parameters=[1.6, 5.0, 384.8, 1.1, 5757, 4.32, 0.21, 1.04],
                expected_temperature=265.0,
                expected_habitability=0.7,
                temperature_tolerance=8.0,
                source="Jenkins et al. 2015, AJ",
                notes="Earth's cousin in habitable zone",
            ),
            BenchmarkPlanet(
                name="TOI-715b",
                parameters=[1.55, 3.02, 19.3, 1.37, 2966, 5.0, -0.4, 0.139],
                expected_temperature=279.0,
                expected_habitability=0.75,
                temperature_tolerance=7.0,
                source="Dransfield et al. 2024, MNRAS",
                notes="Recent TESS discovery in HZ",
            ),
            BenchmarkPlanet(
                name="LHS 1140 b",
                parameters=[1.73, 6.48, 24.7, 0.46, 3216, 5.16, -0.24, 0.146],
                expected_temperature=230.0,
                expected_habitability=0.65,
                temperature_tolerance=10.0,
                source="Ment et al. 2019, AJ",
                notes="Super-Earth in conservative HZ",
            ),
            BenchmarkPlanet(
                name="K2-18b",
                parameters=[2.3, 8.6, 33.0, 1.33, 3457, 4.6, -0.24, 0.36],
                expected_temperature=264.0,
                expected_habitability=0.5,
                temperature_tolerance=12.0,
                source="Benneke et al. 2019, ApJ",
                notes="Sub-Neptune with water vapor detection",
            ),
            BenchmarkPlanet(
                name="GJ 667C c",
                parameters=[1.54, 3.8, 28.1, 0.88, 3440, 4.6, -0.59, 0.31],
                expected_temperature=277.0,
                expected_habitability=0.68,
                temperature_tolerance=8.0,
                source="Feroz & Hobson 2014, MNRAS",
                notes="M-dwarf super-Earth",
            ),
        ]

        return benchmarks

    def validate_model(
        self, model: torch.nn.Module, include_uncertainty: bool = True, save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive model validation following NASA standards.

        Returns detailed validation report with all metrics.
        """
        logger.info("Starting comprehensive model validation...")
        start_time = time.time()

        validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_info": self._get_model_info(model),
            "benchmark_validation": {},
            "physics_validation": {},
            "uncertainty_validation": {},
            "performance_metrics": {},
            "overall_assessment": {},
        }

        # 1. Benchmark Planet Validation
        logger.info("Running benchmark planet validation...")
        benchmark_results = self._validate_benchmark_planets(model, include_uncertainty)
        validation_results["benchmark_validation"] = benchmark_results

        # 2. Physics Constraint Validation
        logger.info("Validating physics constraints...")
        physics_results = self._validate_physics_constraints(model)
        validation_results["physics_validation"] = physics_results

        # 3. Uncertainty Calibration
        if include_uncertainty:
            logger.info("Validating uncertainty calibration...")
            uncertainty_results = self._validate_uncertainty_calibration(model)
            validation_results["uncertainty_validation"] = uncertainty_results

        # 4. Performance Benchmarking
        logger.info("Running performance benchmarks...")
        performance_results = self._benchmark_performance(model)
        validation_results["performance_metrics"] = performance_results

        # 5. Overall Assessment
        overall_assessment = self._compute_overall_assessment(validation_results)
        validation_results["overall_assessment"] = overall_assessment

        validation_time = time.time() - start_time
        validation_results["validation_time"] = validation_time

        # Save results
        if save_results:
            self._save_validation_results(validation_results)

        # Generate report
        self._generate_validation_report(validation_results)

        logger.info(f"Validation completed in {validation_time:.2f} seconds")
        return validation_results

    def _validate_benchmark_planets(
        self, model: torch.nn.Module, include_uncertainty: bool
    ) -> Dict[str, Any]:
        """Validate model against benchmark planets"""

        results = {}
        all_predictions = []
        all_targets = []
        all_errors = []

        for planet in self.benchmark_planets:
            # Convert parameters to tensor
            params = torch.tensor(planet.parameters, dtype=torch.float32).unsqueeze(0)

            if torch.cuda.is_available():
                params = params.cuda()
                model = model.cuda()

            model.eval()
            with torch.no_grad():
                outputs = model(params)

                # Extract predictions
                predicted_temp = float(outputs["surface_temp"].item())
                predicted_habitability = float(torch.sigmoid(outputs["habitability"]).item())

                # Compute errors
                temp_error = abs(predicted_temp - planet.expected_temperature)
                hab_error = abs(predicted_habitability - planet.expected_habitability)

                # Check if within tolerance
                within_tolerance = temp_error <= planet.temperature_tolerance

                planet_result = {
                    "predicted_temperature": predicted_temp,
                    "expected_temperature": planet.expected_temperature,
                    "temperature_error": temp_error,
                    "temperature_tolerance": planet.temperature_tolerance,
                    "within_tolerance": within_tolerance,
                    "predicted_habitability": predicted_habitability,
                    "expected_habitability": planet.expected_habitability,
                    "habitability_error": hab_error,
                    "source": planet.source,
                    "notes": planet.notes,
                }

                # Add uncertainty if available
                if include_uncertainty and hasattr(model, "predict_with_uncertainty"):
                    uncertainty_outputs = model.predict_with_uncertainty(params)
                    planet_result["temperature_uncertainty"] = float(
                        uncertainty_outputs["surface_temp_std"].item()
                    )
                    planet_result["habitability_uncertainty"] = float(
                        uncertainty_outputs["habitability_std"].item()
                    )

                results[planet.name] = planet_result

                # Collect for aggregate metrics
                all_predictions.append(predicted_temp)
                all_targets.append(planet.expected_temperature)
                all_errors.append(temp_error)

        # Compute aggregate metrics
        aggregate_metrics = {
            "mean_absolute_error": np.mean(all_errors),
            "root_mean_squared_error": np.sqrt(np.mean(np.array(all_errors) ** 2)),
            "max_error": np.max(all_errors),
            "r2_score": r2_score(all_targets, all_predictions),
            "planets_within_tolerance": sum(1 for p in results.values() if p["within_tolerance"]),
            "total_planets": len(results),
            "success_rate": sum(1 for p in results.values() if p["within_tolerance"])
            / len(results),
        }

        return {"individual_results": results, "aggregate_metrics": aggregate_metrics}

    def _validate_physics_constraints(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Validate physics constraint satisfaction"""

        # Generate diverse test cases
        n_tests = 1000
        np.random.seed(42)

        # Random planet parameters within physical bounds
        test_params = np.random.rand(n_tests, 8)
        test_params[:, 0] = test_params[:, 0] * 5.0 + 0.1  # radius: 0.1-5.1 R_earth
        test_params[:, 1] = test_params[:, 1] * 20.0 + 0.01  # mass: 0.01-20 M_earth
        test_params[:, 2] = test_params[:, 2] * 1000 + 1  # period: 1-1001 days
        test_params[:, 3] = test_params[:, 3] * 10.0 + 0.01  # insolation: 0.01-10 S_earth
        test_params[:, 4] = test_params[:, 4] * 3000 + 2000  # stellar Teff: 2000-5000K
        test_params[:, 5] = test_params[:, 5] * 2.0 + 3.0  # stellar logg: 3-5
        test_params[:, 6] = test_params[:, 6] * 2.0 - 1.0  # metallicity: -1 to +1
        test_params[:, 7] = test_params[:, 7] * 3.0 + 0.1  # host mass: 0.1-3.1 M_sun

        params_tensor = torch.tensor(test_params, dtype=torch.float32)
        if torch.cuda.is_available():
            params_tensor = params_tensor.cuda()
            model = model.cuda()

        model.eval()
        violations = []
        energy_errors = []
        mass_errors = []

        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 100
            for i in range(0, n_tests, batch_size):
                batch = params_tensor[i : i + batch_size]
                outputs = model(batch)

                # Check energy balance constraint
                if "energy_balance" in outputs:
                    energy_in = batch[:, 3]  # insolation
                    energy_out = outputs["energy_balance"].squeeze()
                    energy_error = torch.abs(energy_in - energy_out)
                    energy_errors.extend(energy_error.cpu().numpy())

                # Check atmospheric mass conservation
                if "atmospheric_composition" in outputs:
                    composition = outputs["atmospheric_composition"]
                    total_mass = composition.sum(dim=1)
                    mass_error = torch.abs(total_mass - 1.0)
                    mass_errors.extend(mass_error.cpu().numpy())

                # Check for unphysical predictions
                if "surface_temp" in outputs:
                    temps = outputs["surface_temp"]
                    unphysical_temps = ((temps < 0) | (temps > 2000)).sum().item()
                    violations.append(unphysical_temps)

        physics_results = {
            "total_tests": n_tests,
            "energy_balance": {
                "mean_error": np.mean(energy_errors) if energy_errors else None,
                "max_error": np.max(energy_errors) if energy_errors else None,
                "violations": np.sum(np.array(energy_errors) > 0.1) if energy_errors else None,
            },
            "mass_conservation": {
                "mean_error": np.mean(mass_errors) if mass_errors else None,
                "max_error": np.max(mass_errors) if mass_errors else None,
                "violations": np.sum(np.array(mass_errors) > 0.01) if mass_errors else None,
            },
            "unphysical_predictions": sum(violations),
            "physics_violation_rate": sum(violations) / n_tests,
        }

        return physics_results

    def _validate_uncertainty_calibration(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Validate uncertainty calibration using reliability diagrams"""

        if not hasattr(model, "predict_with_uncertainty"):
            return {"error": "Model does not support uncertainty quantification"}

        # Use benchmark planets for uncertainty calibration
        predictions = []
        uncertainties = []
        targets = []

        for planet in self.benchmark_planets:
            params = torch.tensor(planet.parameters, dtype=torch.float32).unsqueeze(0)

            if torch.cuda.is_available():
                params = params.cuda()
                model = model.cuda()

            with torch.no_grad():
                uncertainty_outputs = model.predict_with_uncertainty(params)

                pred_temp = float(uncertainty_outputs["surface_temp_mean"].item())
                temp_std = float(uncertainty_outputs["surface_temp_std"].item())

                predictions.append(pred_temp)
                uncertainties.append(temp_std)
                targets.append(planet.expected_temperature)

        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        targets = np.array(targets)

        # Compute calibration metrics
        errors = np.abs(predictions - targets)

        # Coverage analysis
        coverage_68 = np.mean(errors <= 1.0 * uncertainties)  # 68% should be within 1σ
        coverage_95 = np.mean(errors <= 1.96 * uncertainties)  # 95% should be within 1.96σ

        # Continuous Ranked Probability Score (CRPS)
        # Simplified CRPS assuming Gaussian uncertainties
        def gaussian_crps(pred, std, obs):
            z = (obs - pred) / std
            return std * (
                z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
            )

        crps_scores = [
            gaussian_crps(p, s, t) for p, s, t in zip(predictions, uncertainties, targets)
        ]
        mean_crps = np.mean(crps_scores)

        uncertainty_results = {
            "coverage_68": coverage_68,
            "coverage_95": coverage_95,
            "target_coverage_68": 0.68,
            "target_coverage_95": 0.95,
            "coverage_68_calibrated": abs(coverage_68 - 0.68) <= 0.05,
            "coverage_95_calibrated": abs(coverage_95 - 0.95) <= 0.05,
            "mean_crps": mean_crps,
            "uncertainty_reliability": {
                "well_calibrated": abs(coverage_68 - 0.68) <= 0.05
                and abs(coverage_95 - 0.95) <= 0.05,
                "overconfident": coverage_68 < 0.63 or coverage_95 < 0.90,
                "underconfident": coverage_68 > 0.73 or coverage_95 > 1.0,
            },
        }

        return uncertainty_results

    def _benchmark_performance(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Benchmark model performance (speed, memory)"""

        # Warm up
        dummy_input = torch.randn(1, 8)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            model = model.cuda()

        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Single inference timing
        single_times = []
        for _ in range(100):
            start = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            single_times.append(time.time() - start)

        # Batch inference timing
        batch_sizes = [1, 10, 100, 1000]
        batch_times = {}

        for batch_size in batch_sizes:
            batch_input = torch.randn(batch_size, 8)
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()

            batch_time_list = []
            for _ in range(10):
                start = time.time()
                with torch.no_grad():
                    _ = model(batch_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_time_list.append(time.time() - start)

            batch_times[batch_size] = {
                "total_time": np.mean(batch_time_list),
                "time_per_sample": np.mean(batch_time_list) / batch_size,
            }

        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        else:
            memory_allocated = None
            memory_reserved = None

        performance_results = {
            "single_inference": {
                "mean_time": np.mean(single_times),
                "std_time": np.std(single_times),
                "min_time": np.min(single_times),
                "max_time": np.max(single_times),
                "target_met": np.mean(single_times) < 0.4,  # Target: <0.4s
            },
            "batch_inference": batch_times,
            "memory_usage": {"allocated_mb": memory_allocated, "reserved_mb": memory_reserved},
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "device": str(next(model.parameters()).device),
        }

        return performance_results

    def _compute_overall_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall validation assessment"""

        # Extract key metrics
        benchmark_success_rate = validation_results["benchmark_validation"]["aggregate_metrics"][
            "success_rate"
        ]
        benchmark_mae = validation_results["benchmark_validation"]["aggregate_metrics"][
            "mean_absolute_error"
        ]
        benchmark_r2 = validation_results["benchmark_validation"]["aggregate_metrics"]["r2_score"]

        physics_violation_rate = validation_results["physics_validation"]["physics_violation_rate"]

        performance_time = validation_results["performance_metrics"]["single_inference"][
            "mean_time"
        ]
        performance_target_met = validation_results["performance_metrics"]["single_inference"][
            "target_met"
        ]

        # Overall grades
        benchmark_grade = (
            "A"
            if benchmark_success_rate >= 0.9 and benchmark_mae <= 5.0
            else (
                "B"
                if benchmark_success_rate >= 0.8 and benchmark_mae <= 10.0
                else "C" if benchmark_success_rate >= 0.7 else "F"
            )
        )

        physics_grade = (
            "A"
            if physics_violation_rate <= 0.01
            else (
                "B"
                if physics_violation_rate <= 0.05
                else "C" if physics_violation_rate <= 0.1 else "F"
            )
        )

        performance_grade = "A" if performance_target_met else "B"

        # NASA readiness assessment
        nasa_ready = (
            benchmark_grade in ["A", "B"]
            and physics_grade in ["A", "B"]
            and performance_grade == "A"
            and benchmark_r2 >= 0.9
        )

        overall_assessment = {
            "benchmark_grade": benchmark_grade,
            "physics_grade": physics_grade,
            "performance_grade": performance_grade,
            "overall_grade": min(benchmark_grade, physics_grade, performance_grade),
            "nasa_ready": nasa_ready,
            "key_metrics": {
                "benchmark_success_rate": benchmark_success_rate,
                "benchmark_mae_kelvin": benchmark_mae,
                "benchmark_r2": benchmark_r2,
                "physics_violation_rate": physics_violation_rate,
                "inference_time_seconds": performance_time,
            },
            "recommendations": self._generate_recommendations(validation_results),
        }

        return overall_assessment

    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for model improvement"""

        recommendations = []

        # Benchmark performance
        benchmark_success_rate = validation_results["benchmark_validation"]["aggregate_metrics"][
            "success_rate"
        ]
        if benchmark_success_rate < 0.8:
            recommendations.append(
                "Improve benchmark planet accuracy with additional training data"
            )

        # Physics constraints
        physics_violation_rate = validation_results["physics_validation"]["physics_violation_rate"]
        if physics_violation_rate > 0.05:
            recommendations.append("Strengthen physics constraint enforcement in loss function")

        # Performance
        performance_time = validation_results["performance_metrics"]["single_inference"][
            "mean_time"
        ]
        if performance_time > 0.4:
            recommendations.append("Optimize model architecture for faster inference")

        # Uncertainty calibration
        if "uncertainty_validation" in validation_results:
            uncertainty_results = validation_results["uncertainty_validation"]
            if not uncertainty_results.get("uncertainty_reliability", {}).get(
                "well_calibrated", True
            ):
                recommendations.append(
                    "Improve uncertainty calibration with more diverse training data"
                )

        if not recommendations:
            recommendations.append("Model meets all validation criteria - ready for deployment")

        return recommendations

    def _get_model_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Extract model information"""

        return {
            "model_class": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": str(next(model.parameters()).device),
            "mode": getattr(model, "mode", "unknown"),
        }

    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to file"""

        output_dir = Path("validation_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_{timestamp}.json"

        with open(output_dir / filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Validation results saved to {output_dir / filename}")

    def _generate_validation_report(self, results: Dict[str, Any]):
        """Generate comprehensive validation report"""

        output_dir = Path("validation_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"validation_report_{timestamp}.md"

        with open(report_file, "w") as f:
            f.write("# NASA Astrobiology Surrogate Validation Report\n\n")
            f.write(f"**Generated:** {results['timestamp']}\n\n")

            # Overall Assessment
            overall = results["overall_assessment"]
            f.write("## Overall Assessment\n\n")
            f.write(f"- **Overall Grade:** {overall['overall_grade']}\n")
            f.write(f"- **NASA Ready:** {'✅ Yes' if overall['nasa_ready'] else '❌ No'}\n")
            f.write(
                f"- **Benchmark Success Rate:** {overall['key_metrics']['benchmark_success_rate']:.1%}\n"
            )
            f.write(
                f"- **Physics Violation Rate:** {overall['key_metrics']['physics_violation_rate']:.1%}\n"
            )
            f.write(
                f"- **Inference Time:** {overall['key_metrics']['inference_time_seconds']:.3f}s\n\n"
            )

            # Recommendations
            f.write("## Recommendations\n\n")
            for rec in overall["recommendations"]:
                f.write(f"- {rec}\n")
            f.write("\n")

            # Benchmark Results
            f.write("## Benchmark Planet Results\n\n")
            f.write("| Planet | Predicted (K) | Expected (K) | Error (K) | Within Tolerance |\n")
            f.write("|--------|---------------|--------------|-----------|------------------|\n")

            benchmark_results = results["benchmark_validation"]["individual_results"]
            for planet, result in benchmark_results.items():
                status = "✅" if result["within_tolerance"] else "❌"
                f.write(
                    f"| {planet} | {result['predicted_temperature']:.1f} | "
                    f"{result['expected_temperature']:.1f} | "
                    f"{result['temperature_error']:.1f} | {status} |\n"
                )

            f.write("\n")

            # Performance Metrics
            perf = results["performance_metrics"]
            f.write("## Performance Metrics\n\n")
            f.write(
                f"- **Single Inference:** {perf['single_inference']['mean_time']:.3f}s ± {perf['single_inference']['std_time']:.3f}s\n"
            )
            f.write(
                f"- **Target Met:** {'✅' if perf['single_inference']['target_met'] else '❌'} (<0.4s)\n"
            )
            f.write(f"- **Model Parameters:** {perf['model_parameters']:,}\n")
            if perf["memory_usage"]["allocated_mb"]:
                f.write(f"- **GPU Memory:** {perf['memory_usage']['allocated_mb']:.1f} MB\n")
            f.write("\n")

        logger.info(f"Validation report saved to {report_file}")


def run_validation_suite(model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run complete validation suite on a trained model"""

    # Load model
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    # Initialize validation suite
    suite = BenchmarkSuite(config)

    # Run validation
    results = suite.validate_model(model, include_uncertainty=True, save_results=True)

    return results


if __name__ == "__main__":
    # Example usage
    config = {
        "validation": {
            "tolerance": 3.0,
            "physics_constraints": True,
            "uncertainty_calibration": True,
        }
    }

    # This would be called with an actual trained model
    # results = run_validation_suite("path/to/model.pth", config)
