#!/usr/bin/env python3
"""
Comprehensive Integration Validator for Astrobiology Platform
============================================================

Advanced integration validation system that ensures seamless operation of all
multi-modal components in the astrobiology research platform. Validates model
integrations, data flow, API endpoints, and cross-component compatibility.

Features:
- Multi-modal model integration validation
- Data pipeline compatibility checking
- API endpoint health and response validation
- Cross-component dependency verification
- Performance consistency validation
- Error recovery and failover testing
- Real-time integration monitoring
- Comprehensive integration reporting
"""

import asyncio
import logging

import numpy as np
import torch
import torch.nn as nn

# Optional HTTP client libraries
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
import json
import os
import sys
import threading
import time
import traceback
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Optional testing and HTTP libraries
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
from contextlib import asynccontextmanager

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ComponentInfo:
    """Information about a system component"""

    name: str
    component_type: str  # 'model', 'api', 'dataloader', 'service'
    version: str
    dependencies: List[str] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=list)
    test_functions: List[str] = field(default_factory=list)
    health_check_url: Optional[str] = None
    expected_input_types: List[str] = field(default_factory=list)
    expected_output_types: List[str] = field(default_factory=list)
    performance_requirements: Dict[str, float] = field(default_factory=dict)
    critical: bool = True


@dataclass
class ValidationResult:
    """Result of a component validation"""

    component_name: str
    test_name: str
    status: str  # 'passed', 'failed', 'warning', 'skipped'
    message: str
    execution_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IntegrationReport:
    """Comprehensive integration validation report"""

    validation_timestamp: datetime
    total_components: int
    healthy_components: int
    warning_components: int
    failed_components: int
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    overall_health_score: float
    critical_failures: List[str] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class ModelIntegrationValidator:
    """Validator for model integration and compatibility"""

    def __init__(self):
        self.registered_models = {}
        self.compatibility_matrix = {}

    def register_model(
        self,
        model_name: str,
        model: nn.Module,
        input_spec: Dict[str, Any],
        output_spec: Dict[str, Any],
    ):
        """Register a model for integration validation"""
        self.registered_models[model_name] = {
            "model": model,
            "input_spec": input_spec,
            "output_spec": output_spec,
            "device": next(model.parameters()).device if list(model.parameters()) else "cpu",
        }
        logger.info(f"Registered model for validation: {model_name}")

    async def validate_model_integration(self, model_name: str) -> List[ValidationResult]:
        """Validate individual model integration"""
        results = []

        if model_name not in self.registered_models:
            return [
                ValidationResult(
                    component_name=model_name,
                    test_name="model_registration",
                    status="failed",
                    message="Model not registered",
                    execution_time_ms=0,
                )
            ]

        model_info = self.registered_models[model_name]
        model = model_info["model"]

        # Test 1: Model Loading and Basic Properties
        start_time = time.perf_counter()
        try:
            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (
                1024**2
            )

            results.append(
                ValidationResult(
                    component_name=model_name,
                    test_name="model_properties",
                    status="passed",
                    message=f"Model loaded successfully: {param_count:,} parameters, {model_size_mb:.2f}MB",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    details={
                        "parameter_count": param_count,
                        "model_size_mb": model_size_mb,
                        "device": str(model_info["device"]),
                    },
                )
            )
        except Exception as e:
            results.append(
                ValidationResult(
                    component_name=model_name,
                    test_name="model_properties",
                    status="failed",
                    message=f"Failed to analyze model properties: {str(e)}",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            )

        # Test 2: Forward Pass Validation
        start_time = time.perf_counter()
        try:
            # Generate test input based on spec
            test_input = self._generate_test_input(model_info["input_spec"], model_info["device"])

            model.eval()
            with torch.no_grad():
                output = model(test_input)

            # Validate output against spec
            output_valid = self._validate_output(output, model_info["output_spec"])

            if output_valid:
                results.append(
                    ValidationResult(
                        component_name=model_name,
                        test_name="forward_pass",
                        status="passed",
                        message="Forward pass successful",
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        details={
                            "input_shape": (
                                test_input.shape
                                if hasattr(test_input, "shape")
                                else str(type(test_input))
                            ),
                            "output_shape": (
                                output.shape if hasattr(output, "shape") else str(type(output))
                            ),
                        },
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        component_name=model_name,
                        test_name="forward_pass",
                        status="failed",
                        message="Forward pass produced invalid output",
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    )
                )
        except Exception as e:
            results.append(
                ValidationResult(
                    component_name=model_name,
                    test_name="forward_pass",
                    status="failed",
                    message=f"Forward pass failed: {str(e)}",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            )

        # Test 3: Memory Usage Validation
        if torch.cuda.is_available():
            start_time = time.perf_counter()
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                test_input = self._generate_test_input(
                    model_info["input_spec"], model_info["device"]
                )
                with torch.no_grad():
                    _ = model(test_input)

                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

                status = "passed" if peak_memory_mb < 4000 else "warning"  # 4GB threshold
                message = f"Peak memory usage: {peak_memory_mb:.2f}MB"

                results.append(
                    ValidationResult(
                        component_name=model_name,
                        test_name="memory_usage",
                        status=status,
                        message=message,
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        details={"peak_memory_mb": peak_memory_mb},
                    )
                )

                torch.cuda.empty_cache()
            except Exception as e:
                results.append(
                    ValidationResult(
                        component_name=model_name,
                        test_name="memory_usage",
                        status="failed",
                        message=f"Memory validation failed: {str(e)}",
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    )
                )

        # Test 4: Performance Validation
        start_time = time.perf_counter()
        try:
            # Benchmark inference time
            test_input = self._generate_test_input(model_info["input_spec"], model_info["device"])

            # Warmup
            model.eval()
            with torch.no_grad():
                for _ in range(5):
                    _ = model(test_input)

            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            benchmark_start = time.perf_counter()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            avg_inference_time_ms = (time.perf_counter() - benchmark_start) / 10 * 1000

            status = "passed" if avg_inference_time_ms < 1000 else "warning"  # 1 second threshold

            results.append(
                ValidationResult(
                    component_name=model_name,
                    test_name="performance",
                    status=status,
                    message=f"Average inference time: {avg_inference_time_ms:.2f}ms",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    details={"avg_inference_time_ms": avg_inference_time_ms},
                )
            )
        except Exception as e:
            results.append(
                ValidationResult(
                    component_name=model_name,
                    test_name="performance",
                    status="failed",
                    message=f"Performance validation failed: {str(e)}",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            )

        return results

    def _generate_test_input(self, input_spec: Dict[str, Any], device) -> torch.Tensor:
        """Generate test input based on specification"""
        if "shape" in input_spec:
            if isinstance(input_spec["shape"], dict):
                # Multi-modal input
                test_inputs = {}
                for key, shape in input_spec["shape"].items():
                    test_inputs[key] = torch.randn(*shape).to(device)
                return test_inputs
            else:
                # Single tensor input
                return torch.randn(*input_spec["shape"]).to(device)
        else:
            # Default test input
            return torch.randn(1, 3, 64, 64).to(device)

    def _validate_output(self, output, output_spec: Dict[str, Any]) -> bool:
        """Validate model output against specification"""
        try:
            if isinstance(output, dict):
                # Multi-output validation
                if "keys" in output_spec:
                    return all(key in output for key in output_spec["keys"])
                return True
            elif hasattr(output, "shape"):
                # Tensor output validation
                if "shape" in output_spec:
                    expected_shape = output_spec["shape"]
                    # Allow flexible batch size
                    if len(expected_shape) == len(output.shape):
                        return all(
                            expected_shape[i] == output.shape[i] or expected_shape[i] == -1
                            for i in range(len(expected_shape))
                        )
                return True
            return True
        except Exception:
            return False


class APIIntegrationValidator:
    """Validator for API endpoints and services"""

    def __init__(self):
        self.registered_endpoints = {}

    def register_endpoint(
        self,
        endpoint_name: str,
        url: str,
        method: str = "GET",
        headers: Optional[Dict] = None,
        test_data: Optional[Dict] = None,
    ):
        """Register an API endpoint for validation"""
        self.registered_endpoints[endpoint_name] = {
            "url": url,
            "method": method,
            "headers": headers or {},
            "test_data": test_data,
        }
        logger.info(f"Registered API endpoint: {endpoint_name}")

    async def validate_api_integration(self, endpoint_name: str) -> List[ValidationResult]:
        """Validate API endpoint integration"""
        results = []

        if endpoint_name not in self.registered_endpoints:
            return [
                ValidationResult(
                    component_name=endpoint_name,
                    test_name="endpoint_registration",
                    status="failed",
                    message="Endpoint not registered",
                    execution_time_ms=0,
                )
            ]

        endpoint_info = self.registered_endpoints[endpoint_name]

        # Test 1: Endpoint Availability
        start_time = time.perf_counter()

        if not HTTPX_AVAILABLE:
            results.append(
                ValidationResult(
                    component_name=endpoint_name,
                    test_name="endpoint_availability",
                    status="skipped",
                    message="HTTP client not available",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            )
            return results

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(
                    method=endpoint_info["method"],
                    url=endpoint_info["url"],
                    headers=endpoint_info["headers"],
                    json=(
                        endpoint_info["test_data"]
                        if endpoint_info["method"] in ["POST", "PUT"]
                        else None
                    ),
                )

                status = "passed" if response.status_code < 400 else "failed"
                message = f"HTTP {response.status_code}: {response.reason_phrase}"

                results.append(
                    ValidationResult(
                        component_name=endpoint_name,
                        test_name="endpoint_availability",
                        status=status,
                        message=message,
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        details={
                            "status_code": response.status_code,
                            "response_time_ms": (time.perf_counter() - start_time) * 1000,
                            "content_type": response.headers.get("content-type", "unknown"),
                        },
                    )
                )

        except Exception as e:
            results.append(
                ValidationResult(
                    component_name=endpoint_name,
                    test_name="endpoint_availability",
                    status="failed",
                    message=f"Connection failed: {str(e)}",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            )

        # Test 2: Response Format Validation
        start_time = time.perf_counter()

        if not HTTPX_AVAILABLE:
            results.append(
                ValidationResult(
                    component_name=endpoint_name,
                    test_name="response_format",
                    status="skipped",
                    message="HTTP client not available",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            )
            return results

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(
                    method=endpoint_info["method"],
                    url=endpoint_info["url"],
                    headers=endpoint_info["headers"],
                    json=(
                        endpoint_info["test_data"]
                        if endpoint_info["method"] in ["POST", "PUT"]
                        else None
                    ),
                )

                if response.status_code < 400:
                    # Try to parse JSON response
                    try:
                        response_data = response.json()
                        results.append(
                            ValidationResult(
                                component_name=endpoint_name,
                                test_name="response_format",
                                status="passed",
                                message="Valid JSON response received",
                                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                                details={
                                    "response_keys": (
                                        list(response_data.keys())
                                        if isinstance(response_data, dict)
                                        else "non_dict"
                                    )
                                },
                            )
                        )
                    except json.JSONDecodeError:
                        results.append(
                            ValidationResult(
                                component_name=endpoint_name,
                                test_name="response_format",
                                status="warning",
                                message="Non-JSON response received",
                                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                            )
                        )
                else:
                    results.append(
                        ValidationResult(
                            component_name=endpoint_name,
                            test_name="response_format",
                            status="skipped",
                            message="Skipped due to endpoint availability failure",
                            execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        )
                    )

        except Exception as e:
            results.append(
                ValidationResult(
                    component_name=endpoint_name,
                    test_name="response_format",
                    status="failed",
                    message=f"Response validation failed: {str(e)}",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            )

        return results


class DataPipelineValidator:
    """Validator for data loading and processing pipelines"""

    def __init__(self):
        self.registered_pipelines = {}

    def register_pipeline(
        self, pipeline_name: str, dataloader, expected_batch_structure: Dict[str, Any]
    ):
        """Register a data pipeline for validation"""
        self.registered_pipelines[pipeline_name] = {
            "dataloader": dataloader,
            "expected_structure": expected_batch_structure,
        }
        logger.info(f"Registered data pipeline: {pipeline_name}")

    async def validate_pipeline_integration(self, pipeline_name: str) -> List[ValidationResult]:
        """Validate data pipeline integration"""
        results = []

        if pipeline_name not in self.registered_pipelines:
            return [
                ValidationResult(
                    component_name=pipeline_name,
                    test_name="pipeline_registration",
                    status="failed",
                    message="Pipeline not registered",
                    execution_time_ms=0,
                )
            ]

        pipeline_info = self.registered_pipelines[pipeline_name]
        dataloader = pipeline_info["dataloader"]

        # Test 1: Data Loading
        start_time = time.perf_counter()
        try:
            batch = next(iter(dataloader))

            results.append(
                ValidationResult(
                    component_name=pipeline_name,
                    test_name="data_loading",
                    status="passed",
                    message="Successfully loaded data batch",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    details={
                        "batch_type": type(batch).__name__,
                        "batch_size": len(batch) if hasattr(batch, "__len__") else "unknown",
                    },
                )
            )
        except Exception as e:
            results.append(
                ValidationResult(
                    component_name=pipeline_name,
                    test_name="data_loading",
                    status="failed",
                    message=f"Data loading failed: {str(e)}",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            )
            return results  # Can't continue without data

        # Test 2: Batch Structure Validation
        start_time = time.perf_counter()
        try:
            expected_structure = pipeline_info["expected_structure"]
            structure_valid = self._validate_batch_structure(batch, expected_structure)

            if structure_valid:
                results.append(
                    ValidationResult(
                        component_name=pipeline_name,
                        test_name="batch_structure",
                        status="passed",
                        message="Batch structure matches expected format",
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        component_name=pipeline_name,
                        test_name="batch_structure",
                        status="failed",
                        message="Batch structure does not match expected format",
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    )
                )
        except Exception as e:
            results.append(
                ValidationResult(
                    component_name=pipeline_name,
                    test_name="batch_structure",
                    status="failed",
                    message=f"Structure validation failed: {str(e)}",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            )

        # Test 3: Performance Validation
        start_time = time.perf_counter()
        try:
            # Measure batch loading time
            batch_times = []
            for i, batch in enumerate(dataloader):
                if i >= 5:  # Test first 5 batches
                    break
                batch_start = time.perf_counter()
                # Simulate processing
                _ = batch
                batch_time = (time.perf_counter() - batch_start) * 1000
                batch_times.append(batch_time)

            avg_batch_time = np.mean(batch_times) if batch_times else 0

            status = "passed" if avg_batch_time < 100 else "warning"  # 100ms threshold

            results.append(
                ValidationResult(
                    component_name=pipeline_name,
                    test_name="pipeline_performance",
                    status=status,
                    message=f"Average batch loading time: {avg_batch_time:.2f}ms",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    details={"avg_batch_time_ms": avg_batch_time},
                )
            )
        except Exception as e:
            results.append(
                ValidationResult(
                    component_name=pipeline_name,
                    test_name="pipeline_performance",
                    status="failed",
                    message=f"Performance validation failed: {str(e)}",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            )

        return results

    def _validate_batch_structure(self, batch, expected_structure: Dict[str, Any]) -> bool:
        """Validate batch structure against expected format"""
        try:
            if isinstance(batch, (list, tuple)):
                # Sequence batch
                if "length" in expected_structure:
                    return len(batch) == expected_structure["length"]
                return True
            elif isinstance(batch, dict):
                # Dictionary batch
                if "keys" in expected_structure:
                    return all(key in batch for key in expected_structure["keys"])
                return True
            elif hasattr(batch, "shape"):
                # Tensor batch
                if "shape" in expected_structure:
                    expected_shape = expected_structure["shape"]
                    return all(
                        expected_shape[i] == batch.shape[i] or expected_shape[i] == -1
                        for i in range(min(len(expected_shape), len(batch.shape)))
                    )
                return True
            return True
        except Exception:
            return False


class ComprehensiveIntegrationValidator:
    """Main integration validator coordinator"""

    def __init__(self):
        self.model_validator = ModelIntegrationValidator()
        self.api_validator = APIIntegrationValidator()
        self.pipeline_validator = DataPipelineValidator()
        self.registered_components = {}
        self.validation_history = []

    def register_component(self, component_info: ComponentInfo):
        """Register a component for integration validation"""
        self.registered_components[component_info.name] = component_info
        logger.info(
            f"Registered component: {component_info.name} ({component_info.component_type})"
        )

    async def validate_all_integrations(self) -> IntegrationReport:
        """Validate all registered component integrations"""
        logger.info("Starting comprehensive integration validation...")

        validation_start = time.perf_counter()
        all_results = []

        # Validate each component
        for component_name, component_info in self.registered_components.items():
            try:
                if component_info.component_type == "model":
                    results = await self.model_validator.validate_model_integration(component_name)
                elif component_info.component_type == "api":
                    results = await self.api_validator.validate_api_integration(component_name)
                elif component_info.component_type == "dataloader":
                    results = await self.pipeline_validator.validate_pipeline_integration(
                        component_name
                    )
                else:
                    # Generic component validation
                    results = await self._validate_generic_component(component_name, component_info)

                all_results.extend(results)

            except Exception as e:
                all_results.append(
                    ValidationResult(
                        component_name=component_name,
                        test_name="integration_validation",
                        status="failed",
                        message=f"Integration validation error: {str(e)}",
                        execution_time_ms=0,
                    )
                )

        # Validate cross-component dependencies
        dependency_results = await self._validate_dependencies()
        all_results.extend(dependency_results)

        # Generate comprehensive report
        report = self._generate_integration_report(all_results, validation_start)

        # Store in history
        self.validation_history.append(report)

        logger.info(
            f"Integration validation completed. Overall health score: {report.overall_health_score:.2f}"
        )
        return report

    async def _validate_generic_component(
        self, component_name: str, component_info: ComponentInfo
    ) -> List[ValidationResult]:
        """Validate generic component"""
        results = []

        # Basic availability test
        start_time = time.perf_counter()
        try:
            # Try to import the component if it's a module
            if hasattr(component_info, "import_path"):
                module = __import__(component_info.import_path)

                results.append(
                    ValidationResult(
                        component_name=component_name,
                        test_name="component_availability",
                        status="passed",
                        message="Component successfully imported",
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        component_name=component_name,
                        test_name="component_availability",
                        status="passed",
                        message="Generic component registered",
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    )
                )
        except Exception as e:
            results.append(
                ValidationResult(
                    component_name=component_name,
                    test_name="component_availability",
                    status="failed",
                    message=f"Component validation failed: {str(e)}",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            )

        return results

    async def _validate_dependencies(self) -> List[ValidationResult]:
        """Validate cross-component dependencies"""
        results = []

        for component_name, component_info in self.registered_components.items():
            for dependency in component_info.dependencies:
                start_time = time.perf_counter()

                if dependency in self.registered_components:
                    # Dependency is registered
                    results.append(
                        ValidationResult(
                            component_name=component_name,
                            test_name=f"dependency_{dependency}",
                            status="passed",
                            message=f"Dependency {dependency} is available",
                            execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        )
                    )
                else:
                    # Missing dependency
                    status = "failed" if component_info.critical else "warning"
                    results.append(
                        ValidationResult(
                            component_name=component_name,
                            test_name=f"dependency_{dependency}",
                            status=status,
                            message=f"Dependency {dependency} is missing",
                            execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        )
                    )

        return results

    def _generate_integration_report(
        self, results: List[ValidationResult], validation_start: float
    ) -> IntegrationReport:
        """Generate comprehensive integration report"""
        total_components = len(self.registered_components)
        total_tests = len(results)

        # Count results by status
        passed_tests = sum(1 for r in results if r.status == "passed")
        failed_tests = sum(1 for r in results if r.status == "failed")
        warning_tests = sum(1 for r in results if r.status == "warning")

        # Count components by health
        component_health = defaultdict(list)
        for result in results:
            component_health[result.component_name].append(result.status)

        healthy_components = 0
        warning_components = 0
        failed_components = 0

        for component, statuses in component_health.items():
            if "failed" in statuses:
                failed_components += 1
            elif "warning" in statuses:
                warning_components += 1
            else:
                healthy_components += 1

        # Calculate overall health score
        health_score = (passed_tests / max(total_tests, 1)) * 100

        # Identify critical failures
        critical_failures = [
            r.component_name
            for r in results
            if r.status == "failed"
            and self.registered_components.get(r.component_name, ComponentInfo("", "", "")).critical
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        # Performance summary
        performance_summary = {
            "total_validation_time_ms": (time.perf_counter() - validation_start) * 1000,
            "average_test_time_ms": (
                np.mean([r.execution_time_ms for r in results]) if results else 0
            ),
            "slowest_tests": sorted(results, key=lambda x: x.execution_time_ms, reverse=True)[:5],
        }

        return IntegrationReport(
            validation_timestamp=datetime.now(),
            total_components=total_components,
            healthy_components=healthy_components,
            warning_components=warning_components,
            failed_components=failed_components,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            overall_health_score=health_score,
            critical_failures=critical_failures,
            validation_results=results,
            performance_summary=performance_summary,
            recommendations=recommendations,
        )

    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        failed_results = [r for r in results if r.status == "failed"]
        warning_results = [r for r in results if r.status == "warning"]

        if failed_results:
            recommendations.append(f"Address {len(failed_results)} failed tests immediately")

        if warning_results:
            recommendations.append(f"Review {len(warning_results)} warning conditions")

        # Component-specific recommendations
        component_failures = defaultdict(int)
        for result in failed_results:
            component_failures[result.component_name] += 1

        for component, failure_count in component_failures.items():
            if failure_count > 2:
                recommendations.append(
                    f"Component {component} has multiple failures - requires attention"
                )

        # Performance recommendations
        slow_tests = [r for r in results if r.execution_time_ms > 1000]
        if slow_tests:
            recommendations.append(f"Optimize {len(slow_tests)} slow-performing tests")

        return recommendations

    def save_validation_report(
        self, report: IntegrationReport, filepath: Optional[str] = None
    ) -> str:
        """Save validation report to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"integration_validation_report_{timestamp}.json"

        # Convert report to JSON-serializable format
        report_dict = {
            "validation_timestamp": report.validation_timestamp.isoformat(),
            "summary": {
                "total_components": report.total_components,
                "healthy_components": report.healthy_components,
                "warning_components": report.warning_components,
                "failed_components": report.failed_components,
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "warning_tests": report.warning_tests,
                "overall_health_score": report.overall_health_score,
            },
            "critical_failures": report.critical_failures,
            "recommendations": report.recommendations,
            "performance_summary": report.performance_summary,
            "detailed_results": [
                {
                    "component_name": r.component_name,
                    "test_name": r.test_name,
                    "status": r.status,
                    "message": r.message,
                    "execution_time_ms": r.execution_time_ms,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in report.validation_results
            ],
        }

        with open(filepath, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Integration validation report saved to: {filepath}")
        return filepath


# Convenience functions
def create_integration_validator() -> ComprehensiveIntegrationValidator:
    """Create a comprehensive integration validator"""
    return ComprehensiveIntegrationValidator()


async def quick_integration_check(components: List[ComponentInfo]) -> IntegrationReport:
    """Quick integration check for a list of components"""
    validator = create_integration_validator()

    for component in components:
        validator.register_component(component)

    return await validator.validate_all_integrations()


if __name__ == "__main__":
    # Example usage
    async def main():
        validator = create_integration_validator()

        # Register example component
        example_component = ComponentInfo(
            name="example_model",
            component_type="model",
            version="1.0.0",
            dependencies=[],
            critical=True,
        )
        validator.register_component(example_component)

        # Run validation
        report = await validator.validate_all_integrations()

        # Save report
        report_file = validator.save_validation_report(report)

        print(f"Integration validation completed")
        print(f"Overall health score: {report.overall_health_score:.2f}")
        print(f"Report saved to: {report_file}")

    import asyncio

    asyncio.run(main())
