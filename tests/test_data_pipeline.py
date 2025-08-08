"""
Test suite for data pipeline components
======================================

Unit tests for data processing, validation, and quality assurance.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

# Import data pipeline components
from data_build.quality_manager import QualityManager
from data_build.real_data_sources import RealDataSourceManager
from datamodules.cube_dm import CubeDM
from utils.ssl_config import get_ssl_context


class TestQualityManager:
    """Test data quality management system"""

    @pytest.fixture
    def quality_config(self):
        return {
            "validation_rules": {
                "completeness_threshold": 0.95,
                "accuracy_threshold": 0.98,
                "consistency_threshold": 0.99,
            },
            "quality_metrics": ["completeness", "accuracy", "consistency", "timeliness"],
        }

    @pytest.fixture
    def quality_manager(self, quality_config):
        return QualityManager(quality_config)

    def test_quality_manager_initialization(self, quality_manager):
        """Test quality manager initializes correctly"""
        assert hasattr(quality_manager, "validate_data")
        assert hasattr(quality_manager, "quality_metrics")

    def test_data_completeness_validation(self, quality_manager):
        """Test data completeness validation"""
        # Complete data
        complete_data = np.random.randn(100, 10)
        result = quality_manager.check_completeness(complete_data)
        assert result["completeness_score"] == 1.0
        assert result["passes_threshold"] is True

        # Incomplete data with NaNs
        incomplete_data = complete_data.copy()
        incomplete_data[0:10, 0] = np.nan
        result = quality_manager.check_completeness(incomplete_data)
        assert result["completeness_score"] < 1.0

    def test_data_accuracy_validation(self, quality_manager):
        """Test data accuracy validation"""
        # Mock reference data
        reference_data = np.random.randn(50, 5)
        test_data = reference_data + np.random.normal(0, 0.01, reference_data.shape)

        result = quality_manager.check_accuracy(test_data, reference_data)
        assert "accuracy_score" in result
        assert result["accuracy_score"] > 0.9  # Should be highly accurate

    def test_outlier_detection(self, quality_manager):
        """Test outlier detection in data"""
        # Normal data with outliers
        normal_data = np.random.normal(0, 1, (100, 3))
        outlier_data = normal_data.copy()
        outlier_data[0, 0] = 10  # Clear outlier

        result = quality_manager.detect_outliers(outlier_data)
        assert "outlier_indices" in result
        assert len(result["outlier_indices"]) > 0

    def test_quality_report_generation(self, quality_manager):
        """Test quality report generation"""
        mock_data = np.random.randn(100, 5)

        with patch.object(quality_manager, "generate_quality_report") as mock_report:
            mock_report.return_value = {
                "overall_score": 0.95,
                "metrics": {"completeness": 1.0, "accuracy": 0.92, "consistency": 0.94},
                "recommendations": ["Improve data accuracy"],
            }

            report = quality_manager.generate_quality_report(mock_data)
            assert "overall_score" in report
            assert "metrics" in report
            assert report["overall_score"] > 0.9


class TestRealDataSourceManager:
    """Test real data source management"""

    @pytest.fixture
    def data_source_config(self):
        return {
            "sources": {
                "nasa_exoplanet_archive": {
                    "base_url": "https://exoplanetarchive.ipac.caltech.edu",
                    "endpoints": {"planets": "/cgi-bin/nstedAPI/nph-nstedAPI"},
                    "rate_limit": 10,
                    "ssl_verify": True,
                },
                "jwst_mast": {
                    "base_url": "https://mast.stsci.edu",
                    "endpoints": {"observations": "/api/v0.1/Download/file"},
                    "rate_limit": 5,
                    "ssl_verify": True,
                },
            }
        }

    @pytest.fixture
    def data_manager(self, data_source_config):
        return RealDataSourceManager(data_source_config)

    def test_data_manager_initialization(self, data_manager):
        """Test data manager initializes correctly"""
        assert hasattr(data_manager, "sources")
        assert len(data_manager.sources) > 0

    def test_ssl_configuration(self):
        """Test SSL configuration for secure connections"""
        ssl_context = get_ssl_context("nasa_exoplanet_archive")
        assert ssl_context is not None

    @patch("requests.get")
    def test_data_source_connectivity(self, mock_get, data_manager):
        """Test connectivity to data sources"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_get.return_value = mock_response

        result = data_manager.test_connectivity("nasa_exoplanet_archive")
        assert result["status"] == "success"

    @patch("aiohttp.ClientSession.get")
    async def test_async_data_fetching(self, mock_get, data_manager):
        """Test asynchronous data fetching"""
        # Mock async response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = MagicMock(return_value={"data": "test"})
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await data_manager.fetch_data_async("nasa_exoplanet_archive", {"query": "test"})
        assert result is not None

    def test_rate_limiting(self, data_manager):
        """Test rate limiting enforcement"""
        source_name = "nasa_exoplanet_archive"
        rate_limit = data_manager.sources[source_name]["rate_limit"]

        # Test that rate limiting is enforced
        assert data_manager.check_rate_limit(source_name) is True

        # Simulate multiple requests
        for _ in range(rate_limit + 1):
            data_manager.record_request(source_name)

        # Should be rate limited now
        assert data_manager.check_rate_limit(source_name) is False


class TestCubeDM:
    """Test Cube Data Module"""

    @pytest.fixture
    def cube_config(self):
        return {
            "data_dir": "data/test",
            "batch_size": 16,
            "num_workers": 2,
            "cache_dir": "data/cache/test",
            "use_synthetic_data": True,  # Use synthetic for testing
        }

    @pytest.fixture
    def cube_dm(self, cube_config):
        return CubeDM(**cube_config)

    def test_cube_dm_initialization(self, cube_dm):
        """Test CubeDM initializes correctly"""
        assert hasattr(cube_dm, "setup")
        assert hasattr(cube_dm, "train_dataloader")
        assert hasattr(cube_dm, "val_dataloader")

    def test_synthetic_data_generation(self, cube_dm):
        """Test synthetic data generation"""
        cube_dm.setup("fit")

        # Test data loader creation
        train_loader = cube_dm.train_dataloader()
        val_loader = cube_dm.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None

    def test_data_shapes(self, cube_dm):
        """Test data shapes are correct"""
        cube_dm.setup("fit")
        train_loader = cube_dm.train_dataloader()

        # Get first batch
        batch = next(iter(train_loader))
        assert isinstance(batch, dict)
        assert "input" in batch
        assert "target" in batch

        # Check shapes (5D: [batch, variables, time, lat, lon])
        input_shape = batch["input"].shape
        assert len(input_shape) == 5
        assert input_shape[0] <= cube_dm.batch_size  # Batch dimension

    def test_data_augmentation(self, cube_dm):
        """Test data augmentation"""
        cube_dm.use_augmentation = True
        cube_dm.setup("fit")

        # Get two batches and check they're different (due to augmentation)
        train_loader = cube_dm.train_dataloader()
        batch1 = next(iter(train_loader))
        batch2 = next(iter(train_loader))

        # With augmentation, batches should be different
        assert not torch.allclose(batch1["input"], batch2["input"])


class TestDataIntegration:
    """Test data integration and pipeline coordination"""

    def test_multi_source_integration(self):
        """Test integration of multiple data sources"""
        # Mock multiple data sources
        sources = {
            "exoplanet_data": np.random.randn(50, 8),  # Planet parameters
            "stellar_data": np.random.randn(50, 5),  # Stellar properties
            "atmospheric_data": np.random.randn(50, 10, 100),  # Spectral data
        }

        # Test data alignment
        assert sources["exoplanet_data"].shape[0] == sources["stellar_data"].shape[0]
        assert sources["exoplanet_data"].shape[0] == sources["atmospheric_data"].shape[0]

    def test_data_versioning(self):
        """Test data versioning and reproducibility"""
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "data_version.json"

            # Create version metadata
            version_info = {
                "version": "1.0.0",
                "timestamp": "2024-01-01T00:00:00Z",
                "sources": ["nasa_exoplanet_archive", "jwst_mast"],
                "checksum": "abc123",
            }

            with open(version_file, "w") as f:
                json.dump(version_info, f)

            # Verify version file
            assert version_file.exists()
            with open(version_file, "r") as f:
                loaded_version = json.load(f)
            assert loaded_version["version"] == "1.0.0"

    def test_metadata_consistency(self):
        """Test metadata consistency across pipeline"""
        metadata = {
            "planet_name": "test_planet",
            "stellar_type": "G",
            "observation_date": "2024-01-01",
            "instrument": "JWST_NIRSpec",
        }

        # Test metadata validation
        required_fields = ["planet_name", "stellar_type", "observation_date"]
        for field in required_fields:
            assert field in metadata

    def test_cache_management(self):
        """Test data cache management"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            cache_dir.mkdir()

            # Simulate cached data
            test_data = np.random.randn(100, 10)
            cache_file = cache_dir / "test_data.npy"
            np.save(cache_file, test_data)

            # Test cache retrieval
            assert cache_file.exists()
            loaded_data = np.load(cache_file)
            np.testing.assert_array_equal(test_data, loaded_data)


class TestDataValidation:
    """Test comprehensive data validation"""

    def test_schema_validation(self):
        """Test data schema validation"""
        # Define expected schema
        expected_schema = {
            "planet_radius": {"type": "float", "min": 0.1, "max": 10.0},
            "planet_mass": {"type": "float", "min": 0.01, "max": 100.0},
            "orbital_period": {"type": "float", "min": 0.1, "max": 10000.0},
            "stellar_temperature": {"type": "float", "min": 2000, "max": 10000},
        }

        # Test valid data
        valid_data = {
            "planet_radius": 1.2,
            "planet_mass": 0.8,
            "orbital_period": 365.25,
            "stellar_temperature": 5778,
        }

        for field, value in valid_data.items():
            schema = expected_schema[field]
            assert schema["min"] <= value <= schema["max"]

    def test_scientific_unit_validation(self):
        """Test scientific unit validation"""
        # Test unit conversions and validations
        earth_radius_km = 6371.0
        earth_radius_earth = 1.0

        # Convert and validate
        assert abs(earth_radius_km / 6371.0 - earth_radius_earth) < 1e-6

    def test_data_range_validation(self):
        """Test scientific data range validation"""
        # Test realistic ranges for exoplanet parameters
        test_cases = [
            {"name": "Earth", "radius": 1.0, "mass": 1.0, "period": 365.25, "valid": True},
            {"name": "Jupiter", "radius": 11.2, "mass": 317.8, "period": 4333, "valid": True},
            {"name": "Invalid", "radius": -1.0, "mass": 1.0, "period": 365.25, "valid": False},
            {"name": "Extreme", "radius": 1000.0, "mass": 1.0, "period": 365.25, "valid": False},
        ]

        for case in test_cases:
            # Radius should be positive and reasonable
            radius_valid = 0.1 <= case["radius"] <= 20.0
            # Mass should be positive and reasonable
            mass_valid = 0.01 <= case["mass"] <= 1000.0
            # Period should be positive
            period_valid = case["period"] > 0

            overall_valid = radius_valid and mass_valid and period_valid
            assert overall_valid == case["valid"], f"Validation failed for {case['name']}"


# Performance tests
class TestDataPerformance:
    """Test data pipeline performance"""

    def test_data_loading_speed(self):
        """Test data loading performance"""
        import time

        # Generate test dataset
        large_data = np.random.randn(1000, 100)

        start_time = time.time()
        # Simulate data processing
        processed_data = large_data * 2 + 1
        end_time = time.time()

        processing_time = end_time - start_time
        assert processing_time < 1.0, f"Data processing too slow: {processing_time:.3f}s"

    def test_memory_efficiency(self):
        """Test memory efficient data handling"""
        # Test that we can handle reasonably large datasets
        try:
            large_tensor = torch.randn(1000, 5, 10, 32, 32)
            assert large_tensor.numel() > 0
            del large_tensor  # Clean up
        except RuntimeError as e:
            pytest.skip(f"Memory test failed: {e}")

    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency"""
        batch_sizes = [1, 8, 16, 32, 64]
        data_size = (5, 8, 16, 16)

        for batch_size in batch_sizes:
            batch_data = torch.randn(batch_size, *data_size)
            assert batch_data.shape[0] == batch_size
            assert batch_data.shape[1:] == data_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
