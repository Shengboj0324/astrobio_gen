"""
Pytest configuration and shared fixtures
========================================

Common test fixtures and configuration for the astrobio-gen test suite.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session")
def device():
    """Get appropriate device for testing"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_tensor(device):
    """Create small tensor for testing"""
    return torch.randn(2, 3, 4, 4, device=device)


@pytest.fixture
def synthetic_planet_data():
    """Generate synthetic planet data for testing"""
    np.random.seed(42)  # Reproducible
    n_planets = 100

    data = {
        "planet_radius": np.random.uniform(0.5, 2.0, n_planets),  # Earth radii
        "planet_mass": np.random.uniform(0.1, 10.0, n_planets),  # Earth masses
        "orbital_period": np.random.uniform(1, 1000, n_planets),  # Days
        "stellar_temperature": np.random.uniform(3000, 7000, n_planets),  # Kelvin
        "stellar_radius": np.random.uniform(0.5, 2.0, n_planets),  # Solar radii
        "stellar_mass": np.random.uniform(0.5, 2.0, n_planets),  # Solar masses
        "distance": np.random.uniform(10, 1000, n_planets),  # Parsecs
        "metallicity": np.random.uniform(-0.5, 0.5, n_planets),  # [Fe/H]
    }

    # Add some realistic correlations
    # Larger planets tend to be more massive (with scatter)
    data["planet_mass"] = data["planet_radius"] ** 2.5 * np.random.lognormal(0, 0.3, n_planets)

    # Stellar properties correlation
    data["stellar_radius"] = data["stellar_mass"] ** 0.8 * np.random.lognormal(0, 0.1, n_planets)

    return data


@pytest.fixture
def synthetic_spectral_data():
    """Generate synthetic spectral data for testing"""
    np.random.seed(42)

    wavelengths = np.linspace(1.0, 30.0, 1000)  # Microns (JWST range)
    n_spectra = 50

    # Generate realistic-looking spectra with absorption lines
    spectra = np.zeros((n_spectra, len(wavelengths)))

    for i in range(n_spectra):
        # Base continuum
        continuum = np.exp(-wavelengths / 10) * np.random.uniform(0.8, 1.2)

        # Add absorption lines
        for line_center in [1.4, 1.9, 2.7, 4.3, 6.3, 15.0]:  # Typical molecular lines
            if np.min(wavelengths) <= line_center <= np.max(wavelengths):
                line_strength = np.random.uniform(0.1, 0.8)
                line_width = np.random.uniform(0.05, 0.2)
                absorption = line_strength * np.exp(-0.5 * ((wavelengths - line_center) / line_width) ** 2)
                continuum *= (1 - absorption)

        # Add noise
        noise = np.random.normal(0, 0.02, len(wavelengths))
        spectra[i] = continuum + noise

    return {"wavelengths": wavelengths, "spectra": spectra}


@pytest.fixture
def synthetic_datacube():
    """Generate synthetic 5D datacube for testing"""
    # Dimensions: [variables, time, depth, lat, lon]
    shape = (5, 8, 4, 16, 16)
    datacube = torch.randn(*shape)

    # Add some physical structure
    # Temperature decreases with altitude
    datacube[0, :, :, :, :] = 300 - datacube[2, :, :, :, :] * 50  # Temperature vs altitude

    # Pressure decreases exponentially with altitude
    altitude = torch.arange(shape[2]).float().view(1, 1, -1, 1, 1)
    datacube[1, :, :, :, :] = torch.exp(-altitude / 2) * 1000  # Pressure profile

    return datacube


@pytest.fixture
def mock_nasa_api_response():
    """Mock NASA API response data"""
    return {
        "status": "OK",
        "data": [
            {
                "pl_name": "Test Planet b",
                "pl_rade": 1.2,
                "pl_masse": 0.8,
                "pl_orbper": 365.25,
                "st_teff": 5778,
                "st_rad": 1.0,
                "st_mass": 1.0,
                "sy_dist": 100.0,
            }
        ],
    }


@pytest.fixture
def mock_jwst_observation():
    """Mock JWST observation data"""
    return {
        "target_name": "Test Exoplanet",
        "instrument": "NIRSpec",
        "filter": "CLEAR",
        "grating": "G395H",
        "observation_id": "test_obs_001",
        "wavelength_range": [2.9, 5.3],  # Microns
        "spectral_resolution": 2700,
        "integration_time": 3600,  # Seconds
        "date_obs": "2024-01-01T00:00:00",
    }


@pytest.fixture
def quality_thresholds():
    """Standard quality thresholds for testing"""
    return {
        "completeness": 0.95,
        "accuracy": 0.98,
        "consistency": 0.99,
        "timeliness": 0.90,
        "signal_to_noise": 10.0,
        "calibration_precision": 0.01,
    }


@pytest.fixture
def physics_constants():
    """Physical constants for testing"""
    return {
        "earth_radius_km": 6371.0,
        "earth_mass_kg": 5.972e24,
        "solar_radius_km": 695700.0,
        "solar_mass_kg": 1.989e30,
        "au_km": 1.496e8,
        "parsec_km": 3.086e13,
        "stefan_boltzmann": 5.67e-8,  # W m^-2 K^-4
        "gravitational_constant": 6.674e-11,  # m^3 kg^-1 s^-2
    }


# Configure pytest
def pytest_configure(config):
    """Configure pytest settings"""
    # Set environment variables for testing
    os.environ["ASTROBIO_TEST_MODE"] = "true"
    os.environ["ASTROBIO_USE_SYNTHETIC_DATA"] = "true"

    # Configure torch for testing
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pytest_unconfigure(config):
    """Clean up after tests"""
    # Clean up environment variables
    if "ASTROBIO_TEST_MODE" in os.environ:
        del os.environ["ASTROBIO_TEST_MODE"]
    if "ASTROBIO_USE_SYNTHETIC_DATA" in os.environ:
        del os.environ["ASTROBIO_USE_SYNTHETIC_DATA"]


# Custom markers
pytest_plugins = []

# Skip slow tests by default
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers"""
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runintegration", action="store_true", default=False, help="run integration tests"
    )


# Test data fixtures with error handling
@pytest.fixture
def robust_test_data():
    """Generate test data with error handling"""
    try:
        data = torch.randn(10, 5, 8, 16, 16)
        return data
    except Exception as e:
        pytest.skip(f"Could not generate test data: {e}")


@pytest.fixture
def mock_model():
    """Create mock model for testing"""
    model = MagicMock()
    model.forward.return_value = torch.randn(1, 5, 8, 16, 16)
    model.parameters.return_value = [torch.randn(10, 10)]
    model.__class__.__name__ = "MockModel"
    return model


# Database fixtures for integration tests
@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing"""
    connection = MagicMock()
    connection.execute.return_value = {"status": "success", "rows_affected": 1}
    connection.fetch.return_value = [{"id": 1, "name": "test"}]
    return connection


@pytest.fixture
def temporary_config_file(test_data_dir):
    """Create temporary configuration file"""
    config = {
        "model": {"name": "test_model", "parameters": {"learning_rate": 0.001}},
        "data": {"batch_size": 16, "num_workers": 2},
        "training": {"max_epochs": 10, "validation_frequency": 1},
    }

    config_file = test_data_dir / "test_config.yaml"
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return config_file
