#!/usr/bin/env python3
"""
Physics Invariants Testing with Property-Based Testing
======================================================

Comprehensive property-based testing for physics invariants using Hypothesis.
Essential for ISEF competition to demonstrate scientific rigor and physical
consistency of all models and algorithms.

Tests include:
- Energy conservation laws
- Mass conservation principles
- Momentum conservation
- Thermodynamic consistency
- Radiative equilibrium
- Hydrostatic balance
- Atmospheric physics constraints
- Boundary condition preservation

Author: Astrobio Research Team
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from scipy import constants

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore', category=UserWarning)

# Physics constants
STEFAN_BOLTZMANN = constants.Stefan_Boltzmann  # W m^-2 K^-4
PLANCK_CONSTANT = constants.Planck
SPEED_OF_LIGHT = constants.c
BOLTZMANN_CONSTANT = constants.Boltzmann
GAS_CONSTANT = constants.R
AVOGADRO_NUMBER = constants.Avogadro

# Test tolerances
ENERGY_CONSERVATION_TOLERANCE = 1e-6
MASS_CONSERVATION_TOLERANCE = 1e-8
MOMENTUM_CONSERVATION_TOLERANCE = 1e-6
THERMODYNAMIC_TOLERANCE = 1e-5

class PhysicsInvariantTester:
    """Main class for physics invariant testing"""
    
    def __init__(self):
        self.test_results = {}
        
    def test_energy_conservation(
        self, 
        initial_state: np.ndarray, 
        final_state: np.ndarray,
        external_work: float = 0.0
    ) -> Dict[str, Any]:
        """Test energy conservation law"""
        
        # Calculate kinetic energy (assuming velocity components in state)
        if initial_state.shape[-1] >= 6:  # Position + velocity
            initial_kinetic = 0.5 * np.sum(initial_state[..., 3:6]**2, axis=-1)
            final_kinetic = 0.5 * np.sum(final_state[..., 3:6]**2, axis=-1)
        else:
            # Use L2 norm as energy proxy
            initial_kinetic = 0.5 * np.sum(initial_state**2, axis=-1)
            final_kinetic = 0.5 * np.sum(final_state**2, axis=-1)
        
        # Potential energy (simplified gravitational)
        if initial_state.shape[-1] >= 3:
            initial_potential = np.sum(initial_state[..., :3], axis=-1)  # Height proxy
            final_potential = np.sum(final_state[..., :3], axis=-1)
        else:
            initial_potential = np.zeros_like(initial_kinetic)
            final_potential = np.zeros_like(final_kinetic)
        
        # Total energy
        initial_total = initial_kinetic + initial_potential
        final_total = final_kinetic + final_potential
        
        # Energy change
        energy_change = final_total - initial_total
        
        # Check conservation (allowing for external work)
        energy_violation = np.abs(energy_change - external_work)
        max_violation = np.max(energy_violation)
        mean_violation = np.mean(energy_violation)
        
        return {
            'energy_conserved': max_violation < ENERGY_CONSERVATION_TOLERANCE,
            'max_energy_violation': float(max_violation),
            'mean_energy_violation': float(mean_violation),
            'initial_total_energy': float(np.mean(initial_total)),
            'final_total_energy': float(np.mean(final_total)),
            'energy_change': float(np.mean(energy_change)),
            'external_work': external_work
        }
    
    def test_mass_conservation(
        self, 
        initial_mass: np.ndarray, 
        final_mass: np.ndarray,
        mass_sources: float = 0.0,
        mass_sinks: float = 0.0
    ) -> Dict[str, Any]:
        """Test mass conservation principle"""
        
        # Total mass change
        initial_total = np.sum(initial_mass)
        final_total = np.sum(final_mass)
        mass_change = final_total - initial_total
        
        # Expected change from sources and sinks
        expected_change = mass_sources - mass_sinks
        
        # Conservation violation
        mass_violation = abs(mass_change - expected_change)
        relative_violation = mass_violation / max(abs(initial_total), 1e-10)
        
        return {
            'mass_conserved': mass_violation < MASS_CONSERVATION_TOLERANCE,
            'mass_violation': float(mass_violation),
            'relative_mass_violation': float(relative_violation),
            'initial_total_mass': float(initial_total),
            'final_total_mass': float(final_total),
            'mass_change': float(mass_change),
            'expected_change': expected_change
        }
    
    def test_momentum_conservation(
        self,
        initial_momentum: np.ndarray,
        final_momentum: np.ndarray,
        external_forces: Optional[np.ndarray] = None,
        dt: float = 1.0
    ) -> Dict[str, Any]:
        """Test momentum conservation law"""
        
        # Momentum change
        momentum_change = final_momentum - initial_momentum
        
        # Expected change from external forces
        if external_forces is not None:
            expected_change = external_forces * dt
        else:
            expected_change = np.zeros_like(momentum_change)
        
        # Conservation violation
        momentum_violation = np.linalg.norm(momentum_change - expected_change, axis=-1)
        max_violation = np.max(momentum_violation)
        mean_violation = np.mean(momentum_violation)
        
        return {
            'momentum_conserved': max_violation < MOMENTUM_CONSERVATION_TOLERANCE,
            'max_momentum_violation': float(max_violation),
            'mean_momentum_violation': float(mean_violation),
            'initial_momentum_magnitude': float(np.mean(np.linalg.norm(initial_momentum, axis=-1))),
            'final_momentum_magnitude': float(np.mean(np.linalg.norm(final_momentum, axis=-1))),
            'momentum_change_magnitude': float(np.mean(np.linalg.norm(momentum_change, axis=-1)))
        }
    
    def test_thermodynamic_consistency(
        self,
        temperature: np.ndarray,
        pressure: np.ndarray,
        density: np.ndarray,
        molecular_weight: float = 28.97  # Air molecular weight
    ) -> Dict[str, Any]:
        """Test ideal gas law and thermodynamic consistency"""
        
        # Ideal gas law: P = ρRT/M
        # Where R is specific gas constant
        R_specific = GAS_CONSTANT / (molecular_weight * 1e-3)  # J/(kg·K)
        
        # Expected pressure from ideal gas law
        expected_pressure = density * R_specific * temperature
        
        # Pressure violation
        pressure_violation = np.abs(pressure - expected_pressure) / np.maximum(pressure, 1e-10)
        max_violation = np.max(pressure_violation)
        mean_violation = np.mean(pressure_violation)
        
        # Temperature bounds check (physical limits)
        temp_min = 1.0  # Minimum physical temperature (K)
        temp_max = 10000.0  # Maximum reasonable atmospheric temperature (K)
        temp_bounds_ok = np.all((temperature >= temp_min) & (temperature <= temp_max))
        
        # Pressure bounds check
        pressure_min = 1e-6  # Minimum pressure (Pa)
        pressure_max = 1e8   # Maximum reasonable pressure (Pa)
        pressure_bounds_ok = np.all((pressure >= pressure_min) & (pressure <= pressure_max))
        
        # Density bounds check
        density_min = 1e-6  # Minimum density (kg/m³)
        density_max = 1e6   # Maximum reasonable density (kg/m³)
        density_bounds_ok = np.all((density >= density_min) & (density <= density_max))
        
        return {
            'thermodynamically_consistent': max_violation < THERMODYNAMIC_TOLERANCE,
            'max_pressure_violation': float(max_violation),
            'mean_pressure_violation': float(mean_violation),
            'temperature_bounds_ok': bool(temp_bounds_ok),
            'pressure_bounds_ok': bool(pressure_bounds_ok),
            'density_bounds_ok': bool(density_bounds_ok),
            'mean_temperature': float(np.mean(temperature)),
            'mean_pressure': float(np.mean(pressure)),
            'mean_density': float(np.mean(density))
        }
    
    def test_radiative_equilibrium(
        self,
        temperature: np.ndarray,
        incoming_radiation: np.ndarray,
        albedo: np.ndarray,
        emissivity: float = 1.0
    ) -> Dict[str, Any]:
        """Test radiative equilibrium (Stefan-Boltzmann law)"""
        
        # Absorbed radiation
        absorbed_radiation = incoming_radiation * (1 - albedo)
        
        # Outgoing radiation (Stefan-Boltzmann law)
        outgoing_radiation = emissivity * STEFAN_BOLTZMANN * temperature**4
        
        # Radiative balance
        radiative_imbalance = absorbed_radiation - outgoing_radiation
        relative_imbalance = np.abs(radiative_imbalance) / np.maximum(absorbed_radiation, 1e-10)
        
        max_imbalance = np.max(np.abs(radiative_imbalance))
        mean_imbalance = np.mean(np.abs(radiative_imbalance))
        max_relative_imbalance = np.max(relative_imbalance)
        
        # Equilibrium tolerance (5 W/m² is reasonable for climate models)
        equilibrium_tolerance = 5.0  # W/m²
        
        return {
            'radiative_equilibrium': max_imbalance < equilibrium_tolerance,
            'max_radiative_imbalance': float(max_imbalance),
            'mean_radiative_imbalance': float(mean_imbalance),
            'max_relative_imbalance': float(max_relative_imbalance),
            'mean_incoming_radiation': float(np.mean(incoming_radiation)),
            'mean_outgoing_radiation': float(np.mean(outgoing_radiation)),
            'mean_albedo': float(np.mean(albedo)),
            'equilibrium_tolerance': equilibrium_tolerance
        }
    
    def test_hydrostatic_balance(
        self,
        pressure: np.ndarray,
        density: np.ndarray,
        height: np.ndarray,
        gravity: float = 9.81
    ) -> Dict[str, Any]:
        """Test hydrostatic balance in atmospheric columns"""
        
        # Hydrostatic equation: dp/dz = -ρg
        if len(pressure.shape) > 1 and pressure.shape[-1] > 1:
            # Calculate pressure gradient
            dz = np.diff(height, axis=-1)
            dp = np.diff(pressure, axis=-1)
            
            # Average density in each layer
            rho_avg = 0.5 * (density[..., :-1] + density[..., 1:])
            
            # Expected pressure gradient
            expected_dp = -rho_avg * gravity * dz
            
            # Hydrostatic balance violation
            hydrostatic_violation = np.abs(dp - expected_dp)
            max_violation = np.max(hydrostatic_violation)
            mean_violation = np.mean(hydrostatic_violation)
            
            # Relative violation
            relative_violation = hydrostatic_violation / np.maximum(np.abs(expected_dp), 1e-10)
            max_relative_violation = np.max(relative_violation)
            
            hydrostatic_ok = max_relative_violation < 0.1  # 10% tolerance
        else:
            # Single level - cannot test hydrostatic balance
            max_violation = 0.0
            mean_violation = 0.0
            max_relative_violation = 0.0
            hydrostatic_ok = True
        
        return {
            'hydrostatic_balance': hydrostatic_ok,
            'max_hydrostatic_violation': float(max_violation),
            'mean_hydrostatic_violation': float(mean_violation),
            'max_relative_violation': float(max_relative_violation),
            'pressure_range': [float(np.min(pressure)), float(np.max(pressure))],
            'density_range': [float(np.min(density)), float(np.max(density))],
            'height_range': [float(np.min(height)), float(np.max(height))]
        }


# Property-based test strategies
@st.composite
def physical_state_arrays(draw):
    """Generate physically reasonable state arrays"""
    
    # Array dimensions
    batch_size = draw(st.integers(min_value=1, max_value=10))
    state_dim = draw(st.integers(min_value=3, max_value=12))
    
    # Generate physically reasonable values
    positions = draw(arrays(
        dtype=np.float32,
        shape=(batch_size, 3),
        elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    ))
    
    velocities = draw(arrays(
        dtype=np.float32,
        shape=(batch_size, 3),
        elements=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
    ))
    
    # Additional state variables if needed
    if state_dim > 6:
        extra_vars = draw(arrays(
            dtype=np.float32,
            shape=(batch_size, state_dim - 6),
            elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        ))
        state = np.concatenate([positions, velocities, extra_vars], axis=1)
    else:
        state = np.concatenate([positions, velocities], axis=1)[:, :state_dim]
    
    return state

@st.composite
def atmospheric_profiles(draw):
    """Generate physically reasonable atmospheric profiles"""
    
    # Number of levels
    n_levels = draw(st.integers(min_value=5, max_value=50))
    
    # Height levels (increasing)
    heights = np.linspace(0, 50000, n_levels)  # 0 to 50 km
    
    # Temperature profile (decreasing with height, then increasing in stratosphere)
    surface_temp = draw(st.floats(min_value=200.0, max_value=320.0))
    lapse_rate = draw(st.floats(min_value=0.003, max_value=0.01))  # K/m
    
    temperature = np.maximum(
        surface_temp - lapse_rate * heights,
        180.0  # Minimum temperature
    )
    
    # Add stratospheric warming
    stratosphere_start = 11000  # m
    strat_indices = heights > stratosphere_start
    if np.any(strat_indices):
        strat_warming_rate = 0.002  # K/m
        temperature[strat_indices] += strat_warming_rate * (heights[strat_indices] - stratosphere_start)
    
    # Pressure profile (exponential decay)
    surface_pressure = draw(st.floats(min_value=80000.0, max_value=105000.0))  # Pa
    scale_height = 8400.0  # m
    pressure = surface_pressure * np.exp(-heights / scale_height)
    
    # Density from ideal gas law
    R_specific = 287.0  # J/(kg·K) for dry air
    density = pressure / (R_specific * temperature)
    
    return {
        'height': heights,
        'temperature': temperature,
        'pressure': pressure,
        'density': density
    }

@st.composite
def radiation_fields(draw):
    """Generate physically reasonable radiation fields"""
    
    # Array shape
    shape = draw(st.tuples(
        st.integers(min_value=5, max_value=20),
        st.integers(min_value=5, max_value=20)
    ))
    
    # Incoming solar radiation
    solar_constant = draw(st.floats(min_value=1200.0, max_value=1500.0))
    incoming_radiation = draw(arrays(
        dtype=np.float32,
        shape=shape,
        elements=st.floats(
            min_value=solar_constant * 0.1,
            max_value=solar_constant * 1.5,
            allow_nan=False,
            allow_infinity=False
        )
    ))
    
    # Albedo (0 to 1)
    albedo = draw(arrays(
        dtype=np.float32,
        shape=shape,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    
    # Temperature (reasonable range)
    temperature = draw(arrays(
        dtype=np.float32,
        shape=shape,
        elements=st.floats(min_value=180.0, max_value=350.0, allow_nan=False, allow_infinity=False)
    ))
    
    return {
        'incoming_radiation': incoming_radiation,
        'albedo': albedo,
        'temperature': temperature
    }


# Test fixtures
@pytest.fixture
def physics_tester():
    """Create physics invariant tester instance"""
    return PhysicsInvariantTester()


# Property-based tests
class TestEnergyConservation:
    """Test energy conservation laws"""
    
    @given(physical_state_arrays())
    @settings(max_examples=50, deadline=5000)
    def test_energy_conservation_property(self, states):
        """Property: Energy should be conserved in isolated systems"""
        tester = PhysicsInvariantTester()
        
        # Assume states represent initial and final states
        if states.shape[0] >= 2:
            initial_state = states[0:1]
            final_state = states[1:2]
            
            # Test energy conservation (no external work)
            result = tester.test_energy_conservation(initial_state, final_state, external_work=0.0)
            
            # Energy should be conserved
            assert result['max_energy_violation'] < ENERGY_CONSERVATION_TOLERANCE * 10, \
                f"Energy not conserved: violation = {result['max_energy_violation']}"
    
    @given(
        physical_state_arrays(),
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=30, deadline=5000)
    def test_energy_conservation_with_work(self, states, external_work):
        """Property: Energy change should equal external work"""
        tester = PhysicsInvariantTester()
        
        if states.shape[0] >= 2:
            initial_state = states[0:1]
            final_state = states[1:2]
            
            result = tester.test_energy_conservation(initial_state, final_state, external_work)
            
            # Energy change should equal external work (within tolerance)
            energy_change = result['final_total_energy'] - result['initial_total_energy']
            work_error = abs(energy_change - external_work)
            
            assert work_error < ENERGY_CONSERVATION_TOLERANCE * 100, \
                f"Work-energy theorem violated: error = {work_error}"


class TestMassConservation:
    """Test mass conservation principles"""
    
    @given(
        arrays(
            dtype=np.float32,
            shape=st.tuples(st.integers(min_value=5, max_value=50)),
            elements=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        arrays(
            dtype=np.float32,
            shape=st.tuples(st.integers(min_value=5, max_value=50)),
            elements=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_mass_conservation_property(self, initial_mass, final_mass):
        """Property: Mass should be conserved in closed systems"""
        tester = PhysicsInvariantTester()
        
        # Ensure arrays have same shape
        min_size = min(len(initial_mass), len(final_mass))
        initial_mass = initial_mass[:min_size]
        final_mass = final_mass[:min_size]
        
        result = tester.test_mass_conservation(initial_mass, final_mass)
        
        # For closed system (no sources or sinks), mass should be conserved
        assert result['mass_conserved'], \
            f"Mass not conserved: violation = {result['mass_violation']}"


class TestThermodynamicConsistency:
    """Test thermodynamic consistency"""
    
    @given(atmospheric_profiles())
    @settings(max_examples=30, deadline=10000)
    def test_ideal_gas_law_property(self, profile):
        """Property: Atmospheric profiles should satisfy ideal gas law"""
        tester = PhysicsInvariantTester()
        
        result = tester.test_thermodynamic_consistency(
            profile['temperature'],
            profile['pressure'],
            profile['density']
        )
        
        # Thermodynamic consistency should hold
        assert result['thermodynamically_consistent'], \
            f"Thermodynamic inconsistency: max violation = {result['max_pressure_violation']}"
        
        # Physical bounds should be satisfied
        assert result['temperature_bounds_ok'], "Temperature out of physical bounds"
        assert result['pressure_bounds_ok'], "Pressure out of physical bounds"
        assert result['density_bounds_ok'], "Density out of physical bounds"


class TestRadiativeEquilibrium:
    """Test radiative equilibrium"""
    
    @given(radiation_fields())
    @settings(max_examples=30, deadline=10000)
    def test_radiative_balance_property(self, fields):
        """Property: Radiative equilibrium should be maintained"""
        tester = PhysicsInvariantTester()
        
        result = tester.test_radiative_equilibrium(
            fields['temperature'],
            fields['incoming_radiation'],
            fields['albedo']
        )
        
        # Check that radiative imbalance is reasonable
        # (Perfect equilibrium is not expected due to dynamics)
        max_imbalance = result['max_radiative_imbalance']
        equilibrium_tolerance = result['equilibrium_tolerance']
        
        assert max_imbalance < equilibrium_tolerance * 10, \
            f"Excessive radiative imbalance: {max_imbalance} W/m²"
        
        # Check physical reasonableness
        assert result['mean_incoming_radiation'] > 0, "Negative incoming radiation"
        assert result['mean_outgoing_radiation'] > 0, "Negative outgoing radiation"
        assert 0 <= result['mean_albedo'] <= 1, "Albedo out of physical range"


class TestHydrostaticBalance:
    """Test hydrostatic balance"""
    
    @given(atmospheric_profiles())
    @settings(max_examples=30, deadline=10000)
    def test_hydrostatic_balance_property(self, profile):
        """Property: Atmospheric columns should be in hydrostatic balance"""
        tester = PhysicsInvariantTester()
        
        result = tester.test_hydrostatic_balance(
            profile['pressure'],
            profile['density'],
            profile['height']
        )
        
        # Hydrostatic balance should hold (within reasonable tolerance)
        assert result['hydrostatic_balance'], \
            f"Hydrostatic imbalance: max relative violation = {result['max_relative_violation']}"
        
        # Pressure should decrease with height
        pressures = profile['pressure']
        assert np.all(np.diff(pressures) <= 0), "Pressure should decrease with height"


# Integration tests for physics invariants
class TestPhysicsIntegration:
    """Integration tests for physics invariants in models"""
    
    def test_mock_climate_model_physics(self, physics_tester):
        """Test physics invariants in a mock climate model"""
        
        # Create a simple mock climate model
        class MockClimateModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = MockClimateModel()
        
        # Generate test data
        batch_size = 5
        state_dim = 10
        initial_state = torch.randn(batch_size, state_dim)
        
        # Forward pass
        with torch.no_grad():
            final_state = model(initial_state)
        
        # Test energy conservation (mock)
        result = physics_tester.test_energy_conservation(
            initial_state.numpy(),
            final_state.numpy()
        )
        
        # Model should not violate physics too severely
        assert result['max_energy_violation'] < 1.0, \
            "Mock model violates energy conservation too severely"
    
    def test_atmospheric_physics_integration(self, physics_tester):
        """Test integration of multiple atmospheric physics constraints"""
        
        # Generate a realistic atmospheric profile
        n_levels = 20
        heights = np.linspace(0, 30000, n_levels)  # 0 to 30 km
        
        # Temperature profile
        surface_temp = 288.15  # K
        lapse_rate = 0.0065   # K/m
        temperature = surface_temp - lapse_rate * heights
        temperature = np.maximum(temperature, 220.0)  # Tropopause
        
        # Pressure profile
        surface_pressure = 101325.0  # Pa
        scale_height = 8400.0  # m
        pressure = surface_pressure * np.exp(-heights / scale_height)
        
        # Density from ideal gas law
        R_specific = 287.0  # J/(kg·K)
        density = pressure / (R_specific * temperature)
        
        # Test thermodynamic consistency
        thermo_result = physics_tester.test_thermodynamic_consistency(
            temperature, pressure, density
        )
        assert thermo_result['thermodynamically_consistent'], \
            "Atmospheric profile thermodynamically inconsistent"
        
        # Test hydrostatic balance
        hydro_result = physics_tester.test_hydrostatic_balance(
            pressure, density, heights
        )
        assert hydro_result['hydrostatic_balance'], \
            "Atmospheric profile not in hydrostatic balance"
        
        # Test radiative equilibrium (simplified)
        incoming_radiation = np.full_like(temperature, 1361.0)  # Solar constant
        albedo = np.full_like(temperature, 0.3)  # Earth's albedo
        
        rad_result = physics_tester.test_radiative_equilibrium(
            temperature, incoming_radiation, albedo
        )
        
        # Radiative balance should be reasonable
        assert rad_result['max_radiative_imbalance'] < 100.0, \
            "Excessive radiative imbalance in atmospheric profile"


# Regression tests for known physics violations
class TestPhysicsRegressions:
    """Regression tests for previously identified physics violations"""
    
    def test_energy_conservation_regression(self, physics_tester):
        """Regression test for energy conservation violations"""
        
        # Known problematic case
        initial_state = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])
        final_state = np.array([[1.1, 2.1, 3.1, 0.11, 0.21, 0.31]])
        
        result = physics_tester.test_energy_conservation(initial_state, final_state)
        
        # This should pass with reasonable tolerance
        assert result['energy_conserved'] or result['max_energy_violation'] < 0.1, \
            "Energy conservation regression detected"
    
    def test_mass_conservation_regression(self, physics_tester):
        """Regression test for mass conservation violations"""
        
        # Known problematic case
        initial_mass = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        final_mass = np.array([1.1, 1.9, 3.0, 4.1, 4.9])  # Small changes
        
        result = physics_tester.test_mass_conservation(initial_mass, final_mass)
        
        # Should detect violation but not be excessive
        assert result['relative_mass_violation'] < 0.1, \
            "Excessive mass conservation violation detected"


# Performance tests for physics checking
class TestPhysicsPerformance:
    """Performance tests for physics invariant checking"""
    
    def test_energy_conservation_performance(self, physics_tester, benchmark):
        """Benchmark energy conservation testing performance"""
        
        # Large state arrays
        batch_size = 1000
        state_dim = 100
        initial_state = np.random.randn(batch_size, state_dim)
        final_state = np.random.randn(batch_size, state_dim)
        
        # Benchmark the test
        result = benchmark(
            physics_tester.test_energy_conservation,
            initial_state,
            final_state
        )
        
        # Should complete successfully
        assert 'energy_conserved' in result
    
    def test_atmospheric_physics_performance(self, physics_tester, benchmark):
        """Benchmark atmospheric physics testing performance"""
        
        # Large atmospheric profile
        n_levels = 1000
        heights = np.linspace(0, 100000, n_levels)
        temperature = 288.15 - 0.0065 * heights
        temperature = np.maximum(temperature, 180.0)
        
        surface_pressure = 101325.0
        scale_height = 8400.0
        pressure = surface_pressure * np.exp(-heights / scale_height)
        
        R_specific = 287.0
        density = pressure / (R_specific * temperature)
        
        # Benchmark the test
        result = benchmark(
            physics_tester.test_thermodynamic_consistency,
            temperature,
            pressure,
            density
        )
        
        # Should complete successfully
        assert 'thermodynamically_consistent' in result


# Parametrized tests for different scenarios
@pytest.mark.parametrize("external_work", [0.0, 1.0, -1.0, 10.0, -10.0])
def test_energy_conservation_scenarios(external_work):
    """Test energy conservation under different external work scenarios"""
    tester = PhysicsInvariantTester()
    
    # Simple test case
    initial_state = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    final_state = np.array([[1.1, 0.0, 0.0, 1.1, 0.0, 0.0]])
    
    result = tester.test_energy_conservation(initial_state, final_state, external_work)
    
    # Energy change should be reasonable
    energy_change = result['final_total_energy'] - result['initial_total_energy']
    assert abs(energy_change - external_work) < 1.0, \
        f"Energy-work relationship violated for external_work={external_work}"


@pytest.mark.parametrize("molecular_weight", [28.97, 44.01, 18.02, 2.02])  # Air, CO2, H2O, H2
def test_thermodynamic_consistency_gases(molecular_weight):
    """Test thermodynamic consistency for different gases"""
    tester = PhysicsInvariantTester()
    
    # Standard conditions
    temperature = np.array([288.15, 250.0, 300.0])  # K
    pressure = np.array([101325.0, 50000.0, 200000.0])  # Pa
    
    # Calculate density from ideal gas law
    R_specific = GAS_CONSTANT / (molecular_weight * 1e-3)
    density = pressure / (R_specific * temperature)
    
    result = tester.test_thermodynamic_consistency(
        temperature, pressure, density, molecular_weight
    )
    
    # Should be consistent for any reasonable gas
    assert result['thermodynamically_consistent'], \
        f"Thermodynamic inconsistency for molecular weight {molecular_weight}"


# Utility functions for test data generation
def generate_realistic_climate_state(n_points: int = 100) -> Dict[str, np.ndarray]:
    """Generate realistic climate state for testing"""
    
    # Spatial coordinates
    lat = np.random.uniform(-90, 90, n_points)
    lon = np.random.uniform(-180, 180, n_points)
    
    # Temperature (latitude dependent)
    temperature = 288.15 - 0.5 * np.abs(lat) + np.random.normal(0, 5, n_points)
    temperature = np.clip(temperature, 200.0, 350.0)
    
    # Pressure (altitude dependent, simplified)
    altitude = np.random.uniform(0, 20000, n_points)
    pressure = 101325.0 * np.exp(-altitude / 8400.0)
    
    # Density from ideal gas law
    R_specific = 287.0
    density = pressure / (R_specific * temperature)
    
    # Humidity
    humidity = np.random.uniform(0.1, 0.9, n_points)
    
    return {
        'latitude': lat,
        'longitude': lon,
        'altitude': altitude,
        'temperature': temperature,
        'pressure': pressure,
        'density': density,
        'humidity': humidity
    }


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
