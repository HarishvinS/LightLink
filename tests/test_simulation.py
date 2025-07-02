"""
Tests for the physics simulation module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from fsoc_pino.simulation import (
    FSOC_Simulator,
    SimulationConfig,
    PWE_Solver,
    AtmosphericEffects,
    LinkParameters,
    AtmosphericParameters
)


class TestSimulationConfig:
    """Test cases for SimulationConfig class."""

    def test_config_creation(self):
        """Test configuration creation with valid parameters."""
        config = SimulationConfig(
            link_distance=2.5,
            wavelength=1550e-9,
            beam_waist=0.05,
            visibility=3.0,
            temp_gradient=0.05
        )

        assert config.link_distance == 2.5
        assert config.wavelength == 1550e-9
        assert config.beam_waist == 0.05
        assert config.visibility == 3.0
        assert config.temp_gradient == 0.05

    def test_config_serialization(self):
        """Test configuration save/load functionality."""
        config = SimulationConfig(
            link_distance=2.5,
            wavelength=1550e-9,
            beam_waist=0.05,
            visibility=3.0,
            temp_gradient=0.05
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "test_config.json"
            config.save(config_file)

            loaded_config = SimulationConfig.load(config_file)

            assert loaded_config.link_distance == config.link_distance
            assert loaded_config.wavelength == config.wavelength
            assert loaded_config.visibility == config.visibility


class TestAtmosphericEffects:
    """Test cases for AtmosphericEffects class."""

    def test_cn_squared_computation(self):
        """Test Cn² computation."""
        atm_params = AtmosphericParameters(
            visibility=3.0,
            temp_gradient=0.05
        )

        atm_effects = AtmosphericEffects(atm_params, 1550e-9)
        cn_squared = atm_effects.compute_cn_squared()

        # Should be in typical range
        assert 1e-17 <= cn_squared <= 1e-13

    def test_fog_attenuation(self):
        """Test fog attenuation computation."""
        atm_params = AtmosphericParameters(
            visibility=1.0,  # Heavy fog
            temp_gradient=0.05
        )

        atm_effects = AtmosphericEffects(atm_params, 1550e-9)
        alpha_fog = atm_effects.compute_fog_attenuation()

        # Should be positive for fog conditions
        assert alpha_fog > 0

    def test_phase_screen_generation(self):
        """Test turbulence phase screen generation."""
        atm_params = AtmosphericParameters(
            visibility=3.0,
            temp_gradient=0.1
        )

        atm_effects = AtmosphericEffects(atm_params, 1550e-9)
        cn_squared = atm_effects.compute_cn_squared()

        phase_screen = atm_effects.generate_phase_screen(
            grid_size=64,
            grid_spacing=0.01,
            cn_squared=cn_squared
        )

        assert phase_screen.shape == (64, 64)
        assert np.isfinite(phase_screen).all()

    def test_scintillation_index(self):
        """Test scintillation index computation."""
        atm_params = AtmosphericParameters(
            visibility=3.0,
            temp_gradient=0.05
        )

        atm_effects = AtmosphericEffects(atm_params, 1550e-9)

        # Create test irradiance pattern with some variation
        irradiance = np.random.exponential(1.0, (64, 64))

        scint_index = atm_effects.compute_scintillation_index(irradiance)

        assert scint_index >= 0
        assert np.isfinite(scint_index)


class TestPWESolver:
    """Test cases for PWE_Solver class."""

    def test_solver_initialization(self):
        """Test PWE solver initialization."""
        link_params = LinkParameters(
            distance=2.5,
            wavelength=1550e-9,
            beam_waist=0.05,
            grid_size=64
        )

        atm_params = AtmosphericParameters(
            visibility=3.0,
            temp_gradient=0.05
        )
        atm_effects = AtmosphericEffects(atm_params, link_params.wavelength)
        solver = PWE_Solver(link_params, atm_params, atm_effects)

        assert solver.k0 > 0
        assert solver.X.shape == (64, 64)
        assert solver.Y.shape == (64, 64)
        assert solver.num_steps > 0

    def test_initial_field_creation(self):
        """Test initial Gaussian beam field creation."""
        link_params = LinkParameters(
            distance=2.5,
            wavelength=1550e-9,
            beam_waist=0.05,
            grid_size=64
        )

        atm_params = AtmosphericParameters(
            visibility=3.0,
            temp_gradient=0.05
        )
        atm_effects = AtmosphericEffects(atm_params, link_params.wavelength)
        solver = PWE_Solver(link_params, atm_params, atm_effects)
        field = solver.create_initial_field()

        assert field.shape == (64, 64)
        assert field.dtype == np.complex128

        # Check normalization (approximately unit power)
        power = np.sum(np.abs(field)**2) * solver.dx**2
        assert abs(power - 1.0) < 0.1

    def test_diffraction_operator(self):
        """Test diffraction operator computation."""
        link_params = LinkParameters(
            distance=2.5,
            wavelength=1550e-9,
            beam_waist=0.05,
            grid_size=64
        )

        atm_params = AtmosphericParameters(
            visibility=3.0,
            temp_gradient=0.05
        )
        atm_effects = AtmosphericEffects(atm_params, link_params.wavelength)
        solver = PWE_Solver(link_params, atm_params, atm_effects)
        diff_op = solver.compute_diffraction_operator()

        assert diff_op.shape == (64, 64)
        assert diff_op.dtype == np.complex128

        # Should have unit magnitude (unitary operator)
        assert np.allclose(np.abs(diff_op), 1.0)


class TestFSOCSimulator:
    """Test cases for FSOC_Simulator class."""

    def test_simulator_initialization(self):
        """Test that simulator initializes with valid parameters."""
        config = SimulationConfig(
            link_distance=2.5,
            wavelength=1550e-9,
            beam_waist=0.05,
            visibility=3.0,
            temp_gradient=0.05,
            grid_size=64  # Smaller grid for faster testing
        )

        simulator = FSOC_Simulator(config)

        assert simulator.config.link_distance == 2.5
        assert simulator.link_params.distance == 2.5
        assert simulator.atm_params.visibility == 3.0

    def test_simulator_validation(self):
        """Test parameter validation."""
        # Test invalid link distance
        with pytest.raises(ValueError, match="Link distance must be positive"):
            config = SimulationConfig(
                link_distance=-1.0,
                wavelength=1550e-9,
                beam_waist=0.05,
                visibility=3.0,
                temp_gradient=0.05
            )
            FSOC_Simulator(config)

        # Test invalid visibility
        with pytest.raises(ValueError, match="Visibility must be positive"):
            config = SimulationConfig(
                link_distance=2.5,
                wavelength=1550e-9,
                beam_waist=0.05,
                visibility=-1.0,
                temp_gradient=0.05
            )
            FSOC_Simulator(config)

    def test_simulation_run(self):
        """Test running a complete simulation."""
        config = SimulationConfig(
            link_distance=1.0,  # Short distance for fast test
            wavelength=1550e-9,
            beam_waist=0.05,
            visibility=5.0,  # Clear conditions
            temp_gradient=0.01,  # Weak turbulence
            grid_size=32  # Small grid for speed
        )

        simulator = FSOC_Simulator(config)
        result = simulator.run_simulation()

        # Check result structure
        assert result.final_field.shape == (32, 32)
        assert result.irradiance.shape == (32, 32)
        assert result.scintillation_index >= 0
        assert result.bit_error_rate >= 0
        assert result.propagation_time > 0

        # Check physical reasonableness
        assert np.all(result.irradiance >= 0)  # Irradiance should be non-negative
        assert np.isfinite(result.scintillation_index)
        assert result.bit_error_rate <= 1.0  # BER should be <= 1

    def test_link_budget_analysis(self):
        """Test link budget analysis."""
        config = SimulationConfig(
            link_distance=2.0,
            wavelength=1550e-9,
            beam_waist=0.05,
            visibility=2.0,  # Moderate fog
            temp_gradient=0.05,
            grid_size=32
        )

        simulator = FSOC_Simulator(config)
        result = simulator.run_simulation()
        link_budget = simulator.analyze_link_budget(result)

        # Check link budget components
        assert 'fog_loss_db' in link_budget
        assert 'beam_spreading_loss_db' in link_budget
        assert 'total_loss_db' in link_budget
        assert 'received_power_fraction' in link_budget

        # Physical checks
        assert link_budget['fog_loss_db'] >= 0  # Fog causes loss
        assert 0 <= link_budget['received_power_fraction'] <= 1  # Power fraction should be valid

    
