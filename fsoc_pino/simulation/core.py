"""
Core FSOC simulation class.

This module provides the main FSOC_Simulator class that orchestrates
the complete physics simulation pipeline.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from .physics import LinkParameters, AtmosphericParameters
from .ssfm import SplitStepFourierMethod, PropagationResult


@dataclass
class SimulationConfig:
    """Configuration for FSOC simulation."""
    # Link parameters (required)
    link_distance: float  # km
    wavelength: float  # m
    beam_waist: float  # m
    visibility: float  # km
    temp_gradient: float  # K/m

    # Optional parameters with defaults
    grid_size: int = 128
    grid_width: float = 0.5  # m
    pressure: float = 101325.0  # Pa
    temperature: float = 288.15  # K
    humidity: float = 0.5  # 0-1
    save_intermediate: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SimulationConfig':
        """Create from dictionary."""
        return cls(**data)
    
    def save(self, filepath: Path):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'SimulationConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class FSOC_Simulator:
    """
    Main FSOC link simulator.
    
    This class provides a high-level interface for running FSOC link simulations
    using the Split-Step Fourier Method to solve the Parabolic Wave Equation.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None, **kwargs):
        """
        Initialize FSOC simulator.
        
        Args:
            config: Simulation configuration object
            **kwargs: Configuration parameters (alternative to config object)
        """
        if config is None:
            config = SimulationConfig(**kwargs)
        
        self.config = config
        self._validate_config()
        
        # Create parameter objects
        self.link_params = LinkParameters(
            distance=config.link_distance,
            wavelength=config.wavelength,
            beam_waist=config.beam_waist,
            grid_size=config.grid_size,
            grid_width=config.grid_width
        )
        
        self.atm_params = AtmosphericParameters(
            visibility=config.visibility,
            temp_gradient=config.temp_gradient,
            pressure=config.pressure,
            temperature=config.temperature,
            humidity=config.humidity
        )
        
        # Initialize SSFM solver
        self.ssfm = SplitStepFourierMethod(
            self.link_params,
            self.atm_params,
            save_intermediate=config.save_intermediate
        )
        
    def _validate_config(self):
        """Validate simulation configuration parameters."""
        config = self.config
        
        # Link parameter validation
        if config.link_distance <= 0:
            raise ValueError("Link distance must be positive")
        if config.wavelength <= 0:
            raise ValueError("Wavelength must be positive")
        if config.beam_waist <= 0:
            raise ValueError("Beam waist must be positive")
        if config.grid_size < 32:
            raise ValueError("Grid size must be at least 32")
        if config.grid_width <= 0:
            raise ValueError("Grid width must be positive")
            
        # Atmospheric parameter validation
        if config.visibility <= 0:
            raise ValueError("Visibility must be positive")
        if config.temp_gradient < 0:
            raise ValueError("Temperature gradient must be non-negative")
        if config.pressure <= 0:
            raise ValueError("Pressure must be positive")
        if config.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if not 0 <= config.humidity <= 1:
            raise ValueError("Humidity must be between 0 and 1")
    
    def run_simulation(self) -> PropagationResult:
        """
        Run the complete FSOC link simulation.
        
        Returns:
            PropagationResult containing all simulation outputs
        """
        print(f"Starting FSOC simulation...")
        print(f"Link distance: {self.config.link_distance} km")
        print(f"Visibility: {self.config.visibility} km")
        print(f"Temperature gradient: {self.config.temp_gradient} K/m")
        print(f"Wavelength: {self.config.wavelength*1e9:.0f} nm")
        print(f"Beam waist: {self.config.beam_waist*100:.1f} cm")
        print(f"Grid size: {self.config.grid_size}x{self.config.grid_size}")
        
        # Run propagation
        result = self.ssfm.propagate()
        
        print(f"Simulation completed in {result.propagation_time:.2f} seconds")
        print(f"Scintillation index: {result.scintillation_index:.6f}")
        print(f"Bit error rate: {result.bit_error_rate:.2e}")
        
        return result
    
    def analyze_link_budget(self, result: PropagationResult) -> Dict:
        """
        Perform link budget analysis.
        
        Args:
            result: Propagation simulation result
            
        Returns:
            Dictionary containing link budget parameters
        """
        # Beam parameters
        beam_params = self.ssfm.compute_beam_parameters(result.final_field)
        
        # Atmospheric losses
        total_distance_m = self.config.link_distance * 1000
        fog_loss_db = self.ssfm.alpha_fog * total_distance_m * 10 / np.log(10)
        
        # Geometric losses (beam spreading)
        initial_field = self.ssfm.pwe_solver.create_initial_field()
        initial_params = self.ssfm.compute_beam_parameters(initial_field)
        
        beam_spreading_loss_db = -10 * np.log10(
            beam_params['total_power'] / initial_params['total_power']
        )
        
        # Scintillation effects
        scintillation_db = -10 * np.log10(1 + result.scintillation_index)
        
        # Total link loss
        total_loss_db = fog_loss_db + beam_spreading_loss_db + abs(scintillation_db)
        
        return {
            'fog_loss_db': fog_loss_db,
            'beam_spreading_loss_db': beam_spreading_loss_db,
            'scintillation_loss_db': abs(scintillation_db),
            'total_loss_db': total_loss_db,
            'received_power_fraction': beam_params['total_power'] / initial_params['total_power'],
            'link_margin_db': 20 - total_loss_db,  # Assuming 20 dB required SNR
            'beam_parameters': beam_params
        }
    
    def get_simulation_summary(self, result: PropagationResult) -> Dict:
        """
        Get a comprehensive summary of simulation results.
        
        Args:
            result: Propagation simulation result
            
        Returns:
            Dictionary containing simulation summary
        """
        link_budget = self.analyze_link_budget(result)
        quality_metrics = self.ssfm.analyze_propagation_quality(result)
        
        return {
            'configuration': self.config.to_dict(),
            'results': {
                'scintillation_index': result.scintillation_index,
                'bit_error_rate': result.bit_error_rate,
                'propagation_time': result.propagation_time
            },
            'link_budget': link_budget,
            'quality_metrics': quality_metrics,
            'atmospheric_conditions': {
                'cn_squared': self.ssfm.cn_squared,
                'fog_attenuation': self.ssfm.alpha_fog,
                'visibility': self.config.visibility,
                'temp_gradient': self.config.temp_gradient
            }
        }
    
    def save_results(self, result: PropagationResult, output_dir: Path):
        """
        Save simulation results to files.
        
        Args:
            result: Propagation simulation result
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save irradiance map
        np.save(output_dir / "irradiance.npy", result.irradiance)
        
        # Save complex field
        np.save(output_dir / "field_real.npy", np.real(result.final_field))
        np.save(output_dir / "field_imag.npy", np.imag(result.final_field))
        
        # Save simulation summary
        summary = self.get_simulation_summary(result)
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save configuration
        self.config.save(output_dir / "config.json")
        
        print(f"Results saved to: {output_dir}")
    
    @classmethod
    def create_parameter_sweep(
        cls,
        base_config: SimulationConfig,
        parameter_ranges: Dict[str, Tuple[float, float]],
        num_samples: int,
        sampling_method: str = "latin_hypercube"
    ) -> List[SimulationConfig]:
        """
        Create a parameter sweep for dataset generation.
        
        Args:
            base_config: Base configuration
            parameter_ranges: Dictionary of parameter ranges
            num_samples: Number of samples to generate
            sampling_method: Sampling method
            
        Returns:
            List of simulation configurations
        """
        if sampling_method == "latin_hypercube":
            from scipy.stats import qmc
            
            # Create Latin Hypercube sampler
            sampler = qmc.LatinHypercube(d=len(parameter_ranges))
            samples = sampler.random(n=num_samples)
            
        elif sampling_method == "uniform":
            samples = np.random.uniform(0, 1, (num_samples, len(parameter_ranges)))
            
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        # Scale samples to parameter ranges
        configs = []
        param_names = list(parameter_ranges.keys())
        
        for sample in samples:
            config_dict = base_config.to_dict()
            
            for i, param_name in enumerate(param_names):
                min_val, max_val = parameter_ranges[param_name]
                config_dict[param_name] = min_val + sample[i] * (max_val - min_val)
            
            configs.append(SimulationConfig.from_dict(config_dict))
        
        return configs
