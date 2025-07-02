"""
Parameter sampling utilities for dataset generation.

This module provides various sampling methods for generating parameter
combinations for FSOC simulations.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import scipy.stats.qmc as qmc


class ParameterSampler:
    """
    Parameter sampler for generating simulation parameter combinations.
    
    Supports various sampling methods:
    - Uniform random sampling
    - Latin Hypercube Sampling (LHS)
    - Sobol sequences
    - Halton sequences
    """
    
    def __init__(
        self, 
        method: str = "latin_hypercube",
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize parameter sampler.
        
        Args:
            method: Sampling method ('uniform', 'latin_hypercube', 'sobol', 'halton')
            ranges: Dictionary of parameter ranges {name: (min, max)}
            seed: Random seed for reproducibility
        """
        self.method = method
        self.ranges = ranges or {}
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Validate method
        valid_methods = ["uniform", "latin_hypercube", "sobol", "halton"]
        if method not in valid_methods:
            raise ValueError(f"Unknown sampling method: {method}. Valid methods: {valid_methods}")
    
    def add_parameter(self, name: str, min_value: float, max_value: float):
        """Add a parameter range."""
        self.ranges[name] = (min_value, max_value)
    
    def sample(self, num_samples: int) -> np.ndarray:
        """
        Generate parameter samples.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Array of shape (num_samples, num_parameters) with parameter values
        """
        if not self.ranges:
            raise ValueError("No parameter ranges defined")
        
        num_params = len(self.ranges)
        param_names = list(self.ranges.keys())
        
        # Generate unit hypercube samples
        if self.method == "uniform":
            unit_samples = np.random.uniform(0, 1, (num_samples, num_params))
            
        elif self.method == "latin_hypercube":
            sampler = qmc.LatinHypercube(d=num_params, seed=self.seed)
            unit_samples = sampler.random(n=num_samples)
            
        elif self.method == "sobol":
            sampler = qmc.Sobol(d=num_params, seed=self.seed)
            unit_samples = sampler.random(n=num_samples)
            
        elif self.method == "halton":
            sampler = qmc.Halton(d=num_params, seed=self.seed)
            unit_samples = sampler.random(n=num_samples)
        
        # Scale to parameter ranges
        scaled_samples = np.zeros_like(unit_samples)
        for i, param_name in enumerate(param_names):
            min_val, max_val = self.ranges[param_name]
            scaled_samples[:, i] = min_val + unit_samples[:, i] * (max_val - min_val)
        
        return scaled_samples
    
    def sample_dict(self, num_samples: int) -> List[Dict[str, float]]:
        """
        Generate parameter samples as list of dictionaries.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of parameter dictionaries
        """
        samples = self.sample(num_samples)
        param_names = list(self.ranges.keys())
        
        return [
            {param_names[i]: samples[j, i] for i in range(len(param_names))}
            for j in range(num_samples)
        ]
    
    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names."""
        return list(self.ranges.keys())
    
    def get_parameter_bounds(self) -> np.ndarray:
        """
        Get parameter bounds as array.
        
        Returns:
            Array of shape (num_parameters, 2) with [min, max] for each parameter
        """
        bounds = []
        for param_name in self.ranges:
            min_val, max_val = self.ranges[param_name]
            bounds.append([min_val, max_val])
        return np.array(bounds)
    
    def validate_samples(self, samples: np.ndarray) -> bool:
        """
        Validate that samples are within parameter bounds.
        
        Args:
            samples: Array of parameter samples
            
        Returns:
            True if all samples are valid
        """
        bounds = self.get_parameter_bounds()
        
        for i, param_name in enumerate(self.ranges):
            min_val, max_val = bounds[i]
            if np.any(samples[:, i] < min_val) or np.any(samples[:, i] > max_val):
                print(f"Invalid samples for parameter {param_name}")
                return False
        
        return True
    
    def compute_discrepancy(self, samples: np.ndarray) -> float:
        """
        Compute L2-star discrepancy of samples (measure of uniformity).
        
        Args:
            samples: Array of parameter samples
            
        Returns:
            L2-star discrepancy value
        """
        # Scale samples to unit hypercube
        bounds = self.get_parameter_bounds()
        unit_samples = np.zeros_like(samples)
        
        for i in range(samples.shape[1]):
            min_val, max_val = bounds[i]
            unit_samples[:, i] = (samples[:, i] - min_val) / (max_val - min_val)
        
        # Compute L2-star discrepancy using scipy
        return qmc.discrepancy(unit_samples, method="L2-star")
    
    def plot_samples(self, samples: np.ndarray, save_path: Optional[str] = None):
        """
        Plot parameter samples for visualization.
        
        Args:
            samples: Array of parameter samples
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt
        
        num_params = samples.shape[1]
        param_names = self.get_parameter_names()
        
        if num_params == 1:
            plt.figure(figsize=(8, 4))
            plt.hist(samples[:, 0], bins=30, alpha=0.7)
            plt.xlabel(param_names[0])
            plt.ylabel('Frequency')
            plt.title(f'Parameter Distribution ({self.method})')
            
        elif num_params == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6)
            plt.xlabel(param_names[0])
            plt.ylabel(param_names[1])
            plt.title(f'Parameter Samples ({self.method})')
            plt.grid(True, alpha=0.3)
            
        else:
            # Pairwise scatter plots
            fig, axes = plt.subplots(num_params, num_params, figsize=(12, 12))
            
            for i in range(num_params):
                for j in range(num_params):
                    if i == j:
                        # Diagonal: histograms
                        axes[i, j].hist(samples[:, i], bins=20, alpha=0.7)
                        axes[i, j].set_xlabel(param_names[i])
                    else:
                        # Off-diagonal: scatter plots
                        axes[i, j].scatter(samples[:, j], samples[:, i], alpha=0.5, s=1)
                        axes[i, j].set_xlabel(param_names[j])
                        axes[i, j].set_ylabel(param_names[i])
            
            plt.suptitle(f'Parameter Samples ({self.method})')
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Sample plot saved to: {save_path}")
        else:
            plt.show()


def create_fsoc_parameter_sampler(
    link_distance_range: Tuple[float, float] = (1.0, 5.0),
    visibility_range: Tuple[float, float] = (0.5, 10.0),
    temp_gradient_range: Tuple[float, float] = (0.01, 0.2),
    beam_waist_range: Tuple[float, float] = (0.02, 0.10),
    wavelength_range: Tuple[float, float] = (850e-9, 1550e-9),
    pressure_hpa_range: Tuple[float, float] = (950.0, 1050.0),
    temperature_celsius_range: Tuple[float, float] = (0.0, 30.0),
    humidity_range: Tuple[float, float] = (0.2, 0.9),
    altitude_tx_m_range: Tuple[float, float] = (0.0, 100.0),
    altitude_rx_m_range: Tuple[float, float] = (0.0, 100.0),
    method: str = "latin_hypercube",
    seed: Optional[int] = None
) -> ParameterSampler:
    """
    Create a parameter sampler for FSOC simulations with typical ranges.
    
    Args:
        link_distance_range: Link distance range in km
        visibility_range: Visibility range in km
        temp_gradient_range: Temperature gradient range in K/m
        beam_waist_range: Beam waist range in m
        wavelength_range: Wavelength range in m
        pressure_hpa_range: Pressure range in hPa
        temperature_celsius_range: Temperature range in Celsius
        humidity_range: Humidity range (0-1)
        altitude_tx_m_range: Transmitter altitude range in meters
        altitude_rx_m_range: Receiver altitude range in meters
        method: Sampling method
        seed: Random seed
        
    Returns:
        Configured ParameterSampler
    """
    sampler = ParameterSampler(method=method, seed=seed)
    
    sampler.add_parameter("link_distance", *link_distance_range)
    sampler.add_parameter("visibility", *visibility_range)
    sampler.add_parameter("temp_gradient", *temp_gradient_range)
    sampler.add_parameter("beam_waist", *beam_waist_range)
    sampler.add_parameter("wavelength", *wavelength_range)
    sampler.add_parameter("pressure_hpa", *pressure_hpa_range)
    sampler.add_parameter("temperature_celsius", *temperature_celsius_range)
    sampler.add_parameter("humidity", *humidity_range)
    sampler.add_parameter("altitude_tx_m", *altitude_tx_m_range)
    sampler.add_parameter("altitude_rx_m", *altitude_rx_m_range)
    
    return sampler
