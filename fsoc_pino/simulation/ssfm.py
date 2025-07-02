"""
Split-Step Fourier Method (SSFM) implementation for beam propagation.

This module implements the SSFM algorithm for solving the Parabolic Wave Equation
in atmospheric turbulence and fog conditions.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import time

from .physics import PWE_Solver, AtmosphericEffects, LinkParameters, AtmosphericParameters


@dataclass
class PropagationResult:
    """Container for propagation simulation results."""
    final_field: np.ndarray  # Complex field at receiver
    irradiance: np.ndarray  # Irradiance pattern |ψ|²
    scintillation_index: float  # Scintillation index
    bit_error_rate: float  # Bit error rate
    propagation_time: float  # Computation time
    intermediate_fields: Optional[List[np.ndarray]] = None  # For debugging


class SplitStepFourierMethod:
    """
    Split-Step Fourier Method solver for atmospheric beam propagation.
    
    The SSFM alternates between:
    1. Diffraction step (Fourier domain): Free-space propagation
    2. Refraction step (spatial domain): Atmospheric effects
    """
    
    def __init__(
        self,
        link_params: LinkParameters,
        atm_params: AtmosphericParameters,
        link_altitude_m: float = 0.0,
        save_intermediate: bool = False
    ):
        """
        Initialize SSFM solver.

        Args:
            link_params: Link configuration parameters
            atm_params: Atmospheric parameters
            link_altitude_m: Average link altitude in meters for Cn^2 modeling
            save_intermediate: Whether to save intermediate field states
        """
        self.link_params = link_params
        self.atm_params = atm_params
        self.save_intermediate = save_intermediate

        # Initialize physics models
        self.atm_effects = AtmosphericEffects(atm_params, link_params.wavelength, link_altitude_m)
        self.pwe_solver = PWE_Solver(link_params, atm_params, self.atm_effects)
        
        # Precompute operators
        self._precompute_operators()
        
    def _precompute_operators(self):
        """Precompute operators for efficiency."""
        # Diffraction operator (frequency domain)
        self.diffraction_op = self.pwe_solver.compute_diffraction_operator()
        
        # Atmospheric parameters
        self.cn_squared = self.atm_effects.compute_cn_squared()
        self.alpha_fog = self.atm_effects.compute_fog_attenuation()
        
        print(f"Cn² = {self.cn_squared:.2e} m^(-2/3)")
        print(f"Fog attenuation = {self.alpha_fog:.2e} m^(-1)")
        
    def propagate(self) -> PropagationResult:
        """
        Perform full beam propagation simulation.
        
        Returns:
            PropagationResult containing simulation outputs
        """
        start_time = time.time()
        
        # Initialize field
        field = self.pwe_solver.create_initial_field()
        
        # Storage for intermediate results
        intermediate_fields = [] if self.save_intermediate else None
        
        # Propagation loop
        for step in range(self.pwe_solver.num_steps):
            z = step * self.pwe_solver.dz
            
            # Step 1: Diffraction (Fourier domain)
            field = self._diffraction_step(field)
            
            # Step 2: Atmospheric effects (spatial domain)
            field = self._atmospheric_step(field, z)
            
            # Save intermediate field if requested
            if self.save_intermediate:
                intermediate_fields.append(field.copy())
                
            # Progress reporting
            if (step + 1) % max(1, self.pwe_solver.num_steps // 10) == 0:
                progress = (step + 1) / self.pwe_solver.num_steps * 100
                print(f"Propagation progress: {progress:.1f}%")
        
        # Compute final results
        irradiance = np.abs(field)**2
        scintillation_index = self.atm_effects.compute_scintillation_index(irradiance)
        
        # Compute received power (sum over detector area)
        received_power = np.sum(irradiance) * self.pwe_solver.dx**2
        bit_error_rate = self.atm_effects.compute_bit_error_rate(received_power, scintillation_index)
        
        propagation_time = time.time() - start_time
        
        return PropagationResult(
            final_field=field,
            irradiance=irradiance,
            scintillation_index=scintillation_index,
            bit_error_rate=bit_error_rate,
            propagation_time=propagation_time,
            intermediate_fields=intermediate_fields
        )
    
    def _diffraction_step(self, field: np.ndarray) -> np.ndarray:
        """
        Perform diffraction step in Fourier domain.
        
        Applies free-space propagation: ψ' = IFFT[FFT[ψ] * H_diffraction]
        
        Args:
            field: Input complex field
            
        Returns:
            Field after diffraction step
        """
        # Transform to frequency domain
        field_fft = np.fft.fft2(field)
        
        # Apply diffraction operator
        field_fft *= self.diffraction_op
        
        # Transform back to spatial domain
        field_out = np.fft.ifft2(field_fft)
        
        return field_out
    
    def _atmospheric_step(self, field: np.ndarray, z: float) -> np.ndarray:
        """
        Perform atmospheric effects step in spatial domain.
        
        Applies turbulence and fog effects:
        ψ_out = ψ_in * exp(-α_fog/2 * dz) * exp(i * k₀ * δn * dz)
        
        Args:
            field: Input complex field
            z: Current propagation distance
            
        Returns:
            Field after atmospheric step
        """
        # Fog attenuation (amplitude effect)
        attenuation = np.exp(-self.alpha_fog / 2 * self.pwe_solver.dz)
        
        # Turbulence phase screen
        if self.cn_squared > 1e-16:  # Only generate if significant turbulence
            phase_screen = self.atm_effects.generate_phase_screen(
                self.link_params.grid_size,
                self.pwe_solver.dx,
                self.cn_squared
            )
            
            # Convert phase to refractive index fluctuation
            # This is a simplified model - real implementation would be more complex
            delta_n = phase_screen * 1e-6  # Scale factor
            
            # Phase effect
            phase_effect = np.exp(1j * self.pwe_solver.k0 * delta_n * self.pwe_solver.dz)
        else:
            phase_effect = 1.0
        
        # Apply both effects
        field_out = field * attenuation * phase_effect
        
        return field_out
    
    def compute_beam_parameters(self, field: np.ndarray) -> dict:
        """
        Compute beam parameters from field distribution.
        
        Args:
            field: Complex field array
            
        Returns:
            Dictionary of beam parameters
        """
        irradiance = np.abs(field)**2
        
        # Normalize coordinates
        x = np.linspace(-self.link_params.grid_width/2, 
                       self.link_params.grid_width/2, 
                       self.link_params.grid_size)
        y = x.copy()
        X, Y = np.meshgrid(x, y)
        
        # Total power
        total_power = np.sum(irradiance) * self.pwe_solver.dx**2
        
        if total_power > 0:
            # Centroid
            x_centroid = np.sum(X * irradiance) * self.pwe_solver.dx**2 / total_power
            y_centroid = np.sum(Y * irradiance) * self.pwe_solver.dx**2 / total_power
            
            # Second moments
            x2_moment = np.sum((X - x_centroid)**2 * irradiance) * self.pwe_solver.dx**2 / total_power
            y2_moment = np.sum((Y - y_centroid)**2 * irradiance) * self.pwe_solver.dx**2 / total_power
            
            # Beam widths (1/e² radius)
            beam_width_x = 2 * np.sqrt(x2_moment)
            beam_width_y = 2 * np.sqrt(y2_moment)
            
            # Peak irradiance
            peak_irradiance = np.max(irradiance)
            
        else:
            x_centroid = y_centroid = 0.0
            beam_width_x = beam_width_y = 0.0
            peak_irradiance = 0.0
        
        return {
            'total_power': total_power,
            'x_centroid': x_centroid,
            'y_centroid': y_centroid,
            'beam_width_x': beam_width_x,
            'beam_width_y': beam_width_y,
            'peak_irradiance': peak_irradiance
        }
    
    def analyze_propagation_quality(self, result: PropagationResult) -> dict:
        """
        Analyze the quality of beam propagation.
        
        Args:
            result: Propagation result
            
        Returns:
            Dictionary of quality metrics
        """
        beam_params = self.compute_beam_parameters(result.final_field)
        
        # Beam quality metrics
        initial_field = self.pwe_solver.create_initial_field()
        initial_params = self.compute_beam_parameters(initial_field)
        
        # Beam spreading ratio
        spreading_x = beam_params['beam_width_x'] / initial_params['beam_width_x']
        spreading_y = beam_params['beam_width_y'] / initial_params['beam_width_y']
        
        # Power loss (due to grid truncation and absorption)
        power_loss_db = -10 * np.log10(beam_params['total_power'] / initial_params['total_power'])
        
        # Beam wander (centroid displacement)
        beam_wander = np.sqrt(beam_params['x_centroid']**2 + beam_params['y_centroid']**2)
        
        return {
            'beam_spreading_x': spreading_x,
            'beam_spreading_y': spreading_y,
            'power_loss_db': power_loss_db,
            'beam_wander': beam_wander,
            'scintillation_index': result.scintillation_index,
            'bit_error_rate': result.bit_error_rate
        }
