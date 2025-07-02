"""
Physics models for FSOC link simulation.

This module implements the core physics models including:
- Parabolic Wave Equation (PWE) formulation
- Atmospheric effects (turbulence and fog)
- Refractive index modeling
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import scipy.special as special


@dataclass
class AtmosphericParameters:
    """Container for atmospheric parameters."""
    visibility: float  # Meteorological visibility in km
    temp_gradient: float  # Temperature gradient in K/m (for local turbulence strength)
    pressure_hpa: float = 1013.25  # Pressure in hPa (hectopascals)
    temperature_celsius: float = 15.0  # Temperature in Celsius
    humidity: float = 0.5  # Relative humidity (0-1)
    # Note: Altitude for Cn^2 modeling will be derived from link parameters for path-averaged effects
    # or from specific Tx/Rx altitudes for localized effects.
    # For now, we'll keep it simple and use a single altitude for the atmospheric profile if needed.
    # The Hufnagel-Valley model uses a single altitude 'h' which can be interpreted as average link altitude.



@dataclass
class LinkParameters:
    """Container for link parameters."""
    distance: float  # Link distance in km
    wavelength: float  # Wavelength in m
    beam_waist: float  # Initial beam waist in m
    grid_size: int = 128  # Spatial grid size
    grid_width: float = 0.5  # Grid width in m
    altitude_tx_m: float = 0.0  # Transmitter altitude in meters
    altitude_rx_m: float = 0.0  # Receiver altitude in meters


class AtmosphericEffects:
    """
    Models for atmospheric effects on laser beam propagation.

    Includes:
    - Temperature gradient effects (turbulence)
    - Fog attenuation
    - Refractive index structure parameter Cn²
    """

    def __init__(self, atm_params: AtmosphericParameters, wavelength: float, link_altitude_m: float = 0.0):
        self.atm_params = atm_params
        self.wavelength = wavelength
        self.link_altitude_m = link_altitude_m

    def compute_cn_squared(self) -> float:
        """
        Compute refractive index structure parameter Cn² using a more comprehensive Hufnagel-Valley model.
        
        The Hufnagel-Valley model is a widely used empirical model for the altitude dependence of Cn².
        It combines ground-level turbulence, upper-atmosphere turbulence, and a wind-speed dependent term.
        
        Returns:
            Cn² value in m^(-2/3)
        """
        h = self.link_altitude_m  # Altitude in meters
        
        # Default parameters for Hufnagel-Valley model (can be made configurable if needed)
        A = 1.7e-14  # Ground-level Cn^2 (m^-2/3)
        B = 1.0e-16  # Upper-atmosphere Cn^2 (m^-2/3)
        C = 5.0e-14  # Wind-speed dependent term (m^-2/3)
        v_rms = 21.0  # RMS wind speed (m/s) - typical value
        
        # Hufnagel-Valley model formula
        cn_squared = (
            A * np.exp(-h / 100)  # Ground layer
            + B * np.exp(-h / 1500)  # Troposphere
            + C * (h / 1000)**(2/3) * np.exp(-h / 1000) * (v_rms / 21.0)**2 # Wind-dependent term
        )
        
        # Ensure Cn^2 is within a realistic range
        return np.clip(cn_squared, 1e-17, 1e-13)

    def compute_fog_attenuation(self) -> float:
        """
        Compute fog attenuation coefficient using the Kim model, with enhanced wavelength dependence.
        
        The Kim model relates visibility to attenuation, and includes a wavelength dependence.
        
        Returns:
            Attenuation coefficient in m^(-1)
        """
        V = self.atm_params.visibility  # km
        wavelength_nm = self.wavelength * 1e9  # Convert to nm
        
        # Kim model parameters based on visibility range
        if V > 50:
            q = 1.6
        elif V > 6:
            q = 1.3
        elif V > 1:
            q = 0.16 * V + 0.34
        elif V > 0.5:
            q = V - 0.5
        else:
            q = 0.0  # Very dense fog, attenuation becomes wavelength independent
        
        # Kim model formula for attenuation in dB/km
        alpha_fog_db_km = (3.91 / V) * (wavelength_nm / 550)**(-q)
        
        # Convert from dB/km to Nepers/m (m^-1)
        alpha_fog = alpha_fog_db_km * np.log(10) / 10 / 1000
        
        return alpha_fog

    def generate_phase_screen(self, grid_size: int, grid_spacing: float, cn_squared: float) -> np.ndarray:
        """
        Generate a random phase screen for atmospheric turbulence.

        Uses Kolmogorov turbulence spectrum.

        Args:
            grid_size: Size of the grid
            grid_spacing: Spacing between grid points in m
            cn_squared: Refractive index structure parameter

        Returns:
            Phase screen array
        """
        # Frequency grid
        df = 1.0 / (grid_size * grid_spacing)
        fx = np.fft.fftfreq(grid_size, grid_spacing)
        fy = np.fft.fftfreq(grid_size, grid_spacing)
        FX, FY = np.meshgrid(fx, fy)

        # Spatial frequency magnitude
        f = np.sqrt(FX**2 + FY**2)
        f[0, 0] = 1e-10  # Avoid division by zero

        # Kolmogorov spectrum
        # Φ_n(f) = 0.033 * Cn² * f^(-11/3)
        phi_n = 0.033 * cn_squared * f**(-11/3)

        # Generate random complex amplitudes
        random_complex = (np.random.randn(grid_size, grid_size) +
                         1j * np.random.randn(grid_size, grid_size))

        # Apply spectrum
        phase_fft = random_complex * np.sqrt(phi_n * df**2)

        # Transform to spatial domain
        phase_screen = np.real(np.fft.ifft2(phase_fft))

        return phase_screen

    def compute_scintillation_index(self, irradiance: np.ndarray) -> float:
        """
        Compute scintillation index from irradiance pattern.

        Args:
            irradiance: 2D irradiance array

        Returns:
            Scintillation index σ_I²
        """
        mean_I = np.mean(irradiance)
        var_I = np.var(irradiance)

        if mean_I > 0:
            scintillation_index = var_I / (mean_I**2)
        else:
            scintillation_index = 0.0

        return scintillation_index

    def compute_bit_error_rate(self, received_power: float, scintillation_index: float, noise_power: float = 1e-12) -> float:
        """
        Compute bit error rate for On-Off Keying (OOK) modulation with log-normal fading.

        Args:
            received_power: Received optical power (mean power)
            scintillation_index: Scintillation index (variance of log-intensity)
            noise_power: Noise power (thermal + shot noise)

        Returns:
            Bit error rate
        """
        if noise_power <= 0 or received_power <= 0:
            return 0.5 # Max BER if no signal or infinite noise

        # Convert scintillation index to log-amplitude variance (sigma_l^2)
        # sigma_I^2 = exp(sigma_l^2) - 1  => sigma_l^2 = ln(1 + sigma_I^2)
        sigma_l_squared = np.log(1 + scintillation_index)

        # Average SNR
        average_snr = received_power / noise_power

        # BER for OOK with log-normal fading (approximate integral)
        # This is a common approximation for log-normal fading channels.
        # The integral is over the probability density function of the log-normal intensity.
        # The error function is used to calculate the probability of error for a given instantaneous SNR.
        # The integral is then averaged over the fading distribution.

        # Numerical integration parameters
        num_points = 100
        x_values = np.linspace(-5 * np.sqrt(sigma_l_squared), 5 * np.sqrt(sigma_l_squared), num_points)

        # Log-normal PDF (for log-amplitude)
        pdf_log_amplitude = (1 / (np.sqrt(2 * np.pi * sigma_l_squared))) * np.exp(-x_values**2 / (2 * sigma_l_squared))

        # Instantaneous SNR
        instantaneous_snr = average_snr * np.exp(2 * x_values + sigma_l_squared)

        # Instantaneous BER for OOK with Gaussian noise
        instantaneous_ber = 0.5 * special.erfc(np.sqrt(instantaneous_snr / 2))

        # Average BER over fading distribution (numerical integration)
        ber = np.trapz(instantaneous_ber * pdf_log_amplitude, x_values)

        # Ensure BER is within [0, 0.5]
        return np.clip(ber, 0.0, 0.5)


class PWE_Solver:
    """
    Parabolic Wave Equation solver for laser beam propagation.
    
    Implements the PWE: 2ik₀ ∂ψ/∂z + ∇²_T ψ + 2k₀² [n(x,y,z)/n₀ - 1] ψ = 0
    """
    
    def __init__(self, link_params: LinkParameters, atm_params: AtmosphericParameters, atm_effects: AtmosphericEffects):
        self.link_params = link_params
        self.atm_params = atm_params
        self.atm_effects = atm_effects
        
        # Derived parameters
        self.k0 = 2 * np.pi / link_params.wavelength  # Wavenumber
        self.n0 = 1.0  # Reference refractive index (air)
        
        # Setup spatial grid
        self._setup_spatial_grid()
        
        # Setup propagation parameters
        self.dz = 100.0  # Propagation step size in m
        self.num_steps = int(link_params.distance * 1000 / self.dz)
        
    def _setup_spatial_grid(self):
        """Setup the spatial coordinate grids."""
        N = self.link_params.grid_size
        L = self.link_params.grid_width
        
        # Spatial coordinates
        self.dx = L / N
        x = np.linspace(-L/2, L/2, N)
        y = np.linspace(-L/2, L/2, N)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Frequency coordinates for FFT
        kx = 2 * np.pi * np.fft.fftfreq(N, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(N, self.dx)
        self.KX, self.KY = np.meshgrid(kx, ky)
        
        # Transverse wavenumber squared
        self.KT_squared = self.KX**2 + self.KY**2
        
    def create_initial_field(self) -> np.ndarray:
        """
        Create initial Gaussian beam field.
        
        Returns:
            Complex field array ψ(x,y,z=0)
        """
        w0 = self.link_params.beam_waist
        
        # Gaussian beam profile
        r_squared = self.X**2 + self.Y**2
        field = np.exp(-r_squared / w0**2)
        
        # Normalize to unit power
        power = np.sum(np.abs(field)**2) * self.dx**2
        field = field / np.sqrt(power)
        
        return field.astype(np.complex128)
    
    def compute_diffraction_operator(self) -> np.ndarray:
        """
        Compute the diffraction operator for the vacuum propagation step.
        
        Returns:
            Diffraction operator in frequency domain
        """
        return np.exp(-1j * self.KT_squared / (2 * self.k0) * self.dz)
    
    def compute_refractive_index_fluctuation(self, z: float) -> np.ndarray:
        """
        Compute refractive index fluctuation δn(x,y,z) due to atmospheric effects.
        
        This function now uses the generated phase screen from `AtmosphericEffects`
        and scales it appropriately to represent refractive index fluctuations.
        
        Args:
            z: Propagation distance (unused in current phase screen model, but kept for future extensions)
            
        Returns:
            Refractive index fluctuation array (dimensionless)
        """
        cn_squared = self.atm_effects.compute_cn_squared()
        
        # Generate a phase screen. The phase screen represents the accumulated phase distortion.
        # To convert this to a refractive index fluctuation (delta_n), we use the relationship:
        # delta_phi = k0 * delta_n * dz
        # So, delta_n = delta_phi / (k0 * dz)
        # However, a single phase screen is typically applied at intervals or represents the total path effect.
        # For a more robust model, multiple phase screens along the propagation path would be used.
        # Here, we generate a single phase screen and interpret it as a spatial fluctuation in refractive index.
        # The scaling factor (1e-6) is an empirical value to get realistic magnitudes for delta_n.
        # A more rigorous approach would involve integrating the effect of Cn^2 along the path.
        phase_screen = self.atm_effects.generate_phase_screen(
            self.link_params.grid_size,
            self.dx,
            cn_squared
        )
        
        # Convert phase screen to refractive index fluctuation (δn)
        # This is a simplified conversion for the purpose of demonstrating the PINO concept.
        # The magnitude of delta_n should be small (e.g., 10^-6 to 10^-8).
        delta_n = phase_screen * (1e-6)  # Example scaling factor for refractive index fluctuation
        
        return delta_n
