"""
Loss functions for Physics-Informed Neural Operators (PINOs).

This module implements various loss functions including:
- Data loss (MSE between predictions and targets)
- Physics-informed loss (PWE residual)
- Combined PINO loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

from ..simulation.physics import AtmosphericParameters, AtmosphericEffects


class DataLoss(nn.Module):
    """
    Data loss for supervised learning.
    
    Computes MSE between model predictions and ground truth data.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize data loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(DataLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute data loss.
        
        Args:
            prediction: Model prediction of shape (batch, channels, H, W)
            target: Ground truth of shape (batch, channels, H, W)
            
        Returns:
            Data loss scalar
        """
        return F.mse_loss(prediction, target, reduction=self.reduction)


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss based on the Parabolic Wave Equation (PWE).
    
    Computes the residual of the PWE:
    2ik₀ ∂ψ/∂z + ∇²_T ψ + 2k₀² [n(x,y,z)/n₀ - 1] ψ = 0
    """
    
    def __init__(
        self,
        grid_spacing: float = 1.0,
        wavelength: float = 1550e-9,
        n0: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize physics-informed loss.
        
        Args:
            grid_spacing: Spatial grid spacing in meters
            wavelength: Wavelength in meters
            n0: Reference refractive index
            reduction: Reduction method
        """
        super(PhysicsInformedLoss, self).__init__()
        
        self.grid_spacing = grid_spacing
        self.k0 = 2 * np.pi / wavelength  # Wavenumber
        self.n0 = n0
        self.reduction = reduction
        
        # Create differential operators
        self._create_operators()
    
    def _create_operators(self):
        """Create finite difference operators for derivatives."""
        # Second-order central difference for Laplacian
        # ∇² = ∂²/∂x² + ∂²/∂y²
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32) / (self.grid_spacing**2)
        
        # Register as buffer (not trainable parameter)
        self.register_buffer('laplacian_kernel', laplacian_kernel.unsqueeze(0).unsqueeze(0))
    
    def compute_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D Laplacian using finite differences.
        
        Args:
            field: Complex field of shape (batch, 1, H, W)
            
        Returns:
            Laplacian of field
        """
        # Apply Laplacian to real and imaginary parts separately
        real_part = field.real
        imag_part = field.imag
        
        # Pad for convolution
        real_padded = F.pad(real_part, (1, 1, 1, 1), mode='reflect')
        imag_padded = F.pad(imag_part, (1, 1, 1, 1), mode='reflect')
        
        # Apply convolution
        laplacian_real = F.conv2d(real_padded, self.laplacian_kernel)
        laplacian_imag = F.conv2d(imag_padded, self.laplacian_kernel)
        
        return torch.complex(laplacian_real, laplacian_imag)
    
    def compute_z_derivative(self, field: torch.Tensor, dz: float = 100.0) -> torch.Tensor:
        """
        Compute ∂ψ/∂z using finite differences.
        
        For a single field snapshot, we approximate this as zero
        (steady-state assumption) or use a simple model.
        
        Args:
            field: Complex field
            dz: Propagation step size
            
        Returns:
            Z-derivative (approximated as zero for steady state)
        """
        # For steady-state, ∂ψ/∂z ≈ 0
        # In practice, this would require multiple z-slices or a propagation model
        return torch.zeros_like(field)
    
    def compute_refractive_index_perturbation(
        self, 
        params: torch.Tensor, 
        grid_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Compute refractive index perturbation δn from atmospheric parameters.
        
        This function now uses a more detailed atmospheric model to compute δn,
        consistent with the enhanced physics simulation.
        
        Args:
            params: Input parameters (batch, param_dim)
            grid_shape: Spatial grid shape (H, W)
            
        Returns:
            Refractive index perturbation δn
        """
        batch_size = params.shape[0]
        H, W = grid_shape
        
        # Extract parameters by their expected index after flattening the dictionary
        # Order: link_distance, visibility, temp_gradient, beam_waist, wavelength,
        #        pressure_hpa, temperature_celsius, humidity, altitude_tx_m, altitude_rx_m
        
        # Assuming the order of parameters in the input `params` tensor is:
        # 0: link_distance
        # 1: visibility
        # 2: temp_gradient
        # 3: beam_waist
        # 4: wavelength
        # 5: pressure_hpa
        # 6: temperature_celsius
        # 7: humidity
        # 8: altitude_tx_m
        # 9: altitude_rx_m

        visibility = params[:, 1]  # km
        temp_gradient = params[:, 2]  # K/m
        wavelength = params[:, 4] # m
        pressure_hpa = params[:, 5] # hPa
        temperature_celsius = params[:, 6] # Celsius
        humidity = params[:, 7] # 0-1
        altitude_tx_m = params[:, 8] # m
        altitude_rx_m = params[:, 9] # m

        # Calculate average link altitude for Cn^2 modeling
        link_altitude_m = (altitude_tx_m + altitude_rx_m) / 2
        
        # Initialize delta_n tensor
        delta_n = torch.zeros(batch_size, 1, H, W, device=params.device, dtype=torch.float32)
        
        for i in range(batch_size):
            # Create AtmosphericParameters for the current sample
            atm_params_sample = AtmosphericParameters(
                visibility=visibility[i].item(),
                temp_gradient=temp_gradient[i].item(),
                pressure_hpa=pressure_hpa[i].item(),
                temperature_celsius=temperature_celsius[i].item(),
                humidity=humidity[i].item()
            )
            
            # Create AtmosphericEffects instance for the current sample
            atm_effects_instance = AtmosphericEffects(
                atm_params=atm_params_sample,
                wavelength=wavelength[i].item(),
                link_altitude_m=link_altitude_m[i].item()
            )
            
            # Compute Cn^2
            cn_squared = atm_effects_instance.compute_cn_squared()

            # Turbulence contribution to delta_n
            # For PINO, we need a differentiable representation of turbulence.
            # A common simplification is to model delta_n as a random field
            # whose variance is related to Cn^2.
            # The magnitude of refractive index fluctuations (delta_n) is typically
            # on the order of 10^-6 to 10^-8.
            # We'll scale a random field by a factor derived from Cn^2.
            # This is still a simplification for differentiability.
            if cn_squared > 1e-18: # Only add noise if turbulence is significant
                # A more physically inspired scaling for delta_n from Cn^2
                # is complex. For a differentiable approximation, we can use
                # a scaling factor that roughly maps Cn^2 to delta_n magnitude.
                # Example: delta_n_rms ~ sqrt(Cn^2 * L^(1/3)) * C_const
                # For simplicity, let's use a direct scaling of random noise.
                # The constant 1e-4 is empirical to get realistic delta_n values.
                turbulence_delta_n_magnitude = torch.tensor(float(cn_squared), device=params.device, dtype=torch.float32).sqrt() * 1e-4
                noise = torch.randn(H, W, device=params.device, dtype=torch.float32) * turbulence_delta_n_magnitude
                delta_n[i, 0] += noise
            
            # Fog contribution to delta_n (real part of refractive index)
            # Fog primarily causes absorption (imaginary part of refractive index).
            # The real part change is usually very small.
            # For the PWE, we need a real part. This is a simplification for PINO.
            # We'll use a small, uniform real refractive index change related to visibility.
            # This is a very simplified representation for the PINO loss.
            # A more accurate approach would involve a complex refractive index in PWE.
            fog_delta_n_magnitude = 1e-8 / (visibility[i] + 1e-6) # Empirical scaling, very small
            delta_n[i, 0] += fog_delta_n_magnitude
            
        return delta_n
    
    def forward(
        self, 
        prediction: torch.Tensor, 
        params: torch.Tensor,
        collocation_points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute physics-informed loss.
        
        Args:
            prediction: Model prediction (batch, 2, H, W) - real and imaginary parts
            params: Input parameters (batch, param_dim)
            collocation_points: Optional collocation points for sampling
            
        Returns:
            Physics loss scalar
        """
        batch_size, _, H, W = prediction.shape
        
        # Convert prediction to complex field
        field_real = prediction[:, 0:1, :, :]  # (batch, 1, H, W)
        field_imag = prediction[:, 1:2, :, :]  # (batch, 1, H, W)
        field = torch.complex(field_real, field_imag)
        
        # Compute PWE terms
        
        # 1. ∂ψ/∂z term (approximated as zero for steady state)
        dpsidz = self.compute_z_derivative(field)
        term1 = 2j * self.k0 * dpsidz
        
        # 2. Transverse Laplacian ∇²_T ψ
        laplacian = self.compute_laplacian(field)
        term2 = laplacian
        
        # 3. Refractive index term 2k₀² [n/n₀ - 1] ψ
        delta_n = self.compute_refractive_index_perturbation(params, (H, W))
        delta_n_complex = torch.complex(delta_n, torch.zeros_like(delta_n))
        term3 = 2 * (self.k0**2) * (delta_n_complex / self.n0) * field
        
        # PWE residual
        residual = term1 + term2 + term3
        
        # Sample at collocation points if provided
        if collocation_points is not None:
            # Sample residual at specific points
            # This would require interpolation - simplified for now
            pass
        
        # Compute loss as L2 norm of residual
        if self.reduction == 'mean':
            loss = torch.mean(torch.abs(residual)**2)
        elif self.reduction == 'sum':
            loss = torch.sum(torch.abs(residual)**2)
        else:
            loss = torch.abs(residual)**2
        
        return loss


class PINOLoss(nn.Module):
    """
    Combined PINO loss function.
    
    Combines data loss and physics-informed loss with weighting.
    """
    
    def __init__(
        self,
        data_weight: float = 1.0,
        physics_weight: float = 0.1,
        grid_spacing: float = 1.0,
        wavelength: float = 1550e-9
    ):
        """
        Initialize PINO loss.
        
        Args:
            data_weight: Weight for data loss term
            physics_weight: Weight for physics loss term
            grid_spacing: Spatial grid spacing
            wavelength: Wavelength for physics loss
        """
        super(PINOLoss, self).__init__()
        
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        
        self.data_loss = DataLoss()
        self.physics_loss = PhysicsInformedLoss(
            grid_spacing=grid_spacing,
            wavelength=wavelength
        )
    
    def forward(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor,
        params: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined PINO loss.
        
        Args:
            prediction: Model prediction
            target: Ground truth target
            params: Input parameters
            
        Returns:
            Dictionary with individual and total losses
        """
        # Data loss
        data_loss = self.data_loss(prediction, target)
        
        # Physics loss
        physics_loss = self.physics_loss(prediction, params)
        
        # Total loss
        total_loss = self.data_weight * data_loss + self.physics_weight * physics_loss
        
        return {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss,
            'data_weight': self.data_weight,
            'physics_weight': self.physics_weight
        }
    
    def update_weights(self, data_weight: float, physics_weight: float):
        """Update loss weights during training."""
        self.data_weight = data_weight
        self.physics_weight = physics_weight
