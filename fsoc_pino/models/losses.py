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
        
        Args:
            params: Atmospheric parameters (batch, param_dim)
            grid_shape: Spatial grid shape (H, W)
            
        Returns:
            Refractive index perturbation δn
        """
        batch_size = params.shape[0]
        H, W = grid_shape
        
        # Extract atmospheric parameters
        # Assuming params = [link_distance, wavelength, beam_waist, visibility, temp_gradient]
        visibility = params[:, 3]  # km
        temp_gradient = params[:, 4]  # K/m
        
        # Simplified model for δn based on atmospheric conditions
        # In practice, this would be more sophisticated
        
        # Fog contribution (uniform across grid)
        fog_factor = 1.0 / (visibility + 1e-6)  # Avoid division by zero
        
        # Turbulence contribution (random but correlated with temp_gradient)
        turbulence_strength = temp_gradient * 1e-6
        
        # Create spatial perturbation
        delta_n = torch.zeros(batch_size, 1, H, W, device=params.device)
        
        for i in range(batch_size):
            # Uniform fog contribution
            delta_n[i, 0] += fog_factor[i] * 1e-6
            
            # Random turbulence (simplified)
            if turbulence_strength[i] > 0:
                noise = torch.randn(H, W, device=params.device) * turbulence_strength[i]
                delta_n[i, 0] += noise
        
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
