"""
Fourier Neural Operator (FNO) implementation.

This module implements the FNO architecture for learning mappings between
function spaces, as described in Li et al. (2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer for FNO.
    
    Performs convolution in Fourier space by:
    1. FFT of input
    2. Pointwise multiplication with learnable weights
    3. Inverse FFT
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        """
        Initialize spectral convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
        """
        super(SpectralConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply
        self.modes2 = modes2
        
        self.scale = (1 / (in_channels * out_channels))
        
        # Learnable weights for Fourier modes
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication for 2D tensors."""
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spectral convolution.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Output tensor of same shape
        """
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
            dtype=torch.cfloat, device=x.device
        )
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x


class FNOBlock(nn.Module):
    """
    Single FNO block consisting of spectral convolution and local convolution.
    """
    
    def __init__(self, width: int, modes1: int, modes2: int):
        """
        Initialize FNO block.
        
        Args:
            width: Channel width
            modes1: Fourier modes in first dimension
            modes2: Fourier modes in second dimension
        """
        super(FNOBlock, self).__init__()
        
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)  # Local convolution (1x1 conv)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNO block."""
        x1 = self.conv(x)
        x2 = self.w(x)
        return F.gelu(x1 + x2)


class FourierNeuralOperator(nn.Module):
    """
    Fourier Neural Operator for learning mappings between function spaces.
    
    Architecture:
    1. Lifting: Project input to higher dimensional space
    2. FNO layers: Learn in Fourier space
    3. Projection: Project back to output space
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        num_layers: int = 4,
        input_resolution: int = 128
    ):
        """
        Initialize FNO.
        
        Args:
            input_dim: Dimension of input parameters
            output_dim: Dimension of output (e.g., 2 for real/imag parts)
            modes1: Number of Fourier modes in first spatial dimension
            modes2: Number of Fourier modes in second spatial dimension
            width: Width of hidden layers
            num_layers: Number of FNO layers
            input_resolution: Spatial resolution of input/output grids
        """
        super(FourierNeuralOperator, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.input_resolution = input_resolution
        
        # Lifting layer: project input parameters to spatial function
        self.fc0 = nn.Linear(input_dim + 2, self.width)  # +2 for spatial coordinates
        
        # FNO layers
        self.fno_layers = nn.ModuleList([
            FNOBlock(self.width, self.modes1, self.modes2) 
            for _ in range(self.num_layers)
        ])
        
        # Projection layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_dim)
        
        # Create coordinate grid
        self.register_buffer('grid', self._get_grid())
    
    def _get_grid(self) -> torch.Tensor:
        """Create coordinate grid for spatial encoding."""
        # Create normalized coordinate grid [-1, 1] x [-1, 1]
        x = torch.linspace(-1, 1, self.input_resolution)
        y = torch.linspace(-1, 1, self.input_resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([X, Y], dim=-1)  # Shape: (H, W, 2)
        return grid
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FNO.
        
        Args:
            params: Input parameters of shape (batch, input_dim)
            
        Returns:
            Output field of shape (batch, output_dim, height, width)
        """
        batch_size = params.shape[0]
        
        # Expand parameters to spatial grid
        # params: (batch, input_dim) -> (batch, input_dim, H, W)
        params_expanded = params.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, self.input_dim, self.input_resolution, self.input_resolution
        )
        
        # Add spatial coordinates
        # grid: (H, W, 2) -> (batch, 2, H, W)
        grid_expanded = self.grid.permute(2, 0, 1).unsqueeze(0).expand(
            batch_size, 2, self.input_resolution, self.input_resolution
        )
        
        # Concatenate parameters and coordinates
        # Shape: (batch, input_dim + 2, H, W)
        x = torch.cat([params_expanded, grid_expanded], dim=1)
        
        # Lifting: (batch, input_dim + 2, H, W) -> (batch, width, H, W)
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, input_dim + 2)
        x = self.fc0(x)  # (batch, H, W, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, H, W)
        
        # FNO layers
        for layer in self.fno_layers:
            x = layer(x)
        
        # Projection: (batch, width, H, W) -> (batch, output_dim, H, W)
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, width)
        x = F.gelu(self.fc1(x))  # (batch, H, W, 128)
        x = self.fc2(x)  # (batch, H, W, output_dim)
        x = x.permute(0, 3, 1, 2)  # (batch, output_dim, H, W)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'modes1': self.modes1,
            'modes2': self.modes2,
            'width': self.width,
            'num_layers': self.num_layers,
            'input_resolution': self.input_resolution,
            'total_parameters': self.count_parameters()
        }
