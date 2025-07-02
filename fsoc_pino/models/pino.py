"""
Physics-Informed Neural Operator (PINO) for FSOC link prediction.

This module implements the main PINO model that combines the FNO architecture
with physics-informed constraints.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path

from .fno import FourierNeuralOperator
from .losses import PINOLoss


class PINO_FNO(nn.Module):
    """
    Physics-Informed Neural Operator based on Fourier Neural Operator.
    
    This model learns to predict FSOC link performance while respecting
    the underlying physics (Parabolic Wave Equation).
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 2,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        num_layers: int = 4,
        input_resolution: int = 128,
        grid_spacing: float = 1.0,
        wavelength: float = 1550e-9
    ):
        """
        Initialize PINO model.
        
        Args:
            input_dim: Number of input parameters (link_distance, wavelength, etc.)
            output_dim: Number of output channels (2 for real/imag parts)
            modes1: Fourier modes in first spatial dimension
            modes2: Fourier modes in second spatial dimension
            width: Hidden layer width
            num_layers: Number of FNO layers
            input_resolution: Spatial resolution of output grid
            grid_spacing: Physical grid spacing in meters
            wavelength: Reference wavelength for physics loss
        """
        super(PINO_FNO, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.grid_spacing = grid_spacing
        self.wavelength = wavelength
        
        # Core FNO model
        self.fno = FourierNeuralOperator(
            input_dim=input_dim,
            output_dim=output_dim,
            modes1=modes1,
            modes2=modes2,
            width=width,
            num_layers=num_layers,
            input_resolution=input_resolution
        )
        
        # Parameter normalization (will be set during training)
        self.register_buffer('param_mean', torch.zeros(input_dim))
        self.register_buffer('param_std', torch.ones(input_dim))
        self.param_normalized = False
        
    def normalize_parameters(self, params: torch.Tensor) -> torch.Tensor:
        """Normalize input parameters."""
        if self.param_normalized:
            return (params - self.param_mean) / (self.param_std + 1e-8)
        return params
    
    def denormalize_parameters(self, params_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize parameters."""
        if self.param_normalized:
            return params_norm * self.param_std + self.param_mean
        return params_norm
    
    def set_parameter_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """Set parameter normalization statistics."""
        self.param_mean.copy_(mean)
        self.param_std.copy_(std)
        self.param_normalized = True
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PINO model.
        
        Args:
            params: Input parameters of shape (batch, input_dim)
            
        Returns:
            Complex field as (batch, 2, H, W) where channels are [real, imag]
        """
        # Normalize parameters
        params_norm = self.normalize_parameters(params)
        
        # Forward through FNO
        output = self.fno(params_norm)
        
        return output
    
    def predict_irradiance(self, params: torch.Tensor) -> torch.Tensor:
        """
        Predict irradiance pattern |ψ|².
        
        Args:
            params: Input parameters
            
        Returns:
            Irradiance pattern of shape (batch, 1, H, W)
        """
        field = self.forward(params)
        real_part = field[:, 0:1, :, :]
        imag_part = field[:, 1:2, :, :]
        irradiance = real_part**2 + imag_part**2
        return irradiance
    
    def predict_complex_field(self, params: torch.Tensor) -> torch.Tensor:
        """
        Predict complex field ψ = real + i*imag.
        
        Args:
            params: Input parameters
            
        Returns:
            Complex field tensor
        """
        field = self.forward(params)
        real_part = field[:, 0, :, :]
        imag_part = field[:, 1, :, :]
        return torch.complex(real_part, imag_part)
    
    def compute_derived_metrics(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute derived metrics from field prediction.
        
        Args:
            params: Input parameters
            
        Returns:
            Dictionary with computed metrics
        """
        irradiance = self.predict_irradiance(params)
        
        # Scintillation index
        mean_I = torch.mean(irradiance, dim=(2, 3), keepdim=True)
        var_I = torch.var(irradiance, dim=(2, 3), keepdim=True)
        scintillation_index = var_I / (mean_I**2 + 1e-8)
        
        # Total power
        total_power = torch.sum(irradiance, dim=(2, 3))
        
        # Peak irradiance
        peak_irradiance = torch.max(irradiance.view(irradiance.shape[0], -1), dim=1)[0]
        
        return {
            'irradiance': irradiance,
            'scintillation_index': scintillation_index.squeeze(),
            'total_power': total_power,
            'peak_irradiance': peak_irradiance
        }
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        info = self.fno.get_model_info()
        info.update({
            'model_type': 'PINO_FNO',
            'grid_spacing': self.grid_spacing,
            'wavelength': self.wavelength,
            'parameter_normalized': self.param_normalized
        })
        return info
    
    def save(self, filepath: Path):
        """Save model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.get_model_info(),
            'param_mean': self.param_mean,
            'param_std': self.param_std,
            'param_normalized': self.param_normalized
        }, filepath)
    
    @classmethod
    def load(cls, filepath: Path, device: str = 'cpu') -> 'PINO_FNO':
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['model_config']
        
        # Create model
        model = cls(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            modes1=config['modes1'],
            modes2=config['modes2'],
            width=config['width'],
            num_layers=config['num_layers'],
            input_resolution=config['input_resolution'],
            grid_spacing=config.get('grid_spacing', 1.0),
            wavelength=config.get('wavelength', 1550e-9)
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load normalization
        if checkpoint.get('param_normalized', False):
            model.set_parameter_normalization(
                checkpoint['param_mean'],
                checkpoint['param_std']
            )
        
        model.to(device)
        model.eval()
        
        return model
    
    def export_onnx(self, filepath: Path, example_input: Optional[torch.Tensor] = None,
                    use_dynamo: bool = True, fallback_to_torchscript: bool = True):
        """
        Export model to ONNX format for deployment.

        Args:
            filepath: Path to save the ONNX model
            example_input: Example input tensor for tracing
            use_dynamo: Whether to use the new dynamo-based exporter
            fallback_to_torchscript: Whether to fallback to TorchScript if ONNX fails

        Returns:
            str: The actual export format used ('onnx' or 'torchscript')
        """
        if example_input is None:
            # Create dummy input
            example_input = torch.randn(1, self.input_dim)

        # Set to evaluation mode
        self.eval()

        # Try ONNX export first
        onnx_success = False

        if use_dynamo:
            print("Attempting ONNX export with dynamo=True...")
            try:
                torch.onnx.export(
                    self,
                    example_input,
                    filepath,
                    export_params=True,
                    opset_version=21,  # Use newer opset with dynamo
                    do_constant_folding=True,
                    input_names=['parameters'],
                    output_names=['field'],
                    dynamo=True
                )
                print(f"✅ Model exported to ONNX (dynamo): {filepath}")
                onnx_success = True
                return 'onnx'
            except Exception as e:
                print(f"❌ ONNX export with dynamo failed: {str(e)}")

        # Try traditional ONNX export with higher opset
        if not onnx_success:
            print("Attempting ONNX export with traditional method...")
            for opset_version in [20, 19, 18, 17, 16, 15]:
                try:
                    torch.onnx.export(
                        self,
                        example_input,
                        filepath,
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=True,
                        input_names=['parameters'],
                        output_names=['field'],
                        dynamic_axes={
                            'parameters': {0: 'batch_size'},
                            'field': {0: 'batch_size'}
                        }
                    )
                    print(f"✅ Model exported to ONNX (opset {opset_version}): {filepath}")
                    onnx_success = True
                    return 'onnx'
                except Exception as e:
                    print(f"❌ ONNX export with opset {opset_version} failed: {str(e)}")
                    continue

        # Fallback to TorchScript if ONNX fails
        if not onnx_success and fallback_to_torchscript:
            print("⚠️  ONNX export failed. Falling back to TorchScript export...")
            return self.export_torchscript(filepath.with_suffix('.pt'), example_input)

        if not onnx_success:
            raise RuntimeError("All ONNX export methods failed and TorchScript fallback is disabled")

    def export_torchscript(self, filepath: Path, example_input: Optional[torch.Tensor] = None):
        """
        Export model to TorchScript format for deployment.

        Args:
            filepath: Path to save the TorchScript model
            example_input: Example input tensor for tracing

        Returns:
            str: The export format used ('torchscript')
        """
        if example_input is None:
            # Create dummy input
            example_input = torch.randn(1, self.input_dim)

        # Set to evaluation mode
        self.eval()

        try:
            # Trace the model
            traced_model = torch.jit.trace(self, example_input)

            # Save the traced model
            traced_model.save(str(filepath))

            print(f"✅ Model exported to TorchScript: {filepath}")
            return 'torchscript'

        except Exception as e:
            raise RuntimeError(f"TorchScript export failed: {str(e)}")


def create_pino_model(
    input_dim: int = 5,
    grid_size: int = 128,
    modes: int = 12,
    width: int = 64,
    num_layers: int = 4,
    device: str = 'cpu'
) -> PINO_FNO:
    """
    Create a PINO model with standard configuration.
    
    Args:
        input_dim: Number of input parameters
        grid_size: Spatial grid resolution
        modes: Number of Fourier modes
        width: Hidden layer width
        num_layers: Number of FNO layers
        device: Device to place model on
        
    Returns:
        Initialized PINO model
    """
    model = PINO_FNO(
        input_dim=input_dim,
        output_dim=2,  # Real and imaginary parts
        modes1=modes,
        modes2=modes,
        width=width,
        num_layers=num_layers,
        input_resolution=grid_size
    )
    
    model.to(device)
    return model
