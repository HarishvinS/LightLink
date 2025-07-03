"""
Visualization utilities for FSOC-PINO.

This module provides visualization functions for:
- Irradiance maps and field distributions
- Training metrics and loss curves
- Model performance comparisons
- Atmospheric effects visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import seaborn as sns


def plot_irradiance_map(
    irradiance: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Irradiance Map",
    grid_spacing: float = 1.0,
    colormap: str = "hot",
    show_colorbar: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 150,
    log_scale: bool = True,
    beam_waist_circle: Optional[float] = None
) -> None:
    """
    Plot 2D irradiance map with proper scaling and visualization.
    
    Args:
        irradiance: 2D array of irradiance values
        save_path: Optional path to save the plot
        title: Plot title
        grid_spacing: Physical grid spacing in meters
        colormap: Matplotlib colormap name
        show_colorbar: Whether to show colorbar
        figsize: Figure size (width, height)
        dpi: Figure DPI for saving
        log_scale: Whether to use logarithmic color scale
        beam_waist_circle: Optional beam waist radius to overlay as circle
    """
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Create spatial coordinates
    ny, nx = irradiance.shape
    x = np.linspace(-nx//2, nx//2, nx) * grid_spacing
    y = np.linspace(-ny//2, ny//2, ny) * grid_spacing
    X, Y = np.meshgrid(x, y)
    
    # Handle log scale
    if log_scale and np.any(irradiance > 0):
        # Avoid log(0) by adding small epsilon
        irradiance_plot = np.log10(irradiance + 1e-12)
        cbar_label = "Log₁₀(Irradiance) [W/m²]"
    else:
        irradiance_plot = irradiance
        cbar_label = "Irradiance [W/m²]"
    
    # Create the plot
    im = plt.pcolormesh(X, Y, irradiance_plot, cmap=colormap, shading='auto')
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=12)
    
    # Add beam waist circle if specified
    if beam_waist_circle is not None:
        circle = Circle((0, 0), beam_waist_circle, fill=False, 
                       color='white', linewidth=2, linestyle='--', alpha=0.8)
        plt.gca().add_patch(circle)
    
    # Formatting
    plt.xlabel("X [m]", fontsize=12)
    plt.ylabel("Y [m]", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"Peak: {np.max(irradiance):.2e} W/m²\n"
    stats_text += f"Mean: {np.mean(irradiance):.2e} W/m²\n"
    stats_text += f"Total Power: {np.sum(irradiance) * grid_spacing**2:.2e} W"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Irradiance map saved to: {save_path}")
    
    plt.show()


def plot_field_components(
    field: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Field Components",
    grid_spacing: float = 1.0,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot real, imaginary, and magnitude components of complex field.
    
    Args:
        field: Complex 2D field array
        save_path: Optional path to save the plot
        title: Plot title
        grid_spacing: Physical grid spacing in meters
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Create spatial coordinates
    ny, nx = field.shape
    x = np.linspace(-nx//2, nx//2, nx) * grid_spacing
    y = np.linspace(-ny//2, ny//2, ny) * grid_spacing
    X, Y = np.meshgrid(x, y)
    
    # Real part
    im1 = axes[0].pcolormesh(X, Y, np.real(field), cmap='RdBu_r', shading='auto')
    axes[0].set_title("Real Part")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Imaginary part
    im2 = axes[1].pcolormesh(X, Y, np.imag(field), cmap='RdBu_r', shading='auto')
    axes[1].set_title("Imaginary Part")
    axes[1].set_xlabel("X [m]")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # Magnitude
    im3 = axes[2].pcolormesh(X, Y, np.abs(field), cmap='hot', shading='auto')
    axes[2].set_title("Magnitude")
    axes[2].set_xlabel("X [m]")
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Field components plot saved to: {save_path}")
    
    plt.show()


def plot_training_metrics(
    metrics_dict: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot training metrics including losses and validation scores.
    
    Args:
        metrics_dict: Dictionary containing training metrics
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Training and validation loss
    if 'train_loss' in metrics_dict and 'val_loss' in metrics_dict:
        axes[0, 0].plot(metrics_dict['train_loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(metrics_dict['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
    
    # Physics loss component
    if 'physics_loss' in metrics_dict:
        axes[0, 1].plot(metrics_dict['physics_loss'], label='Physics Loss', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Physics Loss')
        axes[0, 1].set_title('Physics Constraint')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
    
    # Learning rate
    if 'learning_rate' in metrics_dict:
        axes[1, 0].plot(metrics_dict['learning_rate'], label='Learning Rate', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Validation metrics
    if 'val_accuracy' in metrics_dict:
        axes[1, 1].plot(metrics_dict['val_accuracy'], label='Validation Accuracy', color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Validation Performance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training metrics plot saved to: {save_path}")
    
    plt.show()


def plot_prediction_comparison(
    simulation_result: np.ndarray,
    prediction_result: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Simulation vs Prediction",
    grid_spacing: float = 1.0,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Compare simulation and prediction results side by side.
    
    Args:
        simulation_result: Ground truth simulation irradiance
        prediction_result: Model prediction irradiance
        save_path: Optional path to save the plot
        title: Plot title
        grid_spacing: Physical grid spacing in meters
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Create spatial coordinates
    ny, nx = simulation_result.shape
    x = np.linspace(-nx//2, nx//2, nx) * grid_spacing
    y = np.linspace(-ny//2, ny//2, ny) * grid_spacing
    X, Y = np.meshgrid(x, y)
    
    # Determine common color scale
    vmin = min(np.min(simulation_result), np.min(prediction_result))
    vmax = max(np.max(simulation_result), np.max(prediction_result))
    
    # Simulation
    im1 = axes[0].pcolormesh(X, Y, simulation_result, cmap='hot', 
                            vmin=vmin, vmax=vmax, shading='auto')
    axes[0].set_title("Simulation (Ground Truth)")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Prediction
    im2 = axes[1].pcolormesh(X, Y, prediction_result, cmap='hot',
                            vmin=vmin, vmax=vmax, shading='auto')
    axes[1].set_title("PINO Prediction")
    axes[1].set_xlabel("X [m]")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # Difference
    difference = np.abs(simulation_result - prediction_result)
    im3 = axes[2].pcolormesh(X, Y, difference, cmap='viridis', shading='auto')
    axes[2].set_title("Absolute Difference")
    axes[2].set_xlabel("X [m]")
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    # Add error statistics
    mse = np.mean((simulation_result - prediction_result)**2)
    mae = np.mean(np.abs(simulation_result - prediction_result))
    relative_error = np.mean(np.abs(simulation_result - prediction_result) / 
                           (np.abs(simulation_result) + 1e-12)) * 100
    
    error_text = f"MSE: {mse:.2e}\nMAE: {mae:.2e}\nRel. Error: {relative_error:.2f}%"
    fig.text(0.02, 0.02, error_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def plot_benchmark_results(
    results: Dict[str, Any],
    output_dir: Path,
    figsize: Tuple[int, int] = (15, 10),
    dpi: int = 150
) -> None:
    """
    Generate comprehensive benchmark plots comparing PINO predictions to simulations.

    Args:
        results: Dictionary containing benchmark results
        output_dir: Directory to save plots
        figsize: Figure size (width, height)
        dpi: Figure DPI for saving
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Inference time comparison
    if 'pino_inference_times' in results and 'simulation_times' in results:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(results['pino_inference_times'], bins=20, alpha=0.7, label='PINO', color='blue')
        plt.xlabel('Inference Time (s)')
        plt.ylabel('Frequency')
        plt.title('PINO Inference Time Distribution')
        plt.legend()

        plt.subplot(1, 2, 2)
        if results['simulation_times']:
            plt.hist(results['simulation_times'], bins=20, alpha=0.7, label='Simulation', color='red')
            plt.xlabel('Simulation Time (s)')
            plt.ylabel('Frequency')
            plt.title('Physics Simulation Time Distribution')
            plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "timing_comparison.png", dpi=dpi, bbox_inches='tight')
        plt.close()

    # Plot 2: Accuracy metrics
    if 'metrics' in results and results['metrics']:
        metrics = results['metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen'][:len(metric_names)])
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Benchmark Accuracy Metrics')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_metrics.png", dpi=dpi, bbox_inches='tight')
        plt.close()

    print(f"Benchmark plots saved to {output_dir}")
