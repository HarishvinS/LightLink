#!/usr/bin/env python3
"""
Demo script to test the FSOC simulation functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from fsoc_pino import FSOC_Simulator, SimulationConfig


def main():
    """Run a demonstration of the FSOC simulation."""
    print("=== FSOC-PINO Simulation Demo ===\n")
    
    # Create simulation configuration
    config = SimulationConfig(
        link_distance=2.0,  # 2 km link
        wavelength=1550e-9,  # 1550 nm
        beam_waist=0.05,  # 5 cm beam waist
        visibility=3.0,  # 3 km visibility (moderate fog)
        temp_gradient=0.05,  # 0.05 K/m temperature gradient
        grid_size=64,  # 64x64 grid
        grid_width=0.5  # 0.5 m grid width
    )
    
    print("Simulation Configuration:")
    print(f"  Link distance: {config.link_distance} km")
    print(f"  Wavelength: {config.wavelength*1e9:.0f} nm")
    print(f"  Beam waist: {config.beam_waist*100:.1f} cm")
    print(f"  Visibility: {config.visibility} km")
    print(f"  Temperature gradient: {config.temp_gradient} K/m")
    print(f"  Grid size: {config.grid_size}x{config.grid_size}")
    print()
    
    # Create and run simulator
    simulator = FSOC_Simulator(config)
    result = simulator.run_simulation()
    
    print("\nSimulation Results:")
    print(f"  Scintillation index: {result.scintillation_index:.6f}")
    print(f"  Bit error rate: {result.bit_error_rate:.2e}")
    print(f"  Computation time: {result.propagation_time:.3f} seconds")
    
    # Analyze link budget
    link_budget = simulator.analyze_link_budget(result)
    print(f"\nLink Budget Analysis:")
    print(f"  Fog loss: {link_budget['fog_loss_db']:.2f} dB")
    print(f"  Beam spreading loss: {link_budget['beam_spreading_loss_db']:.2f} dB")
    print(f"  Total loss: {link_budget['total_loss_db']:.2f} dB")
    print(f"  Received power fraction: {link_budget['received_power_fraction']:.4f}")
    print(f"  Link margin: {link_budget['link_margin_db']:.2f} dB")
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    # Plot irradiance map
    plt.subplot(1, 3, 1)
    extent = [-config.grid_width/2, config.grid_width/2, 
              -config.grid_width/2, config.grid_width/2]
    plt.imshow(result.irradiance, extent=extent, origin='lower', cmap='hot')
    plt.colorbar(label='Irradiance')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Received Irradiance Pattern')
    
    # Plot cross-section
    plt.subplot(1, 3, 2)
    center_idx = config.grid_size // 2
    x_coords = np.linspace(-config.grid_width/2, config.grid_width/2, config.grid_size)
    plt.plot(x_coords, result.irradiance[center_idx, :], 'b-', linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('Irradiance')
    plt.title('Cross-section through Center')
    plt.grid(True, alpha=0.3)
    
    # Plot beam parameters
    plt.subplot(1, 3, 3)
    beam_params = simulator.ssfm.compute_beam_parameters(result.final_field)
    
    params = ['Total Power', 'Peak Irradiance', 'Beam Width X', 'Beam Width Y']
    values = [
        beam_params['total_power'],
        beam_params['peak_irradiance'],
        beam_params['beam_width_x'],
        beam_params['beam_width_y']
    ]
    
    plt.bar(range(len(params)), values)
    plt.xticks(range(len(params)), params, rotation=45)
    plt.ylabel('Value')
    plt.title('Beam Parameters')
    
    plt.tight_layout()
    plt.savefig('fsoc_simulation_demo.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: fsoc_simulation_demo.png")
    
    # Save results
    output_dir = Path("demo_results")
    simulator.save_results(result, output_dir)
    
    print(f"\nDemo completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
