"""
Dataset generation command for FSOC-PINO CLI.

This module implements the 'generate-dataset' subcommand that creates training
datasets by running physics simulations across a parameter space.
"""

import click
import time
from pathlib import Path
from typing import Tuple


@click.command()
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for generated dataset"
)
@click.option(
    "--num-samples", "-n",
    type=int,
    default=1000,
    help="Number of simulation samples to generate"
)
@click.option(
    "--link-distance-range",
    type=(float, float),
    default=(1.0, 5.0),
    help="Link distance range in km (min, max)"
)
@click.option(
    "--visibility-range", 
    type=(float, float),
    default=(0.5, 10.0),
    help="Meteorological visibility range in km (min, max)"
)
@click.option(
    "--temp-gradient-range",
    type=(float, float), 
    default=(0.01, 0.2),
    help="Temperature gradient range in K/m (min, max)"
)
@click.option(
    "--beam-waist-range",
    type=(float, float),
    default=(0.02, 0.10),
    help="Initial beam waist range in m (min, max)"
)
@click.option(
    "--wavelength-range",
    type=(float, float),
    default=(850e-9, 1550e-9),
    help="Wavelength range in m (min, max)"
)
@click.option(
    "--pressure-hpa-range",
    type=(float, float),
    default=(950.0, 1050.0),
    help="Atmospheric pressure range in hPa (min, max)"
)
@click.option(
    "--temperature-celsius-range",
    type=(float, float),
    default=(0.0, 30.0),
    help="Temperature range in Celsius (min, max)"
)
@click.option(
    "--humidity-range",
    type=(float, float),
    default=(0.2, 0.9),
    help="Relative humidity range (0-1) (min, max)"
)
@click.option(
    "--altitude-tx-m-range",
    type=(float, float),
    default=(0.0, 100.0),
    help="Transmitter altitude range in m (min, max)"
)
@click.option(
    "--altitude-rx-m-range",
    type=(float, float),
    default=(0.0, 100.0),
    help="Receiver altitude range in m (min, max)"
)
@click.option(
    "--grid-size",
    type=int,
    default=128,
    help="Spatial grid size for simulation"
)
@click.option(
    "--parallel-jobs", "-j",
    type=int,
    default=1,
    help="Number of parallel simulation jobs"
)
@click.option(
    "--sampling-method",
    type=click.Choice(["uniform", "latin_hypercube", "sobol"]),
    default="latin_hypercube",
    help="Parameter sampling method"
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Batch size for saving data"
)
@click.pass_context
def generate_dataset(
    ctx,
    output_dir: Path,
    num_samples: int,
    link_distance_range: Tuple[float, float],
    visibility_range: Tuple[float, float],
    temp_gradient_range: Tuple[float, float],
    beam_waist_range: Tuple[float, float],
    wavelength_range: Tuple[float, float],
    pressure_hpa_range: Tuple[float, float],
    temperature_celsius_range: Tuple[float, float],
    humidity_range: Tuple[float, float],
    altitude_tx_m_range: Tuple[float, float],
    altitude_rx_m_range: Tuple[float, float],
    grid_size: int,
    parallel_jobs: int,
    sampling_method: str,
    batch_size: int
):
    """
    Generate training dataset from physics simulations.
    
    This command runs FSOC link simulations across a specified parameter space
    and saves the results in HDF5 format for training PINO models.
    """
    logger = ctx.obj['logger']
    logger.info("Starting dataset generation...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log parameters
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Link distance range: {link_distance_range} km")
    logger.info(f"Visibility range: {visibility_range} km")
    logger.info(f"Temperature gradient range: {temp_gradient_range} K/m")
    logger.info(f"Beam waist range: {beam_waist_range} m")
    logger.info(f"Wavelength range: {wavelength_range} m")
    logger.info(f"Pressure range: {pressure_hpa_range} hPa")
    logger.info(f"Temperature range: {temperature_celsius_range} °C")
    logger.info(f"Humidity range: {humidity_range}")
    logger.info(f"Transmitter altitude range: {altitude_tx_m_range} m")
    logger.info(f"Receiver altitude range: {altitude_rx_m_range} m")
    logger.info(f"Grid size: {grid_size}")
    logger.info(f"Sampling method: {sampling_method}")
    logger.info(f"Parallel jobs: {parallel_jobs}")
    
    try:
        # Import required modules
        from fsoc_pino.data import DatasetGenerator, ParameterSampler
        from fsoc_pino.simulation import FSOC_Simulator

        # Create parameter sampler with correct order for SimulationConfig
        sampler = ParameterSampler(
            method=sampling_method,
            ranges={
                'link_distance': link_distance_range,
                'wavelength': wavelength_range,
                'beam_waist': beam_waist_range,
                'visibility': visibility_range,
                'temp_gradient': temp_gradient_range,
                'pressure_hpa': pressure_hpa_range,
                'temperature_celsius': temperature_celsius_range,
                'humidity': humidity_range,
                'altitude_tx_m': altitude_tx_m_range,
                'altitude_rx_m': altitude_rx_m_range
            }
        )

        # Generate parameter samples
        logger.info("Generating parameter samples...")
        parameters = sampler.sample_dict(num_samples)

        # Create dataset generator
        generator = DatasetGenerator(
            output_dir=output_dir,
            grid_size=grid_size,
            batch_size=batch_size,
            parallel_jobs=parallel_jobs
        )

        # Generate dataset
        logger.info("Running simulations...")
        start_time = time.time()

        dataset_info = generator.generate(parameters)

        elapsed_time = time.time() - start_time
        logger.info(f"Dataset generation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Generated {dataset_info['num_samples']} samples")
        logger.info(f"Dataset saved to: {dataset_info['output_files']}")

        # Save metadata
        metadata_file = output_dir / "metadata.json"
        generator.save_metadata(metadata_file, {
            'num_samples': num_samples,
            'parameter_ranges': {
                'link_distance': link_distance_range,
                'visibility': visibility_range,
                'temp_gradient': temp_gradient_range,
                'beam_waist': beam_waist_range,
                'wavelength': wavelength_range,
                'pressure_hpa': pressure_hpa_range,
                'temperature_celsius': temperature_celsius_range,
                'humidity': humidity_range,
                'altitude_tx_m': altitude_tx_m_range,
                'altitude_rx_m': altitude_rx_m_range
            },
            'grid_size': grid_size,
            'sampling_method': sampling_method,
            'generation_time': elapsed_time
        })

        logger.info(f"Dataset generation completed successfully!")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Generated {dataset_info['num_samples']} samples")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Success rate: {dataset_info['success_rate']*100:.1f}%")

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        ctx.exit(1)
