"""
Prediction command for FSOC-PINO CLI.

This module implements the 'predict' subcommand that uses trained PINO models
for fast inference on new parameter combinations.
"""

import click
import time
from pathlib import Path
from typing import Optional


@click.command()
@click.option(
    "--model-path", "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model (.pth or .onnx)"
)
@click.option(
    "--link-distance",
    type=float,
    required=True,
    help="Link distance in km"
)
@click.option(
    "--visibility",
    type=float,
    required=True,
    help="Meteorological visibility in km"
)
@click.option(
    "--temp-gradient",
    type=float,
    required=True,
    help="Temperature gradient in K/m"
)
@click.option(
    "--beam-waist",
    type=float,
    default=0.05,
    help="Initial beam waist in m"
)
@click.option(
    "--wavelength",
    type=float,
    default=1550e-9,
    help="Wavelength in m"
)
@click.option(
    "--pressure-hpa",
    type=float,
    default=1013.25,
    help="Atmospheric pressure in hPa"
)
@click.option(
    "--temperature-celsius",
    type=float,
    default=15.0,
    help="Atmospheric temperature in Celsius"
)
@click.option(
    "--humidity",
    type=float,
    default=0.5,
    help="Relative humidity (0-1)"
)
@click.option(
    "--altitude-tx-m",
    type=float,
    default=0.0,
    help="Transmitter altitude in meters"
)
@click.option(
    "--altitude-rx-m",
    type=float,
    default=0.0,
    help="Receiver altitude in meters"
)
@click.option(
    "--output-file", "-o",
    type=click.Path(path_type=Path),
    help="Output file for prediction results (.h5 or .npz)"
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    help="Device to use for inference"
)
@click.option(
    "--grid-size",
    type=int,
    default=64,
    help="Output grid size (must match model training grid size)"
)
@click.option(
    "--compute-metrics",
    is_flag=True,
    help="Compute derived metrics (BER, scintillation index)"
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Generate visualization plots"
)
@click.pass_context
def predict(
    ctx,
    model_path: Path,
    link_distance: float,
    visibility: float,
    temp_gradient: float,
    beam_waist: float,
    wavelength: float,
    pressure_hpa: float,
    temperature_celsius: float,
    humidity: float,
    altitude_tx_m: float,
    altitude_rx_m: float,
    output_file: Optional[Path],
    device: str,
    grid_size: int,
    compute_metrics: bool,
    visualize: bool
):
    """
    Make predictions using trained PINO model.
    
    This command uses a trained PINO model to predict FSOC link performance
    for specified atmospheric and link parameters.
    """
    logger = ctx.obj['logger']
    logger.info("Starting PINO prediction...")
    
    # Log parameters
    logger.info(f"Model path: {model_path}")
    logger.info(f"Link distance: {link_distance} km")
    logger.info(f"Visibility: {visibility} km")
    logger.info(f"Temperature gradient: {temp_gradient} K/m")
    logger.info(f"Beam waist: {beam_waist} m")
    logger.info(f"Wavelength: {wavelength} m")
    logger.info(f"Pressure: {pressure_hpa} hPa")
    logger.info(f"Temperature: {temperature_celsius} °C")
    logger.info(f"Humidity: {humidity}")
    logger.info(f"Transmitter altitude: {altitude_tx_m} m")
    logger.info(f"Receiver altitude: {altitude_rx_m} m")
    logger.info(f"Grid size: {grid_size}")
    
    try:
        # Import required modules
        from fsoc_pino.models import PINO_FNO
        from fsoc_pino.utils.visualization import plot_irradiance_map
        from fsoc_pino.simulation.physics import AtmosphericEffects, LinkParameters, AtmosphericParameters
        from fsoc_pino.simulation.ssfm import SplitStepFourierMethod
        import torch
        import numpy as np

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load model
        logger.info("Loading model...")
        if model_path.suffix == ".onnx":
            # Load ONNX model for edge deployment
            import onnxruntime as ort
            session = ort.InferenceSession(str(model_path))
            model_type = "onnx"
        elif model_path.suffix == ".pt":
            # Load TorchScript model
            model = torch.jit.load(str(model_path), map_location=device)
            model.eval()
            model_type = "torchscript"
        else:
            # Load PyTorch model
            model = PINO_FNO.load(model_path, device=device)
            model_type = "pytorch"

        # Prepare input parameters (must match the order used in training)
        input_params = np.array([
            link_distance,
            wavelength,
            beam_waist,
            visibility,
            temp_gradient,
            pressure_hpa,
            temperature_celsius,
            humidity,
            altitude_tx_m,
            altitude_rx_m
        ], dtype=np.float32)

        # Make prediction
        logger.info("Making prediction...")
        start_time = time.time()

        if model_type == "onnx":
            # ONNX inference
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: input_params.reshape(1, -1)})
            prediction = output[0]
        else:
            # PyTorch inference
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_params).unsqueeze(0).to(device)
                prediction = model(input_tensor).cpu().numpy()

        inference_time = time.time() - start_time
        logger.info(f"Prediction completed in {inference_time:.4f} seconds")

        # Extract real and imaginary parts
        field_real = prediction[0, 0, :, :]
        field_imag = prediction[0, 1, :, :]
        predicted_field = field_real + 1j * field_imag

        # Compute irradiance
        irradiance = np.abs(predicted_field)**2

        # Compute derived metrics if requested
        metrics = {}
        if compute_metrics:
            logger.info("Computing derived metrics using physics module...")

            # Setup physics parameters for metric calculation
            link_params = LinkParameters(
                distance=link_distance,
                wavelength=wavelength,
                beam_waist=beam_waist,
                grid_size=grid_size,
                altitude_tx_m=altitude_tx_m,
                altitude_rx_m=altitude_rx_m,
            )
            atm_params = AtmosphericParameters(
                visibility=visibility,
                temp_gradient=temp_gradient,
                pressure_hpa=pressure_hpa,
                temperature_celsius=temperature_celsius,
                humidity=humidity,
            )
            average_altitude_m = (altitude_tx_m + altitude_rx_m) / 2
            atm_effects = AtmosphericEffects(atm_params, link_params.wavelength, average_altitude_m)
            ssfm_solver = SplitStepFourierMethod(link_params, atm_params, link_altitude_m=average_altitude_m)


            # Scintillation index
            scintillation_index = atm_effects.compute_scintillation_index(irradiance)
            metrics['scintillation_index'] = scintillation_index

            # Received Power
            dx = link_params.grid_width / grid_size
            received_power = np.sum(irradiance) * dx**2
            metrics['received_power_watts'] = received_power

            # Bit Error Rate (BER)
            noise_power = 1e-12  # W
            ber = atm_effects.compute_bit_error_rate(received_power, scintillation_index, noise_power)
            metrics['bit_error_rate'] = ber

            # Actionable Metrics
            beam_params = ssfm_solver.compute_beam_parameters(predicted_field)
            initial_field = ssfm_solver.pwe_solver.create_initial_field()
            initial_params = ssfm_solver.compute_beam_parameters(initial_field)

            beam_wander = np.sqrt(beam_params['x_centroid']**2 + beam_params['y_centroid']**2)
            metrics['beam_wander_m'] = beam_wander

            spreading_x = beam_params['beam_width_x'] / initial_params['beam_width_x']
            spreading_y = beam_params['beam_width_y'] / initial_params['beam_width_y']
            metrics['beam_spreading_ratio_x'] = spreading_x
            metrics['beam_spreading_ratio_y'] = spreading_y


            logger.info(f"Scintillation index: {scintillation_index:.6f}")
            logger.info(f"Received Power: {received_power:.2e} W")
            logger.info(f"Bit Error Rate: {ber:.2e}")
            logger.info(f"Beam Wander: {beam_wander * 100:.2f} cm")
            logger.info(f"Beam Spreading (X): {spreading_x:.2f}x")
            logger.info(f"Beam Spreading (Y): {spreading_y:.2f}x")
        
        # Save results if output file specified
        if output_file:
            logger.info(f"Saving results to: {output_file}")
            
            if output_file.suffix == ".h5":
                import h5py
                with h5py.File(output_file, 'w') as f:
                    f.create_dataset('irradiance', data=irradiance)
                    f.create_dataset('field_real', data=field_real)
                    f.create_dataset('field_imag', data=field_imag)
                    f.attrs['link_distance'] = link_distance
                    f.attrs['wavelength'] = wavelength
                    f.attrs['beam_waist'] = beam_waist
                    f.attrs['visibility'] = visibility
                    f.attrs['temp_gradient'] = temp_gradient
                    f.attrs['pressure_hpa'] = pressure_hpa
                    f.attrs['temperature_celsius'] = temperature_celsius
                    f.attrs['humidity'] = humidity
                    f.attrs['altitude_tx_m'] = altitude_tx_m
                    f.attrs['altitude_rx_m'] = altitude_rx_m
                    f.attrs['inference_time'] = inference_time
                    
                    if metrics:
                        for key, value in metrics.items():
                            f.attrs[key] = value
            else:
                # Save as NPZ
                save_data = {
                    'irradiance': irradiance,
                    'field_real': field_real,
                    'field_imag': field_imag,
                    'parameters': input_params,
                    'inference_time': inference_time
                }
                if metrics:
                    save_data.update(metrics)
                np.savez(output_file, **save_data)
        
        # Generate visualization if requested
        if visualize:
            logger.info("Generating visualization...")
            plot_path = output_file.parent / f"{output_file.stem}_plot.png" if output_file else Path("prediction_plot.png")
            plot_irradiance_map(irradiance, save_path=plot_path)
            logger.info(f"Visualization saved to: {plot_path}")
        
        logger.info(f"Prediction completed successfully!")
        logger.info(f"Inference time: {inference_time:.4f} seconds")
        logger.info(f"Irradiance map shape: {irradiance.shape}")
        logger.info(f"Peak irradiance: {np.max(irradiance):.2e}")
        
        if metrics:
            logger.info(f"Scintillation index: {metrics['scintillation_index']:.6f}")
            logger.info(f"Bit error rate: {metrics['bit_error_rate']:.2e}")
        
        if output_file:
            logger.info(f"Results saved to: {output_file}")
        
    except ImportError as e:
        logger.error(f"Required modules not yet implemented: {e}")
        logger.error("Prediction not yet implemented.")
        logger.error("This command will be available after implementing the PINO models.")
        ctx.exit(1)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        ctx.exit(1)
