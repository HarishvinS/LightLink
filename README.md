# Lightlink: Physics-Informed Neural Operators for FSOC Link Performance Prediction

## Overview

LightLink is a toolkit for utilzing and training Physics-Informed Neural Operators (PINOs) for real-time performance prediction on edge computing devices. Currently, the system generates synthetic data using high fidelity physics simulations. This data is used to train PINOs that can predict FSOC link behaviour and performance in real time.

Physics Informed Neural Operators are neural operators that adhere to physics at greater rate than traditional machine learning methods. They are able to accomplish this as they are designed to learn mappings between funciton spaces while penalizing results that violate the underlying physics equations during training. This allows them to accurate predictions even on limited datasets or inputs as they take into account the underlying physics of the system.

The goal of this project is to provide a tool to efficiently build, train, and deploy PINOs to edge-computing devices (e.g Raspberry Pi, NVIDIA Jetson Nano) located at physical FSOC stations to optimize their operation in real time. Furthermore, a network of edge devices, connected by a protocol such as LoRaWAN, can coordinate and correct for overall optimized data transmission rates. 

## What it does

- **Performs high-fidelity physics simulations**: Accurate beam propagation modeling using the Parabolic Wave Equation (PWE) and Split-Step Fourier Method (SSFM).  Accounts for turbulence (from temperature           gradients) and fog attenuation.
- **Trains a PINO model**: Useing the data from the physics sims, it trains a PINO architecture that respects physical laws (e.g. Parabolic Wave Equations) during training for more accurate and generalizable         predictions.
- **Preps models for deployment**: The trained PINO models can be exported in ONNX, PyTorch, and TorchScript compatible formats, allowing for easier deployment.
- 
### Other important things

- **Comprehensive CLI**: An easy-to-use command-line interface for dataset generation, model training, prediction, and benchmarking.
- **Python API**: A Python API for programmatic control and integration.

## Installation

### Prerequisites
- Python 3.8 or higher
- A CUDA-capable GPU is recommended for training.

### Steps
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/fsoc-pino.git
    cd fsoc-pino
    ```
2.  **Install the package:**
    ```bash
    # For standard installation
    pip install -e .

    # For development (includes testing and linting tools)
    pip install -e ".[dev]"

    # For GPU support
    pip install -e ".[gpu]"
    ```
3.  **(Optional) Setup pre-commit hooks for development:**
    ```bash
    pre-commit install
    ```

## Usage

The project can be used via its command-line interface or its Python API.

### Command-Line Interface (CLI)

The main CLI tool is `fso-pino-cli`. It has four primary commands:

#### `generate-dataset`
Generate a training dataset by running physics simulations across a range of parameters.
```bash
fso-pino-cli generate-dataset --output-dir ./data/training --num-samples 100 --grid-size 64 --link-distance-range 1.0 5.0 --visibility-range 0.5 10.0 --temp-gradient-range 0.01 0.2 --beam-waist-range 0.02 0.10 --wavelength-range 850e-9 1550e-9 --parallel-jobs 4

```

#### `train`
Train a PINO model on a generated dataset.
```bash
fso-pino-cli train --dataset-path data --output-dir models --epochs 50 --learning-rate 1e-3 --physics-loss-weight 0.1
```

#### `predict`
Make fast predictions using a trained model.
```bash
fso-pino-cli predict --model-path ./models/best_model.pth --link-distance 2.5 --visibility 3.0 --temp-gradient 0.05 --beam-waist 0.05 --wavelength 1550e-9 --pressure-hpa 1013.25 --temperature-celsius 15.0 --humidity 0.5 --altitude-tx-m 10.0 --altitude-rx-m 10.0 --compute-metrics
```
Add the flag `--visualize` for a visual depiction of the irradiance map.

#### `benchmark`
Benchmark the performance (accuracy and speed) of a trained model against the physics simulation.
```bash
fso-pino-cli benchmark  --model-path ./models/best_model.pth --test-dataset ./data/test --output-dir ./benchmarks
```

### Python API

The project can also be used as a Python library for more complex workflows.

```python
from fsoc_pino import FSOC_Simulator, PINO_FNO
import torch

# 1. Run a physics simulation
simulator = FSOC_Simulator(
    link_distance=2.5,  # km
    visibility=3.0,     # km
    temp_gradient=0.05, # K/m
    wavelength=1550e-9, # m
    beam_waist=0.05     # m
)
result = simulator.run_simulation()
print(f"Simulation Scintillation Index: {result.scintillation_index:.4f}")
print(f"Simulation BER: {result.bit_error_rate:.2e}")

# 2. Use a trained PINO model for prediction
model = PINO_FNO.load("./models/best_model.pth")
input_params = torch.tensor([[2.5, 1550e-9, 0.05, 3.0, 0.05]]) # Must match training order
metrics = model.compute_derived_metrics(input_params)
print(f"PINO Predicted Scintillation Index: {metrics['scintillation_index'].item():.4f}")
```

## Mathematical Modeling

The project's physics simulation and PINO model are based on the following mathematical principles:

### Parabolic Wave Equation (PWE)
The propagation of the laser beam is modeled by the Parabolic Wave Equation (PWE), an approximation of the Helmholtz equation suitable for paraxial beams:
```
2ik₀ ∂ψ/∂z + ∇²_T ψ + 2k₀² [n(x,y,z)/n₀ - 1] ψ = 0
```
- `ψ`: Complex envelope of the electric field.
- `k₀`: Wavenumber in vacuum.
- `∇²_T`: Transverse Laplacian operator.
- `n(x,y,z)`: Refractive index of the medium.

### Split-Step Fourier Method (SSFM)
The PWE is solved numerically using the SSFM (Split Step Fourier Method), which splits the propagation into two steps for each segment of the path:
1.  **Diffraction Step**: Solved in the Fourier domain to model free-space propagation (aka no atmospheric effects are applied)
2.  **Refraction/Absorption Step**: Solved in the spatial domain to apply atmospheric effects.

### Atmospheric Effects
- **Turbulence**: Modeled using the Kolmogorov turbulence theory. The strength is characterized by the refractive index structure parameter `Cₙ²`, calculated using the Hufnagel-Valley model (note: still personally leanring the more nuanced physics at this step). Turbulence is implemented as random phase screens applied to the beam.
- **Fog Attenuation**: Modeled using the Kim model, which relates the attenuation coefficient to meteorological visibility and wavelength. (note: atmospheric physics was harder than expected. still also figuring out nuanced implementation and structure.)

### Physics-Informed Neural Operator (PINO)
The PINO model's loss function combines data-driven learning with physical constraints:
- **Data Loss**: The Mean Squared Error (MSE) between the model's prediction and the ground truth from the simulation.
- **Physics Loss**: The residual of the Parabolic Wave Equation. This term penalizes predictions that violate the PWE, ensuring the model's output is physically consistent.

The total loss is a weighted sum: `L_total = λ_data * L_data + λ_physics * L_physics`.

## Testing

To run the test suite, use `pytest`:
```bash
pytest
```
To include code coverage:
```bash
pytest --cov=fsoc_pino --cov-report=html
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##Contact
Hey! This project was developed by me: Harishvin Sasikumar! Check out my website [here](https://harishvin.framer.website).
Email me at [harishsasi17@gmail.com](mailto:harishsasi17@gmail.com)
