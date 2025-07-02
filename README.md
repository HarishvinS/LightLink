# FSOC-PINO: Edge-Deployed Physics-Informed Neural Operator for FSOC Link Performance Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive toolkit for simulating Free Space Optical Communication (FSOC) links and training Physics-Informed Neural Operators (PINOs) for real-time performance prediction on edge devices.

## 🎯 Project Overview

This project develops a CLI tool that:

1. **Simulates FSOC link performance** under specified fog and temperature gradient conditions using the Parabolic Wave Equation (PWE) and Split-Step Fourier Method (SSFM)
2. **Trains a Fourier Neural Operator (FNO)** with physics-informed constraints to predict link performance
3. **Deploys the trained model** on edge devices for near-real-time prediction, bypassing computationally expensive simulations

## 🚀 Key Features

- **High-Fidelity Physics Simulation**: Implementation of PWE with SSFM for accurate beam propagation modeling
- **Atmospheric Effects Modeling**: Temperature gradients (turbulence) and fog attenuation effects
- **Physics-Informed Learning**: PINO architecture that respects physical laws during training
- **Edge Deployment Ready**: ONNX model conversion for deployment on resource-constrained devices
- **Comprehensive CLI**: Easy-to-use command-line interface for all operations
- **Extensive Benchmarking**: Performance comparison between simulation and ML prediction

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-org/fsoc-pino.git
cd fsoc-pino

# Install the package
pip install -e .

# For development installation
pip install -e ".[dev]"

# For GPU support
pip install -e ".[gpu]"
```

### Development Setup

```bash
# Setup development environment
python scripts/setup_environment.py --dev --gpu

# Install pre-commit hooks
pre-commit install
```

## 🔧 Usage

### CLI Commands

The main CLI tool `fso-pino-cli` provides four primary commands:

#### 1. Generate Dataset
```bash
fso-pino-cli generate-dataset \
    --output-dir ./data/training \
    --num-samples 10000 \
    --link-distance-range 1.0 5.0 \
    --visibility-range 0.5 10.0 \
    --temp-gradient-range 0.01 0.2 \
    --parallel-jobs 8
```

#### 2. Train PINO Model
```bash
fso-pino-cli train \
    --dataset-path ./data/training \
    --output-dir ./models \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --physics-loss-weight 0.1
```

#### 3. Make Predictions
```bash
fso-pino-cli predict \
    --model-path ./models/best_model.onnx \
    --link-distance 2.5 \
    --visibility 3.0 \
    --temp-gradient 0.05 \
    --output-file prediction.h5
```

#### 4. Benchmark Performance
```bash
fso-pino-cli benchmark \
    --model-path ./models/best_model.onnx \
    --test-dataset ./data/test \
    --output-dir ./benchmarks
```

### Python API

```python
from fsoc_pino import FSOC_Simulator, PINO_FNO

# Physics simulation
simulator = FSOC_Simulator(
    link_distance=2.5,  # km
    visibility=3.0,     # km
    temp_gradient=0.05, # K/m
    wavelength=1550e-9, # m
    beam_waist=0.05     # m
)

result = simulator.run_simulation()
irradiance_map = result.irradiance
scintillation_index = result.scintillation_index
ber = result.bit_error_rate

# PINO prediction
model = PINO_FNO.load("./models/best_model.pth")
prediction = model.predict(
    link_distance=2.5,
    visibility=3.0, 
    temp_gradient=0.05
)
```

## 📊 Physics Background

### Parabolic Wave Equation (PWE)
The fundamental model for laser beam propagation:

```
2ik₀ ∂ψ/∂z + ∇²_T ψ + 2k₀² [n(x,y,z)/n₀ - 1] ψ = 0
```

### Atmospheric Effects
- **Temperature Gradients**: Modeled using Kolmogorov turbulence theory with Cₙ² parameter
- **Fog Attenuation**: Kim model relating attenuation to meteorological visibility

### Split-Step Fourier Method (SSFM)
Numerical solution alternating between:
1. Diffraction step (Fourier domain)
2. Refraction/absorption step (spatial domain)

## 🧠 PINO Architecture

The Physics-Informed Neural Operator combines:
- **Fourier Neural Operator (FNO)**: Learns mappings between function spaces
- **Physics-Informed Loss**: Enforces PWE constraints during training
- **Data Loss**: Matches simulation ground truth

Total Loss: `L_total = λ_data * L_data + λ_physics * L_physics`

## 📈 Performance

Expected performance improvements:
- **Speed**: 100x - 1000x faster than physics simulation
- **Accuracy**: <5% relative error on key metrics
- **Memory**: Deployable on edge devices with <1GB RAM

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fsoc_pino --cov-report=html

# Run specific test module
pytest tests/test_simulation.py -v
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:
- API Reference
- Tutorials and Examples
- Mathematical Background
- Deployment Guide

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Reach out

 [Harishvin Sasikumar](mailto:harishsasi17@gmail.com)

---

**Note**: This project is under active development. APIs may change before the 1.0 release.
