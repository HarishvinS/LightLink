#!/usr/bin/env python3
"""
Setup script for FSOC-PINO package.
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Core requirements
requirements = [
    "numpy>=1.21.0",
    "scipy>=1.7.0", 
    "torch>=1.12.0",
    "torchvision>=0.13.0",
    "h5py>=3.7.0",
    "click>=8.0.0",
    "pyyaml>=6.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "tqdm>=4.64.0",
    "scikit-learn>=1.1.0",
    "onnx>=1.12.0",
    "onnxruntime>=1.12.0",
    "tensorboard>=2.9.0",
    "wandb>=0.13.0",
]

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.950",
    "pre-commit>=2.19.0",
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]

setup(
    name="fsoc-pino",
    version="0.1.0",
    author="FSOC-PINO Development Team",
    author_email="contact@fsoc-pino.org",
    description="Edge-Deployed Physics-Informed Neural Operator for FSOC Link Performance Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/fsoc-pino",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "gpu": ["torch>=1.12.0", "torchvision>=0.13.0"],
    },
    entry_points={
        "console_scripts": [
            "fso-pino-cli=fsoc_pino.cli.main:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
