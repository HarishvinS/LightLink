"""
Data generation and processing module.

This module handles:
- Dataset generation from physics simulations
- Parameter space sampling
- Data preprocessing and augmentation
- HDF5 data management
"""

from .generator import DatasetGenerator, generate_fsoc_dataset
from .sampling import ParameterSampler, create_fsoc_parameter_sampler
# from .preprocessing import DataPreprocessor  # Will be implemented later
from .storage import HDF5Manager, FSOCDataset

__all__ = [
    "DatasetGenerator",
    "generate_fsoc_dataset",
    "ParameterSampler",
    "create_fsoc_parameter_sampler",
    # "DataPreprocessor",
    "HDF5Manager",
    "FSOCDataset"
]
