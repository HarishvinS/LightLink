#!/usr/bin/env python3
"""
Script to verify the generated dataset.
"""

import h5py
import numpy as np
import json
from pathlib import Path

def main():
    """Verify the generated dataset."""
    dataset_dir = Path("test_dataset")
    
    # Load metadata
    with open(dataset_dir / "dataset_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print("=== Dataset Metadata ===")
    print(json.dumps(metadata, indent=2))
    
    # Load HDF5 data
    batch_file = dataset_dir / "batch_0000.h5"
    
    print(f"\n=== HDF5 File: {batch_file} ===")
    
    with h5py.File(batch_file, 'r') as f:
        print("Datasets:")
        for key in f.keys():
            dataset = f[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        print("\nAttributes:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        # Load and examine data
        parameters = f['parameters'][:]
        irradiance = f['irradiance'][:]
        field_real = f['field_real'][:]
        field_imag = f['field_imag'][:]
        
        print(f"\n=== Data Analysis ===")
        print(f"Number of samples: {len(parameters)}")
        print(f"Parameter ranges:")
        param_names = ['link_distance', 'wavelength', 'beam_waist', 'visibility', 'temp_gradient']
        for i, name in enumerate(param_names):
            print(f"  {name}: {np.min(parameters[:, i]):.6f} to {np.max(parameters[:, i]):.6f}")
        
        print(f"\nIrradiance statistics:")
        print(f"  Min: {np.min(irradiance):.6f}")
        print(f"  Max: {np.max(irradiance):.6f}")
        print(f"  Mean: {np.mean(irradiance):.6f}")
        print(f"  Std: {np.std(irradiance):.6f}")
        
        print(f"\nField statistics:")
        print(f"  Real part - Min: {np.min(field_real):.6f}, Max: {np.max(field_real):.6f}")
        print(f"  Imag part - Min: {np.min(field_imag):.6f}, Max: {np.max(field_imag):.6f}")
        
        # Check for any issues
        print(f"\n=== Data Quality Checks ===")
        print(f"Any NaN in parameters: {np.any(np.isnan(parameters))}")
        print(f"Any NaN in irradiance: {np.any(np.isnan(irradiance))}")
        print(f"Any NaN in fields: {np.any(np.isnan(field_real)) or np.any(np.isnan(field_imag))}")
        print(f"Any negative irradiance: {np.any(irradiance < 0)}")
        
        # Verify irradiance = |field|²
        computed_irradiance = field_real**2 + field_imag**2
        irradiance_error = np.mean(np.abs(irradiance - computed_irradiance))
        print(f"Irradiance consistency error: {irradiance_error:.2e}")


if __name__ == "__main__":
    main()
