"""
Dataset generation for FSOC-PINO training.

This module handles the generation of training datasets by running
physics simulations across parameter spaces.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import h5py

from ..simulation import FSOC_Simulator, SimulationConfig
from .sampling import ParameterSampler


class DatasetGenerator:
    """
    Generator for FSOC simulation datasets.
    
    Handles parallel execution of simulations and data storage.
    """
    
    def __init__(
        self,
        output_dir: Path,
        grid_size: int = 128,
        batch_size: int = 100,
        parallel_jobs: int = 1,
        save_intermediate: bool = False
    ):
        """
        Initialize dataset generator.
        
        Args:
            output_dir: Output directory for dataset files
            grid_size: Spatial grid size for simulations
            batch_size: Number of samples per batch file
            parallel_jobs: Number of parallel simulation processes
            save_intermediate: Whether to save intermediate field states
        """
        self.output_dir = Path(output_dir)
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.parallel_jobs = parallel_jobs
        self.save_intermediate = save_intermediate
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.total_samples = 0
        self.successful_samples = 0
        self.failed_samples = 0
        self.generation_time = 0.0
    
    def generate(self, parameters: List[Dict[str, float]]) -> Dict:
        """
        Generate dataset from parameter list of dictionaries.
        
        Args:
            parameters: List of dictionaries, each representing a set of simulation parameters
            
        Returns:
            Dictionary with generation statistics and output files
        """
        start_time = time.time()
        self.total_samples = len(parameters)
        
        print(f"Generating dataset with {self.total_samples} samples...")
        print(f"Output directory: {self.output_dir}")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Parallel jobs: {self.parallel_jobs}")
        
        # Split parameters into batches
        batches = self._create_batches(parameters)
        output_files = []
        
        # Process batches
        for batch_idx, batch_params in enumerate(batches):
            print(f"\nProcessing batch {batch_idx + 1}/{len(batches)}...")
            
            # Run simulations for this batch
            batch_results = self._process_batch(batch_params, batch_idx)
            
            # Save batch to HDF5
            if batch_results:
                output_file = self.output_dir / f"batch_{batch_idx:04d}.h5"
                self._save_batch_hdf5(batch_results, output_file)
                output_files.append(output_file)
                
                print(f"Saved {len(batch_results)} samples to {output_file}")
        
        self.generation_time = time.time() - start_time
        
        # Save dataset metadata
        metadata = self._create_metadata(parameters, output_files)
        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset generation completed!")
        print(f"Total samples: {self.total_samples}")
        print(f"Successful: {self.successful_samples}")
        print(f"Failed: {self.failed_samples}")
        print(f"Success rate: {self.successful_samples/self.total_samples*100:.1f}%")
        print(f"Total time: {self.generation_time:.2f} seconds")
        print(f"Average time per sample: {self.generation_time/self.total_samples:.3f} seconds")
        
        return {
            'num_samples': self.successful_samples,
            'output_files': output_files,
            'metadata_file': metadata_file,
            'generation_time': self.generation_time,
            'success_rate': self.successful_samples / self.total_samples
        }
    
    def _create_batches(self, parameters: List[Dict[str, float]]) -> List[List[Dict[str, float]]]:
        """Split parameters into batches."""
        batches = []
        for i in range(0, len(parameters), self.batch_size):
            batch = parameters[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def _process_batch(self, batch_params: np.ndarray, batch_idx: int) -> List[Dict]:
        """Process a batch of simulations."""
        batch_results = []
        
        if self.parallel_jobs > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.parallel_jobs) as executor:
                # Submit all simulations
                futures = []
                for i, params in enumerate(batch_params):
                    future = executor.submit(self._run_single_simulation, params, batch_idx, i)
                    futures.append(future)
                
                # Collect results with progress bar
                for future in tqdm(as_completed(futures), total=len(futures), 
                                 desc=f"Batch {batch_idx + 1}"):
                    result = future.result()
                    if result is not None:
                        batch_results.append(result)
                        self.successful_samples += 1
                    else:
                        self.failed_samples += 1
        else:
            # Sequential processing
            for i, params in enumerate(tqdm(batch_params, desc=f"Batch {batch_idx + 1}")):
                result = self._run_single_simulation(params, batch_idx, i)
                if result is not None:
                    batch_results.append(result)
                    self.successful_samples += 1
                else:
                    self.failed_samples += 1
        
        return batch_results
    
    def _run_single_simulation(self, params: Dict[str, float], batch_idx: int, sample_idx: int) -> Optional[Dict]:
        """Run a single simulation."""
        try:
            # Create simulation configuration from dictionary
            config = SimulationConfig(
                **params,
                grid_size=self.grid_size
            )

            # Run simulation
            simulator = FSOC_Simulator(config, save_intermediate=self.save_intermediate)
            result = simulator.run_simulation()
            
            # Package results
            return {
                'parameters': params, # Store the dictionary directly
                'irradiance': result.irradiance.copy(),
                'field_real': np.real(result.final_field).copy(),
                'field_imag': np.imag(result.final_field).copy(),
                'scintillation_index': result.scintillation_index,
                'bit_error_rate': result.bit_error_rate,
                'propagation_time': result.propagation_time,
                'batch_idx': batch_idx,
                'sample_idx': sample_idx
            }
            
        except Exception as e:
            print(f"Simulation failed for sample {batch_idx}:{sample_idx}: {e}")
            return None
    
    def _save_batch_hdf5(self, batch_results: List[Dict], output_file: Path):
        """Save batch results to HDF5 file."""
        with h5py.File(output_file, 'w') as f:
            # Get dimensions and parameter names from the first sample
            num_samples = len(batch_results)
            grid_size = batch_results[0]['irradiance'].shape[0]
            parameter_names = list(batch_results[0]['parameters'].keys())
            num_params = len(parameter_names)
            
            # Create datasets for each parameter
            for param_name in parameter_names:
                f.create_dataset(f'params/{param_name}', (num_samples,), dtype=np.float32)

            # Create datasets for targets and metrics
            f.create_dataset('irradiance', (num_samples, grid_size, grid_size), dtype=np.float32)
            f.create_dataset('field_real', (num_samples, grid_size, grid_size), dtype=np.float32)
            f.create_dataset('field_imag', (num_samples, grid_size, grid_size), dtype=np.float32)
            f.create_dataset('scintillation_index', (num_samples,), dtype=np.float32)
            f.create_dataset('bit_error_rate', (num_samples,), dtype=np.float32)
            f.create_dataset('propagation_time', (num_samples,), dtype=np.float32)
            
            # Fill datasets
            for i, result in enumerate(batch_results):
                for param_name in parameter_names:
                    f[f'params/{param_name}'][i] = result['parameters'][param_name]
                
                f['irradiance'][i] = result['irradiance']
                f['field_real'][i] = result['field_real']
                f['field_imag'][i] = result['field_imag']
                f['scintillation_index'][i] = result['scintillation_index']
                f['bit_error_rate'][i] = result['bit_error_rate']
                f['propagation_time'][i] = result['propagation_time']
            
            # Add metadata
            f.attrs['num_samples'] = num_samples
            f.attrs['grid_size'] = grid_size
            f.attrs['num_parameters'] = num_params
            f.attrs['parameter_names'] = parameter_names
    
    def _create_metadata(self, parameters: List[Dict[str, float]], output_files: List[Path]) -> Dict:
        """Create dataset metadata."""
        if not parameters:
            return {}

        param_names = list(parameters[0].keys())
        
        # Compute parameter statistics
        param_stats = {}
        for name in param_names:
            values = np.array([p[name] for p in parameters])
            param_stats[name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        return {
            'dataset_info': {
                'total_samples': self.total_samples,
                'successful_samples': self.successful_samples,
                'failed_samples': self.failed_samples,
                'success_rate': self.successful_samples / self.total_samples,
                'generation_time': self.generation_time,
                'grid_size': self.grid_size,
                'batch_size': self.batch_size,
                'parallel_jobs': self.parallel_jobs
            },
            'parameter_statistics': param_stats,
            'output_files': [str(f) for f in output_files],
            'parameter_names': param_names
        }
    
    def save_metadata(self, filepath: Path, additional_metadata: Dict):
        """Save additional metadata to file."""
        with open(filepath, 'w') as f:
            json.dump(additional_metadata, f, indent=2)


def generate_fsoc_dataset(
    output_dir: Path,
    num_samples: int,
    parameter_ranges: Dict[str, Tuple[float, float]],
    sampling_method: str = "latin_hypercube",
    grid_size: int = 128,
    batch_size: int = 100,
    parallel_jobs: int = 1,
    seed: Optional[int] = None
) -> Dict:
    """
    High-level function to generate FSOC dataset.
    
    Args:
        output_dir: Output directory
        num_samples: Number of samples to generate
        parameter_ranges: Parameter ranges dictionary
        sampling_method: Sampling method
        grid_size: Spatial grid size
        batch_size: Batch size for processing
        parallel_jobs: Number of parallel jobs
        seed: Random seed
        
    Returns:
        Generation statistics dictionary
    """
    # Create parameter sampler
    sampler = ParameterSampler(method=sampling_method, ranges=parameter_ranges, seed=seed)
    
    # Generate parameter samples
    parameters = sampler.sample_dict(num_samples)
    
    # Create dataset generator
    generator = DatasetGenerator(
        output_dir=output_dir,
        grid_size=grid_size,
        batch_size=batch_size,
        parallel_jobs=parallel_jobs
    )
    
    # Generate dataset
    return generator.generate(parameters)
