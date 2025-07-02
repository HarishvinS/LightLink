"""
CLI command modules.

This package contains the implementation of all CLI subcommands:
- generate_dataset: Dataset generation from physics simulations
- train: PINO model training
- predict: Model inference
- benchmark: Performance benchmarking
"""

from . import generate_dataset, train, predict, benchmark

__all__ = ["generate_dataset", "train", "predict", "benchmark"]
