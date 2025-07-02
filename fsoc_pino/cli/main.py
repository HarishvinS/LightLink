#!/usr/bin/env python3
"""
Main CLI entry point for FSOC-PINO.

This module provides the main command-line interface with subcommands for:
- generate-dataset: Generate training data from physics simulations
- train: Train PINO models
- predict: Make predictions using trained models
- benchmark: Compare simulation vs. model performance
"""

import click
from pathlib import Path

from fsoc_pino.utils.logging_utils import setup_logger

from fsoc_pino.utils.logging_utils import setup_logger
from fsoc_pino.cli.commands import (
    generate_dataset,
    train,
    predict, 
    benchmark
)


@click.group()
@click.option(
    "--verbose", "-v", 
    is_flag=True, 
    help="Enable verbose logging"
)
@click.option(
    "--log-file",
    type=click.Path(),
    help="Log file path (default: logs to console)"
)
@click.pass_context
def cli(ctx, verbose, log_file):
    """
    FSOC-PINO: Edge-Deployed Physics-Informed Neural Operator for FSOC Link Performance Prediction.
    
    A comprehensive toolkit for simulating Free Space Optical Communication (FSOC) links
    and training Physics-Informed Neural Operators (PINOs) for real-time performance prediction.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger = setup_logger(level=log_level, log_file=log_file)
    ctx.obj['logger'] = logger
    
    # Store global options
    ctx.obj['verbose'] = verbose
    ctx.obj['log_file'] = log_file
    
    logger.info("FSOC-PINO CLI started")


# Add subcommands
cli.add_command(generate_dataset.generate_dataset)
cli.add_command(train.train)
cli.add_command(predict.predict)
cli.add_command(benchmark.benchmark)


@cli.command()
def version():
    """Show version information."""
    try:
        from fsoc_pino import __version__
        logger = setup_logger() # Get a logger instance
        logger.info(f"FSOC-PINO version {__version__}")
    except ImportError:
        logger = setup_logger() # Get a logger instance
        logger.info("FSOC-PINO version: development")


@cli.command()
def info():
    """Show system and package information."""
    import platform
    import torch
    import numpy as np
    import scipy
    
    logger = setup_logger() # Get a logger instance
    logger.info("=== FSOC-PINO System Information ===")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"SciPy version: {scipy.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")


if __name__ == "__main__":
    cli()
