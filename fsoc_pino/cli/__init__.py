"""
Command Line Interface module.

This module provides the main CLI entry points for:
- Dataset generation
- Model training
- Inference/prediction
- Benchmarking
"""

from .main import cli
from .commands import *

__all__ = ["cli"]
