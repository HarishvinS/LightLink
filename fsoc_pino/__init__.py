"""
FSOC-PINO: Edge-Deployed Physics-Informed Neural Operator for FSOC Link Performance Prediction

A comprehensive toolkit for simulating Free Space Optical Communication (FSOC) links
and training Physics-Informed Neural Operators (PINOs) for real-time performance prediction.
"""

__version__ = "0.1.0"
__author__ = "FSOC-PINO Development Team"
__email__ = "contact@fsoc-pino.org"

from .simulation import FSOC_Simulator, SimulationConfig
from .models import PINO_FNO, create_pino_model, PINOTrainer
from .utils import *

__all__ = [
    "FSOC_Simulator",
    "SimulationConfig",
    "PINO_FNO",
    "create_pino_model",
    "PINOTrainer",
]
