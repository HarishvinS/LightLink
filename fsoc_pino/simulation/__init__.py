"""
Physics simulation module for FSOC link modeling.

This module contains the core physics simulation components including:
- Parabolic Wave Equation (PWE) solver
- Split-Step Fourier Method (SSFM) implementation
- Atmospheric effects modeling (turbulence and fog)
- FSOC_Simulator main class
"""

from .core import FSOC_Simulator, SimulationConfig
from .physics import PWE_Solver, AtmosphericEffects, LinkParameters, AtmosphericParameters
from .ssfm import SplitStepFourierMethod, PropagationResult

__all__ = [
    "FSOC_Simulator",
    "SimulationConfig",
    "PWE_Solver",
    "AtmosphericEffects",
    "LinkParameters",
    "AtmosphericParameters",
    "SplitStepFourierMethod",
    "PropagationResult"
]
