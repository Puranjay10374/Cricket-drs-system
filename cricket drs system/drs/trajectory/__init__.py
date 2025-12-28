"""
Trajectory Package

Modular trajectory prediction and analysis components.

This package contains:
- polynomial_fitter.py: Polynomial fitting and quality metrics
- path_predictor.py: Path prediction and extrapolation
- intersection_finder.py: Stump intersection detection
- trajectory_analyzer.py: Main analysis pipeline
"""

from .polynomial_fitter import PolynomialFitter
from .path_predictor import PathPredictor
from .intersection_finder import IntersectionFinder
from .trajectory_analyzer import TrajectoryAnalyzer, TrajectoryPredictor

__all__ = [
    'PolynomialFitter',
    'PathPredictor',
    'IntersectionFinder',
    'TrajectoryAnalyzer',
    'TrajectoryPredictor',  # Backward compatibility alias
]
