"""
Trajectory Prediction Configuration

Configuration for ball trajectory prediction and polynomial fitting.
"""

from dataclasses import dataclass


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory prediction"""
    
    # Polynomial fitting
    polynomial_degree: int = 2
    min_points_required: int = 3
    
    # Prediction
    prediction_points: int = 50
    extrapolation_points: int = 30


# Default configuration
DEFAULT_TRAJECTORY_CONFIG = TrajectoryConfig()
