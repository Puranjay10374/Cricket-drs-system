"""
Ball Detection Configuration

Configuration for ball detection parameters.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BallDetectionConfig:
    """Configuration for ball detection parameters"""
    
    # Color ranges (HSV) for red cricket ball
    color_lower1: Tuple[int, int, int] = (0, 120, 70)
    color_upper1: Tuple[int, int, int] = (10, 255, 255)
    color_lower2: Tuple[int, int, int] = (170, 120, 70)
    color_upper2: Tuple[int, int, int] = (180, 255, 255)
    
    # Size constraints
    min_area: int = 50
    max_area: int = 5000
    min_radius: int = 5
    max_radius: int = 50  # Maximum ball radius in pixels
    
    # Morphological operations
    kernel_size: int = 5
    erosion_iterations: int = 1
    dilation_iterations: int = 2


# Default configuration
DEFAULT_DETECTION_CONFIG = BallDetectionConfig()
