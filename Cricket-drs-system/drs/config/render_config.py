"""
Rendering Configuration

Configuration for visual overlays and trajectory rendering.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class RenderConfig:
    """Configuration for rendering overlays"""
    
    # Ball marker
    ball_circle_color: Tuple[int, int, int] = (0, 0, 255)  # Red (BGR)
    ball_circle_thickness: int = 2
    ball_center_color: Tuple[int, int, int] = (255, 0, 0)  # Blue (BGR)
    ball_center_radius: int = 3
    
    # Trajectory path
    path_color: Tuple[int, int, int] = (0, 255, 0)  # Green (BGR)
    path_thickness: int = 2
    
    # Predicted path
    predicted_path_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow (BGR)
    predicted_path_thickness: int = 2
    predicted_path_style: str = 'dashed'  # 'solid' or 'dashed'
    
    # Text overlay
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White
    text_font_scale: float = 0.6
    text_thickness: int = 2


# Default configuration
DEFAULT_RENDER_CONFIG = RenderConfig()
