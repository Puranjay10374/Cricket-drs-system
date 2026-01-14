"""
Renderer Package

Modular rendering components for video overlays.

This package contains:
- ball_renderer.py: Ball marker rendering
- path_renderer.py: Trajectory path rendering (actual and predicted)
- overlay_renderer.py: Stump regions, impact points, decision text
- trajectory_renderer.py: Unified facade interface
"""

from .ball_renderer import BallRenderer
from .path_renderer import PathRenderer
from .overlay_renderer import OverlayRenderer
from .trajectory_renderer import TrajectoryRenderer

__all__ = [
    'BallRenderer',
    'PathRenderer',
    'OverlayRenderer',
    'TrajectoryRenderer',
]
