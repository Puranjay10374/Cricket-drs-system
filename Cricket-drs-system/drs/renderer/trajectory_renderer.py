"""
Trajectory Renderer - Unified Interface

Combines all rendering components into a single interface.
"""

import numpy as np
from typing import List, Tuple

from ..config import RenderConfig
from .ball_renderer import BallRenderer
from .path_renderer import PathRenderer
from .overlay_renderer import OverlayRenderer


class TrajectoryRenderer:
    """
    Unified trajectory renderer
    
    Facade that combines ball, path, and overlay renderers
    """
    
    def __init__(self, config: RenderConfig):
        """
        Initialize trajectory renderer
        
        Args:
            config: Rendering configuration
        """
        self.config = config
        self.ball_renderer = BallRenderer(config)
        self.path_renderer = PathRenderer(config)
        self.overlay_renderer = OverlayRenderer(config)
    
    def draw_ball(self, frame: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
        """Draw ball marker (delegates to BallRenderer)"""
        return self.ball_renderer.draw(frame, x, y, radius)
    
    def draw_trajectory_path(self, frame: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
        """Draw trajectory path (delegates to PathRenderer)"""
        return self.path_renderer.draw_actual_path(frame, points)
    
    def draw_predicted_path(self, frame: np.ndarray, points: List[Tuple[float, float]]) -> np.ndarray:
        """Draw predicted path (delegates to PathRenderer)"""
        return self.path_renderer.draw_predicted_path(frame, points)
    
    def draw_stump_region(self, frame: np.ndarray, stump_x: float, 
                         stump_y_top: float, stump_y_bottom: float, 
                         color: Tuple[int, int, int] = (255, 255, 0),
                         thickness: int = 2) -> np.ndarray:
        """Draw stump region (delegates to OverlayRenderer)"""
        return self.overlay_renderer.draw_stump_region(frame, stump_x, stump_y_top, 
                                                       stump_y_bottom, color, thickness)
    
    def draw_impact_point(self, frame: np.ndarray, x: float, y: float,
                         label: str = "IMPACT", 
                         color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """Draw impact point (delegates to OverlayRenderer)"""
        return self.overlay_renderer.draw_impact_point(frame, x, y, label, color)
    
    def draw_decision_overlay(self, frame: np.ndarray, decision: str,
                             confidence: float, position: str = 'top-left') -> np.ndarray:
        """Draw decision overlay (delegates to OverlayRenderer)"""
        return self.overlay_renderer.draw_decision_overlay(frame, decision, confidence, position)
