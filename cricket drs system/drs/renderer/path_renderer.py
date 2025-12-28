"""
Path Rendering Module

Renders trajectory paths (actual and predicted).
"""

import numpy as np
from typing import List, Tuple

from ..config import RenderConfig
from .utils import LineDrawer


class PathRenderer:
    """Renders trajectory paths"""
    
    def __init__(self, config: RenderConfig):
        """
        Initialize path renderer
        
        Args:
            config: Rendering configuration
        """
        self.config = config
    
    def draw_actual_path(self, 
                        frame: np.ndarray, 
                        points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Draw actual trajectory path as connected lines
        
        Args:
            frame: Video frame
            points: List of (x, y) coordinates
            
        Returns:
            Frame with path drawn
        """
        if len(points) < 2:
            return frame
        
        # Draw solid line through points
        LineDrawer.draw_solid_line(
            frame,
            points,
            self.config.path_color,
            self.config.path_thickness
        )
        
        return frame
    
    def draw_predicted_path(self,
                           frame: np.ndarray,
                           points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Draw predicted trajectory path
        
        Args:
            frame: Video frame
            points: List of predicted (x, y) coordinates
            
        Returns:
            Frame with predicted path drawn
        """
        if len(points) < 2:
            return frame
        
        # Convert to integer coordinates
        int_points = [(int(x), int(y)) for x, y in points]
        
        if self.config.predicted_path_style == 'dashed':
            # Draw dashed line
            LineDrawer.draw_dashed_line(
                frame,
                int_points,
                self.config.predicted_path_color,
                self.config.predicted_path_thickness
            )
        else:
            # Draw solid line
            LineDrawer.draw_solid_line(
                frame,
                int_points,
                self.config.predicted_path_color,
                self.config.predicted_path_thickness
            )
        
        return frame