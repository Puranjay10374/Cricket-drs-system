"""
Ball Rendering Module

Renders ball markers on video frames.
"""

import cv2
import numpy as np
from typing import Tuple

from ..config import RenderConfig


class BallRenderer:
    """Renders ball position markers"""
    
    def __init__(self, config: RenderConfig):
        """
        Initialize ball renderer
        
        Args:
            config: Rendering configuration
        """
        self.config = config
    
    def draw(self, 
             frame: np.ndarray, 
             x: int, 
             y: int, 
             radius: int) -> np.ndarray:
        """
        Draw ball marker on frame
        
        Args:
            frame: Video frame
            x, y: Ball center coordinates
            radius: Ball radius
            
        Returns:
            Frame with ball drawn
        """
        # Draw circle around ball
        cv2.circle(
            frame, 
            (x, y), 
            radius, 
            self.config.ball_circle_color, 
            self.config.ball_circle_thickness
        )
        
        # Draw center point
        cv2.circle(
            frame, 
            (x, y), 
            self.config.ball_center_radius, 
            self.config.ball_center_color, 
            -1
        )
        
        return frame
    
    def draw_multiple(self,
                     frame: np.ndarray,
                     positions: list) -> np.ndarray:
        """
        Draw multiple ball positions
        
        Args:
            frame: Video frame
            positions: List of (x, y, radius) tuples
            
        Returns:
            Frame with all balls drawn
        """
        for x, y, radius in positions:
            self.draw(frame, x, y, radius)
        
        return frame
