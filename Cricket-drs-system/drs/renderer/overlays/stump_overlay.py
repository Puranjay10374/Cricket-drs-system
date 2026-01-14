"""
Stump Overlay Renderer

Renders stump region indicators.
"""

import cv2
import numpy as np
from typing import Tuple


class StumpOverlay:
    """Renders stump region visual indicators"""
    
    @staticmethod
    def draw(frame: np.ndarray,
            stump_x: float,
            stump_y_top: float,
            stump_y_bottom: float,
            color: Tuple[int, int, int] = (255, 255, 0),
            thickness: int = 2,
            bar_width: int = 20) -> np.ndarray:
        """
        Draw stump region indicator
        
        Args:
            frame: Video frame
            stump_x: X coordinate of stumps
            stump_y_top: Top Y coordinate
            stump_y_bottom: Bottom Y coordinate
            color: Line color (BGR)
            thickness: Line thickness
            bar_width: Width of horizontal bars
            
        Returns:
            Frame with stump region drawn
        """
        x = int(stump_x)
        y_top = int(stump_y_top)
        y_bottom = int(stump_y_bottom)
        
        # Draw vertical line representing stumps
        cv2.line(frame, (x, y_top), (x, y_bottom), color, thickness)
        
        # Draw horizontal bars at top and bottom
        cv2.line(frame, (x - bar_width, y_top), (x + bar_width, y_top), color, thickness)
        cv2.line(frame, (x - bar_width, y_bottom), (x + bar_width, y_bottom), color, thickness)
        
        return frame
