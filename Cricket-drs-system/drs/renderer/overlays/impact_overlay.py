"""
Impact Point Overlay Renderer

Renders impact point markers and crosshairs.
"""

import cv2
import numpy as np
from typing import Tuple


class ImpactOverlay:
    """Renders impact point markers"""
    
    @staticmethod
    def draw(frame: np.ndarray,
            x: float,
            y: float,
            label: str = "IMPACT",
            color: Tuple[int, int, int] = (0, 0, 255),
            crosshair_size: int = 15,
            circle_radius: int = 10) -> np.ndarray:
        """
        Draw impact point marker
        
        Args:
            frame: Video frame
            x, y: Impact coordinates
            label: Text label
            color: Marker color (BGR)
            crosshair_size: Size of crosshair
            circle_radius: Radius of circle
            
        Returns:
            Frame with impact point drawn
        """
        x_int, y_int = int(x), int(y)
        
        # Draw crosshair
        ImpactOverlay._draw_crosshair(frame, x_int, y_int, crosshair_size, color)
        
        # Draw circle
        cv2.circle(frame, (x_int, y_int), circle_radius, color, 2)
        
        # Draw label
        ImpactOverlay._draw_label(frame, x_int, y_int, label, color)
        
        return frame
    
    @staticmethod
    def _draw_crosshair(frame: np.ndarray,
                       x: int,
                       y: int,
                       size: int,
                       color: Tuple[int, int, int]) -> None:
        """Draw crosshair at position"""
        cv2.line(frame, (x - size, y), (x + size, y), color, 2)
        cv2.line(frame, (x, y - size), (x, y + size), color, 2)
    
    @staticmethod
    def _draw_label(frame: np.ndarray,
                   x: int,
                   y: int,
                   label: str,
                   color: Tuple[int, int, int]) -> None:
        """Draw text label near impact point"""
        cv2.putText(
            frame,
            label,
            (x + 15, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
