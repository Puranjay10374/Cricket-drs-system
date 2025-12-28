"""
Text Overlay Renderer

Renders decision text and other informational overlays.
"""

import cv2
import numpy as np
from typing import Tuple


class TextOverlay:
    """Renders text overlays and decision information"""
    
    @staticmethod
    def draw_decision(frame: np.ndarray,
                     decision: str,
                     confidence: float,
                     position: str = 'top-left') -> np.ndarray:
        """
        Draw decision text overlay
        
        Args:
            frame: Video frame
            decision: Decision text (OUT/NOT OUT)
            confidence: Confidence score
            position: Text position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
            
        Returns:
            Frame with decision text
        """
        # Prepare text
        text1 = f"Decision: {decision}"
        text2 = f"Confidence: {confidence:.1%}"
        
        # Choose color based on decision
        color = TextOverlay._get_decision_color(decision)
        
        # Calculate position
        x, y = TextOverlay._calculate_position(frame, position)
        
        # Draw semi-transparent background
        TextOverlay._draw_background(frame, x, y)
        
        # Draw text
        cv2.putText(frame, text1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, text2, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    @staticmethod
    def draw_info_text(frame: np.ndarray,
                      text: str,
                      position: Tuple[int, int],
                      color: Tuple[int, int, int] = (255, 255, 255),
                      font_scale: float = 0.6,
                      thickness: int = 2) -> np.ndarray:
        """
        Draw simple info text
        
        Args:
            frame: Video frame
            text: Text to display
            position: (x, y) position
            color: Text color (BGR)
            font_scale: Font size
            thickness: Text thickness
            
        Returns:
            Frame with text
        """
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness
        )
        return frame
    
    @staticmethod
    def _get_decision_color(decision: str) -> Tuple[int, int, int]:
        """Get color based on decision type"""
        if decision == "OUT":
            return (0, 0, 255)  # Red
        elif decision == "NOT OUT":
            return (0, 255, 0)  # Green
        else:
            return (0, 255, 255)  # Yellow
    
    @staticmethod
    def _calculate_position(frame: np.ndarray, position: str) -> Tuple[int, int]:
        """Calculate text position based on frame and position string"""
        if position == 'top-left':
            return (10, 30)
        elif position == 'top-right':
            return (frame.shape[1] - 300, 30)
        elif position == 'bottom-left':
            return (10, frame.shape[0] - 40)
        else:  # bottom-right
            return (frame.shape[1] - 300, frame.shape[0] - 40)
    
    @staticmethod
    def _draw_background(frame: np.ndarray, x: int, y: int) -> None:
        """Draw semi-transparent background for text"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - 25), (x + 290, y + 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
