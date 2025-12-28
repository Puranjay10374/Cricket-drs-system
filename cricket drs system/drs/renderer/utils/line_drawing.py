"""
Line Drawing Utilities

Utilities for drawing different line styles on video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple


class LineDrawer:
    """Utilities for drawing lines on frames"""
    
    @staticmethod
    def draw_solid_line(frame: np.ndarray,
                       points: List[Tuple[int, int]],
                       color: Tuple[int, int, int],
                       thickness: int) -> None:
        """
        Draw solid line through points
        
        Args:
            frame: Video frame
            points: List of (x, y) coordinates
            color: Line color (BGR)
            thickness: Line thickness
        """
        for i in range(1, len(points)):
            cv2.line(
                frame,
                points[i-1],
                points[i],
                color,
                thickness
            )
    
    @staticmethod
    def draw_dashed_line(frame: np.ndarray,
                        points: List[Tuple[int, int]],
                        color: Tuple[int, int, int],
                        thickness: int,
                        dash_length: int = 10) -> None:
        """
        Draw dashed line through points
        
        Args:
            frame: Video frame
            points: List of (x, y) coordinates
            color: Line color (BGR)
            thickness: Line thickness
            dash_length: Length of each dash in pixels
        """
        for i in range(1, len(points)):
            p1, p2 = points[i-1], points[i]
            
            # Calculate line parameters
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            if dist == 0:
                continue
            
            # Draw dashes along the line
            num_dashes = int(dist / (2 * dash_length))
            
            for j in range(num_dashes):
                # Calculate dash start and end
                t1 = j * 2 * dash_length / dist
                t2 = (j * 2 + 1) * dash_length / dist
                
                if t2 > 1.0:
                    t2 = 1.0
                
                x1 = int(p1[0] + t1 * (p2[0] - p1[0]))
                y1 = int(p1[1] + t1 * (p2[1] - p1[1]))
                x2 = int(p1[0] + t2 * (p2[0] - p1[0]))
                y2 = int(p1[1] + t2 * (p2[1] - p1[1]))
                
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    
    @staticmethod
    def draw_dotted_line(frame: np.ndarray,
                        points: List[Tuple[int, int]],
                        color: Tuple[int, int, int],
                        dot_spacing: int = 10) -> None:
        """
        Draw dotted line through points
        
        Args:
            frame: Video frame
            points: List of (x, y) coordinates
            color: Line color (BGR)
            dot_spacing: Spacing between dots in pixels
        """
        for i in range(1, len(points)):
            p1, p2 = points[i-1], points[i]
            
            # Calculate line parameters
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            if dist == 0:
                continue
            
            # Draw dots along the line
            num_dots = int(dist / dot_spacing)
            
            for j in range(num_dots):
                t = j * dot_spacing / dist
                x = int(p1[0] + t * (p2[0] - p1[0]))
                y = int(p1[1] + t * (p2[1] - p1[1]))
                cv2.circle(frame, (x, y), 2, color, -1)
