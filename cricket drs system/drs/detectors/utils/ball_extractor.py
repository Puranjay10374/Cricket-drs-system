"""
Ball Position Extraction Utilities

Extracts ball position and radius from contours.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from ...config import BallDetectionConfig


class BallExtractor:
    """Extracts ball position from contours"""
    
    def __init__(self, config: BallDetectionConfig):
        """
        Initialize ball extractor
        
        Args:
            config: Ball detection configuration
        """
        self.config = config
    
    def extract_position(self, contour: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Extract ball center and radius from contour
        
        Args:
            contour: Ball contour
            
        Returns:
            Tuple of (x, y, radius) or None if invalid
        """
        # Find minimum enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        
        # Validate radius meets minimum requirement
        if radius > self.config.min_radius:
            return (int(x), int(y), int(radius))
        
        return None
