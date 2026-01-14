"""
Contour Filtering Utilities

Finds and filters contours for ball detection.
"""

import cv2
import numpy as np
from typing import Optional, List

from ...config import BallDetectionConfig


class ContourFilter:
    """Filters and selects ball contours"""
    
    def __init__(self, config: BallDetectionConfig):
        """
        Initialize contour filter
        
        Args:
            config: Ball detection configuration
        """
        self.config = config
    
    def find_ball_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the most likely ball contour
        
        Args:
            mask: Binary mask
            
        Returns:
            Ball contour or None if not found
        """
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Filter by area
        valid_contours = self._filter_by_area(contours)
        
        if not valid_contours:
            return None
        
        # Return largest valid contour
        return max(valid_contours, key=cv2.contourArea)
    
    def _filter_by_area(self, contours: List) -> List:
        """
        Filter contours by area constraints
        
        Args:
            contours: List of contours to filter
            
        Returns:
            Filtered list of contours within valid area range
        """
        return [
            c for c in contours 
            if self.config.min_area < cv2.contourArea(c) < self.config.max_area
        ]
