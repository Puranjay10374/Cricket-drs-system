"""
Shape Analysis Utilities

Analyzes contour shapes for circularity and ball detection.
"""

import cv2
import numpy as np
from typing import Optional, List

from ...config import BallDetectionConfig


class ShapeAnalyzer:
    """Analyzes contour shapes for ball detection"""
    
    def __init__(self, config: BallDetectionConfig, circularity_threshold: float = 0.7):
        """
        Initialize shape analyzer
        
        Args:
            config: Ball detection configuration
            circularity_threshold: Minimum circularity (0-1) for valid ball
        """
        self.config = config
        self.circularity_threshold = circularity_threshold
    
    def calculate_circularity(self, contour: np.ndarray) -> float:
        """
        Calculate circularity of contour (1.0 = perfect circle)
        
        Args:
            contour: Contour to analyze
            
        Returns:
            Circularity score (0-1)
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Circularity formula: 4π*area / perimeter²
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return min(1.0, circularity)
    
    def find_most_circular(self, contours: List) -> Optional[np.ndarray]:
        """
        Find the most circular contour within size constraints
        
        Args:
            contours: List of contours to analyze
            
        Returns:
            Most circular contour or None if none meet criteria
        """
        best_contour = None
        best_circularity = 0.0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Check area constraints
            if not (self.config.min_area < area < self.config.max_area):
                continue
            
            # Calculate circularity
            circularity = self.calculate_circularity(contour)
            
            # Check if this is better and meets threshold
            if circularity > best_circularity and circularity >= self.circularity_threshold:
                best_circularity = circularity
                best_contour = contour
        
        return best_contour
