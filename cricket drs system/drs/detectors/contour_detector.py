"""
Contour-Based Ball Detector

Uses shape circularity analysis to detect balls.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from ..config import BallDetectionConfig
from .base_detector import BallDetector
from .utils import ShapeAnalyzer


class ContourBasedDetector(BallDetector):
    """Detects ball using contour shape analysis (circularity)"""
    
    def __init__(self, config: BallDetectionConfig, circularity_threshold: float = 0.7):
        """
        Initialize contour-based detector
        
        Args:
            config: Ball detection configuration
            circularity_threshold: Minimum circularity (0-1) for valid ball
        """
        self.config = config
        self.circularity_threshold = circularity_threshold
        self.shape_analyzer = ShapeAnalyzer(config, circularity_threshold)
        
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect ball using shape circularity"""
        
        # Preprocessing
        edges = self._preprocess_frame(frame)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find most circular contour
        best_contour = self.shape_analyzer.find_most_circular(contours)
        
        if best_contour is None:
            return None
        
        # Extract position
        return self._extract_position(best_contour)
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for edge detection
        
        Args:
            frame: BGR image
            
        Returns:
            Edge-detected image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def _extract_position(self, contour: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Extract ball position from contour"""
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        
        if radius > self.config.min_radius:
            return (int(x), int(y), int(radius))
        
        return None
