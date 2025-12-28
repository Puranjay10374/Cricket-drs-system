"""
Hybrid Ball Detector

Combines multiple detection strategies for better accuracy.
"""

import numpy as np
from typing import Optional, Tuple

from ..config import BallDetectionConfig
from .base_detector import BallDetector
from .color_detector import ColorBasedDetector
from .contour_detector import ContourBasedDetector


class HybridDetector(BallDetector):
    """Combines multiple detection strategies for better accuracy"""
    
    def __init__(self, config: BallDetectionConfig):
        """
        Initialize hybrid detector
        
        Args:
            config: Ball detection configuration
        """
        self.color_detector = ColorBasedDetector(config)
        self.contour_detector = ContourBasedDetector(config)
        
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Try multiple detection strategies
        
        Tries color-based detection first (faster), then falls back
        to contour-based detection if needed.
        
        Args:
            frame: BGR image from video
            
        Returns:
            Tuple of (x, y, radius) if ball found, None otherwise
        """
        # Try color-based first (faster)
        result = self.color_detector.detect(frame)
        
        if result:
            return result
        
        # Fall back to contour-based
        return self.contour_detector.detect(frame)
