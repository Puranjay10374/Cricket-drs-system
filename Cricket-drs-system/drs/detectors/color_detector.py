"""
Color-Based Ball Detector

Uses HSV color space to detect red cricket balls.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from ..config import BallDetectionConfig
from .base_detector import BallDetector
from .utils import MorphologyProcessor, ContourFilter, BallExtractor


class ColorBasedDetector(BallDetector):
    """Detects ball using HSV color-based detection"""
    
    def __init__(self, config: BallDetectionConfig):
        """
        Initialize color-based detector
        
        Args:
            config: Ball detection configuration
        """
        self.config = config
        self.lower_red1 = np.array(config.color_lower1)
        self.upper_red1 = np.array(config.color_upper1)
        self.lower_red2 = np.array(config.color_lower2)
        self.upper_red2 = np.array(config.color_upper2)
        
        # Utility processors
        self.morphology = MorphologyProcessor(config)
        self.contour_filter = ContourFilter(config)
        self.ball_extractor = BallExtractor(config)
        
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect ball using color masking and contour detection"""
        
        # Convert BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for red color (red wraps in HSV)
        mask = self._create_red_mask(hsv)
        
        # Apply morphological operations
        mask = self.morphology.process(mask)
        
        # Find and filter contours
        ball_contour = self.contour_filter.find_ball_contour(mask)
        
        if ball_contour is None:
            return None
        
        # Extract ball position
        return self.ball_extractor.extract_position(ball_contour)
    
    def _create_red_mask(self, hsv: np.ndarray) -> np.ndarray:
        """
        Create binary mask for red color
        
        Red wraps around in HSV space, so we need two ranges
        
        Args:
            hsv: Image in HSV color space
            
        Returns:
            Binary mask with red regions
        """
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        return cv2.bitwise_or(mask1, mask2)
    
    def apply_color_mask(self, hsv: np.ndarray) -> np.ndarray:
        """
        Apply color mask (public method for MultiColorDetector)
        
        Args:
            hsv: Image in HSV color space
            
        Returns:
            Binary mask with color regions
        """
        return self._create_red_mask(hsv)
