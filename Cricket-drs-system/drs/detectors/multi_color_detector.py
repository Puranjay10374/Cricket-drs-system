"""
Multi-Color Ball Detector

Detects cricket balls of different colors (red, white, pink).
Automatically tries all color configurations and returns the best match.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

from .base_detector import BallDetector
from .hybrid_detector import HybridDetector
from ..config import BallDetectionConfig


class BallColorConfig:
    """Pre-configured color ranges for different cricket balls"""
    
    @staticmethod
    def get_red_config() -> BallDetectionConfig:
        """
        Red cricket ball configuration (Test cricket)
        HSV: Hue 0-10 and 170-180 (red wraps around)
        """
        return BallDetectionConfig(
            color_lower1=(0, 100, 80),  # Increased saturation/value to avoid skin tones
            color_upper1=(10, 255, 255),
            color_lower2=(170, 100, 80),  # Increased saturation/value
            color_upper2=(180, 255, 255),
            min_area=30,
            max_area=800,  # Reduced to reject large objects (bowler's head)
            min_radius=3,
            max_radius=25  # Reduced max radius to reject large detections
        )
    
    @staticmethod
    def get_white_config() -> BallDetectionConfig:
        """
        White cricket ball configuration (ODI, T20)
        HSV: Low saturation, high value
        """
        return BallDetectionConfig(
            color_lower1=(0, 0, 160),
            color_upper1=(180, 50, 255),
            color_lower2=(0, 0, 160),
            color_upper2=(180, 50, 255),
            min_area=80,
            max_area=2500,
            min_radius=6,
            max_radius=45
        )
    
    @staticmethod
    def get_pink_config() -> BallDetectionConfig:
        """
        Pink cricket ball configuration (Day-Night Test)
        HSV: Pink/Magenta range
        """
        return BallDetectionConfig(
            color_lower1=(145, 80, 80),  # Tighter pink range, avoid skin
            color_upper1=(165, 255, 255),
            color_lower2=(0, 80, 80),
            color_upper2=(10, 255, 255),
            min_area=40,
            max_area=600,  # Reduced to reject large objects
            min_radius=4,
            max_radius=20  # Reduced max radius
        )


class MultiColorDetector(BallDetector):
    """
    Detects cricket balls of multiple colors automatically
    
    Tries red, white, and pink detection and returns the best match
    based on detection confidence (circle size and mask quality).
    """
    
    def __init__(self):
        """Initialize detectors for all ball colors"""
        self.detectors = {
            'red': HybridDetector(BallColorConfig.get_red_config()),
            'white': HybridDetector(BallColorConfig.get_white_config()),
            'pink': HybridDetector(BallColorConfig.get_pink_config())
        }
        self.last_detected_color = None
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect ball in frame using multiple color detectors with smart filtering
        
        Args:
            frame: BGR image from video
            
        Returns:
            Tuple of (x, y, radius) for best detection, None if no ball found
        """
        detections: List[Tuple[str, Tuple[int, int, int], float]] = []
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        
        # Try each color detector
        for color_name, detector in self.detectors.items():
            result = detector.detect(frame)
            
            if result is not None:
                x, y, radius = result
                
                # STRICT filtering to reject bowler body parts
                
                # 1. SIZE FILTER: Ball must be small (reject head/shoulder)
                if radius > 20:  # Maximum 20 pixels (relaxed from 15)
                    continue
                if radius < 2:   # Minimum size - too small to be reliable
                    continue
                
                # 2. POSITION FILTER: Ball must be in lower 90% of frame (pitch area)
                vertical_position = y / frame_height
                if vertical_position < 0.1:  # Reject detections in top 10% only
                    continue
                
                # 3. ROUNDNESS CHECK: Ball must be reasonably circular
                circularity = self._check_circularity(frame, x, y, radius, detector)
                if circularity < 0.5:  # Must be at least 50% circular (relaxed from 70%)
                    continue
                
                # 4. Small objects (balls) score higher
                size_score = max(0, 1.0 - (radius / 20.0))
                
                # 5. Objects in lower frame score higher
                position_score = min(vertical_position * 1.2, 1.0)
                
                # 6. Objects near center score higher
                horizontal_center = abs(x - frame_width/2) / (frame_width/2)
                center_score = 1.0 - (horizontal_center * 0.2)
                
                # 7. Base confidence from color/shape matching
                confidence = self._calculate_confidence(frame, x, y, radius, detector)
                
                # Combined score: balanced weighting
                final_score = (
                    confidence * 0.30 +      # Color match
                    size_score * 0.25 +      # Small size
                    circularity * 0.20 +     # Round shape
                    position_score * 0.15 +  # Low position
                    center_score * 0.10      # Centered
                )
                
                detections.append((color_name, result, final_score))
        
        # Return best detection (highest score)
        if detections:
            best_color, best_result, best_confidence = max(detections, key=lambda x: x[2])
            self.last_detected_color = best_color
            return best_result
        
        return None
    
    def _check_circularity(self,
                          frame: np.ndarray,
                          x: int,
                          y: int,
                          radius: int,
                          detector: HybridDetector) -> float:
        """
        Check how circular the detected object is
        
        Rejects irregular shapes like heads, shoulders, hands
        
        Args:
            frame: Original frame
            x, y, radius: Detected circle
            detector: Detector used
            
        Returns:
            Circularity score 0-1 (1 = perfect circle)
        """
        # Get color mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = detector.color_detector.apply_color_mask(hsv)
        
        # Extract region around detection
        r = radius + 5
        y1, y2 = max(0, y-r), min(frame.shape[0], y+r)
        x1, x2 = max(0, x-r), min(frame.shape[1], x+r)
        roi = color_mask[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.0
        
        # Find contours in ROI
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 10:
            return 0.0
        
        # Calculate circularity: 4π×area / perimeter²
        # Perfect circle = 1.0, irregular shape < 0.7
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        return min(1.0, circularity)
    
    def _calculate_confidence(self, 
                             frame: np.ndarray,
                             x: int, 
                             y: int, 
                             radius: int,
                             detector: HybridDetector) -> float:
        """
        Calculate detection confidence score
        
        Args:
            frame: Original frame
            x, y, radius: Detected circle
            detector: Detector that found the ball
            
        Returns:
            Confidence score (0-1, higher is better)
        """
        # Create mask for detected circle
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        
        # Get color mask from detector's color component
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = detector.color_detector.apply_color_mask(hsv)
        
        # Calculate overlap between circle and color mask
        overlap = cv2.bitwise_and(mask, color_mask)
        overlap_ratio = np.count_nonzero(overlap) / np.count_nonzero(mask)
        
        # Radius score (prefer reasonable sizes)
        radius_score = min(radius / 30.0, 1.0)  # Normalize to 0-1
        
        # Combined confidence
        confidence = (overlap_ratio * 0.7) + (radius_score * 0.3)
        
        return confidence
    
    def get_last_detected_color(self) -> Optional[str]:
        """
        Get the color of the last detected ball
        
        Returns:
            'red', 'white', 'pink', or None
        """
        return self.last_detected_color
    
    def get_detection_info(self) -> Dict:
        """
        Get information about available detectors
        
        Returns:
            Dictionary with detector information
        """
        return {
            'supported_colors': list(self.detectors.keys()),
            'last_detected_color': self.last_detected_color,
            'detector_count': len(self.detectors)
        }
