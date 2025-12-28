"""
Adaptive Ball Detector

Learns ball color from manual selection and adapts detection parameters.
Much more effective than generic color ranges.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from .base_detector import BallDetector


class AdaptiveBallDetector(BallDetector):
    """Detector that adapts to the specific ball color in the video"""
    
    def __init__(self):
        self.ball_template = None
        self.ball_color_range = None
        self.template_size = (30, 30)
        self.learned = False
        
    def learn_from_selection(self, frame: np.ndarray, x: int, y: int, radius: int):
        """
        Learn ball appearance from manual selection
        
        Args:
            frame: BGR image
            x, y: Ball center position
            radius: Ball radius
        """
        # Extract ball region (smaller to avoid background)
        r = max(int(radius * 0.7), 5)  # Use 70% of radius to focus on ball center
        y1, y2 = max(0, y - r), min(frame.shape[0], y + r)
        x1, x2 = max(0, x - r), min(frame.shape[1], x + r)
        
        ball_region = frame[y1:y2, x1:x2]
        
        if ball_region.size == 0:
            return
        
        # Store template for matching
        self.ball_template = cv2.resize(ball_region, self.template_size)
        
        # Analyze actual ball color in this video
        hsv_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
        
        # Use median instead of mean (more robust to outliers)
        h_median = np.median(hsv_region[:, :, 0])
        s_median = np.median(hsv_region[:, :, 1])
        v_median = np.median(hsv_region[:, :, 2])
        
        # Calculate MAD (median absolute deviation) for robust std estimate
        h_mad = np.median(np.abs(hsv_region[:, :, 0] - h_median))
        s_mad = np.median(np.abs(hsv_region[:, :, 1] - s_median))
        v_mad = np.median(np.abs(hsv_region[:, :, 2] - v_median))
        
        # Create tighter color range (±1.5 MAD, more conservative)
        h_min = max(0, int(h_median - 1.5 * h_mad))
        h_max = min(179, int(h_median + 1.5 * h_mad))
        s_min = max(30, int(s_median - 2 * s_mad))  # Keep some saturation
        s_max = min(255, int(s_median + 2 * s_mad))
        v_min = max(50, int(v_median - 2 * v_mad))  # Keep some brightness
        v_max = min(255, int(v_median + 2 * v_mad))
        
        self.ball_color_range = {
            'lower': np.array([h_min, s_min, v_min]),
            'upper': np.array([h_max, s_max, v_max])
        }
        
        self.learned = True
        
        print(f"✓ Learned ball color: H={h_median:.0f}±{h_mad:.0f}, S={s_median:.0f}±{s_mad:.0f}, V={v_median:.0f}±{v_mad:.0f}")
        print(f"  Color range: H[{h_min}-{h_max}] S[{s_min}-{s_max}] V[{v_min}-{v_max}]")
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect ball using learned color and template matching
        
        Args:
            frame: BGR image
            
        Returns:
            (x, y, radius) or None
        """
        if not self.learned:
            # Fallback to basic detection
            return self._detect_basic(frame)
        
        # Method 1: Adaptive color detection (finds multiple candidates)
        color_candidates = self._detect_by_color_multi(frame)
        
        # Method 2: Template matching (more precise)
        template_detection = self._detect_by_template(frame)
        
        # If template matching found something, prefer it
        if template_detection is not None:
            # Check if any color candidate is near template match
            tx, ty, tr = template_detection
            for cx, cy, cr in color_candidates:
                dist = np.sqrt((tx - cx)**2 + (ty - cy)**2)
                if dist < 30:  # Within 30px
                    # Use color detection position but trust template
                    return (cx, cy, cr)
            # No color match near template, use template
            return template_detection
        
        # No template match, use best color candidate
        if color_candidates:
            return color_candidates[0]  # Already sorted by score
        
        return None
    
    def _detect_by_color_multi(self, frame: np.ndarray) -> list:
        """Detect multiple candidates using learned color range"""
        if self.ball_color_range is None:
            return []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask with learned color range
        mask = cv2.inRange(hsv, self.ball_color_range['lower'], self.ball_color_range['upper'])
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Find all candidates
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20 or area > 2000:  # Size filter
                continue
            
            # Get enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if radius < 2 or radius > 30:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Score based on size and circularity
            size_score = 1 - abs(radius - 10) / 20  # Prefer ~10px radius
            score = circularity * 0.7 + size_score * 0.3
            
            if score > 0.3:
                candidates.append((int(x), int(y), int(radius), score))
        
        # Sort by score (best first)
        candidates.sort(key=lambda c: c[3], reverse=True)
        
        # Return top 3 candidates without score
        return [(x, y, r) for x, y, r, s in candidates[:3]]
    
    def _detect_by_template(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect using template matching"""
        if self.ball_template is None:
            return None
        
        # Convert to grayscale for template matching
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.ball_template, cv2.COLOR_BGR2GRAY)
        
        # Match template
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Only accept good matches
        if max_val > 0.6:  # Confidence threshold
            x = max_loc[0] + self.template_size[0] // 2
            y = max_loc[1] + self.template_size[1] // 2
            radius = self.template_size[0] // 2
            return (x, y, radius)
        
        return None
    
    def _detect_basic(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Basic detection without learning (fallback)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Generic pink/red detection
        lower_pink = np.array([145, 50, 50])
        upper_pink = np.array([165, 255, 255])
        
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        
        if radius > 2 and radius < 30:
            return (int(x), int(y), int(radius))
        
        return None
