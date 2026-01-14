"""
YOLO-based Ball Detector

Uses YOLOv8 for robust ball detection with deep learning.
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from .base_detector import BallDetector


class YOLODetector(BallDetector):
    """
    YOLOv8-based ball detector
    
    Uses pre-trained YOLOv8 model or custom-trained cricket ball model.
    Handles small balls, occlusions, and varying lighting conditions.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.25,
                 use_pretrained_sports_ball: bool = True):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to custom trained model (e.g., 'cricket_ball.pt')
                       If None, uses pretrained YOLOv8n
            confidence_threshold: Minimum detection confidence (0-1)
            use_pretrained_sports_ball: Use COCO 'sports ball' class if no custom model
        
        Raises:
            ImportError: If ultralytics package not installed
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics package required for YOLODetector. "
                "Install with: pip install ultralytics"
            )
        
        self.confidence_threshold = confidence_threshold
        self.use_pretrained_sports_ball = use_pretrained_sports_ball
        
        # Load model
        if model_path and Path(model_path).exists():
            # Custom trained model
            self.model = YOLO(model_path)
            self.custom_model = True
            print(f"✓ Loaded custom YOLO model: {model_path}")
        else:
            # Pretrained model (YOLOv8n = nano, fastest)
            self.model = YOLO('yolov8n.pt')
            self.custom_model = False
            print("✓ Loaded pretrained YOLOv8n model (sports ball detection)")
        
        # COCO class ID for sports ball (used with pretrained model)
        self.sports_ball_class = 32
        
        self.last_detection_confidence = 0.0
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect ball using YOLOv8
        
        Args:
            frame: BGR image from video
            
        Returns:
            Tuple of (x, y, radius) if ball found, None otherwise
        """
        # Run inference
        if self.custom_model:
            # Custom model: detect all classes
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        else:
            # Pretrained: filter to sports ball class only
            results = self.model(
                frame, 
                classes=[self.sports_ball_class] if self.use_pretrained_sports_ball else None,
                conf=self.confidence_threshold,
                verbose=False
            )
        
        # Extract best detection
        best_detection = self._extract_best_detection(results[0])
        
        return best_detection
    
    def _extract_best_detection(self, result) -> Optional[Tuple[int, int, int]]:
        """
        Extract ball position from YOLO results
        
        Selects detection with highest confidence.
        If multiple detections, prefers lower position (actual ball near ground).
        
        Args:
            result: YOLO detection result
            
        Returns:
            (x, y, radius) or None
        """
        if result.boxes is None or len(result.boxes) == 0:
            self.last_detection_confidence = 0.0
            return None
        
        boxes = result.boxes
        
        # Get all detections
        detections = []
        for i in range(len(boxes)):
            # Get bounding box center and size
            x_center, y_center, width, height = boxes.xywh[i].cpu().numpy()
            confidence = float(boxes.conf[i].cpu().numpy())
            
            # Calculate radius from bounding box
            radius = int(max(width, height) / 2)
            
            detections.append({
                'x': int(x_center),
                'y': int(y_center),
                'radius': radius,
                'confidence': confidence
            })
        
        if not detections:
            self.last_detection_confidence = 0.0
            return None
        
        # Select best detection
        # Prefer higher confidence, but also consider vertical position
        # (actual ball is typically lower in frame)
        best = max(detections, key=lambda d: d['confidence'] + (d['y'] / 10000))
        
        self.last_detection_confidence = best['confidence']
        
        return (best['x'], best['y'], best['radius'])
    
    def get_last_confidence(self) -> float:
        """Get confidence of last detection"""
        return self.last_detection_confidence
    
    def get_last_detected_color(self) -> Optional[str]:
        """
        Get color of last detected ball
        
        YOLO doesn't track color information, returns None.
        This method exists for compatibility with color-based detectors.
        """
        return None
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
