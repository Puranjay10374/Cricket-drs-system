"""
Ball Detectors Package

Modular ball detection strategies for cricket DRS.

This package contains:
- base_detector.py: Abstract detector interface
- color_detector.py: HSV color-based detection
- contour_detector.py: Shape circularity-based detection
- hybrid_detector.py: Combined detection strategies
- multi_color_detector.py: Multi-color ball detection (red/white/pink)
- yolo_detector.py: YOLOv8 deep learning detection
- utils/: Shared detection utilities
"""

from .base_detector import BallDetector
from .color_detector import ColorBasedDetector
from .contour_detector import ContourBasedDetector
from .hybrid_detector import HybridDetector
from .multi_color_detector import MultiColorDetector, BallColorConfig
from .yolo_detector import YOLODetector
from .adaptive_detector import AdaptiveBallDetector

__all__ = [
    'BallDetector',
    'ColorBasedDetector',
    'ContourBasedDetector',
    'HybridDetector',
    'MultiColorDetector',
    'BallColorConfig',
    'YOLODetector',
    'AdaptiveBallDetector',
]
