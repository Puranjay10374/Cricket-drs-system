"""
Tracker Package

Modular ball tracking components for cricket DRS.

This package contains:
- ball_tracker.py: Core ball tracking logic
- trajectory_processor.py: Trajectory data processing
- video_annotator.py: Video annotation and visualization
- quality_metrics.py: Tracking quality calculations
- kalman_filter.py: Kalman filter for prediction and smoothing
- manual_selector.py: Manual ball selection UI
"""

from .ball_tracker import BallTracker
from .trajectory_processor import TrajectoryProcessor
from .video_annotator import VideoAnnotator
from .quality_metrics import QualityMetrics
from .kalman_filter import BallKalmanFilter
from .manual_selector import ManualBallSelector, get_manual_ball_position

__all__ = [
    'BallTracker',
    'TrajectoryProcessor',
    'VideoAnnotator',
    'QualityMetrics',
    'BallKalmanFilter',
    'ManualBallSelector',
    'get_manual_ball_position'
]

