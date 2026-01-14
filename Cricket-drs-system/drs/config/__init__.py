"""
Configuration Package

Modular configuration for the Cricket DRS System.

This package contains:
- detection_config.py: Ball detection parameters
- video_config.py: Video processing settings
- trajectory_config.py: Trajectory prediction settings
- stump_config.py: Stump region definition
- decision_config.py: Decision making thresholds
- render_config.py: Rendering and visualization settings
"""

from .detection_config import BallDetectionConfig, DEFAULT_DETECTION_CONFIG
from .video_config import VideoConfig, DEFAULT_VIDEO_CONFIG
from .trajectory_config import TrajectoryConfig, DEFAULT_TRAJECTORY_CONFIG
from .stump_config import StumpRegion, DEFAULT_STUMP_REGION, create_stump_region_for_frame
from .decision_config import DecisionConfig, DEFAULT_DECISION_CONFIG
from .render_config import RenderConfig, DEFAULT_RENDER_CONFIG

__all__ = [
    # Config classes
    'BallDetectionConfig',
    'VideoConfig',
    'TrajectoryConfig',
    'StumpRegion',
    'DecisionConfig',
    'RenderConfig',
    
    # Default instances
    'DEFAULT_DETECTION_CONFIG',
    'DEFAULT_VIDEO_CONFIG',
    'DEFAULT_TRAJECTORY_CONFIG',
    'DEFAULT_STUMP_REGION',
    'DEFAULT_DECISION_CONFIG',
    'DEFAULT_RENDER_CONFIG',
    
    # Functions
    'create_stump_region_for_frame',
]
