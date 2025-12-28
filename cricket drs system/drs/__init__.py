"""
Cricket DRS (Decision Review System) Package

A modular system for cricket ball tracking, trajectory prediction, and automated decisions.

This package contains:
- config/: Configuration management for all components
- detectors/: Ball detection strategies (color, contour, hybrid)
- tracker/: Ball tracking and quality metrics
- trajectory/: Polynomial fitting and trajectory prediction
- decision/: Decision making with confidence scoring
- renderer/: Video rendering and overlays
- video_io/: Video input/output operations
"""

__version__ = "2.0.0"

# Core tracking and analysis
from .tracker import BallTracker, TrajectoryProcessor, VideoAnnotator, QualityMetrics
from .trajectory import (
    TrajectoryAnalyzer, 
    TrajectoryPredictor,  # Backward compatibility
    PolynomialFitter, 
    PathPredictor,
    IntersectionFinder
)
from .decision import (
    DecisionMaker,
    DRSDecision,  # Backward compatibility
    ConfidenceCalculator,
    DecisionValidator,
    ResponseFormatter
)

# Detection strategies
from .detectors import (
    BallDetector, 
    ColorBasedDetector, 
    ContourBasedDetector, 
    HybridDetector
)

# Video I/O
from .video_io import VideoReader, VideoWriter, VideoProcessor

# Rendering
from .renderer import TrajectoryRenderer

# Configuration
from .config import (
    BallDetectionConfig,
    VideoConfig,
    TrajectoryConfig,
    StumpRegion,
    DecisionConfig,
    RenderConfig,
    DEFAULT_DETECTION_CONFIG,
    DEFAULT_VIDEO_CONFIG,
    DEFAULT_TRAJECTORY_CONFIG,
    DEFAULT_DECISION_CONFIG,
    DEFAULT_RENDER_CONFIG,
    DEFAULT_STUMP_REGION
)

__all__ = [
    # Main classes
    'BallTracker',
    'TrajectoryAnalyzer',
    'DecisionMaker',
    
    # Backward compatibility aliases
    'TrajectoryPredictor',
    'DRSDecision',
    
    # Tracker components
    'TrajectoryProcessor',
    'VideoAnnotator',
    'QualityMetrics',
    
    # Trajectory components
    'PolynomialFitter',
    'PathPredictor',
    'IntersectionFinder',
    
    # Decision components
    'ConfidenceCalculator',
    'DecisionValidator',
    'ResponseFormatter',
    
    # Detection strategies
    'BallDetector',
    'ColorBasedDetector',
    'ContourBasedDetector',
    'HybridDetector',
    
    # Video I/O
    'VideoReader',
    'VideoWriter',
    'VideoProcessor',
    
    # Rendering
    'TrajectoryRenderer',
    
    # Configuration classes
    'BallDetectionConfig',
    'VideoConfig',
    'TrajectoryConfig',
    'StumpRegion',
    'DecisionConfig',
    'RenderConfig',
    
    # Default configurations
    'DEFAULT_DETECTION_CONFIG',
    'DEFAULT_VIDEO_CONFIG',
    'DEFAULT_TRAJECTORY_CONFIG',
    'DEFAULT_DECISION_CONFIG',
    'DEFAULT_RENDER_CONFIG',
    'DEFAULT_STUMP_REGION',
]
