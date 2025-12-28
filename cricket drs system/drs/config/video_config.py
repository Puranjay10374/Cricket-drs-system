"""
Video Processing Configuration

Configuration for video input/output and processing.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VideoConfig:
    """Configuration for video processing"""
    
    # Processing
    frame_skip: int = 1
    max_file_size_mb: int = 100
    allowed_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov')
    
    # Output
    output_codec: str = 'mp4v'
    output_extension: str = '.mp4'


# Default configuration
DEFAULT_VIDEO_CONFIG = VideoConfig()
