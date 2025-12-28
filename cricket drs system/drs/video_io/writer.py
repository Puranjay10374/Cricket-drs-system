"""
Video Writer Module

Handles video file writing and frame output.
"""

import cv2
import numpy as np
from typing import Tuple


class VideoWriter:
    """Write video files with context manager support"""
    
    def __init__(self, 
                 output_path: str,
                 fps: float,
                 frame_size: Tuple[int, int],
                 codec: str = 'mp4v'):
        """
        Initialize video writer
        
        Args:
            output_path: Path for output video
            fps: Frames per second
            frame_size: (width, height) of frames
            codec: Video codec (default: mp4v for .mp4 files)
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.frame_count = 0
        
        # Create codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Create video writer
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if not self.writer.isOpened():
            raise ValueError(f"Could not create video writer for: {output_path}")
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a single frame to video
        
        Args:
            frame: Image frame to write
        """
        self.writer.write(frame)
        self.frame_count += 1
    
    def write_frames(self, frames: list):
        """
        Write multiple frames to video
        
        Args:
            frames: List of image frames
        """
        for frame in frames:
            self.write_frame(frame)
    
    def get_frame_count(self) -> int:
        """Get number of frames written"""
        return self.frame_count
    
    def release(self):
        """Release video writer resources"""
        if self.writer:
            self.writer.release()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.release()
