"""
Video Reader Module

Handles video file reading and frame iteration.
"""

import cv2
import numpy as np
from typing import Iterator, Tuple, Dict


class VideoReader:
    """Read video files frame by frame with context manager support"""
    
    def __init__(self, video_path: str, frame_skip: int = 1):
        """
        Initialize video reader
        
        Args:
            video_path: Path to video file
            frame_skip: Process every Nth frame (1 = all frames)
        """
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self._extract_properties()
    
    def _extract_properties(self):
        """Extract video metadata properties"""
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (self.width, self.height)
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
    
    def get_properties(self) -> Dict:
        """
        Get video properties
        
        Returns:
            Dictionary with video metadata
        """
        return {
            'total_frames': self.total_frames,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'frame_size': self.frame_size,
            'duration': self.duration
        }
    
    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterate through video frames
        
        Yields:
            Tuple of (frame_number, frame_image)
        """
        frame_num = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Yield frame if it matches skip pattern
            if frame_num % self.frame_skip == 0:
                yield (frame_num, frame)
            
            frame_num += 1
    
    def read_frame(self, frame_number: int) -> np.ndarray:
        """
        Read specific frame by number
        
        Args:
            frame_number: Frame index to read
            
        Returns:
            Frame image or None if not available
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """Release video capture resources"""
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.release()
