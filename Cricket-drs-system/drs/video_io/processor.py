"""
Video Processor Module

High-level video processing utilities and validation.
"""

import cv2
from pathlib import Path
from typing import Tuple, Dict, Callable

from ..config import VideoConfig
from .reader import VideoReader
from .writer import VideoWriter


class VideoProcessor:
    """High-level video processing and validation utilities"""
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict:
        """
        Get video file information without full processing
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video properties
        """
        with VideoReader(video_path) as reader:
            return reader.get_properties()
    
    @staticmethod
    def validate_video_file(file_path: str, config: VideoConfig) -> Tuple[bool, str]:
        """
        Validate video file against configuration constraints
        
        Args:
            file_path: Path to video file
            config: Video configuration with validation rules
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            return False, "File does not exist"
        
        # Check extension
        if path.suffix.lower() not in config.allowed_extensions:
            return False, f"Invalid extension. Allowed: {config.allowed_extensions}"
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > config.max_file_size_mb:
            return False, f"File too large: {size_mb:.1f}MB (max: {config.max_file_size_mb}MB)"
        
        # Try to open video
        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                return False, "Could not open video file"
            
            # Check if video has frames
            ret, _ = cap.read()
            if not ret:
                cap.release()
                return False, "Video file has no readable frames"
            
            cap.release()
        except Exception as e:
            return False, f"Error opening file: {str(e)}"
        
        return True, ""
    
    @staticmethod
    def copy_video_with_processing(input_path: str, 
                                   output_path: str,
                                   frame_processor: Callable) -> str:
        """
        Copy video while processing each frame
        
        Args:
            input_path: Input video path
            output_path: Output video path
            frame_processor: Function(frame_num, frame) -> processed_frame
            
        Returns:
            Output path
        """
        with VideoReader(input_path) as reader:
            props = reader.get_properties()
            
            with VideoWriter(output_path, props['fps'], props['frame_size']) as writer:
                for frame_num, frame in reader.frames():
                    # Process frame
                    processed_frame = frame_processor(frame_num, frame)
                    
                    # Write to output
                    writer.write_frame(processed_frame)
        
        return output_path
    
    @staticmethod
    def extract_frames(video_path: str, 
                      frame_numbers: list,
                      output_format: str = "frame_{:04d}.jpg") -> list:
        """
        Extract specific frames from video
        
        Args:
            video_path: Path to video file
            frame_numbers: List of frame indices to extract
            output_format: Format string for output filenames
            
        Returns:
            List of saved frame paths
        """
        saved_paths = []
        
        with VideoReader(video_path) as reader:
            for frame_num in frame_numbers:
                frame = reader.read_frame(frame_num)
                if frame is not None:
                    output_path = output_format.format(frame_num)
                    cv2.imwrite(output_path, frame)
                    saved_paths.append(output_path)
        
        return saved_paths
