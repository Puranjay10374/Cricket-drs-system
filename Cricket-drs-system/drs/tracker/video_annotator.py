"""
Video Annotation

Creates annotated videos with ball tracking visualization.
"""

import numpy as np
from typing import List, Tuple, Optional

from ..video_io import VideoReader, VideoWriter
from ..renderer import TrajectoryRenderer
from ..config import RenderConfig
from .trajectory_processor import TrajectoryProcessor


class VideoAnnotator:
    """Annotates videos with ball trajectory visualization"""
    
    def __init__(self, renderer: Optional[TrajectoryRenderer] = None):
        """
        Initialize video annotator
        
        Args:
            renderer: Optional custom renderer (uses default if not provided)
        """
        self.renderer = renderer or TrajectoryRenderer(RenderConfig())
    
    def annotate_video(self,
                      input_video: str,
                      output_video: str,
                      trajectory: List[Tuple]) -> str:
        """
        Create annotated video with ball trajectory
        
        Args:
            input_video: Path to input video
            output_video: Path to save annotated video
            trajectory: List of (frame_num, x, y, radius) tuples
            
        Returns:
            Path to output video
        """
        # Prepare trajectory data
        traj_dict = TrajectoryProcessor.create_lookup_dict(trajectory)
        all_positions = TrajectoryProcessor.extract_positions(trajectory)
        
        # Process video
        with VideoReader(input_video) as reader:
            props = reader.get_properties()
            
            with VideoWriter(output_video, props['fps'], props['frame_size']) as writer:
                for frame_num, frame in reader.frames():
                    # Annotate frame
                    annotated_frame = self._annotate_frame(
                        frame, 
                        frame_num, 
                        trajectory, 
                        traj_dict, 
                        all_positions
                    )
                    
                    writer.write_frame(annotated_frame)
        
        return output_video
    
    def _annotate_frame(self,
                       frame: np.ndarray,
                       frame_num: int,
                       trajectory: List[Tuple],
                       traj_dict: dict,
                       all_positions: List[Tuple]) -> np.ndarray:
        """
        Annotate a single frame with trajectory and ball position
        
        Args:
            frame: Video frame
            frame_num: Current frame number
            trajectory: Full trajectory data
            traj_dict: Frame number to position lookup
            all_positions: All ball positions
            
        Returns:
            Annotated frame
        """
        # Draw trajectory path up to current frame
        visible_positions = TrajectoryProcessor.filter_by_frame(trajectory, frame_num)
        
        if len(visible_positions) > 1:
            frame = self.renderer.draw_trajectory_path(frame, visible_positions)
        
        # Draw current ball position
        if frame_num in traj_dict:
            x, y, radius = traj_dict[frame_num]
            frame = self.renderer.draw_ball(frame, x, y, radius)
        
        return frame
