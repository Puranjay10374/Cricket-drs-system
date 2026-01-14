"""
Trajectory Processing

Processes and transforms trajectory data.
"""

from typing import List, Tuple, Dict


class TrajectoryProcessor:
    """Processes trajectory data from ball tracking"""
    
    @staticmethod
    def create_lookup_dict(trajectory: List[Tuple]) -> Dict[int, Tuple[int, int, int]]:
        """
        Create frame number to position lookup dictionary
        
        Args:
            trajectory: List of (frame_num, x, y, radius) tuples
            
        Returns:
            Dictionary mapping frame_num to (x, y, radius)
        """
        return {frame_num: (x, y, radius) for frame_num, x, y, radius in trajectory}
    
    @staticmethod
    def extract_positions(trajectory: List[Tuple]) -> List[Tuple[int, int]]:
        """
        Extract (x, y) positions from trajectory
        
        Args:
            trajectory: List of (frame_num, x, y, radius) tuples
            
        Returns:
            List of (x, y) coordinate tuples
        """
        return [(x, y) for _, x, y, _ in trajectory]
    
    @staticmethod
    def filter_by_frame(trajectory: List[Tuple], max_frame: int) -> List[Tuple[int, int]]:
        """
        Get positions up to specified frame number
        
        Args:
            trajectory: List of (frame_num, x, y, radius) tuples
            max_frame: Maximum frame number to include
            
        Returns:
            List of (x, y) positions up to max_frame
        """
        return [
            (x, y) for frame_num, x, y, _ in trajectory 
            if frame_num <= max_frame
        ]
    
    @staticmethod
    def smooth_trajectory(trajectory: List[Tuple], window_size: int = 3) -> List[Tuple]:
        """
        Apply moving average smoothing to trajectory
        
        Args:
            trajectory: List of (frame_num, x, y, radius) tuples
            window_size: Size of smoothing window
            
        Returns:
            Smoothed trajectory
        """
        if len(trajectory) < window_size:
            return trajectory
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(trajectory)):
            frame_num, x, y, radius = trajectory[i]
            
            # Calculate window bounds
            start = max(0, i - half_window)
            end = min(len(trajectory), i + half_window + 1)
            
            # Average positions in window
            window = trajectory[start:end]
            avg_x = sum(t[1] for t in window) / len(window)
            avg_y = sum(t[2] for t in window) / len(window)
            
            smoothed.append((frame_num, int(avg_x), int(avg_y), radius))
        
        return smoothed
