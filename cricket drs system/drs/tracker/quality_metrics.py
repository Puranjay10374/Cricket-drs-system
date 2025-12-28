"""
Quality Metrics Calculation

Calculates tracking quality and performance metrics.
"""

from typing import Dict


class QualityMetrics:
    """Calculates quality metrics for ball tracking"""
    
    @staticmethod
    def calculate_tracking_quality(frames_tracked: int, frames_processed: int) -> float:
        """
        Calculate tracking quality ratio
        
        Args:
            frames_tracked: Number of frames where ball was detected
            frames_processed: Total number of frames processed
            
        Returns:
            Quality ratio (0-1)
        """
        if frames_processed == 0:
            return 0.0
        return frames_tracked / frames_processed
    
    @staticmethod
    def calculate_continuity(trajectory: list) -> float:
        """
        Calculate trajectory continuity (how consistent frame numbers are)
        
        Args:
            trajectory: List of (frame_num, x, y, radius) tuples
            
        Returns:
            Continuity score (0-1)
        """
        if len(trajectory) < 2:
            return 0.0
        
        frame_nums = [t[0] for t in trajectory]
        gaps = [frame_nums[i+1] - frame_nums[i] for i in range(len(frame_nums)-1)]
        
        # Perfect continuity = all gaps are 1
        avg_gap = sum(gaps) / len(gaps)
        return 1.0 / avg_gap if avg_gap > 0 else 0.0
    
    @staticmethod
    def get_tracking_stats(tracking_result: Dict) -> Dict:
        """
        Get comprehensive tracking statistics
        
        Args:
            tracking_result: Result from BallTracker.track_video()
            
        Returns:
            Dictionary with detailed statistics
        """
        trajectory = tracking_result.get('trajectory', [])
        frames_tracked = tracking_result.get('frames_tracked', 0)
        total_frames = tracking_result.get('total_frames', 0)
        
        continuity = QualityMetrics.calculate_continuity(trajectory)
        
        return {
            'tracking_quality': tracking_result.get('tracking_quality', 0.0),
            'continuity': continuity,
            'frames_tracked': frames_tracked,
            'total_frames': total_frames,
            'coverage_percent': (frames_tracked / total_frames * 100) if total_frames > 0 else 0
        }
