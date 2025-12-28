"""
Statistics Formatting Module

Formats tracking and analysis statistics.
"""

from typing import Dict


class StatsFormatter:
    """Formats statistics for API responses"""
    
    @staticmethod
    def format_tracking_stats(details: Dict, precision: int = 3) -> Dict:
        """
        Format tracking statistics
        
        Args:
            details: Details dictionary from decision result
            precision: Decimal places
            
        Returns:
            Formatted tracking stats
        """
        return {
            'tracking_quality': round(details.get('tracking_quality', 0), precision),
            'fit_quality': round(details.get('fit_quality', 0), precision),
            'frames_tracked': details.get('frames_tracked', 0),
            'points_tracked': details.get('points_tracked', 0)
        }
    
    @staticmethod
    def format_video_stats(tracking_info: Dict) -> Dict:
        """
        Format video processing statistics
        
        Args:
            tracking_info: Tracking information
            
        Returns:
            Formatted video stats
        """
        return {
            'total_frames': tracking_info.get('total_frames', 0),
            'frames_processed': tracking_info.get('frames_tracked', 0),
            'fps': tracking_info.get('fps', 0),
            'duration': tracking_info.get('duration', 0)
        }
