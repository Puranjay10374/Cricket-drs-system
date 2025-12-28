"""
Tracking Validation Module

Validates ball tracking results.
"""

from typing import Dict, Optional, Tuple


class TrackingValidator:
    """Validates tracking data"""
    
    def __init__(self, min_tracking_quality: float = 0.3):
        """
        Initialize validator
        
        Args:
            min_tracking_quality: Minimum acceptable quality
        """
        self.min_tracking_quality = min_tracking_quality
    
    def validate(self, tracking_info: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate tracking information
        
        Args:
            tracking_info: Results from BallTracker
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not tracking_info:
            return False, "Tracking information is empty"
        
        tracking_quality = tracking_info.get('tracking_quality', 0)
        
        if tracking_quality < self.min_tracking_quality:
            return False, f'Poor tracking quality: {tracking_quality:.2%}'
        
        if tracking_info.get('frames_tracked', 0) == 0:
            return False, "No frames tracked"
        
        return True, None
    
    def has_sufficient_points(self, tracking_info: Dict, min_points: int = 5) -> bool:
        """
        Check if enough points were tracked
        
        Args:
            tracking_info: Tracking results
            min_points: Minimum required points
            
        Returns:
            True if sufficient points tracked
        """
        points = tracking_info.get('points_tracked', 0)
        return points >= min_points
