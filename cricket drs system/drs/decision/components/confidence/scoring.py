"""
Confidence Scoring Module

Calculates raw confidence scores from quality metrics.
"""

from typing import NamedTuple


class ConfidenceWeights(NamedTuple):
    """Weights for confidence calculation"""
    tracking: float = 0.4
    fit: float = 0.4
    points: float = 0.2


class ConfidenceScorer:
    """Calculates raw confidence scores"""
    
    def __init__(self, weights: ConfidenceWeights = None):
        """
        Initialize scorer
        
        Args:
            weights: Weight configuration for different factors
        """
        self.weights = weights or ConfidenceWeights()
    
    def calculate_points_score(self, points_tracked: int, min_points: int) -> float:
        """
        Calculate score based on number of points tracked
        
        Args:
            points_tracked: Number of points tracked
            min_points: Minimum points for good confidence
            
        Returns:
            Points score (0-1)
        """
        return min(1.0, points_tracked / (min_points * 2))
    
    def calculate(self,
                 tracking_quality: float,
                 fit_quality: float,
                 points_tracked: int,
                 min_points: int = 10) -> float:
        """
        Calculate overall confidence score
        
        Args:
            tracking_quality: Tracking quality (0-1)
            fit_quality: Fit quality (0-1)
            points_tracked: Number of points tracked
            min_points: Minimum points needed
            
        Returns:
            Confidence score (0-1)
        """
        points_score = self.calculate_points_score(points_tracked, min_points)
        
        confidence = (
            tracking_quality * self.weights.tracking +
            fit_quality * self.weights.fit +
            points_score * self.weights.points
        )
        
        return min(1.0, max(0.0, confidence))
