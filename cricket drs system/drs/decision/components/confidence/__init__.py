"""
Confidence Package

Modular confidence calculation components.
"""

from .scoring import ConfidenceScorer, ConfidenceWeights
from .levels import ConfidenceLevelClassifier, ConfidenceLevel, ConfidenceLevelThresholds
from .thresholds import ThresholdChecker

from ....config import DecisionConfig


class ConfidenceCalculator:
    """
    Unified confidence calculator
    
    Facade that combines scorer, classifier, and threshold checker
    """
    
    def __init__(self, config: DecisionConfig):
        """Initialize with config"""
        self.config = config
        self.scorer = ConfidenceScorer()
        self.classifier = ConfidenceLevelClassifier()
        self.threshold_checker = ThresholdChecker(config.confidence_threshold)
    
    def calculate(self, tracking_quality: float, fit_quality: float, 
                 points_tracked: int) -> float:
        """Calculate confidence score"""
        return self.scorer.calculate(
            tracking_quality, 
            fit_quality, 
            points_tracked,
            self.config.min_points_for_good_confidence
        )
    
    def get_confidence_level(self, confidence: float) -> str:
        """Get confidence level string"""
        return self.classifier.get_level_string(confidence)
    
    def meets_threshold(self, confidence: float) -> bool:
        """Check if meets threshold"""
        return self.threshold_checker.meets_threshold(confidence)


__all__ = [
    'ConfidenceCalculator',
    'ConfidenceScorer',
    'ConfidenceLevelClassifier',
    'ThresholdChecker',
    'ConfidenceWeights',
    'ConfidenceLevel'
]
