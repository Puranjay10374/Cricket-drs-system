"""
Confidence Level Categorization Module

Categorizes confidence scores into human-readable levels.
"""

from enum import Enum
from typing import NamedTuple


class ConfidenceLevel(Enum):
    """Confidence level categories"""
    VERY_HIGH = "Very High"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    VERY_LOW = "Very Low"


class ConfidenceLevelThresholds(NamedTuple):
    """Thresholds for confidence levels"""
    very_high: float = 0.9
    high: float = 0.8
    medium: float = 0.6
    low: float = 0.4


class ConfidenceLevelClassifier:
    """Classifies confidence scores into levels"""
    
    def __init__(self, thresholds: ConfidenceLevelThresholds = None):
        """
        Initialize classifier
        
        Args:
            thresholds: Custom thresholds for classification
        """
        self.thresholds = thresholds or ConfidenceLevelThresholds()
    
    def classify(self, confidence: float) -> ConfidenceLevel:
        """
        Get confidence level from score
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            ConfidenceLevel enum
        """
        if confidence >= self.thresholds.very_high:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= self.thresholds.high:
            return ConfidenceLevel.HIGH
        elif confidence >= self.thresholds.medium:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.thresholds.low:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def get_level_string(self, confidence: float) -> str:
        """
        Get confidence level as string
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Level string
        """
        return self.classify(confidence).value
