"""
Decision Making Configuration

Configuration for confidence thresholds and decision weights.
"""

from dataclasses import dataclass


@dataclass
class DecisionConfig:
    """Configuration for decision making"""
    
    # Thresholds
    confidence_threshold: float = 0.6
    min_tracking_quality: float = 0.3
    min_points_for_good_confidence: int = 5
    
    # Weights for confidence calculation
    tracking_weight: float = 0.4
    fit_weight: float = 0.4
    points_weight: float = 0.2
    
    def validate_weights(self) -> bool:
        """
        Validate that weights sum to 1.0
        
        Returns:
            True if weights are valid
        """
        total = self.tracking_weight + self.fit_weight + self.points_weight
        return abs(total - 1.0) < 0.01


# Default configuration
DEFAULT_DECISION_CONFIG = DecisionConfig()
