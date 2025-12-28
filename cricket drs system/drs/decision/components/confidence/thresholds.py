"""
Threshold Checking Module

Checks if confidence meets minimum requirements.
"""


class ThresholdChecker:
    """Checks confidence against thresholds"""
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize threshold checker
        
        Args:
            min_confidence: Minimum acceptable confidence
        """
        self.min_confidence = min_confidence
    
    def meets_threshold(self, confidence: float) -> bool:
        """
        Check if confidence meets minimum threshold
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            True if confidence is sufficient
        """
        return confidence >= self.min_confidence
    
    def get_deficit(self, confidence: float) -> float:
        """
        Get how much below threshold the confidence is
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Deficit (0 if meets threshold)
        """
        return max(0.0, self.min_confidence - confidence)
    
    def is_marginal(self, confidence: float, margin: float = 0.1) -> bool:
        """
        Check if confidence is marginally above/below threshold
        
        Args:
            confidence: Confidence score (0-1)
            margin: Margin around threshold
            
        Returns:
            True if within margin of threshold
        """
        return abs(confidence - self.min_confidence) <= margin
