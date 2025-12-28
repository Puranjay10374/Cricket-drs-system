"""
Result Building Module

Constructs decision result dictionaries with all required fields.
"""

from typing import Dict, Optional


class ResultBuilder:
    """Builds structured decision results"""
    
    def __init__(self, confidence_calculator):
        """
        Initialize result builder
        
        Args:
            confidence_calculator: ConfidenceCalculator instance
        """
        self.confidence_calc = confidence_calculator
    
    def build_decision_result(self,
                             decision: str,
                             confidence: float,
                             impact_point: Optional[tuple],
                             reason: str,
                             quality_metrics: Dict) -> Dict:
        """
        Build complete decision result
        
        Args:
            decision: OUT or NOT OUT
            confidence: Confidence score
            impact_point: Impact coordinates
            reason: Decision reason
            quality_metrics: Quality metrics dictionary
            
        Returns:
            Complete decision result dictionary
        """
        # Add confidence qualifier if below threshold
        decision_with_qualifier = self._add_confidence_qualifier(decision, confidence)
        
        return {
            'decision': decision,
            'decision_with_confidence': decision_with_qualifier,
            'confidence': confidence,
            'confidence_level': self.confidence_calc.get_confidence_level(confidence),
            'impact_point': impact_point,
            'reason': reason,
            'details': self._build_details(quality_metrics)
        }
    
    def _add_confidence_qualifier(self, decision: str, confidence: float) -> str:
        """
        Add confidence qualifier to decision
        
        Args:
            decision: Base decision
            confidence: Confidence score
            
        Returns:
            Decision with qualifier if needed
        """
        if not self.confidence_calc.meets_threshold(confidence):
            return f'{decision} (Low Confidence)'
        return decision
    
    @staticmethod
    def _build_details(quality_metrics: Dict) -> Dict:
        """
        Build details section of result
        
        Args:
            quality_metrics: Quality metrics
            
        Returns:
            Details dictionary
        """
        return {
            'tracking_quality': quality_metrics.get('tracking_quality', 0),
            'fit_quality': quality_metrics.get('fit_quality', 0),
            'frames_tracked': quality_metrics.get('frames_tracked', 0),
            'total_frames': quality_metrics.get('total_frames', 0),
            'points_tracked': quality_metrics.get('points_tracked', 0)
        }
