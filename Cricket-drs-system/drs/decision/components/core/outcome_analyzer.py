"""
Outcome Analysis Module

Analyzes trajectory intersection to determine OUT/NOT OUT decision.
"""

from typing import Dict, Tuple, Optional


class OutcomeAnalyzer:
    """Analyzes trajectory outcomes and determines cricket decisions"""
    
    @staticmethod
    def analyze(intersection_data: Dict) -> Tuple[str, str]:
        """
        Determine decision outcome and reason
        
        Args:
            intersection_data: Dictionary with intersection information
            
        Returns:
            Tuple of (decision, reason)
        """
        intersects = intersection_data.get('intersects', False)
        
        if intersects:
            return OutcomeAnalyzer._analyze_out_decision()
        else:
            return OutcomeAnalyzer._analyze_not_out_decision(intersection_data)
    
    @staticmethod
    def _analyze_out_decision() -> Tuple[str, str]:
        """
        Analyze OUT decision
        
        Returns:
            Tuple of (decision, reason)
        """
        return 'OUT', 'Ball trajectory intersects with stumps'
    
    @staticmethod
    def _analyze_not_out_decision(intersection_data: Dict) -> Tuple[str, str]:
        """
        Analyze NOT OUT decision and determine reason
        
        Args:
            intersection_data: Intersection information
            
        Returns:
            Tuple of (decision, reason)
        """
        predicted_y = intersection_data.get('predicted_y')
        stump_range = intersection_data.get('stump_range', (0, 0))
        
        if predicted_y is None:
            reason = 'Ball trajectory does not reach stumps'
        elif predicted_y < stump_range[0]:
            reason = 'Ball trajectory passes above stumps'
        elif predicted_y > stump_range[1]:
            reason = 'Ball trajectory passes below stumps'
        else:
            reason = 'Ball trajectory misses stumps'
        
        return 'NOT OUT', reason
    
    @staticmethod
    def is_marginal_decision(intersection_data: Dict, margin: float = 10.0) -> bool:
        """
        Check if decision is marginal (close to stumps)
        
        Args:
            intersection_data: Intersection information
            margin: Pixel margin for marginal decision
            
        Returns:
            True if decision is marginal
        """
        predicted_y = intersection_data.get('predicted_y')
        stump_range = intersection_data.get('stump_range', (0, 0))
        
        if predicted_y is None:
            return False
        
        # Check if within margin of stump boundaries
        distance_to_top = abs(predicted_y - stump_range[0])
        distance_to_bottom = abs(predicted_y - stump_range[1])
        
        return min(distance_to_top, distance_to_bottom) <= margin
