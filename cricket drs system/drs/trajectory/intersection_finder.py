"""
Intersection Finding Module

Detects if ball trajectory intersects with stump region.
"""

import numpy as np
from typing import Dict

from ..config import StumpRegion


class IntersectionFinder:
    """Finds intersections between trajectory and stumps"""
    
    @staticmethod
    def find_intersection(polynomial: np.poly1d,
                         stump_region: StumpRegion) -> Dict:
        """
        Find if trajectory intersects with stumps
        
        Args:
            polynomial: Fitted polynomial function
            stump_region: Stump region configuration
            
        Returns:
            Dictionary with intersection analysis results
        """
        # Predict y value at stump x position
        predicted_y = float(polynomial(stump_region.x))
        
        # Check if predicted y falls within stump height range
        intersects = stump_region.contains_point(stump_region.x, predicted_y)
        
        impact_point = (float(stump_region.x), predicted_y) if intersects else None
        
        return {
            'intersects': intersects,
            'impact_point': impact_point,
            'predicted_y': predicted_y,
            'stump_range': stump_region.get_range(),
            'stump_x': stump_region.x
        }
    
    @staticmethod
    def calculate_miss_distance(polynomial: np.poly1d,
                                stump_region: StumpRegion) -> float:
        """
        Calculate vertical distance between trajectory and stumps
        
        Args:
            polynomial: Fitted polynomial
            stump_region: Stump region
            
        Returns:
            Distance in pixels (0 if intersects, positive if above/below)
        """
        predicted_y = float(polynomial(stump_region.x))
        
        # Check if intersects
        if stump_region.y_top <= predicted_y <= stump_region.y_bottom:
            return 0.0
        
        # Calculate distance to nearest edge
        if predicted_y < stump_region.y_top:
            return stump_region.y_top - predicted_y  # Above stumps
        else:
            return predicted_y - stump_region.y_bottom  # Below stumps
