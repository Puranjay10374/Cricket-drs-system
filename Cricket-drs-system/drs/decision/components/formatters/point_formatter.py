"""
Point Formatting Module

Formats coordinate points for API responses.
"""

from typing import Optional, Dict, Tuple


class PointFormatter:
    """Formats coordinate points"""
    
    @staticmethod
    def format_impact_point(point: Optional[Tuple[float, float]], 
                           precision: int = 2) -> Optional[Dict]:
        """
        Format impact point coordinates
        
        Args:
            point: (x, y) tuple or None
            precision: Decimal places
            
        Returns:
            Formatted point dict or None
        """
        if not point:
            return None
        
        return {
            'x': round(point[0], precision),
            'y': round(point[1], precision)
        }
    
    @staticmethod
    def format_point_list(points: list, precision: int = 2) -> list:
        """
        Format list of points
        
        Args:
            points: List of (x, y) tuples
            precision: Decimal places
            
        Returns:
            List of formatted point dicts
        """
        return [
            {'x': round(p[0], precision), 'y': round(p[1], precision)}
            for p in points if p
        ]
