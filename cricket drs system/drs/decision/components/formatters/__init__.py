"""
Formatters Package

Modular formatting components.
"""

from typing import Dict, Optional
from .point_formatter import PointFormatter
from .stats_formatter import StatsFormatter


class ResponseFormatter:
    """
    Unified response formatter
    
    Facade combining point and stats formatters
    """
    
    def __init__(self):
        """Initialize formatters"""
        self.point_formatter = PointFormatter()
        self.stats_formatter = StatsFormatter()
    
    def format_decision(self, decision_result: Dict) -> Dict:
        """Format complete decision response"""
        return {
            'decision': decision_result['decision'],
            'confidence': round(decision_result['confidence'], 3),
            'impact_point': self.point_formatter.format_impact_point(
                decision_result.get('impact_point')
            ),
            'reason': decision_result['reason'],
            'tracking_stats': self.stats_formatter.format_tracking_stats(
                decision_result.get('details', {})
            )
        }
    
    def format_inconclusive(self, reason: str, tracking_info: Dict,
                          confidence: float = 0.0) -> Dict:
        """Format inconclusive decision"""
        return {
            'decision': 'INCONCLUSIVE',
            'confidence': round(confidence, 3),
            'impact_point': None,
            'reason': reason,
            'details': {
                'tracking_quality': tracking_info.get('tracking_quality', 0),
                'frames_tracked': tracking_info.get('frames_tracked', 0),
                'total_frames': tracking_info.get('total_frames', 0),
                'points_tracked': tracking_info.get('points_tracked', 0)
            }
        }


__all__ = [
    'ResponseFormatter',
    'PointFormatter',
    'StatsFormatter'
]
