"""
Trajectory Validation Module

Validates trajectory analysis results.
"""

from typing import Dict, Optional, Tuple


class TrajectoryValidator:
    """Validates trajectory analysis data"""
    
    @staticmethod
    def validate(trajectory_analysis: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate trajectory analysis results
        
        Args:
            trajectory_analysis: Results from TrajectoryPredictor
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not trajectory_analysis:
            return False, "Trajectory analysis is empty"
        
        if not trajectory_analysis.get('success', False):
            error = trajectory_analysis.get('error', 'Trajectory analysis failed')
            return False, error
        
        if 'intersection' not in trajectory_analysis:
            return False, "Missing intersection data"
        
        return True, None
    
    @staticmethod
    def has_valid_fit(trajectory_analysis: Dict, min_r_squared: float = 0.7) -> bool:
        """
        Check if trajectory has good polynomial fit
        
        Args:
            trajectory_analysis: Trajectory results
            min_r_squared: Minimum R-squared value
            
        Returns:
            True if fit quality is good
        """
        fit_quality = trajectory_analysis.get('fit_quality', 0)
        return fit_quality >= min_r_squared
