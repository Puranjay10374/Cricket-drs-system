"""
Validators Package

Modular validation components.
"""

from typing import Dict, Optional, Tuple
from .trajectory_validator import TrajectoryValidator
from .tracking_validator import TrackingValidator


class DecisionValidator:
    """
    Unified validator
    
    Facade combining trajectory and tracking validators
    """
    
    def __init__(self, min_tracking_quality: float = 0.3):
        """Initialize validators"""
        self.trajectory_validator = TrajectoryValidator()
        self.tracking_validator = TrackingValidator(min_tracking_quality)
    
    def validate_trajectory_analysis(self, trajectory_analysis: Dict) -> Tuple[bool, Optional[str]]:
        """Validate trajectory analysis"""
        return self.trajectory_validator.validate(trajectory_analysis)
    
    def validate_tracking_info(self, tracking_info: Dict) -> Tuple[bool, Optional[str]]:
        """Validate tracking info"""
        return self.tracking_validator.validate(tracking_info)
    
    def validate_all(self, trajectory_analysis: Dict, 
                    tracking_info: Dict) -> Tuple[bool, Optional[str]]:
        """Validate all inputs"""
        # Validate tracking first
        valid, error = self.validate_tracking_info(tracking_info)
        if not valid:
            return False, error
        
        # Then validate trajectory
        valid, error = self.validate_trajectory_analysis(trajectory_analysis)
        if not valid:
            return False, error
        
        return True, None


__all__ = [
    'DecisionValidator',
    'TrajectoryValidator',
    'TrackingValidator'
]
