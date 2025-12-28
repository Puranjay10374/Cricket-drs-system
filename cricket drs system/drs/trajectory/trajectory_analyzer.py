"""
Trajectory Analysis Module

Main pipeline for complete trajectory analysis.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

from ..config import TrajectoryConfig, StumpRegion
from .polynomial_fitter import PolynomialFitter
from .path_predictor import PathPredictor
from .intersection_finder import IntersectionFinder


class TrajectoryAnalyzer:
    """Complete trajectory analysis pipeline"""
    
    def __init__(self, config: Optional[TrajectoryConfig] = None):
        """
        Initialize trajectory analyzer
        
        Args:
            config: Trajectory configuration (uses defaults if not provided)
        """
        self.config = config or TrajectoryConfig()
        self.fitter = PolynomialFitter(degree=self.config.polynomial_degree)
    
    def analyze_trajectory(self, 
                          trajectory_data: List[Tuple[int, int, int, int]],
                          stump_region: StumpRegion) -> Dict:
        """
        Complete trajectory analysis pipeline
        
        Args:
            trajectory_data: List of (frame_num, x, y, radius) tuples from tracker
            stump_region: Stump region configuration object
            
        Returns:
            Complete analysis results with success/error status
        """
        # Extract (x, y) points from trajectory data
        points = [(x, y) for _, x, y, _ in trajectory_data]
        
        # Validate sufficient points
        if len(points) < self.config.min_points_required:
            return self._error_result(
                f'Insufficient tracking points. Need at least {self.config.min_points_required}, got {len(points)}',
                len(points)
            )
        
        # Fit polynomial to trajectory
        fit_result = self.fitter.fit(points)
        
        if not fit_result:
            return self._error_result('Failed to fit polynomial to trajectory', len(points))
        
        polynomial = fit_result['polynomial']
        
        # Predict future path if stumps are ahead
        predicted_path = self._predict_future_path(points, polynomial, stump_region)
        
        # Find intersection with stumps
        intersection = IntersectionFinder.find_intersection(polynomial, stump_region)
        
        return self._success_result(points, fit_result, predicted_path, intersection)
    
    def _predict_future_path(self, 
                            points: List[Tuple[int, int]],
                            polynomial: np.poly1d,
                            stump_region: StumpRegion) -> List[Tuple[float, float]]:
        """Predict path if stumps are ahead of last position"""
        last_x = points[-1][0]
        stump_x = stump_region.x
        
        if stump_x > last_x:
            return PathPredictor.extrapolate(
                polynomial, 
                last_x, 
                stump_x, 
                num_points=self.config.extrapolation_points
            )
        return []
    
    def _error_result(self, error: str, points_tracked: int) -> Dict:
        """Build error result dictionary"""
        return {
            'success': False,
            'error': error,
            'points_tracked': points_tracked
        }
    
    def _success_result(self, 
                       points: List[Tuple[int, int]],
                       fit_result: Dict,
                       predicted_path: List[Tuple[float, float]],
                       intersection: Dict) -> Dict:
        """Build success result dictionary"""
        return {
            'success': True,
            'points_tracked': len(points),
            'coefficients': fit_result['coefficients'].tolist(),
            'fit_quality': fit_result['fit_quality'],
            'equation_str': fit_result['equation_str'],
            'predicted_path': predicted_path,
            'intersection': intersection,
            'tracked_points': points
        }


# Backward compatibility alias
TrajectoryPredictor = TrajectoryAnalyzer
