"""
Polynomial Fitting Module

Fits polynomial curves to trajectory data and calculates fit quality.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class PolynomialFitter:
    """Handles polynomial fitting operations"""
    
    def __init__(self, degree: int = 2):
        """
        Initialize polynomial fitter
        
        Args:
            degree: Degree of polynomial (2 = parabola, 3 = cubic)
        """
        self.degree = degree
    
    def fit(self, points: List[Tuple[int, int]], weighted: bool = True) -> Optional[Dict]:
        """
        Fit polynomial to trajectory points with optional weighting
        
        Args:
            points: List of (x, y) coordinates
            weighted: If True, recent points get higher weight
            
        Returns:
            Dictionary with fit results or None if insufficient points
        """
        if len(points) < self.degree + 1:
            return None
        
        # Separate x and y coordinates
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])
        
        # Create weights - recent points (end of trajectory) get higher weight
        if weighted and len(points) > 5:
            # Exponential weighting: later points are more important
            weights = np.exp(np.linspace(0, 2, len(points)))
            weights = weights / np.sum(weights) * len(points)  # Normalize
        else:
            weights = np.ones(len(points))
        
        # Fit weighted polynomial (y as function of x)
        coefficients = np.polyfit(x_coords, y_coords, self.degree, w=weights)
        polynomial = np.poly1d(coefficients)
        
        # Calculate fit quality
        fit_quality = self.calculate_r_squared(x_coords, y_coords, polynomial)
        
        return {
            'coefficients': coefficients,
            'polynomial': polynomial,
            'fit_quality': fit_quality,
            'equation_str': str(polynomial),
            'weighted': weighted
        }
    
    def calculate_r_squared(self, 
                           x_coords: np.ndarray, 
                           y_coords: np.ndarray,
                           polynomial: np.poly1d) -> float:
        """
        Calculate R-squared value for fit quality
        
        Args:
            x_coords: X coordinates
            y_coords: Y coordinates  
            polynomial: Fitted polynomial
            
        Returns:
            R-squared score (0-1, higher is better)
        """
        y_pred = polynomial(x_coords)
        ss_res = np.sum((y_coords - y_pred) ** 2)
        ss_tot = np.sum((y_coords - np.mean(y_coords)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, min(1.0, r_squared))  # Clamp between 0 and 1
