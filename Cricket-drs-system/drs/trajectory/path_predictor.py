"""
Path Prediction Module

Predicts future ball positions along fitted trajectory.
"""

import numpy as np
from typing import List, Tuple


class PathPredictor:
    """Predicts ball path along polynomial trajectory"""
    
    @staticmethod
    def predict(polynomial: np.poly1d, 
               x_values: np.ndarray) -> np.ndarray:
        """
        Predict y values for given x values
        
        Args:
            polynomial: Fitted polynomial function
            x_values: X coordinates to predict
            
        Returns:
            Predicted y values
        """
        return polynomial(x_values)
    
    @staticmethod
    def predict_path(polynomial: np.poly1d, 
                    start_x: float, 
                    end_x: float, 
                    num_points: int = 50) -> List[Tuple[float, float]]:
        """
        Predict future ball positions along trajectory
        
        Args:
            polynomial: Fitted polynomial function
            start_x: Starting x coordinate
            end_x: Ending x coordinate
            num_points: Number of points to predict
            
        Returns:
            List of (x, y) predicted coordinates
        """
        # Generate x values from start to end
        x_values = np.linspace(start_x, end_x, num_points)
        
        # Predict y values using polynomial
        y_values = PathPredictor.predict(polynomial, x_values)
        
        # Combine into (x, y) tuples
        return list(zip(x_values.tolist(), y_values.tolist()))
    
    @staticmethod
    def extrapolate(polynomial: np.poly1d,
                   start_x: float,
                   end_x: float,
                   num_points: int = 50) -> List[Tuple[float, float]]:
        """
        Extrapolate points beyond known trajectory
        
        Alias for predict_path for clarity when extrapolating
        
        Args:
            polynomial: Fitted polynomial function
            start_x: Starting x coordinate
            end_x: Ending x coordinate
            num_points: Number of points to generate
            
        Returns:
            List of (x, y) predicted coordinates
        """
        return PathPredictor.predict_path(polynomial, start_x, end_x, num_points)
