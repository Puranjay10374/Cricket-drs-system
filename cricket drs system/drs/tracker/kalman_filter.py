"""
Kalman Filter for Ball Tracking

Predicts ball position during occlusions and smooths trajectory.
"""

import numpy as np
from typing import Optional, Tuple


class BallKalmanFilter:
    """
    Kalman filter optimized for cricket ball tracking
    
    Predicts ball position and velocity, handles missing detections,
    and smooths noisy measurements.
    """
    
    def __init__(self):
        """Initialize Kalman filter for 2D ball tracking"""
        # State: [x, y, vx, vy] - position and velocity
        self.state = np.zeros(4)
        
        # State covariance (uncertainty)
        self.P = np.eye(4) * 1000
        
        # Process noise (how much we trust motion model)
        # INCREASED for fast-moving cricket ball
        self.Q = np.array([
            [2, 0, 0, 0],    # Position noise
            [0, 2, 0, 0],    # Position noise
            [0, 0, 50, 0],   # HIGH velocity uncertainty (fast ball!)
            [0, 0, 0, 50]    # HIGH velocity uncertainty
        ])
        
        # Measurement noise (how much we trust detections)
        # REDUCED - trust detections more for fast objects
        self.R = np.array([
            [5, 0],    # Lower noise = trust detections more
            [0, 5]
        ])
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 0.95, 0],  # vx with slight decay (air resistance)
            [0, 0, 0, 0.95]   # vy with slight decay
        ])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.initialized = False
        self.frames_without_detection = 0
        self.max_predict_frames = 5  # Reduced from 10 - don't predict too far for fast ball
    
    def initialize(self, x: int, y: int):
        """
        Initialize filter with first detection
        
        Args:
            x: Ball x position
            y: Ball y position
        """
        self.state = np.array([x, y, 0, 0], dtype=float)
        self.P = np.eye(4) * 1000
        self.initialized = True
        self.frames_without_detection = 0
    
    def predict(self) -> Tuple[int, int]:
        """
        Predict next ball position
        
        Returns:
            Tuple of (x, y) predicted position
        """
        # Predict state: x = F * x
        self.state = self.F @ self.state
        
        # Predict covariance: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return (int(self.state[0]), int(self.state[1]))
    
    def update(self, measurement: Tuple[int, int]) -> Tuple[int, int]:
        """
        Update filter with new measurement
        
        Args:
            measurement: Detected (x, y) position
            
        Returns:
            Corrected (x, y) position
        """
        z = np.array(measurement, dtype=float)
        
        # Innovation: y = z - H * x
        y = z - self.H @ self.state
        
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * y
        self.state = self.state + K @ y
        
        # Update covariance: P = (I - K * H) * P
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        self.frames_without_detection = 0
        
        return (int(self.state[0]), int(self.state[1]))
    
    def process(self, detection: Optional[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
        """
        Process detection with Kalman filter
        
        Args:
            detection: (x, y, radius) or None if no detection
            
        Returns:
            Filtered (x, y, radius) or predicted position, None if lost track
        """
        if not self.initialized:
            if detection is not None:
                self.initialize(detection[0], detection[1])
                return detection
            return None
        
        # Predict
        predicted_x, predicted_y = self.predict()
        
        if detection is not None:
            # Have detection - update filter
            corrected_x, corrected_y = self.update((detection[0], detection[1]))
            self.frames_without_detection = 0
            return (corrected_x, corrected_y, detection[2])
        else:
            # No detection - use prediction
            self.frames_without_detection += 1
            
            # If too many frames without detection, lost track
            if self.frames_without_detection > self.max_predict_frames:
                self.initialized = False
                return None
            
            # Return predicted position with estimated radius
            estimated_radius = 10  # Default radius when predicting
            return (predicted_x, predicted_y, estimated_radius)
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current estimated velocity
        
        Returns:
            (vx, vy) velocity in pixels/frame
        """
        return (self.state[2], self.state[3])
    
    def reset(self):
        """Reset filter"""
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1000
        self.initialized = False
        self.frames_without_detection = 0
