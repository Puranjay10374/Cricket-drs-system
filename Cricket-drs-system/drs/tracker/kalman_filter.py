"""
Kalman Filter for Ball Tracking

Predicts ball position during occlusions and smooths trajectory.
Enhanced with gravity model, outlier rejection, and adaptive noise.
"""

import numpy as np
from typing import Optional, Tuple


class BallKalmanFilter:
    """
    Kalman filter optimized for cricket ball tracking
    
    Predicts ball position and velocity, handles missing detections,
    and smooths noisy measurements.
    
    Enhanced features:
    - Gravity/acceleration model for realistic ball physics
    - Mahalanobis distance for outlier rejection
    - Adaptive noise matrices based on tracking conditions
    - Confidence tracking for prediction reliability
    """
    
    def __init__(self, use_gravity: bool = True):
        """Initialize Kalman filter for 2D ball tracking
        
        Args:
            use_gravity: Use 6-state model with acceleration (default: True)
        """
        self.use_gravity = use_gravity
        
        if use_gravity:
            # State: [x, y, vx, vy, ax, ay] - position, velocity, and acceleration
            self.state = np.zeros(6)
            
            # State covariance (uncertainty)
            self.P = np.eye(6) * 1000
            
            # Process noise (how much we trust motion model)
            self.Q = np.array([
                [2, 0, 0, 0, 0, 0],      # Position noise
                [0, 2, 0, 0, 0, 0],      # Position noise
                [0, 0, 50, 0, 0, 0],     # Velocity uncertainty
                [0, 0, 0, 50, 0, 0],     # Velocity uncertainty
                [0, 0, 0, 0, 10, 0],     # Acceleration noise
                [0, 0, 0, 0, 0, 10]      # Acceleration noise (gravity)
            ])
            
            # State transition matrix (constant acceleration model)
            self.F = np.array([
                [1, 0, 1, 0, 0.5, 0],    # x = x + vx + 0.5*ax
                [0, 1, 0, 1, 0, 0.5],    # y = y + vy + 0.5*ay
                [0, 0, 1, 0, 1, 0],      # vx = vx + ax
                [0, 0, 0, 1, 0, 1],      # vy = vy + ay
                [0, 0, 0, 0, 0.98, 0],   # ax with slight decay
                [0, 0, 0, 0, 0, 0.98]    # ay with slight decay (gravity varies)
            ])
            
            # Measurement matrix (we only measure position)
            self.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ])
        else:
            # State: [x, y, vx, vy] - position and velocity
            self.state = np.zeros(4)
            
            # State covariance (uncertainty)
            self.P = np.eye(4) * 1000
            
            # Process noise
            self.Q = np.array([
                [2, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 50, 0],
                [0, 0, 0, 50]
            ])
            
            # State transition matrix (constant velocity model)
            self.F = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 0.95, 0],
                [0, 0, 0, 0.95]
            ])
            
            # Measurement matrix
            self.H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
        
        # Measurement noise (how much we trust detections)
        self.R = np.array([
            [5, 0],
            [0, 5]
        ])
        
        # Base noise values for adaptation
        self.R_base = self.R.copy()
        self.Q_base = self.Q.copy()
        
        self.initialized = False
        self.frames_without_detection = 0
        self.max_predict_frames = 5
        self.mahalanobis_threshold = 3.0  # 3-sigma outlier rejection
    
    def initialize(self, x: int, y: int):
        """
        Initialize filter with first detection
        
        Args:
            x: Ball x position
            y: Ball y position
        """
        if self.use_gravity:
            # Initialize with position, zero velocity, and gravity estimate
            self.state = np.array([x, y, 0, 0, 0, 9.8], dtype=float)  # ay ~ gravity
            self.P = np.eye(6) * 1000
        else:
            self.state = np.array([x, y, 0, 0], dtype=float)
            self.P = np.eye(4) * 1000
        
        self.initialized = True
        self.frames_without_detection = 0
    
    def predict(self) -> Tuple[int, int]:
        """
        Predict next ball position with physics-based model
        
        Returns:
            Tuple of (x, y) predicted position
        """
        # Apply air resistance before prediction
        self._apply_air_resistance()
        
        # Predict state: x = F * x
        self.state = self.F @ self.state
        
        # Predict covariance: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return (int(self.state[0]), int(self.state[1]))
    
    def update(self, measurement: Tuple[int, int], confidence: float = 1.0) -> Tuple[int, int]:
        """
        Update filter with new measurement (with outlier rejection)
        
        Args:
            measurement: Detected (x, y) position
            confidence: Detection confidence (0-1), used for adaptive noise
            
        Returns:
            Corrected (x, y) position
        """
        z = np.array(measurement, dtype=float)
        
        # Check if measurement is valid (Mahalanobis distance test)
        if not self._is_valid_detection(measurement):
            # Outlier detected - skip update, just return prediction
            return (int(self.state[0]), int(self.state[1]))
        
        # Adapt noise matrices based on confidence
        self._update_noise_matrices(confidence)
        
        # Innovation: y = z - H * x
        innovation = z - self.H @ self.state
        
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * y
        self.state = self.state + K @ innovation
        
        # Update covariance: P = (I - K * H) * P
        I = np.eye(len(self.state))
        self.P = (I - K @ self.H) @ self.P
        
        self.frames_without_detection = 0
        
        return (int(self.state[0]), int(self.state[1]))
    
    def process(self, detection: Optional[Tuple[int, int, int]], 
                confidence: float = 1.0) -> Optional[Tuple[int, int, int]]:
        """
        Process detection with Kalman filter
        
        Args:
            detection: (x, y, radius) or None if no detection
            confidence: Detection confidence score (0-1)
            
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
            corrected_x, corrected_y = self.update((detection[0], detection[1]), confidence)
            self.frames_without_detection = 0
            return (corrected_x, corrected_y, detection[2])
        else:
            # No detection - use prediction
            self.frames_without_detection += 1
            
            # If too many frames without detection or low confidence, lost track
            pred_confidence = self.get_prediction_confidence()
            if (self.frames_without_detection > self.max_predict_frames or 
                pred_confidence < 0.1):
                self.initialized = False
                return None
            
            # Return predicted position with estimated radius
            estimated_radius = 10
            return (predicted_x, predicted_y, estimated_radius)
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current estimated velocity
        
        Returns:
            (vx, vy) velocity in pixels/frame
        """
        return (self.state[2], self.state[3])
    
    def get_acceleration(self) -> Tuple[float, float]:
        """
        Get current estimated acceleration
        
        Returns:
            (ax, ay) acceleration in pixels/frame^2, or (0, 0) if not using gravity model
        """
        if self.use_gravity and len(self.state) >= 6:
            return (self.state[4], self.state[5])
        return (0.0, 0.0)
    
    def get_prediction_confidence(self) -> float:
        """
        Get confidence in current prediction (0-1)
        
        Lower position uncertainty = higher confidence
        
        Returns:
            Confidence score between 0 and 1
        """
        # Calculate position uncertainty (trace of position covariance)
        position_uncertainty = np.trace(self.P[:2, :2])
        
        # Convert to confidence (0-1 scale)
        confidence = 1.0 / (1.0 + position_uncertainty / 100.0)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _is_valid_detection(self, measurement: Tuple[int, int]) -> bool:
        """
        Check if measurement is statistically valid using Mahalanobis distance
        
        Args:
            measurement: Detected (x, y) position
            
        Returns:
            True if detection is within threshold, False if outlier
        """
        z = np.array(measurement, dtype=float)
        
        # Innovation: difference between measurement and prediction
        innovation = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Mahalanobis distance
        try:
            distance = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)
            
            # Return True if within threshold (3-sigma)
            return distance < self.mahalanobis_threshold
        except np.linalg.LinAlgError:
            # If covariance is singular, accept the measurement
            return True
    
    def _update_noise_matrices(self, detection_confidence: float):
        """
        Adapt noise matrices based on detection confidence and velocity
        
        Args:
            detection_confidence: Confidence score (0-1) from detector
        """
        # High confidence → trust measurements more (lower R)
        confidence_factor = max(0.5, 1.0 - detection_confidence * 0.5)
        self.R = self.R_base * confidence_factor
        
        # Fast ball → higher process noise
        vx, vy = self.state[2], self.state[3]
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        speed_factor = min(2.0, 1.0 + velocity_magnitude / 30.0)
        
        if self.use_gravity:
            # Scale velocity and acceleration noise
            self.Q[2:6, 2:6] = self.Q_base[2:6, 2:6] * speed_factor
        else:
            # Scale velocity noise only
            self.Q[2:4, 2:4] = self.Q_base[2:4, 2:4] * speed_factor
    
    def _apply_air_resistance(self):
        """
        Apply realistic air resistance (proportional to velocity squared)
        """
        vx, vy = self.state[2], self.state[3]
        speed = np.sqrt(vx**2 + vy**2)
        
        if speed > 0.1:  # Avoid division by zero
            # Drag coefficient (tune based on cricket ball aerodynamics)
            drag_coefficient = 0.02
            
            # Velocity decay factor
            decay_factor = 1.0 - drag_coefficient * speed / 30.0
            decay_factor = max(0.8, min(1.0, decay_factor))  # Clamp between 0.8 and 1.0
            
            # Apply decay to velocity
            self.state[2] *= decay_factor
            self.state[3] *= decay_factor
    
    def reset(self):
        """Reset filter to initial state"""
        if self.use_gravity:
            self.state = np.zeros(6)
            self.P = np.eye(6) * 1000
        else:
            self.state = np.zeros(4)
            self.P = np.eye(4) * 1000
        
        self.initialized = False
        self.frames_without_detection = 0
