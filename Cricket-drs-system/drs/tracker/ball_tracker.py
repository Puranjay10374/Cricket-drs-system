"""
Ball Tracker - Core Tracking Logic

Handles ball detection and tracking across video frames.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict

from ..detectors import BallDetector, ColorBasedDetector
from ..video_io import VideoReader
from ..config import BallDetectionConfig
from .quality_metrics import QualityMetrics
from .kalman_filter import BallKalmanFilter
from .manual_selector import get_manual_ball_position


class BallTracker:
    """Tracks a cricket ball in video frames using pluggable detection strategies"""
    
    def __init__(self, 
                 detector: Optional[BallDetector] = None,
                 config: Optional[BallDetectionConfig] = None,
                 use_kalman: bool = True,
                 use_manual_selection: bool = False):
        """
        Initialize ball tracker
        
        Args:
            detector: Ball detection strategy (defaults to ColorBasedDetector)
            config: Detection configuration (uses defaults if not provided)
            use_kalman: Enable Kalman filter for prediction and smoothing
            use_manual_selection: Enable manual ball selection from first frame
        """
        self.config = config or BallDetectionConfig()
        self.detector = detector or ColorBasedDetector(self.config)
        self.use_kalman = use_kalman
        self.use_manual_selection = use_manual_selection
        self.kalman_filter = BallKalmanFilter() if use_kalman else None
        self.initial_ball_position = None  # Store manual selection
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect ball in a single frame using configured detector
        
        Args:
            frame: BGR image from video
            
        Returns:
            Tuple of (x, y, radius) if ball found, None otherwise
        """
        return self.detector.detect(frame)
    
    def track_video(self, video_path: str, frame_skip: int = 1) -> Dict:
        """
        Track ball throughout entire video with smart filtering and interpolation
        
        Args:
            video_path: Path to video file
            frame_skip: Process every Nth frame (1 = all frames, RECOMMENDED for fast balls)
            
        Returns:
            Dictionary containing tracking results and metadata
        """
        trajectory = []
        frames_processed = 0
        frames_tracked = 0
        last_position = None
        last_frame = None
        last_velocity = None  # Track velocity for smoothness check
        max_movement = 250  # INCREASED for fast-moving cricket ball (was 180)
        max_acceleration = 100  # NEW: Max change in velocity (px/frame²)
        max_search_radius = 250  # INCREASED: Search radius for manual selection (was 150)
        
        # Reset Kalman filter for new video
        if self.use_kalman and self.kalman_filter:
            self.kalman_filter.reset()
        
        # Get manual ball position if enabled
        if self.use_manual_selection:
            print("\n🎯 Manual Ball Selection Mode")
            print("Please click on the ball in the first frame...")
            self.initial_ball_position = get_manual_ball_position(video_path)
            if self.initial_ball_position is None:
                print("⚠️ Manual selection cancelled - falling back to automatic detection")
                self.use_manual_selection = False
            else:
                print(f"✓ Ball position locked: {self.initial_ball_position[:2]}")
                
                # Learn ball appearance from selection (if detector supports it)
                if hasattr(self.detector, 'learn_from_selection'):
                    with VideoReader(video_path, frame_skip=1) as reader:
                        first_frame = next(reader.frames())[1]
                        x, y, r = self.initial_ball_position
                        self.detector.learn_from_selection(first_frame, x, y, r)
                        print("✓ Adaptive detector learned ball appearance")
                
                # Initialize Kalman with manual selection
                if self.use_kalman and self.kalman_filter:
                    self.kalman_filter.update(self.initial_ball_position[:2])
        
        # Use VideoReader for clean video I/O
        with VideoReader(video_path, frame_skip=frame_skip) as reader:
            props = reader.get_properties()
            frame_height = props['frame_size'][1]
            frame_width = props['frame_size'][0]
            
            # Define ROI (focus on pitch area - lower 70% and center 80%)
            roi_top = int(frame_height * 0.3)
            roi_left = int(frame_width * 0.1)
            roi_right = int(frame_width * 0.9)
            
            for frame_num, frame in reader.frames():
                frames_processed += 1
                
                # For first frame with manual selection, use that position
                if frame_num == 0 and self.initial_ball_position is not None:
                    ball_pos = self.initial_ball_position
                    x, y, radius = ball_pos[0], ball_pos[1], ball_pos[2]
                    trajectory.append((frame_num, x, y, radius))
                    frames_tracked += 1
                    last_position = (x, y)
                    last_frame = frame_num
                    continue
                
                # Detect ball in this frame
                raw_detection = self.detect_ball(frame)
                
                # Apply Kalman filter if enabled
                if self.use_kalman and self.kalman_filter:
                    ball_pos = self.kalman_filter.process(raw_detection)
                else:
                    ball_pos = raw_detection
                
                if ball_pos:
                    x, y, radius = ball_pos
                    
                    # Manual selection mode: only accept detections near expected path
                    # But relax constraint if we've successfully tracked for a while
                    if self.use_manual_selection and last_position is not None:
                        expected_x, expected_y = last_position
                        if self.use_kalman and self.kalman_filter:
                            # Use Kalman prediction for expected position
                            predicted = self.kalman_filter.predict()
                            if predicted is not None:
                                expected_x, expected_y = predicted
                        
                        distance_from_expected = np.sqrt((x - expected_x)**2 + (y - expected_y)**2)
                        
                        # Adaptive search radius based on tracking success
                        adaptive_radius = max_search_radius
                        if len(trajectory) > 10:  # After 10 good detections, trust the detector more
                            adaptive_radius = max_search_radius * 1.5
                        
                        if distance_from_expected > adaptive_radius:
                            # Too far from expected path - likely false detection
                            if frames_processed % 10 == 0:  # Debug every 10 frames
                                print(f"   Frame {frame_num}: Detection at ({x},{y}) rejected - {distance_from_expected:.0f}px from expected ({expected_x:.0f},{expected_y:.0f})")
                            ball_pos = None
                            continue
                        else:
                            if frames_processed <= 20:  # Debug early frames
                                print(f"   Frame {frame_num}: Detection at ({x},{y}) accepted - {distance_from_expected:.0f}px from expected")
                    
                    # ROI filtering: prefer detections in pitch area
                    in_roi = (y >= roi_top and roi_left <= x <= roi_right)
                    
                    # Motion-based filtering (less strict with Kalman)
                    if last_position is not None and last_frame is not None:
                        last_x, last_y = last_position
                        distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                        frame_gap = frame_num - last_frame
                        
                        # Calculate velocity
                        if frame_gap > 0:
                            current_velocity = (x - last_x, y - last_y)
                            
                            # Velocity consistency check (smooth ball motion)
                            if last_velocity is not None and len(trajectory) > 5:
                                # Calculate acceleration (change in velocity)
                                accel_x = abs(current_velocity[0] - last_velocity[0])
                                accel_y = abs(current_velocity[1] - last_velocity[1])
                                acceleration = np.sqrt(accel_x**2 + accel_y**2)
                                
                                # Reject erratic movements (body parts jerk around)
                                # Ball has smooth motion, body parts move erratically
                                if acceleration > max_acceleration:
                                    continue  # Too erratic - likely not a ball
                            
                            last_velocity = current_velocity
                        
                        # Adjust max movement based on frame gap
                        adjusted_max_movement = max_movement * (1 + frame_gap * 0.5)
                        
                        # With Kalman, be more lenient (predictions can be far)
                        if self.use_kalman and raw_detection is None:
                            adjusted_max_movement *= 1.5  # Allow predictions to be farther
                        
                        # Reject if moved too far
                        if distance > adjusted_max_movement:
                            # Unless it's in ROI and previous wasn't
                            if not in_roi:
                                continue
                    
                    # Size consistency check (skip for Kalman predictions)
                    if trajectory and raw_detection is not None:
                        last_radius = trajectory[-1][3]
                        # Allow more variation for small balls
                        max_radius_change = 15 if radius < 15 else 10
                        if abs(radius - last_radius) > max_radius_change:
                            continue
                    
                    # Prioritize detections in ROI
                    # If we have detections outside ROI and this is inside, trust it more
                    if in_roi or not trajectory or len(trajectory) < 3:
                        trajectory.append((frame_num, x, y, radius))
                        frames_tracked += 1
                        last_position = (x, y)
                        last_frame = frame_num
        
        # Remove outliers before processing
        trajectory = self._remove_outliers(trajectory)
        
        # Smooth trajectory for natural ball motion
        trajectory = self._smooth_trajectory(trajectory)
        
        # Interpolate missing frames for smoother trajectory
        trajectory = self._interpolate_trajectory(trajectory)
        
        # Calculate tracking quality
        tracking_quality = QualityMetrics.calculate_tracking_quality(
            len(trajectory),  # Use smoothed trajectory length
            frames_processed
        )
        
        return {
            'trajectory': trajectory,
            'total_frames': props['total_frames'],
            'frames_tracked': frames_tracked,
            'fps': props['fps'],
            'frame_size': props['frame_size'],
            'tracking_quality': tracking_quality
        }
    
    def _interpolate_trajectory(self, trajectory: List[Tuple]) -> List[Tuple]:
        """
        Interpolate missing frames in trajectory for smoother visualization
        
        Args:
            trajectory: List of (frame_num, x, y, radius) tuples
            
        Returns:
            Trajectory with interpolated points
        """
        if len(trajectory) < 2:
            return trajectory
        
        interpolated = []
        
        for i in range(len(trajectory) - 1):
            frame1, x1, y1, r1 = trajectory[i]
            frame2, x2, y2, r2 = trajectory[i + 1]
            
            # Add current point
            interpolated.append((frame1, x1, y1, r1))
            
            # Interpolate if gap is small (< 5 frames)
            frame_gap = frame2 - frame1
            if 1 < frame_gap < 5:
                # Linear interpolation for missing frames
                for j in range(1, frame_gap):
                    t = j / frame_gap
                    interp_frame = frame1 + j
                    interp_x = int(x1 + (x2 - x1) * t)
                    interp_y = int(y1 + (y2 - y1) * t)
                    interp_r = int(r1 + (r2 - r1) * t)
                    interpolated.append((interp_frame, interp_x, interp_y, interp_r))
        
        # Add last point
        if trajectory:
            interpolated.append(trajectory[-1])
        
        return interpolated
    
    def _remove_outliers(self, trajectory: List[Tuple]) -> List[Tuple]:
        """
        Remove outlier detections using distance-based filtering
        
        Args:
            trajectory: List of (frame_num, x, y, radius) tuples
            
        Returns:
            Cleaned trajectory without outliers
        """
        if len(trajectory) < 3:
            return trajectory
        
        cleaned = []
        
        for i in range(len(trajectory)):
            frame, x, y, radius = trajectory[i]
            
            # For first and last points, always keep
            if i == 0 or i == len(trajectory) - 1:
                cleaned.append(trajectory[i])
                continue
            
            # Check if point is consistent with neighbors
            prev_frame, prev_x, prev_y, prev_r = trajectory[i - 1]
            next_frame, next_x, next_y, next_r = trajectory[i + 1]
            
            # Calculate distances to neighbors
            dist_to_prev = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            dist_to_next = np.sqrt((x - next_x)**2 + (y - next_y)**2)
            
            # Expected distance based on neighbor positions
            neighbor_dist = np.sqrt((next_x - prev_x)**2 + (next_y - prev_y)**2)
            avg_dist = (dist_to_prev + dist_to_next) / 2
            
            # If this point is too far from both neighbors, it's likely an outlier
            # Allow point if it's reasonably positioned between neighbors
            if avg_dist < neighbor_dist * 1.5:  # Within reasonable path
                cleaned.append(trajectory[i])
        
        return cleaned if len(cleaned) > 0 else trajectory
    
    def _smooth_trajectory(self, trajectory: List[Tuple]) -> List[Tuple]:
        """
        Smooth trajectory using moving average for natural ball motion
        
        Args:
            trajectory: List of (frame_num, x, y, radius) tuples
            
        Returns:
            Smoothed trajectory
        """
        if len(trajectory) < 3:
            return trajectory
        
        smoothed = []
        window_size = 3  # Use 3-point moving average
        
        for i in range(len(trajectory)):
            frame, x, y, radius = trajectory[i]
            
            # For edges, use less smoothing
            if i == 0 or i == len(trajectory) - 1:
                smoothed.append(trajectory[i])
                continue
            
            # Calculate moving average
            if i > 0 and i < len(trajectory) - 1:
                _, prev_x, prev_y, prev_r = trajectory[i - 1]
                _, next_x, next_y, next_r = trajectory[i + 1]
                
                # Average with neighbors
                smooth_x = int((prev_x + x + next_x) / 3)
                smooth_y = int((prev_y + y + next_y) / 3)
                smooth_r = int((prev_r + radius + next_r) / 3)
                
                smoothed.append((frame, smooth_x, smooth_y, smooth_r))
        
        return smoothed
