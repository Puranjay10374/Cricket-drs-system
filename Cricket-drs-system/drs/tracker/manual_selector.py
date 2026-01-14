"""
Manual Ball Selection Tool

Allows user to click on the ball in the first frame for accurate tracking.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class ManualBallSelector:
    """Interactive ball selection from first video frame"""
    
    def __init__(self):
        self.ball_position = None
        self.frame = None
        self.window_name = "Cricket DRS - Click on the Ball"
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ball_position = (x, y)
            # Draw a circle where user clicked
            display_frame = self.frame.copy()
            cv2.circle(display_frame, (x, y), 15, (0, 255, 0), 2)
            cv2.putText(display_frame, "Ball Selected! Press ENTER to continue",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(self.window_name, display_frame)
    
    def select_ball(self, video_path: str) -> Optional[Tuple[int, int]]:
        """
        Display first frame and let user click on ball
        
        Args:
            video_path: Path to video file
            
        Returns:
            (x, y) position of ball, or None if cancelled
        """
        # Open video and get first frame
        cap = cv2.VideoCapture(video_path)
        ret, self.frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Error: Could not read video")
            return None
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Display instructions
        display_frame = self.frame.copy()
        cv2.putText(display_frame, "Click on the cricket ball",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, "Press ESC to cancel",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(self.window_name, display_frame)
        
        # Wait for user input
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Enter/Space - confirm selection
            if key in [13, 32] and self.ball_position is not None:
                break
            
            # ESC - cancel
            if key == 27:
                self.ball_position = None
                break
        
        cv2.destroyAllWindows()
        
        if self.ball_position:
            print(f"✓ Ball selected at position: {self.ball_position}")
        else:
            print("⚠️ Ball selection cancelled")
        
        return self.ball_position


def get_manual_ball_position(video_path: str) -> Optional[Tuple[int, int, int]]:
    """
    Get ball position from user click
    
    Args:
        video_path: Path to video file
        
    Returns:
        (x, y, radius) tuple, or None if cancelled
    """
    selector = ManualBallSelector()
    position = selector.select_ball(video_path)
    
    if position:
        # Assume a reasonable ball radius (adjust based on zoom)
        estimated_radius = 10
        return (position[0], position[1], estimated_radius)
    
    return None
