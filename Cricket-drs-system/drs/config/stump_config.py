"""
Stump Region Configuration

Configuration for stump region definition and intersection checking.
"""

from dataclasses import dataclass


@dataclass
class StumpRegion:
    """Stump region configuration"""
    
    x: float
    y_top: float
    y_bottom: float
    x_tolerance: float = 10.0
    
    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if point falls within stump region
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is within stump region
        """
        x_match = abs(x - self.x) <= self.x_tolerance
        y_match = self.y_top <= y <= self.y_bottom
        return x_match and y_match
    
    def get_range(self) -> tuple:
        """
        Get stump y-range
        
        Returns:
            Tuple of (y_top, y_bottom)
        """
        return (self.y_top, self.y_bottom)


# Default stump region (for 1280x720 video)
DEFAULT_STUMP_REGION = StumpRegion(
    x=640,
    y_top=300,
    y_bottom=450  # Changed from 500 to fit within typical frame heights
)


def create_stump_region_for_frame(frame_width: int, frame_height: int) -> StumpRegion:
    """
    Create stump region automatically scaled for frame size
    
    Args:
        frame_width: Video frame width in pixels
        frame_height: Video frame height in pixels
        
    Returns:
        StumpRegion scaled for the given frame size
    """
    # Stumps are typically:
    # - Horizontally: at center of frame (where batsman stands)
    # - Vertically: from ~55% to ~85% down the frame
    
    stump_x = frame_width / 2
    stump_y_top = frame_height * 0.55   # 55% down from top
    stump_y_bottom = frame_height * 0.85  # 85% down from top
    
    return StumpRegion(
        x=stump_x,
        y_top=stump_y_top,
        y_bottom=stump_y_bottom,
        x_tolerance=frame_width * 0.05  # 5% of width as tolerance
    )
