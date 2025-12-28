"""
Base Ball Detector

Abstract interface for ball detection strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BallDetector(ABC):
    """Abstract base class for ball detection strategies"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect ball in a single frame
        
        Args:
            frame: BGR image from video
            
        Returns:
            Tuple of (x, y, radius) if ball found, None otherwise
        """
        pass
