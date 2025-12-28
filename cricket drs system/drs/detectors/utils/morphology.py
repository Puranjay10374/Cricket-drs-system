"""
Morphology Processing Utilities

Handles morphological operations on binary masks.
"""

import cv2
import numpy as np

from ...config import BallDetectionConfig


class MorphologyProcessor:
    """Handles morphological operations on masks"""
    
    def __init__(self, config: BallDetectionConfig):
        """
        Initialize morphology processor
        
        Args:
            config: Ball detection configuration
        """
        self.config = config
        self.kernel = np.ones(
            (config.kernel_size, config.kernel_size), 
            np.uint8
        )
    
    def process(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to reduce noise
        
        Args:
            mask: Binary mask
            
        Returns:
            Processed mask with noise removed
        """
        # Erosion to remove small noise
        mask = cv2.erode(
            mask, 
            self.kernel, 
            iterations=self.config.erosion_iterations
        )
        
        # Dilation to restore size
        mask = cv2.dilate(
            mask, 
            self.kernel, 
            iterations=self.config.dilation_iterations
        )
        
        return mask
