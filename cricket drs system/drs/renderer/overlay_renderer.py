"""
Overlay Rendering Module

Unified interface for all overlay rendering components.
"""

import numpy as np
from typing import Tuple

from ..config import RenderConfig
from .overlays import StumpOverlay, ImpactOverlay, TextOverlay


class OverlayRenderer:
    """
    Unified overlay renderer
    
    Facade combining stump, impact, and text overlay renderers
    """
    
    def __init__(self, config: RenderConfig):
        """
        Initialize overlay renderer
        
        Args:
            config: Rendering configuration
        """
        self.config = config
    
    def draw_stump_region(self,
                         frame: np.ndarray,
                         stump_x: float,
                         stump_y_top: float,
                         stump_y_bottom: float,
                         color: Tuple[int, int, int] = (255, 255, 0),
                         thickness: int = 2) -> np.ndarray:
        """Draw stump region (delegates to StumpOverlay)"""
        return StumpOverlay.draw(
            frame, stump_x, stump_y_top, stump_y_bottom, color, thickness
        )
    
    def draw_impact_point(self,
                         frame: np.ndarray,
                         x: float,
                         y: float,
                         label: str = "IMPACT",
                         color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """Draw impact point (delegates to ImpactOverlay)"""
        return ImpactOverlay.draw(frame, x, y, label, color)
    
    def draw_decision_overlay(self,
                             frame: np.ndarray,
                             decision: str,
                             confidence: float,
                             position: str = 'top-left') -> np.ndarray:
        """Draw decision overlay (delegates to TextOverlay)"""
        return TextOverlay.draw_decision(frame, decision, confidence, position)
