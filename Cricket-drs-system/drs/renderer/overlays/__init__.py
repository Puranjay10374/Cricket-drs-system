"""
Overlay Components

Modular overlay rendering components.

This package contains:
- stump_overlay.py: Stump region rendering
- impact_overlay.py: Impact point markers
- text_overlay.py: Decision text and info overlays
"""

from .stump_overlay import StumpOverlay
from .impact_overlay import ImpactOverlay
from .text_overlay import TextOverlay

__all__ = [
    'StumpOverlay',
    'ImpactOverlay',
    'TextOverlay',
]
