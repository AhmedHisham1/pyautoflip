"""
Signal fusion package for AutoFlip.

This package provides components for fusing different detection types
with configurable priorities and weights.
"""

from autoflip.fusion.signal_fuser import (
    SignalFuser,
    SignalType,
    StandardSignalType,
    SignalSettings,
    SalientRegion
)

__all__ = [
    'SignalFuser',
    'SignalType',
    'StandardSignalType',
    'SignalSettings',
    'SalientRegion'
] 