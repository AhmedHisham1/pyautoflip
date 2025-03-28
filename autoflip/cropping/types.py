"""
Shared types and enums for the autoflip cropping modules.
"""

from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Callable, Union


class LogLevel(Enum):
    """Log levels for more granular control beyond standard Python logging."""
    DEBUG_DETAIL = 5  # Even more detailed than DEBUG
    DEBUG = 10        # Standard DEBUG
    INFO = 20         # Standard INFO
    WARNING = 30      # Standard WARNING
    ERROR = 40        # Standard ERROR


class CameraMotionMode(Enum):
    """Camera motion modes similar to MediaPipe AutoFlip."""
    STATIONARY = 0
    PANNING = 1
    TRACKING = 2


class CoverageType(Enum):
    """Coverage types for crop regions."""
    NOT_COVERED = 0
    PARTIALLY_COVERED = 1
    FULLY_COVERED = 2 