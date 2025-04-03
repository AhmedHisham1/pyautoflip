from enum import Enum

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