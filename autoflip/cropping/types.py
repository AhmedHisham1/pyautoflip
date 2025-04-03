from enum import Enum

class CameraMotionMode(Enum):
    """Camera motion modes similar to MediaPipe AutoFlip."""
    STATIONARY = 0
    PANNING = 1
    TRACKING = 2
