"""
Autoflip video cropping module.

This module provides functionality for automatically cropping video scenes
to a target aspect ratio while preserving important content.
"""

from pyautoflip.cropping.scene_cropper import SceneCropper
from pyautoflip.cropping.types import CameraMotionMode
from pyautoflip.cropping.camera_motion import CameraMotionHandler
from pyautoflip.cropping.frame_crop_region import FrameCropRegionComputer
from pyautoflip.cropping.padding_effects import PaddingEffectGenerator
from pyautoflip.cropping.detection_utils import DetectionProcessor

__all__ = [
    "SceneCropper",
    "CameraMotionMode",
    "CameraMotionHandler",
    "FrameCropRegionComputer",
    "PaddingEffectGenerator",
    "DetectionProcessor",
]
