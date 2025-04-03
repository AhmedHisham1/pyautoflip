"""
Autoflip video cropping module.

This module provides functionality for automatically cropping video scenes
to a target aspect ratio while preserving important content.
"""

from autoflip.cropping.scene_cropper import SceneCropper
from autoflip.cropping.types import CameraMotionMode
from autoflip.cropping.camera_motion import CameraMotionHandler
from autoflip.cropping.frame_crop_region import FrameCropRegionComputer
from autoflip.cropping.padding_effects import PaddingEffectGenerator
from autoflip.cropping.detection_utils import DetectionProcessor

__all__ = [
    "SceneCropper",
    "CameraMotionMode",
    "CameraMotionHandler",
    "FrameCropRegionComputer",
    "PaddingEffectGenerator",
    "DetectionProcessor",
]
