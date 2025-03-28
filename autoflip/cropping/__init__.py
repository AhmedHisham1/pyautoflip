"""
Autoflip video cropping module.

This module provides functionality for automatically cropping video scenes
to a target aspect ratio while preserving important content.
"""

from autoflip.cropping.scene_cropper import SceneCropper
from autoflip.cropping.types import LogLevel, CameraMotionMode, CoverageType
from autoflip.cropping.camera_motion import CameraMotionHandler
from autoflip.cropping.frame_crop_region import FrameCropRegionComputer
from autoflip.cropping.padding_effects import PaddingEffectGenerator
from autoflip.cropping.detection_utils import DetectionProcessor
from autoflip.cropping.visualization import VisualizationUtils

__all__ = [
    'SceneCropper',
    'LogLevel',
    'CameraMotionMode',
    'CoverageType',
    'CameraMotionHandler',
    'FrameCropRegionComputer',
    'PaddingEffectGenerator',
    'DetectionProcessor',
    'VisualizationUtils'
]
