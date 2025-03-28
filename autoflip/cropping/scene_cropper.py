"""
Scene cropper for autoflip video reframing.

This module contains the SceneCropper class that coordinates the process of
determining optimal crop windows for scenes and applying them to frames.
"""

import logging
import time
from typing import List, Dict, Tuple, Any

import numpy as np

from autoflip.cropping.frame_crop_region import FrameCropRegionComputer
from autoflip.cropping.camera_motion import CameraMotionHandler
from autoflip.cropping.padding_effects import PaddingEffectGenerator
from autoflip.cropping.detection_utils import DetectionProcessor

# Create module-level logger
logger = logging.getLogger("autoflip.cropping.scene_cropper")


class SceneCropper:
    """
    Coordinates video scene cropping in the autoflip pipeline.
    
    This class is responsible for:
    1. Taking detections for keyframes in a scene
    2. Computing optimal crop regions
    3. Deciding on camera motion strategy
    4. Applying crop windows to all frames with smoothing
    5. Handling padding if needed
    
    It delegates specific tasks to specialized modules for better organization.
    """
    
    def __init__(
        self,
        target_aspect_ratio: float,
        motion_threshold: float = 0.5,
        padding_method: str = "blur",
        debug_mode: bool = False,
    ):
        """
        Initialize the scene cropper.
        
        Args:
            target_aspect_ratio: Target aspect ratio as width/height (e.g., 9/16)
            motion_threshold: Threshold for camera motion (0.0-1.0)
            padding_method: Method for padding ("blur" or "solid_color")
            debug_mode: If True, generate debug visualizations
        """
        self.target_aspect_ratio = target_aspect_ratio
        self.motion_threshold = motion_threshold
        self.padding_method = padding_method
        self.debug_mode = debug_mode
        
        # Initialize component modules
        self.camera_motion_handler = CameraMotionHandler(motion_threshold)
        self.padding_generator = PaddingEffectGenerator()
        self.detection_processor = DetectionProcessor()
        
        # Initialize state
        self.frame_crop_computer = None  # Will be initialized when we know frame dimensions
        
        logger.info(f"Initialized SceneCropper with target AR: {target_aspect_ratio}, "
                   f"motion threshold: {motion_threshold}")
    
    def _calculate_target_dimensions(
        self, frame_width: int, frame_height: int, target_aspect_ratio: float
    ) -> Tuple[int, int]:
        """
        Calculate target dimensions based on aspect ratio.
        
        Args:
            frame_width: Original frame width
            frame_height: Original frame height
            target_aspect_ratio: Target aspect ratio (width/height)
            
        Returns:
            Tuple of (target_width, target_height)
        """
        original_aspect_ratio = frame_width / frame_height
        
        if target_aspect_ratio > original_aspect_ratio:
            # Width constrained case (portrait target from landscape source)
            target_width = frame_width
            target_height = int(frame_width / target_aspect_ratio)
        else:
            # Height constrained case (landscape target from portrait source)
            target_height = frame_height
            target_width = int(frame_height * target_aspect_ratio)
        
        # Ensure even dimensions for video encoding
        if target_width % 2 == 1:
            target_width -= 1
        if target_height % 2 == 1:
            target_height -= 1
            
        logger.debug(f"Calculated target dimensions: {target_width}x{target_height} "
                    f"from {frame_width}x{frame_height}, target AR: {target_aspect_ratio:.4f}")
            
        return target_width, target_height

    def process_scene_streaming(
        self,
        key_frames: Dict[int, np.ndarray],
        face_detections: Dict[int, List[Dict[str, Any]]],
        object_detections: Dict[int, List[Dict[str, Any]]],
        frame_count: int,
        frame_dimensions: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Process a scene in streaming mode to generate crop windows without storing all frames.
        
        This method analyzes key frames to determine the crop strategy, then generates
        crop windows for all frames in the scene, which can be applied later during streaming.
        
        Args:
            key_frames: Dictionary mapping frame indices to key frame images
            face_detections: Dictionary mapping frame indices to face detections
            object_detections: Dictionary mapping frame indices to object detections
            frame_count: Total number of frames in the scene
            frame_dimensions: Tuple of (height, width) for the frames
            
        Returns:
            List of crop windows (x, y, width, height) for each frame
        """
        if not key_frames:
            logger.warning("No key frames provided to process_scene_streaming")
            return []
            
        # Start timing
        start_time = time.time()
            
        # Get frame dimensions from the input tuple
        frame_height, frame_width = frame_dimensions
        logger.info(f"Processing scene with {frame_count} frames, dimensions: {frame_width}x{frame_height}")
        
        # Create crop region computer with target dimensions
        target_width, target_height = self._calculate_target_dimensions(
            frame_width, frame_height, self.target_aspect_ratio)
        
        self.frame_crop_computer = FrameCropRegionComputer(
            target_width=target_width, 
            target_height=target_height
        )
        
        # Use the provided key frame indices
        key_frame_indices = sorted(key_frames.keys())
            
        if not key_frame_indices:
            logger.warning("Empty key frame dictionary provided")
            # Fall back to using evenly spaced frames (without actual frames)
            num_key_frames = max(3, frame_count // 60 + 3)
            key_frame_indices = [i * frame_count // num_key_frames for i in range(num_key_frames)]
            
        logger.info(f"Using {len(key_frame_indices)} key frames at indices: {key_frame_indices}")
        
        # Process detections (assign priorities)
        processed_detections = self.detection_processor.process_detections(
            face_detections, object_detections)
        
        # Check if this is a talking head video and adjust detection priorities if needed
        is_talking_head = self.detection_processor.identify_talking_head(
            face_detections, key_frame_indices)
            
        if is_talking_head:
            # Boost face priorities for talking head videos
            processed_detections = self.detection_processor.boost_talking_head_priorities(
                processed_detections)
            
        # Compute crop regions for key frames
        key_crop_regions = []
        
        for idx in key_frame_indices:
            # Get detections for this frame
            frame_detections = processed_detections.get(idx, [])
            
            if frame_detections:
                # Compute optimal crop region
                crop_region, crop_score, required_covered = self.frame_crop_computer.compute_frame_crop_region(
                    frame_detections, frame_width, frame_height)
                
                logger.debug(f"Frame {idx} - crop region: {crop_region}, score: {crop_score:.2f}")
                key_crop_regions.append((crop_region, crop_score))
            else:
                logger.debug(f"Frame {idx} - no detections, using default center crop")
                # Default to center crop if no detections
                default_x = (frame_width - target_width) // 2
                default_y = (frame_height - target_height) // 2
                key_crop_regions.append(((default_x, default_y, target_width, target_height), 0.0))
        
        # Select camera motion mode
        camera_mode = self.camera_motion_handler.select_camera_motion_mode(
            key_crop_regions, key_frame_indices)
            
        logger.info(f"Selected camera motion mode: {camera_mode.name}")
        
        # Generate crop windows for all frames
        all_crop_windows = self.camera_motion_handler.interpolate_crop_windows(
            key_crop_regions, key_frame_indices, frame_count, camera_mode)
        
        # Apply smoothing
        smoothed_windows = self.camera_motion_handler.smooth_trajectory(
            all_crop_windows, camera_mode)
            
        # Save the camera mode for later reference
        self.camera_mode = camera_mode
        
        # Save the last crop window for streaming use
        if smoothed_windows:
            self.current_crop_window = smoothed_windows[-1]
            
        elapsed_time = time.time() - start_time
        logger.info(f"Processed scene strategy in {elapsed_time:.2f} seconds")
        
        return smoothed_windows
    
    def apply_crop_window(
        self, 
        frame: np.ndarray, 
        crop_window: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Apply a crop window to a single frame.
        
        Args:
            frame: Video frame to crop
            crop_window: Tuple of (x, y, width, height) for cropping
            
        Returns:
            Cropped frame
        """
        if frame is None:
            logger.error("Null frame provided to apply_crop_window")
            return None
            
        frame_height, frame_width = frame.shape[:2]
        x, y, crop_width, crop_height = crop_window
        
        # Ensure crop window stays within frame
        x = max(0, min(x, frame_width - crop_width))
        y = max(0, min(y, frame_height - crop_height))
        
        # Extract crop region
        crop_region = frame[y:y+crop_height, x:x+crop_width]
        
        # Determine if padding is needed
        crop_aspect_ratio = crop_width / crop_height
        
        if not hasattr(self, 'target_aspect_ratio'):
            self.target_aspect_ratio = crop_width / crop_height
        
        if abs(crop_aspect_ratio - self.target_aspect_ratio) > 0.01:
            logger.debug(f"Frame - applying padding to match target aspect ratio")
            # Compute target dimensions that match the target aspect ratio
            target_width, target_height = self._calculate_target_dimensions(
                frame_width, frame_height, self.target_aspect_ratio)
                
            padded_frame = self.padding_generator.apply_padding(
                frame=frame, 
                crop_region=crop_region, 
                x=x, 
                y=y,
                crop_width=crop_width,
                crop_height=crop_height,
                target_width=target_width,
                target_height=target_height,
                padding_method=self.padding_method
            )
            result = padded_frame
        else:
            result = crop_region
        
        # Ensure the frame has the correct data type for VideoWriter (8-bit unsigned)
        if result.dtype != np.uint8:
            logger.warning(f"Converting frame from {result.dtype} to uint8 for VideoWriter compatibility")
            if np.issubdtype(result.dtype, np.floating):
                # Scale float values to 0-255 range
                result = (result * 255).clip(0, 255).astype(np.uint8)
            else:
                # For other types, just convert directly
                result = result.astype(np.uint8)
                
        return result
