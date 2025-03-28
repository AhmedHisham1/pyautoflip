"""
Scene cropper for autoflip video reframing.

This module contains the SceneCropper class that coordinates the process of
determining optimal crop windows for scenes and applying them to frames.
"""

import logging
import time
from typing import List, Dict, Tuple, Any, Optional

import numpy as np

from autoflip.cropping.frame_crop_region import FrameCropRegionComputer
from autoflip.cropping.camera_motion import CameraMotionHandler
from autoflip.cropping.padding_effects import PaddingEffectGenerator
from autoflip.cropping.detection_utils import DetectionProcessor
from autoflip.cropping.visualization import VisualizationUtils

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
    
    def process_scene(
        self,
        frames: List[np.ndarray],
        face_detections: Dict[int, List[Dict[str, Any]]],
        object_detections: Dict[int, List[Dict[str, Any]]],
        key_frame_indices: Optional[List[int]] = None
    ) -> Tuple[List[np.ndarray], Dict[int, np.ndarray]]:
        """
        Process a scene to create optimally cropped frames.
        
        This is the main entry point that coordinates the entire process.
        
        Args:
            frames: List of video frames for the scene
            face_detections: Dictionary mapping frame indices to face detections
            object_detections: Dictionary mapping frame indices to object detections
            key_frame_indices: Indices of key frames (if None, all frames with detections are key frames)
            
        Returns:
            Tuple containing:
            - List of cropped frames
            - Dictionary of debug visualization frames (if debug_mode is True)
        """
        if not frames:
            logger.warning("Empty frame list provided to process_scene")
            return [], {}
            
        # Start timing
        start_time = time.time()
            
        # Get frame dimensions
        frame_height, frame_width = frames[0].shape[:2]
        logger.info(f"Processing scene with {len(frames)} frames, dimensions: {frame_width}x{frame_height}")
        
        # Create crop region computer with target dimensions
        target_width, target_height = self._calculate_target_dimensions(
            frame_width, frame_height, self.target_aspect_ratio)
        
        self.frame_crop_computer = FrameCropRegionComputer(
            target_width=target_width, 
            target_height=target_height
        )
        
        # If key_frame_indices not provided, use all frames with detections
        if key_frame_indices is None:
            key_frame_indices = sorted(set(face_detections.keys()) | set(object_detections.keys()))
            
        if not key_frame_indices:
            logger.warning("No key frames with detections found")
            # Fall back to using evenly spaced frames
            num_key_frames = max(3, len(frames) // 10)  # At least 3, or 10% of frames
            key_frame_indices = [i * len(frames) // num_key_frames for i in range(num_key_frames)]
            
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
            key_crop_regions, key_frame_indices, len(frames), camera_mode)
        
        # Apply smoothing
        smoothed_windows = self.camera_motion_handler.smooth_trajectory(
            all_crop_windows, camera_mode)
            
        # Crop all frames
        cropped_frames = []
        
        # Debug visualization frames (only populated if debug_mode is True)
        debug_frames = {}
        
        for i, frame in enumerate(frames):
            x, y, crop_width, crop_height = smoothed_windows[i]
            
            # Ensure crop window stays within frame
            x = max(0, min(x, frame_width - crop_width))
            y = max(0, min(y, frame_height - crop_height))
            
            # Extract crop region
            crop_region = frame[y:y+crop_height, x:x+crop_width]
            
            # Determine if padding is needed
            crop_aspect_ratio = crop_width / crop_height
            
            if abs(crop_aspect_ratio - self.target_aspect_ratio) > 0.01:
                logger.debug(f"Frame {i} - applying padding to match target aspect ratio")
                # Compute target dimensions that match the target aspect ratio
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
                cropped_frames.append(padded_frame)
            else:
                cropped_frames.append(crop_region)
            
            # Create debug visualization if in debug mode
            if self.debug_mode and i in key_frame_indices:
                key_idx = key_frame_indices.index(i)
                
                # Get the processed detections for this frame which have priorities and required flags
                frame_processed_detections = processed_detections.get(i, [])
                
                # Separate processed detections into faces and objects
                processed_face_dets = []
                processed_obj_dets = []
                
                for det in frame_processed_detections:
                    if det.get('class', '').lower() == 'face':
                        processed_face_dets.append(det)
                    else:
                        processed_obj_dets.append(det)
                
                debug_frame = VisualizationUtils.create_debug_frame(
                    frame=frame,
                    x=x,
                    y=y,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    face_dets=processed_face_dets,  # Use processed face detections
                    object_dets=processed_obj_dets,  # Use processed object detections
                    saliency_score=key_crop_regions[key_idx][1] if key_idx < len(key_crop_regions) else None,
                    camera_mode=camera_mode
                )
                # Store debug frame
                debug_frames[i] = debug_frame
        
        # Optionally create a motion trajectory visualization
        if self.debug_mode and len(smoothed_windows) > 1:
            # Create a trajectory visualization for the first frame
            trajectory_vis = VisualizationUtils.create_trajectory_vis(
                frames[0], smoothed_windows, 0)
            debug_frames[-1] = trajectory_vis  # Use -1 as a special key
        
        # Ensure all frames have the correct data type for VideoWriter
        for i in range(len(cropped_frames)):
            if cropped_frames[i].dtype != np.uint8:
                logger.warning(f"Converting frame {i} from {cropped_frames[i].dtype} to uint8 for VideoWriter compatibility")
                if np.issubdtype(cropped_frames[i].dtype, np.floating):
                    # Scale float values to 0-255 range
                    cropped_frames[i] = (cropped_frames[i] * 255).clip(0, 255).astype(np.uint8)
                else:
                    # For other types, just convert directly
                    cropped_frames[i] = cropped_frames[i].astype(np.uint8)
                    
        # Save the last crop window for streaming use
        if smoothed_windows:
            self.current_crop_window = smoothed_windows[-1]
            
        elapsed_time = time.time() - start_time
        logger.info(f"Processed scene in {elapsed_time:.2f} seconds "
                   f"({len(frames) / elapsed_time:.1f} frames/sec)")
        
        return cropped_frames, debug_frames
    
    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop a single frame using the current cropping strategy.
        
        This method is used for streaming mode after initial scene analysis.
        
        Args:
            frame: Video frame to crop
            
        Returns:
            Cropped frame
        """
        # Default to center crop if no cropping strategy established
        if not hasattr(self, 'current_crop_window'):
            height, width = frame.shape[:2]
            target_width, target_height = self._calculate_target_dimensions(
                width, height, self.target_aspect_ratio)
                
            x = (width - target_width) // 2
            y = (height - target_height) // 2
            
            self.current_crop_window = (x, y, target_width, target_height)
            logger.debug(f"Initialized default center crop: {self.current_crop_window}")
        
        # Extract crop region
        x, y, crop_width, crop_height = self.current_crop_window
        
        # Ensure coordinates are within frame bounds
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        crop_width = min(crop_width, frame.shape[1] - x)
        crop_height = min(crop_height, frame.shape[0] - y)
        
        # Extract the crop region
        crop_region = frame[y:y+crop_height, x:x+crop_width]
        
        # Ensure the frame has the correct data type for VideoWriter (8-bit unsigned)
        if crop_region.dtype != np.uint8:
            logger.warning(f"Converting frame from {crop_region.dtype} to uint8 for VideoWriter compatibility")
            if np.issubdtype(crop_region.dtype, np.floating):
                # Scale float values to 0-255 range
                crop_region = (crop_region * 255).clip(0, 255).astype(np.uint8)
            else:
                # For other types, just convert directly
                crop_region = crop_region.astype(np.uint8)
        
        return crop_region
    
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
