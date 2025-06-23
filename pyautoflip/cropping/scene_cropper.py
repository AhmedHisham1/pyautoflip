"""
Scene cropper for autoflip video reframing.

This module contains the SceneCropper class that coordinates the process of
determining optimal crop windows for scenes and applying them to frames.
"""

import logging
import time
from typing import List, Dict, Tuple, Any

import numpy as np

from pyautoflip.cropping.frame_crop_region import FrameCropRegionComputer
from pyautoflip.cropping.camera_motion import CameraMotionHandler
from pyautoflip.cropping.padding_effects import PaddingEffectGenerator
from pyautoflip.cropping.detection_utils import DetectionProcessor
from pyautoflip.utils.visualizer import Visualizer

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
        self.frame_crop_computer = (
            None  # Will be initialized when we know frame dimensions
        )

        logger.debug(
            f"Initialized SceneCropper with target AR: {target_aspect_ratio}, "
            f"motion threshold: {motion_threshold}"
        )

    def process_scene(
        self,
        key_frames: Dict[int, np.ndarray],
        face_detections: Dict[int, List[Dict[str, Any]]],
        object_detections: Dict[int, List[Dict[str, Any]]],
        frame_count: int,
    ) -> List[Tuple]:
        """
        Process a scene in streaming mode to generate crop windows without storing all frames.

        This method analyzes key frames to determine the crop strategy, then generates
        crop windows for all frames in the scene, which can be applied later during streaming.

        Args:
            key_frames: Dictionary mapping frame indices to key frame images
            face_detections: Dictionary mapping frame indices to face detections
            object_detections: Dictionary mapping frame indices to object detections
            frame_count: Total number of frames in the scene

        Returns:
            List of crop windows as relative coordinates (x_rel, y_rel, w_rel, h_rel)
            where each value is between 0.0 and 1.0
        """
        if not key_frames:
            logger.warning("No key frames provided to process_scene_streaming")
            return []

        # Start timing
        start_time = time.time()

        key_frame_indices = sorted(key_frames.keys())
        frame_height, frame_width = key_frames[key_frame_indices[0]].shape[:2]

        # Step 1: Initialize crop region computer
        target_width, target_height = self._initialize_crop_computer(
            frame_width, frame_height
        )

        # Step 2: Process detections
        processed_detections = self._process_scene_detections(
            face_detections, object_detections, key_frame_indices
        )

        # Step 3: Compute crop regions for key frames
        key_crop_regions = self._compute_key_crop_regions(
            key_frame_indices,
            processed_detections,
            frame_width,
            frame_height,
            target_width,
            target_height,
        )

        if self.debug_mode:
            self.__visualize_crop_regions(
                key_frames, key_crop_regions, processed_detections
            )

        # Step 4: Select camera motion mode and generate windows
        camera_mode = self.camera_motion_handler.select_camera_motion_mode(
            key_crop_regions
        )
        logger.debug(f"Selected camera motion mode: {camera_mode.name}")

        # Step 5: Generate and smooth crop windows
        all_crop_windows = self._generate_crop_windows(
            key_crop_regions, key_frame_indices, frame_count, camera_mode
        )

        # Step 6: Convert to relative coordinates
        relative_windows = self._convert_to_relative_coordinates(
            all_crop_windows, frame_width, frame_height
        )

        elapsed_time = time.time() - start_time
        logger.debug(f"Processed scene strategy in {elapsed_time:.2f} seconds")

        return relative_windows

    def _initialize_crop_computer(
        self, frame_width: int, frame_height: int
    ) -> Tuple[int, int]:
        """Initialize the crop region computer with target dimensions."""

        target_width, target_height = self._calculate_target_dimensions(
            frame_width, frame_height, self.target_aspect_ratio
        )

        self.frame_crop_computer = FrameCropRegionComputer(
            target_width=target_width, target_height=target_height
        )
        return target_width, target_height

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
            # Width constrained case (landscape target from portrait source)
            target_width = frame_width
            target_height = int(frame_width / target_aspect_ratio)
            logger.debug(
                f"Width fixed: keeping original width ({frame_width}), adjusting height to {target_height}"
            )
        else:
            # Height constrained case (portrait target from landscape source)
            target_height = frame_height
            target_width = int(frame_height * target_aspect_ratio)
            logger.debug(
                f"Height fixed: keeping original height ({frame_height}), adjusting width to {target_width}"
            )
        # Ensure even dimensions for video encoding: subtract 1 if odd
        target_width -= 1 if target_width % 2 == 1 else 0
        target_height -= 1 if target_height % 2 == 1 else 0

        logger.debug(
            f"Calculated target dimensions (wxh): {target_width}x{target_height} "
            f"from (wxh): {frame_width}x{frame_height}, target AR: {target_aspect_ratio:.4f}, "
            f"original AR: {original_aspect_ratio:.4f}"
        )

        return target_width, target_height

    def _process_scene_detections(
        self,
        face_detections: Dict[int, List[Dict[str, Any]]],
        object_detections: Dict[int, List[Dict[str, Any]]],
        key_frame_indices: List[int],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Process detection data for the scene."""
        # Process detections (assign priorities)
        processed_detections = self.detection_processor.process_detections(
            face_detections, object_detections
        )

        # Check if this is a talking head video and adjust detection priorities if needed
        is_talking_head = self.detection_processor.identify_talking_head(
            face_detections, key_frame_indices
        )

        if is_talking_head:
            # Boost face priorities for talking head videos
            processed_detections = (
                self.detection_processor.boost_talking_head_priorities(
                    processed_detections
                )
            )

        return processed_detections

    def _compute_key_crop_regions(
        self,
        key_frame_indices: List[int],
        processed_detections: Dict[int, List[Dict[str, Any]]],
        frame_width: int,
        frame_height: int,
        target_width: int,
        target_height: int,
    ) -> List[Tuple]:
        """Compute crop regions for key frames."""
        crop_regions_start = time.time()
        key_crop_regions = []

        for idx in key_frame_indices:
            # Get detections for this frame
            frame_detections = processed_detections.get(idx, [])

            if frame_detections:
                # Compute optimal crop region
                (
                    crop_region,
                    crop_score,
                ) = self.frame_crop_computer.compute_frame_crop_region(
                    frame_detections, frame_width, frame_height
                )

                logger.debug(
                    f"Frame {idx} - crop region: {crop_region}, score: {crop_score:.2f}"
                )
                key_crop_regions.append((crop_region, crop_score))
            else:
                logger.debug(f"Frame {idx} - no detections, using default center crop")
                # Default to center crop if no detections
                default_x = (frame_width - target_width) // 2
                default_y = (frame_height - target_height) // 2
                key_crop_regions.append(
                    ((default_x, default_y, target_width, target_height), 0.0)
                )

        crop_regions_end = time.time()
        logger.debug(
            f"Crop regions computation took {crop_regions_end - crop_regions_start:.4f} seconds"
        )

        return key_crop_regions

    def __visualize_crop_regions(
        self,
        key_frames: Dict[int, np.ndarray],
        key_crop_regions: List[Tuple],
        processed_detections: Dict[int, List[Dict[str, Any]]],
    ):
        """Visualize the crop regions."""
        import cv2

        # cleanup Visualizer SceneCropper frames
        Visualizer.clear_frames_with_prefix("SceneCropper - Frame")

        for idx in range(len(key_crop_regions)):
            crop_region, crop_score = key_crop_regions[idx]
            frame_idx = list(key_frames.keys())[idx]
            frame = key_frames[frame_idx].copy()

            frame_processed_detections = processed_detections.get(frame_idx, [])
            # draw the detections
            for detection in frame_processed_detections:
                x, y, w, h = (
                    detection["x"],
                    detection["y"],
                    detection["width"],
                    detection["height"],
                )
                # convert to absolute coordinates
                frame_height, frame_width = frame.shape[:2]
                x, y = int(x * frame_width), int(y * frame_height)
                w, h = int(w * frame_width), int(h * frame_height)
                priority = detection["priority"]
                class_name = detection["class"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # red
                cv2.putText(
                    frame,
                    f"Priority: {priority:.2f}",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Class: {class_name}",
                    (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )

            # draw the crop regions and scores
            x, y, w, h = crop_region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green
            cv2.putText(
                frame,
                f"Frame: {frame_idx}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )  # blue
            cv2.putText(
                frame,
                f"Score: {crop_score:.2f}",
                (x, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )  # blue

            # show frame shape and crop region shape in the top right corner of the frame
            cv2.putText(
                frame,
                f"Frame shape: {frame.shape}",
                (frame.shape[1] - 300, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Crop region: {crop_region}",
                (frame.shape[1] - 300, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

            Visualizer.report_frame(frame, f"SceneCropper - Frame {frame_idx}")

    def _generate_crop_windows(
        self,
        key_crop_regions: List[Tuple],
        key_frame_indices: List[int],
        frame_count: int,
        camera_mode,
    ) -> List[Tuple]:
        """Generate and smooth crop windows for all frames."""
        # Generate crop windows for all frames
        interpolation_start = time.time()
        all_crop_windows = self.camera_motion_handler.interpolate_crop_windows(
            key_crop_regions, key_frame_indices, frame_count, camera_mode
        )
        interpolation_end = time.time()
        logger.debug(
            f"Crop windows interpolation took {interpolation_end - interpolation_start:.4f} seconds"
        )

        # Apply smoothing
        smoothing_start = time.time()
        smoothed_windows = self.camera_motion_handler.smooth_trajectory(
            all_crop_windows, camera_mode
        )
        smoothing_end = time.time()
        logger.debug(
            f"Trajectory smoothing took {smoothing_end - smoothing_start:.4f} seconds"
        )

        return smoothed_windows

    def _convert_to_relative_coordinates(
        self, crop_windows: List[Tuple], frame_width: int, frame_height: int
    ) -> List[Tuple]:
        """Convert absolute crop windows to relative coordinates."""
        conversion_start = time.time()

        relative_windows = []
        for x, y, w, h in crop_windows:
            x_rel = x / frame_width
            y_rel = y / frame_height
            w_rel = w / frame_width
            h_rel = h / frame_height
            relative_windows.append((x_rel, y_rel, w_rel, h_rel))

        # Save the last crop window for streaming use
        if relative_windows:
            self.current_crop_window = relative_windows[-1]

        conversion_end = time.time()
        logger.debug(
            f"Coordinates conversion took {conversion_end - conversion_start:.4f} seconds"
        )

        return relative_windows

    def apply_crop_window(
        self, frame: np.ndarray, crop_window: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Apply a crop window to a single frame.

        Args:
            frame: Video frame to crop
            crop_window: Tuple of (x_rel, y_rel, w_rel, h_rel) as relative coordinates (0.0-1.0)

        Returns:
            Cropped frame
        """
        start_time = time.time()

        if frame is None:
            logger.error("Null frame provided to apply_crop_window")
            return None

        # Get frame dimensions and convert coordinates - fast operations
        frame_height, frame_width = frame.shape[:2]
        x_rel, y_rel, w_rel, h_rel = crop_window
        x = int(x_rel * frame_width)
        y = int(y_rel * frame_height)
        crop_width = int(w_rel * frame_width)
        crop_height = int(h_rel * frame_height)

        # Ensure even dimensions for video encoding
        if crop_width % 2 == 1:
            crop_width -= 1
        if crop_height % 2 == 1:
            crop_height -= 1

        # Ensure crop window stays within frame
        x = max(0, min(x, frame_width - crop_width))
        y = max(0, min(y, frame_height - crop_height))

        # Extract crop region - core operation
        crop_region = frame[y : y + crop_height, x : x + crop_width]

        # Check if padding is needed
        crop_aspect_ratio = crop_width / crop_height

        if not hasattr(self, "target_aspect_ratio"):
            self.target_aspect_ratio = crop_width / crop_height

        # Only apply padding if absolutely necessary (aspect ratio difference > 1%)
        padding_start = time.time()
        if abs(crop_aspect_ratio - self.target_aspect_ratio) > 0.01:
            # Compute target dimensions that match the target aspect ratio
            target_width, target_height = self._calculate_target_dimensions(
                frame_width, frame_height, self.target_aspect_ratio
            )

            # Apply padding - potentially expensive operation
            padded_frame = self.padding_generator.apply_padding(
                frame=frame,
                crop_region=crop_region,
                x=x,
                y=y,
                crop_width=crop_width,
                crop_height=crop_height,
                target_width=target_width,
                target_height=target_height,
                padding_method=self.padding_method,
            )
            result = padded_frame
        else:
            # Fast path - no padding needed
            result = crop_region
        padding_end = time.time()
        logger.debug(f"Padding took {padding_end - padding_start:.4f} seconds")

        # Type conversion if needed
        if result.dtype != np.uint8:
            conversion_start = time.time()
            logger.warning(
                f"Converting frame from {result.dtype} to uint8 for VideoWriter compatibility"
            )
            if np.issubdtype(result.dtype, np.floating):
                # Scale float values to 0-255 range
                result = (result * 255).clip(0, 255).astype(np.uint8)
            else:
                # For other types, just convert directly
                result = result.astype(np.uint8)
            conversion_end = time.time()
            logger.debug(
                f"Type conversion took {conversion_end - conversion_start:.4f} seconds"
            )

        total_time = time.time() - start_time
        # Only log at INFO level if processing takes more than 5ms
        logger.debug(f"Total frame crop processing took {total_time:.4f} seconds")

        return result
