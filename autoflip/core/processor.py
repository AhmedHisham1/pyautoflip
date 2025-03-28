"""
Main processor for AutoFlip video reframing.
"""

import os
import time
import logging
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from autoflip.detection.shot_boundary import ShotBoundaryDetector
from autoflip.detection.face_detector import FaceDetector
from autoflip.detection.object_detector import ObjectDetector
from autoflip.cropping.scene_cropper import SceneCropper
from autoflip.cropping.visualization import VisualizationUtils
from autoflip.utils.video import VideoReader, VideoWriter

logger = logging.getLogger("autoflip")


class AutoFlipProcessor:
    """
    Main processor for AutoFlip video reframing.
    
    This class orchestrates the entire process of reframing a video:
    1. Breaking the video into shots/scenes
    2. Detecting important content in each frame
    3. Determining optimal crop windows
    4. Generating the reframed video
    
    Attributes:
        target_aspect_ratio (str): Target aspect ratio in "width:height" format
        motion_threshold (float): Threshold for camera motion (0.0-1.0)
        padding_method (str): Method for padding ("blur" or "solid_color")
    """
    
    def __init__(
        self,
        target_aspect_ratio: str = "9:16",
        motion_threshold: float = 0.5,
        padding_method: str = "blur",
        debug_mode: bool = False,
    ):
        """
        Initialize the AutoFlip processor.
        
        Args:
            target_aspect_ratio: Target aspect ratio as "width:height" (e.g., "9:16")
            motion_threshold: Threshold for camera motion (0.0-1.0)
            padding_method: Method for padding ("blur" or "solid_color")
            debug_mode: If True, draw rectangles to show crop regions instead of actually cropping
        """
        self.target_aspect_ratio = self._parse_aspect_ratio(target_aspect_ratio)
        self.motion_threshold = motion_threshold
        self.padding_method = padding_method
        self.debug_mode = debug_mode
        
        logger.info(f"Initializing AutoFlipProcessor with target AR: {target_aspect_ratio}, motion threshold: {motion_threshold}")
        logger.info(f"Debug mode: {debug_mode}, Padding method: {padding_method}")
        
        # Initialize detectors
        self.shot_detector = ShotBoundaryDetector()
        self.face_detector = FaceDetector()
        self.object_detector = ObjectDetector()
        
        # Directory for debug output
        self.debug_dir = "debug_frames"
        
        # Timing information
        self.timing_info = {}
        
    def _parse_aspect_ratio(self, aspect_ratio_str: str) -> float:
        """
        Parse aspect ratio string into a float.
        
        Args:
            aspect_ratio_str: Aspect ratio as "width:height" (e.g., "9:16")
            
        Returns:
            float: Aspect ratio as width/height
        """
        try:
            width, height = map(int, aspect_ratio_str.split(':'))
            ratio = width / height
            logger.debug(f"Parsed aspect ratio {aspect_ratio_str} to {ratio:.4f}")
            return ratio
        except (ValueError, ZeroDivisionError):
            error_msg = f"Invalid aspect ratio: {aspect_ratio_str}. Format should be 'width:height' (e.g., '9:16')."
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
    ) -> str:
        """
        Process a video file and generate a reframed version.
        
        Args:
            input_path: Path to the input video file
            output_path: Path to save the output video
        Returns:
            str: Path to the output video file
        """
        # Start total timing
        total_start_time = time.time()
        
        # Setup debug mode if enabled
        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)
            logger.info(f"Debug frames will be saved to {self.debug_dir}")
            logger.info("Debug mode enabled with simplified visualization")
            
            # Initialize visualization system
            VisualizationUtils.set_debug_directory(self.debug_dir)
        
        # Step 1: Initialize video reader
        logger.info(f"Reading video: {input_path}")
        start_time = time.time()
        video_reader = VideoReader(input_path)
        
        # Get video metadata
        metadata = video_reader.get_metadata()
        logger.info(f"Video info: {metadata['width']}x{metadata['height']} @ {metadata['fps']} fps")
        logger.info(f"Total frames: {metadata['frame_count']} ({metadata['duration']:.2f} seconds)")
            
        self.timing_info["video_setup"] = time.time() - start_time
        logger.info(f"Video setup completed in {self.timing_info['video_setup']:.2f} seconds")
        
        # Step 2: Detect shot boundaries (using direct file access which is more efficient)
        logger.info("Detecting scene boundaries...")
        start_time = time.time()
        
        try:
            shot_boundaries = self.shot_detector.detect(input_path)
            
            self.timing_info["shot_detection"] = time.time() - start_time
            logger.info(f"Shot detection completed in {self.timing_info['shot_detection']:.2f} seconds")
            logger.info(f"Found {len(shot_boundaries)} boundaries at frames {shot_boundaries}")
            
            # If no boundaries detected, treat the entire video as one scene
            if not shot_boundaries:
                logger.info("No scene changes detected. Treating the video as a single scene.")
                scene_boundaries = [(0, metadata['frame_count'])]
            else:
                # Convert boundaries to scene ranges
                scene_boundaries = []
                last_boundary = 0
                for boundary in shot_boundaries:
                    scene_boundaries.append((last_boundary, boundary))
                    last_boundary = boundary
                # Add the last scene
                scene_boundaries.append((last_boundary, metadata['frame_count']))
                
        except Exception as e:
            self.timing_info["shot_detection"] = time.time() - start_time
            logger.error(f"Scene detection failed: {e}")
            logger.info("Falling back to processing the video as a single scene")
            scene_boundaries = [(0, metadata['frame_count'])]
        
        logger.info(f"Processing {len(scene_boundaries)} scenes...")
        
        # Initialize video writer with EXACT same fps as input
        video_writer = VideoWriter(
            output_path,
            fps=video_reader.fps,
            audio_path=video_reader.extract_audio()
        )
        
        # Pass input metadata to the writer to help with verification
        video_writer.set_input_metadata(
            frame_count=video_reader.frame_count,
            duration=video_reader.frame_count / video_reader.fps
        )
        
        # Make sure we're starting at the beginning of the video
        video_reader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        total_detection_time = 0
        total_cropping_time = 0
        total_frames_processed = 0
        frame_count = 0
        
        # Process each scene sequentially
        for scene_idx, (start_frame, end_frame) in enumerate(scene_boundaries):
            scene_length = end_frame - start_frame
            logger.info(f"Processing scene {scene_idx+1}/{len(scene_boundaries)} with {scene_length} frames...")
            
            # Skip to the correct starting frame position
            if frame_count < start_frame:
                frames_to_skip = start_frame - frame_count
                logger.info(f"Skipping {frames_to_skip} frames to reach scene start")
                
                for _ in range(frames_to_skip):
                    ret = video_reader.cap.grab()  # Just grab the frame without decoding it (faster)
                    if not ret:
                        logger.warning(f"Could not skip to frame {start_frame}, video ended prematurely")
                        break
                    frame_count += 1
            
            # Read all frames in this scene
            scene_frames = []
            scene_frame_count = 0
            
            while frame_count < end_frame:
                ret, frame = video_reader.cap.read()
                if not ret:
                    logger.warning(f"Reached end of video at frame {frame_count}, expected end at {end_frame}")
                    break
                
                scene_frames.append(frame)
                frame_count += 1
                scene_frame_count += 1
            
            if not scene_frames:
                logger.warning(f"No frames read for scene {scene_idx+1}, skipping")
                continue
            
            # Step 3.1: Detect faces and objects in key frames
            detection_start_time = time.time()
            key_frame_indices = self._select_key_frames(scene_frames)
            
            face_detections = {}
            object_detections = {}
            
            # Using tqdm to show progress of detection
            for idx in tqdm(key_frame_indices, desc="Detecting content"):
                frame = self._downsample_frame(scene_frames[idx])
                try:
                    face_detections[idx] = self.face_detector.detect(frame)
                except Exception as e:
                    logger.error(f"Face detection failed for frame {idx}: {e}")
                    face_detections[idx] = []
                    
                try:
                    object_detections[idx] = self.object_detector.detect(frame)
                except Exception as e:
                    logger.error(f"Object detection failed for frame {idx}: {e}")
                    object_detections[idx] = []
            
            detection_time = time.time() - detection_start_time
            total_detection_time += detection_time
            
            # Step 3.2: Create scene cropper
            cropping_start_time = time.time()
            cropper = SceneCropper(
                target_aspect_ratio=self.target_aspect_ratio,
                motion_threshold=self.motion_threshold,
                padding_method=self.padding_method,
                debug_mode=self.debug_mode
            )
            
            # Process scene to establish cropping strategy
            try:
                cropped_frames, debug_vis_frames = cropper.process_scene(
                    scene_frames,
                    face_detections,
                    object_detections,
                    key_frame_indices
                )
                
                # Write frames to output
                for frame in cropped_frames:
                    video_writer.write_frame(frame)
                    total_frames_processed += 1
                
                # Process debug visualization if needed
                if self.debug_mode:
                    first_key_idx = key_frame_indices[0]
                    frame_info = f"Scene: {scene_idx+1}/{len(scene_boundaries)}, Frame: {first_key_idx}/{len(scene_frames)}"
                    
                    # Get detection visualization frame
                    detection_frame = debug_vis_frames.get(first_key_idx)
                    
                    # Get trajectory visualization frame
                    trajectory_frame = debug_vis_frames.get(-1)  # Special key for trajectory
                    
                    # Get crop window
                    current_crop_window = cropper.current_crop_window if hasattr(cropper, 'current_crop_window') else None
                    
                    # Debug visualization
                    debug_path = VisualizationUtils.get_standard_debug_path(
                        self.debug_dir, scene_idx, first_key_idx)
                    
                    VisualizationUtils.display_processing_view(
                        original_frame=scene_frames[first_key_idx],
                        detection_frame=detection_frame,
                        cropped_frame=cropped_frames[first_key_idx] if first_key_idx < len(cropped_frames) else None,
                        crop_window=current_crop_window,
                        trajectory_frame=trajectory_frame,
                        frame_info=frame_info,
                        save_path=debug_path
                    )
                
            except Exception as e:
                logger.error(f"Scene cropping failed: {e}. Falling back to center crop.")
                
                # Fall back to center crop
                height, width = scene_frames[0].shape[:2]
                target_width = int(height * self.target_aspect_ratio)
                
                # Calculate center crop dimensions
                if target_width <= width:
                    # Need to crop width
                    x = (width - target_width) // 2
                    y = 0
                    crop_width = target_width
                    crop_height = height
                else:
                    # Need to crop height
                    target_height = int(width / self.target_aspect_ratio)
                    x = 0
                    y = (height - target_height) // 2
                    crop_width = width
                    crop_height = target_height
                
                logger.info(f"Using center crop: ({x}, {y}, {crop_width}, {crop_height})")
                
                # Apply center crop to all frames in the scene
                for frame in scene_frames:
                    cropped = frame[y:y+crop_height, x:x+crop_width]
                    video_writer.write_frame(cropped)
                    total_frames_processed += 1
            
            cropping_time = time.time() - cropping_start_time
            total_cropping_time += cropping_time
            
            logger.info(f"Scene {scene_idx+1} processing summary:")
            logger.info(f"    - Detection time: {detection_time:.2f} seconds")
            logger.info(f"    - Cropping time: {cropping_time:.2f} seconds")
            logger.info(f"    - Processed {scene_frame_count} frames")
        
        self.timing_info["detection"] = total_detection_time
        self.timing_info["cropping"] = total_cropping_time
        
        # Finalize output video
        output_path = video_writer.finalize()
        
        # Calculate and print total time
        total_time = time.time() - total_start_time
        self.timing_info["total"] = total_time
        logger.info("\n===== Processing Summary =====")
        logger.info(f"Total processing time: {self.timing_info['total']:.2f} seconds")
        logger.info(f"Frames processed: {total_frames_processed}")
        logger.info(f"Frames per second: {total_frames_processed / self.timing_info['total']:.2f}")
        logger.info(f"Video setup: {self.timing_info.get('video_setup', 0):.2f} seconds ({self.timing_info.get('video_setup', 0) / self.timing_info['total'] * 100:.1f}%)")
        logger.info(f"Shot detection: {self.timing_info['shot_detection']:.2f} seconds ({self.timing_info['shot_detection'] / self.timing_info['total'] * 100:.1f}%)")
        logger.info(f"Content detection: {self.timing_info['detection']:.2f} seconds ({self.timing_info['detection'] / self.timing_info['total'] * 100:.1f}%)")
        logger.info(f"Cropping: {self.timing_info['cropping']:.2f} seconds ({self.timing_info['cropping'] / self.timing_info['total'] * 100:.1f}%)")
        logger.info("===========================")
        
        # Clean up visualization if debug mode was enabled
        if self.debug_mode:
            # Show processing finished message
            logger.info("Processing finished. Close visualization window to continue...")
            
            # Create a simple summary image
            summary_img = self._create_summary_visualization(total_frames_processed, total_time)
            if summary_img is not None:
                # Create a final frame info string for the summary
                frame_info = f"Total time: {total_time:.2f}s, Frames: {total_frames_processed}"
                
                # Display the summary
                VisualizationUtils.display_processing_view(
                    original_frame=summary_img,
                    detection_frame=None,
                    cropped_frame=None,
                    crop_window=None,
                    trajectory_frame=None,
                    frame_info=frame_info,
                    save_path=os.path.join(self.debug_dir, "processing_summary.jpg")
                )
                
                # Keep window open until user closes it
                cv2.waitKey(0)
            
            # Close debug window
            VisualizationUtils.close_debug_window()
        
        logger.info(f"Completed processing. Output saved to: {output_path}")
        return output_path
        
    def _downsample_frame(self, frame: np.ndarray) -> np.ndarray:
        """Downsample a frame for more efficient processing."""
        target_width = 480
        h, w = frame.shape[:2]
        scale = target_width / w
        new_size = (target_width, int(h * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    
    def _select_key_frames(self, scene_frames: List[np.ndarray]) -> List[int]:
        """
        Select key frames from a scene for content detection.
        
        Args:
            scene_frames: List of frames in the scene
            
        Returns:
            List of indices for key frames
        """
        frame_count = len(scene_frames)
        
        if frame_count < 3:
            return list(range(frame_count))
            
        # Determine number of samples based on scene length
        num_samples = min(8, max(3, frame_count // 60 + 3)) if frame_count > 10 else 3
        
        # Generate evenly spaced frame indices
        return sorted([int(i) for i in np.linspace(0, frame_count - 1, num_samples)])

    def _create_summary_visualization(self, total_frames: int, total_time: float) -> Optional[np.ndarray]:
        """Create a summary visualization with timing information."""
        try:
            # Create a blank image
            img = np.ones((600, 800, 3), dtype=np.uint8) * 255
            
            # Add title
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "AutoFlip Processing Summary", (50, 50), font, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Add timing information
            y_pos = 100
            line_height = 30
            
            # Add fps and total frames
            cv2.putText(img, f"Total frames processed: {total_frames}", (50, y_pos), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            y_pos += line_height
            cv2.putText(img, f"Frames per second: {total_frames / total_time:.2f}", (50, y_pos), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            y_pos += line_height * 2
            
            # Add detailed timing
            cv2.putText(img, f"Total processing time: {total_time:.2f} seconds", (50, y_pos), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            y_pos += line_height
            
            # Add breakdown of time spent
            for key, value in self.timing_info.items():
                if key != "total":
                    percentage = value / total_time * 100
                    cv2.putText(img, f"{key.capitalize()}: {value:.2f}s ({percentage:.1f}%)", 
                               (70, y_pos), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                    y_pos += line_height
            
            # Add bar chart
            y_pos += line_height
            cv2.putText(img, "Time Distribution:", (50, y_pos), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            y_pos += line_height
            
            # Draw bar chart
            chart_width = 600
            chart_height = 30
            chart_left = 100
            
            for key, value in self.timing_info.items():
                if key != "total":
                    percentage = value / total_time
                    bar_width = int(chart_width * percentage)
                    
                    # Pick color based on key
                    if key == "video_setup":
                        color = (255, 150, 150)  # Light red
                    elif key == "shot_detection":
                        color = (150, 255, 150)  # Light green
                    elif key == "detection":
                        color = (150, 150, 255)  # Light blue
                    elif key == "cropping":
                        color = (255, 255, 150)  # Light yellow
                    else:
                        color = (200, 200, 200)  # Light gray
                    
                    # Draw the bar
                    cv2.rectangle(img, (chart_left, y_pos), (chart_left + bar_width, y_pos + chart_height), color, -1)
                    cv2.rectangle(img, (chart_left, y_pos), (chart_left + bar_width, y_pos + chart_height), (0, 0, 0), 1)
                    
                    # Add label
                    cv2.putText(img, f"{key}", (chart_left - 90, y_pos + 20), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, f"{percentage*100:.1f}%", (chart_left + bar_width + 10, y_pos + 20), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    y_pos += chart_height + 10
            
            return img
        except Exception as e:
            logger.error(f"Error creating summary visualization: {e}")
            return None 