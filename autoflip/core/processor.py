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
        
        Uses a streaming approach to avoid loading all frames in memory at once.
        
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
        
        total_detection_time = 0
        total_cropping_time = 0
        total_frames_processed = 0
        
        # Process each scene sequentially
        for scene_idx, (start_frame, end_frame) in enumerate(scene_boundaries):
            scene_length = end_frame - start_frame
            logger.info(f"Processing scene {scene_idx+1}/{len(scene_boundaries)} with {scene_length} frames...")
            
            # PASS 1: Sample key frames and detect content
            detection_start_time = time.time()
            
            # Select key frame indices (sparse sampling)
            frame_count = scene_length
            num_samples = min(8, max(3, frame_count // 60 + 3)) if frame_count > 10 else 3
            relative_key_indices = sorted([int(i) for i in np.linspace(0, frame_count - 1, num_samples)])
            # Convert to absolute frame indices
            key_frame_indices = [idx + start_frame for idx in relative_key_indices]
            
            logger.info(f"Selected {len(key_frame_indices)} key frames for content detection")
            
            # Read only the key frames
            key_frames = {}
            face_detections = {}
            object_detections = {}
            
            # Go to the beginning of the scene
            video_reader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Read and process key frames only
            current_frame_idx = start_frame
            
            for key_idx in tqdm(key_frame_indices, desc="Detecting content"):
                # Skip frames if needed to reach the next key frame
                frames_to_skip = key_idx - current_frame_idx
                
                if frames_to_skip > 0:
                    for _ in range(frames_to_skip):
                        ret = video_reader.cap.grab()  # Just grab frames without decoding
                        if not ret:
                            logger.warning(f"Could not skip to frame {key_idx}, video ended prematurely")
                            break
                        current_frame_idx += 1
                
                # Read the key frame
                ret, frame = video_reader.cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame at position {key_idx}")
                    continue
                
                current_frame_idx += 1
                
                # Store the key frame
                key_frames[key_idx - start_frame] = frame
                
                # Downsample frame for detection
                downsampled_frame = self._downsample_frame(frame)
                
                # Detect faces
                try:
                    faces = self.face_detector.detect(downsampled_frame)
                    face_detections[key_idx - start_frame] = faces
                except Exception as e:
                    logger.error(f"Face detection failed for frame {key_idx}: {e}")
                    face_detections[key_idx - start_frame] = []
                
                # Detect objects
                try:
                    objects = self.object_detector.detect(downsampled_frame)
                    object_detections[key_idx - start_frame] = objects
                except Exception as e:
                    logger.error(f"Object detection failed for frame {key_idx}: {e}")
                    object_detections[key_idx - start_frame] = []
            
            detection_time = time.time() - detection_start_time
            total_detection_time += detection_time
            
            # PASS 2: Setup cropping strategy using key frames
            cropping_start_time = time.time()
            
            # Create scene cropper
            cropper = SceneCropper(
                target_aspect_ratio=self.target_aspect_ratio,
                motion_threshold=self.motion_threshold,
                padding_method=self.padding_method,
                debug_mode=self.debug_mode
            )
            
            # Get frame dimensions
            first_key_frame = next(iter(key_frames.values()))
            frame_height, frame_width = first_key_frame.shape[:2]
            frame_dimensions = (frame_height, frame_width)
            
            # Process scene in streaming mode (generate crop windows but don't store frames)
            try:
                crop_windows = cropper.process_scene_streaming(
                    key_frames,
                    face_detections,
                    object_detections,
                    scene_length,
                    frame_dimensions
                )
                
                if not crop_windows:
                    raise ValueError("No crop windows generated")
                    
                # PASS 3: Stream through all frames again and apply crop windows
                
                # Return to the beginning of the scene
                video_reader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # Process frames in small batches to optimize I/O
                batch_size = 30  # Process 30 frames at a time (adjustable)
                for batch_start in range(0, scene_length, batch_size):
                    batch_end = min(batch_start + batch_size, scene_length)
                    
                    for i in range(batch_start, batch_end):
                        # Read frame
                        ret, frame = video_reader.cap.read()
                        if not ret:
                            logger.warning(f"Failed to read frame at position {start_frame + i}")
                            continue
                        
                        # Get crop window for this frame
                        crop_window = crop_windows[i]
                        
                        # Apply crop window
                        cropped_frame = cropper.apply_crop_window(frame, crop_window)
                        
                        # Write to output
                        video_writer.write_frame(cropped_frame)
                        total_frames_processed += 1
                
                # Process debug visualization if needed
                if self.debug_mode:
                    # Convert to scene-relative index
                    first_key_relative_idx = relative_key_indices[0]
                    first_key_abs_idx = key_frame_indices[0]
                    
                    frame_info = f"Scene: {scene_idx+1}/{len(scene_boundaries)}, Frame: {first_key_relative_idx}/{scene_length}"
                    
                    # Use the first key frame for visualization
                    sample_frame = key_frames[first_key_relative_idx]
                    crop_window = crop_windows[first_key_relative_idx]
                    
                    # Create visualization of detections
                    processed_face_dets = face_detections.get(first_key_relative_idx, [])
                    processed_obj_dets = object_detections.get(first_key_relative_idx, [])
                    
                    # Create a debug frame
                    x, y, crop_width, crop_height = crop_window
                    debug_frame = VisualizationUtils.create_debug_frame(
                        frame=sample_frame,
                        x=x,
                        y=y,
                        crop_width=crop_width,
                        crop_height=crop_height,
                        face_dets=processed_face_dets,
                        object_dets=processed_obj_dets,
                        saliency_score=None,  # We don't have this in streaming mode
                        camera_mode=cropper.camera_mode if hasattr(cropper, 'camera_mode') else None
                    )
                    
                    # Get crop window
                    current_crop_window = crop_window
                    
                    # Create trajectory visualization with a few samples
                    trajectory_samples = crop_windows[::max(1, len(crop_windows) // 10)]
                    trajectory_frame = VisualizationUtils.create_trajectory_vis(
                        sample_frame, trajectory_samples, 0)
                    
                    # Debug visualization
                    debug_path = VisualizationUtils.get_standard_debug_path(
                        self.debug_dir, scene_idx, first_key_relative_idx)
                    
                    cropped_sample = cropper.apply_crop_window(sample_frame, crop_window)
                    
                    VisualizationUtils.display_processing_view(
                        original_frame=sample_frame,
                        detection_frame=debug_frame,
                        cropped_frame=cropped_sample,
                        crop_window=current_crop_window,
                        trajectory_frame=trajectory_frame,
                        frame_info=frame_info,
                        save_path=debug_path
                    )
                
            except Exception as e:
                logger.error(f"Scene cropping failed: {e}. Falling back to center crop.")
                
                # Return to the beginning of the scene
                video_reader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # Calculate center crop dimensions
                if not 'frame_width' in locals():
                    # Read a frame to get dimensions if we don't have them
                    ret, frame = video_reader.cap.read()
                    if not ret:
                        logger.error(f"Could not read frame to determine dimensions, skipping scene {scene_idx+1}")
                        continue
                    # Return to beginning of scene again
                    video_reader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    frame_height, frame_width = frame.shape[:2]
                
                # Calculate center crop based on target aspect ratio
                target_width = int(frame_height * self.target_aspect_ratio)
                
                # Calculate center crop dimensions
                if target_width <= frame_width:
                    # Need to crop width
                    x = (frame_width - target_width) // 2
                    y = 0
                    crop_width = target_width
                    crop_height = frame_height
                else:
                    # Need to crop height
                    target_height = int(frame_width / self.target_aspect_ratio)
                    x = 0
                    y = (frame_height - target_height) // 2
                    crop_width = frame_width
                    crop_height = target_height
                
                center_crop = (x, y, crop_width, crop_height)
                logger.info(f"Using center crop: {center_crop}")
                
                # Apply center crop to all frames in the scene
                for _ in range(scene_length):
                    ret, frame = video_reader.cap.read()
                    if not ret:
                        break
                    
                    cropped = frame[y:y+crop_height, x:x+crop_width]
                    video_writer.write_frame(cropped)
                    total_frames_processed += 1
            
            cropping_time = time.time() - cropping_start_time
            total_cropping_time += cropping_time
            
            logger.info(f"Scene {scene_idx+1} processing summary:")
            logger.info(f"    - Detection time: {detection_time:.2f} seconds")
            logger.info(f"    - Cropping time: {cropping_time:.2f} seconds")
            logger.info(f"    - Processed {scene_length} frames")
        
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
    
    def _select_key_frames_indices(self, frame_count: int) -> List[int]:
        """
        Select key frame indices from a scene without loading all frames.
        
        Args:
            frame_count: Number of frames in the scene
            
        Returns:
            List of indices for key frames
        """
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