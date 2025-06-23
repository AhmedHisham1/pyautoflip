import time
import logging
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures
import os

from pyautoflip.detection.shot_boundary import ShotBoundaryDetector
from pyautoflip.detection.face_detector import FaceDetector
from pyautoflip.detection.mediapipe_object_detector import ObjectDetector
from pyautoflip.cropping.scene_cropper import SceneCropper
from pyautoflip.utils.video import VideoReader, VideoWriter

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
            debug_mode: If True, draw debug visualizations
        """
        self.target_aspect_ratio = self._parse_aspect_ratio(target_aspect_ratio)
        self.motion_threshold = motion_threshold
        self.padding_method = padding_method
        self.debug_mode = debug_mode

        logger.debug(
            f"Initializing AutoFlipProcessor with target AR: {target_aspect_ratio}, motion threshold: {motion_threshold}"
        )
        logger.debug(f"Debug mode: {debug_mode}, Padding method: {padding_method}")

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
            width, height = map(int, aspect_ratio_str.split(":"))
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

        # Step 1: Initialize video reader
        video_reader, metadata = self._initialize_video(input_path)

        # Step 2: Initialize video writer
        video_writer = self._initialize_writer(output_path, video_reader)

        # Step 3: Detect scene boundaries
        scene_boundaries = self._detect_scenes(input_path, metadata["frame_count"])

        # Step 4: Process each scene
        total_frames_processed = self._process_scenes(
            scene_boundaries, video_reader, video_writer
        )

        # Step 5: Finalize output video
        output_path = video_writer.finalize()

        # Log summary statistics
        self._log_processing_summary(total_start_time, total_frames_processed)

        logger.debug(f"Completed processing. Output saved to: {output_path}")
        return output_path

    def _initialize_video(self, input_path: str):
        """
        Initialize the video reader and get metadata.

        Args:
            input_path: Path to the input video file

        Returns:
            video_reader: VideoReader object
            metadata: Metadata of the video: width, height, fps, frame_count, aspect_ratio, duration
        """
        logger.debug(f"Reading video: {input_path}")
        start_time = time.time()

        video_reader = VideoReader(input_path)
        metadata = video_reader.get_metadata()

        logger.debug(
            f"Video info: {metadata['width']}x{metadata['height']} @ {metadata['fps']} fps"
        )
        logger.debug(
            f"Total frames: {metadata['frame_count']} ({metadata['duration']:.2f} seconds)"
        )

        self.timing_info["video_setup"] = time.time() - start_time
        logger.debug(
            f"Video setup completed in {self.timing_info['video_setup']:.2f} seconds"
        )

        return video_reader, metadata

    def _detect_scenes(
        self, input_path: str, frame_count: int
    ) -> List[Tuple[int, int]]:
        """
        Detect scene boundaries in the video.

        Args:
            input_path: Path to the input video file
            frame_count: Number of frames in the video

        Returns:
            List of scene boundaries: [(start_frame, end_frame), ...]
        """
        logger.debug("Detecting scene boundaries...")
        start_time = time.time()

        try:
            shot_boundaries = self.shot_detector.detect(input_path)

            self.timing_info["shot_detection"] = time.time() - start_time
            logger.debug(
                f"Shot detection completed in {self.timing_info['shot_detection']:.2f} seconds"
            )
            logger.debug(
                f"Found {len(shot_boundaries)} boundaries at frames {shot_boundaries}"
            )

            # If no boundaries detected, treat the entire video as one scene
            if not shot_boundaries:
                logger.warning(
                    "No scene changes detected. Treating the video as a single scene."
                )
                scene_boundaries = [(0, frame_count)]
            else:
                # Convert boundaries to scene ranges
                scene_boundaries = []
                last_boundary = 0
                for boundary in shot_boundaries:
                    scene_boundaries.append((last_boundary, boundary))
                    last_boundary = boundary
                # Add the last scene
                scene_boundaries.append((last_boundary, frame_count))

        except Exception as e:
            self.timing_info["shot_detection"] = time.time() - start_time
            logger.error(f"Scene detection failed: {e}")
            logger.warning("Falling back to processing the video as a single scene")
            scene_boundaries = [(0, frame_count)]

        logger.debug(f"Processing {len(scene_boundaries)} scenes...")
        return scene_boundaries

    def _initialize_writer(self, output_path: str, video_reader: VideoReader):
        """
        Initialize the video writer.

        Args:
            output_path: Path to save the output video
            video_reader: VideoReader object
        """
        video_writer = VideoWriter(
            output_path, fps=video_reader.fps, audio_path=video_reader.extract_audio()
        )

        # Pass input metadata to the writer to help with verification
        video_writer.set_input_metadata(
            frame_count=video_reader.frame_count,
            duration=video_reader.frame_count / video_reader.fps,
        )

        return video_writer

    def _process_scenes(
        self,
        scene_boundaries: List[Tuple[int, int]],
        video_reader: VideoReader,
        video_writer: VideoWriter,
    ) -> int:
        """Process each scene in the video.

        Args:
            scene_boundaries: List of scene boundaries: [(start_frame, end_frame), ...]
            video_reader: VideoReader object
            video_writer: VideoWriter object

        Returns:
            Total number of frames processed
        """
        total_detection_time = 0
        total_cropping_time = 0
        total_frames_processed = 0

        # Process each scene sequentially
        for scene_idx, (start_frame, end_frame) in tqdm(
            enumerate(scene_boundaries),
            total=len(scene_boundaries),
            desc="Processing scenes",
        ):
            scene_length = end_frame - start_frame
            logger.debug(
                f"Processing scene {scene_idx+1}/{len(scene_boundaries)} with {scene_length} frames..."
            )

            # Get key frames and detections
            detection_start_time = time.time()
            key_frame_data = self._process_key_frames(
                video_reader, start_frame, scene_length
            )

            if not key_frame_data:
                logger.error("No key frames available for scene processing")
                continue

            detection_time = time.time() - detection_start_time
            total_detection_time += detection_time

            # Process the scene with crop windows
            cropping_start_time = time.time()
            frames_processed = self._apply_cropping(
                video_reader, video_writer, start_frame, scene_length, key_frame_data
            )

            total_frames_processed += frames_processed
            cropping_time = time.time() - cropping_start_time
            total_cropping_time += cropping_time

            logger.debug(f"Scene {scene_idx+1} processing summary:")
            logger.debug(f"    - Detection time: {detection_time:.2f} seconds")
            logger.debug(f"    - Cropping time: {cropping_time:.2f} seconds")
            logger.debug(f"    - Processed {scene_length} frames")

        self.timing_info["detection"] = total_detection_time
        self.timing_info["cropping"] = total_cropping_time

        return total_frames_processed

    def _process_key_frames(
        self, video_reader: VideoReader, start_frame: int, scene_length: int
    ) -> Dict[str, Any]:
        """Sample and process key frames for content detection."""
        # Select key frame indices (sparse sampling)
        frame_count = scene_length
        num_samples = min(15, max(3, frame_count // 30)) # approx 30fps: 1 sample per second; max 15 samples
        relative_key_indices = sorted(
            [int(i) for i in np.linspace(0, frame_count - 1, num_samples)]
        )
        # Convert to absolute frame indices
        key_frame_indices = [idx + start_frame for idx in relative_key_indices]

        logger.debug(
            f"Selected {len(key_frame_indices)} key frames for content detection"
        )

        # Read only the key frames
        key_frames = {}
        face_detections = {}
        object_detections = {}

        # Read and process key frames only
        for key_idx in key_frame_indices:
            # Skip frames if needed to reach the next key frame
            video_reader.cap.set(cv2.CAP_PROP_POS_FRAMES, key_idx)

            # Read the key frame
            ret, frame = video_reader.cap.read()
            if not ret:
                logger.warning(f"Failed to read frame at position {key_idx}")
                continue

            # Store frame
            key_frames[key_idx - start_frame] = frame

            resized_frame = frame.copy()
            resized_frame = cv2.resize(resized_frame, (640, 640))
            # Detect faces
            try:
                faces = self.face_detector.detect(resized_frame)
                face_detections[key_idx - start_frame] = faces
            except Exception as e:
                logger.error(f"Face detection failed for frame {key_idx}: {e}")
                face_detections[key_idx - start_frame] = []

            # Detect objects
            try:
                objects = self.object_detector.detect(resized_frame)
                object_detections[key_idx - start_frame] = objects
            except Exception as e:
                logger.error(f"Object detection failed for frame {key_idx}: {e}")
                object_detections[key_idx - start_frame] = []

        if not key_frames:
            return None

        return {
            "key_frames": key_frames,
            "face_detections": face_detections,
            "object_detections": object_detections,
        }

    def _apply_cropping(
        self,
        video_reader: VideoReader,
        video_writer: VideoWriter,
        start_frame: int,
        scene_length: int,
        key_frame_data: Dict[str, Any],
    ) -> int:
        """Apply cropping to the scene using the key frame detections.

        Args:
            video_reader: VideoReader object
            video_writer: VideoWriter object
            start_frame: Start frame of the scene
            scene_length: Length of the scene
            key_frame_data: Key frame data

        Returns:
            Total number of frames processed
        """
        frames_processed = 0

        # Create scene cropper
        cropper = SceneCropper(
            target_aspect_ratio=self.target_aspect_ratio,
            motion_threshold=self.motion_threshold,
            padding_method=self.padding_method,
            debug_mode=self.debug_mode,
        )

        try:
            # Process scene to get crop windows
            rel_crop_windows = cropper.process_scene(
                key_frame_data["key_frames"],
                key_frame_data["face_detections"],
                key_frame_data["object_detections"],
                scene_length,
            )

            if not rel_crop_windows:
                raise ValueError("No crop windows generated")

            # Return to the beginning of the scene
            video_reader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Process frames in batches to avoid memory issues
            batch_size = 60
            current_frame = 0
            
            while current_frame < scene_length:
                # Calculate batch range
                batch_end = min(current_frame + batch_size, scene_length)
                batch_frames = []
                
                # Read batch of frames
                for i in range(current_frame, batch_end):
                    ret, frame = video_reader.cap.read()
                    if not ret:
                        logger.warning(
                            f"Failed to read frame at position {start_frame + i}"
                        )
                        continue
                    
                    rel_crop_window = rel_crop_windows[i]
                    batch_frames.append((frame, rel_crop_window))
                
                # Process batch in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, os.cpu_count()-2 or 4)) as executor:
                    cropped_batch = list(executor.map(
                        lambda x: cropper.apply_crop_window(x[0], x[1]),
                        batch_frames
                    ))
                
                # Write batch to output
                for cropped_frame in cropped_batch:
                    video_writer.write_frame(cropped_frame)
                    frames_processed += 1
                
                # Move to next batch
                current_frame = batch_end

        except Exception as e:
            logger.error(f"Scene cropping failed: {e}.")
            raise e

        return frames_processed

    def _log_processing_summary(self, total_start_time, total_frames_processed):
        """Log the processing summary statistics."""
        total_time = time.time() - total_start_time
        self.timing_info["total"] = total_time

        print("\n===== Processing Summary =====")
        print(f"Total processing time: {self.timing_info['total']:.2f} seconds")
        print(f"Frames processed: {total_frames_processed}")
        print(
            f"Frames per second: {total_frames_processed / self.timing_info['total']:.2f}"
        )
        print(
            f"Video setup: {self.timing_info.get('video_setup', 0):.2f} seconds ({self.timing_info.get('video_setup', 0) / self.timing_info['total'] * 100:.1f}%)"
        )
        print(
            f"Shot detection: {self.timing_info['shot_detection']:.2f} seconds ({self.timing_info['shot_detection'] / self.timing_info['total'] * 100:.1f}%)"
        )
        print(
            f"Content detection: {self.timing_info['detection']:.2f} seconds ({self.timing_info['detection'] / self.timing_info['total'] * 100:.1f}%)"
        )
        print(
            f"Cropping: {self.timing_info['cropping']:.2f} seconds ({self.timing_info['cropping'] / self.timing_info['total'] * 100:.1f}%)"
        )
        print("===========================")
