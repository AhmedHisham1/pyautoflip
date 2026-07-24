import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures
import os

from pyautoflip.detection.shot_boundary import ShotBoundaryDetector
from pyautoflip.detection.face_detector import FaceDetector
from pyautoflip.detection.mediapipe_object_detector import ObjectDetector
from pyautoflip.cropping.scene_cropper import SceneCropper
from pyautoflip.cropping.saliency_cropper import SaliencyCropper
from pyautoflip.utils.video import VideoReader, VideoWriter

logger = logging.getLogger("autoflip")


# ─── Crop Analysis Data Classes ──────────────────────────────────────────────


@dataclass
class CropRegion:
    """A single crop region in relative coordinates (0-1)."""
    x: float
    y: float
    w: float
    h: float


@dataclass
class CropKeyframe:
    """A keyframe with one or more crop regions."""
    frame: int
    time: float  # seconds
    regions: List[CropRegion]


@dataclass
class CropWindow:
    """A dense crop window sample at a specific time."""
    time: float  # seconds
    regions: List[CropRegion]


@dataclass
class SceneInfo:
    """Metadata about a detected scene."""
    start_frame: int
    end_frame: int
    camera_mode: str


@dataclass
class CropAnalysis:
    """Complete crop analysis result returned by analyze_video()."""
    frame_width: int
    frame_height: int
    fps: float
    total_frames: int
    target_aspect_ratio: str  # e.g. "9:16"
    mode: str  # "full" or "stacked"

    keyframes: List[CropKeyframe]
    crop_windows: List[CropWindow]
    crop_windows_fps: float  # sampling rate

    camera_mode: str  # dominant mode across scenes
    is_talking_head: bool
    scenes: List[SceneInfo]

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "CropAnalysis":
        """Deserialize from dict."""
        return cls(
            frame_width=data["frame_width"],
            frame_height=data["frame_height"],
            fps=data["fps"],
            total_frames=data["total_frames"],
            target_aspect_ratio=data["target_aspect_ratio"],
            mode=data["mode"],
            keyframes=[
                CropKeyframe(
                    frame=kf["frame"],
                    time=kf["time"],
                    regions=[CropRegion(**r) for r in kf["regions"]],
                )
                for kf in data["keyframes"]
            ],
            crop_windows=[
                CropWindow(
                    time=cw["time"],
                    regions=[CropRegion(**r) for r in cw["regions"]],
                )
                for cw in data["crop_windows"]
            ],
            crop_windows_fps=data["crop_windows_fps"],
            camera_mode=data["camera_mode"],
            is_talking_head=data["is_talking_head"],
            scenes=[SceneInfo(**s) for s in data["scenes"]],
        )


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
        detection_method: str = "saliency",
    ):
        """
        Initialize the AutoFlip processor.

        Args:
            target_aspect_ratio: Target aspect ratio as "width:height" (e.g., "9:16")
            motion_threshold: Threshold for camera motion (0.0-1.0)
            padding_method: Method for padding ("blur" or "solid_color")
            debug_mode: If True, draw debug visualizations
            detection_method: "saliency" for UNISAL saliency-based pipeline (default),
                            "detection" for face/object detection pipeline
        """
        self.target_aspect_ratio = self._parse_aspect_ratio(target_aspect_ratio)
        self.target_aspect_ratio_str = target_aspect_ratio
        self.motion_threshold = motion_threshold
        self.padding_method = padding_method
        self.debug_mode = debug_mode
        self.detection_method = detection_method

        logger.debug(
            f"Initializing AutoFlipProcessor with target AR: {target_aspect_ratio}, "
            f"motion threshold: {motion_threshold}, method: {detection_method}"
        )
        logger.debug(f"Debug mode: {debug_mode}, Padding method: {padding_method}")

        # Initialize detectors
        self.shot_detector = ShotBoundaryDetector()

        if detection_method == "detection":
            self.face_detector = FaceDetector()
            self.object_detector = ObjectDetector()
        # Saliency cropper is initialized lazily per scene

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
        if self.detection_method == "saliency":
            total_frames_processed = self._process_scenes_saliency(
                scene_boundaries, video_reader, video_writer
            )
        else:
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

        # skip scene detection for short videos (<30 seconds at 30fps)
        duration_seconds = frame_count / 30  # assume 30fps
        if duration_seconds < 30:
            logger.debug(f"Video is short ({duration_seconds:.1f}s), skipping scene detection")
            self.timing_info["shot_detection"] = time.time() - start_time
            return [(0, frame_count)]

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

    def _process_scenes_saliency(
        self,
        scene_boundaries: List[Tuple[int, int]],
        video_reader: VideoReader,
        video_writer: VideoWriter,
    ) -> int:
        """Process scenes using the saliency-based pipeline."""
        total_detection_time = 0
        total_cropping_time = 0
        total_frames_processed = 0

        saliency_cropper = SaliencyCropper(
            target_aspect_ratio=self.target_aspect_ratio,
            motion_threshold=self.motion_threshold,
            padding_method=self.padding_method,
        )

        max_workers = os.cpu_count() or 4

        for scene_idx, (start_frame, end_frame) in tqdm(
            enumerate(scene_boundaries),
            total=len(scene_boundaries),
            desc="Processing scenes",
        ):
            scene_length = end_frame - start_frame

            # Sample key frames for this scene
            detection_start = time.time()
            n_samples = min(scene_length, saliency_cropper.max_frames_per_scene)
            sample_indices = sorted(
                [int(i) for i in np.linspace(0, scene_length - 1, max(2, n_samples))]
            )

            key_frames = []
            for idx in sample_indices:
                video_reader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + idx)
                ret, frame = video_reader.cap.read()
                if ret:
                    key_frames.append(frame)

            if not key_frames:
                continue

            # Process scene to get crop windows
            rel_crop_windows = saliency_cropper.process_scene(key_frames, scene_length)
            detection_time = time.time() - detection_start
            total_detection_time += detection_time

            if not rel_crop_windows:
                continue

            # Apply crops to all frames
            cropping_start = time.time()
            video_reader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            batch_size = 60
            current_frame = 0

            while current_frame < scene_length:
                batch_end = min(current_frame + batch_size, scene_length)
                batch_frames = []

                for i in range(current_frame, batch_end):
                    ret, frame = video_reader.cap.read()
                    if not ret:
                        continue
                    batch_frames.append((frame, rel_crop_windows[i]))

                if saliency_cropper.needs_split_screen():
                    for frame, _ in batch_frames:
                        result = saliency_cropper.apply_split_screen(frame)
                        if result is None:
                            result = saliency_cropper.apply_crop_window(frame, rel_crop_windows[current_frame])
                        video_writer.write_frame(result)
                        total_frames_processed += 1
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        cropped_batch = list(executor.map(
                            lambda x: saliency_cropper.apply_crop_window(x[0], x[1]),
                            batch_frames
                        ))
                    for cropped_frame in cropped_batch:
                        video_writer.write_frame(cropped_frame)
                        total_frames_processed += 1

                current_frame = batch_end

            total_cropping_time += time.time() - cropping_start

        self.timing_info["detection"] = total_detection_time
        self.timing_info["cropping"] = total_cropping_time
        return total_frames_processed

    def _quick_talking_head_check(
        self, video_reader: VideoReader, start_frame: int, scene_length: int
    ) -> bool:
        """
        Quick check if scene is likely a talking head (face-only content).

        Samples 3 frames and checks if faces are consistently present.
        This allows us to skip expensive object detection.

        Args:
            video_reader: VideoReader object
            start_frame: Start frame of the scene
            scene_length: Length of the scene

        Returns:
            True if likely a talking head scene, False otherwise
        """
        # Sample just 3 frames (start, middle, end)
        sample_indices = [0, scene_length // 2, scene_length - 1]
        face_count = 0

        for idx in sample_indices:
            video_reader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + idx)
            ret, frame = video_reader.cap.read()
            if ret:
                # Use small frame for quick detection
                small_frame = cv2.resize(frame, (320, 320))
                try:
                    faces = self.face_detector.detect(small_frame)
                    if len(faces) > 0:
                        face_count += 1
                except Exception as e:
                    logger.debug(f"Face detection failed in quick check: {e}")

        # If 2 out of 3 frames have faces, likely talking head
        is_talking_head = face_count >= 2
        logger.debug(f"Quick talking head check: {face_count}/3 frames with faces -> {is_talking_head}")
        return is_talking_head

    def _process_key_frames(
        self, video_reader: VideoReader, start_frame: int, scene_length: int
    ) -> Dict[str, Any]:
        """Sample and process key frames for content detection."""
        # Select key frame indices (sparse sampling)
        frame_count = scene_length
        # sample less frequently for longer scenes (>30s at 30fps)
        if frame_count > 900:  # >30 seconds at 30fps
            num_samples = min(10, max(3, frame_count // 60))  # 1 every 2 seconds
        else:
            num_samples = min(15, max(3, frame_count // 30))  # 1 every second
        relative_key_indices = sorted(
            [int(i) for i in np.linspace(0, frame_count - 1, num_samples)]
        )
        # Convert to absolute frame indices
        key_frame_indices = [idx + start_frame for idx in relative_key_indices]

        logger.debug(
            f"Selected {len(key_frame_indices)} key frames for content detection"
        )

        # quick check if this is a talking head scene
        is_likely_talking_head = self._quick_talking_head_check(
            video_reader, start_frame, scene_length
        )
        if is_likely_talking_head:
            logger.debug("Detected talking head scene - skipping object detection")

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
            # use 320x320 for faster detection
            resized_frame = cv2.resize(resized_frame, (320, 320))
            # Detect faces
            try:
                faces = self.face_detector.detect(resized_frame)
                face_detections[key_idx - start_frame] = faces
            except Exception as e:
                logger.error(f"Face detection failed for frame {key_idx}: {e}")
                face_detections[key_idx - start_frame] = []

            # detect objects (skip if talking head to save time)
            if not is_likely_talking_head:
                try:
                    objects = self.object_detector.detect(resized_frame)
                    object_detections[key_idx - start_frame] = objects
                except Exception as e:
                    logger.error(f"Object detection failed for frame {key_idx}: {e}")
                    object_detections[key_idx - start_frame] = []
            else:
                # skip object detection for talking heads
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
                # use all available CPU cores
                max_workers = os.cpu_count() or 4
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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

    def analyze_video(
        self,
        input_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        sample_fps: float = 5.0,
    ) -> CropAnalysis:
        """
        Analyze a video and return crop window data without producing output video.

        This is the "dry run" mode — runs detection and crop computation but skips
        video writing. Returns a CropAnalysis with sparse keyframes (for editing)
        and dense sampled windows (for Remotion preview).

        Args:
            input_path: Path to the input video file
            start_time: Optional start time in seconds (analyze a subrange)
            end_time: Optional end time in seconds
            sample_fps: FPS for dense crop window sampling (default 5.0)

        Returns:
            CropAnalysis dataclass with all crop window data
        """
        total_start = time.time()

        # Step 1: Initialize video
        video_reader, metadata = self._initialize_video(input_path)
        fps = metadata["fps"]
        frame_width = metadata["width"]
        frame_height = metadata["height"]
        total_video_frames = metadata["frame_count"]

        # Compute frame range if start/end time provided
        if start_time is not None:
            start_frame = int(start_time * fps)
        else:
            start_frame = 0
        if end_time is not None:
            end_frame = min(int(end_time * fps), total_video_frames)
        else:
            end_frame = total_video_frames

        analysis_frame_count = end_frame - start_frame
        if analysis_frame_count <= 0:
            raise ValueError(f"Invalid frame range: start={start_frame}, end={end_frame}")

        logger.info(
            f"Analyzing {analysis_frame_count} frames "
            f"({start_frame}-{end_frame}, {analysis_frame_count/fps:.1f}s)"
        )

        # Step 2: Detect scene boundaries within the range
        # For short segments (<30s) we treat as single scene, which is typical for clips
        scene_boundaries = [(0, analysis_frame_count)]
        if analysis_frame_count / fps >= 30:
            try:
                # Scan only the analyzed range — a full-file scan of a long
                # source used to dominate analysis time for short segments
                raw_boundaries = self.shot_detector.detect(
                    input_path, start_frame=start_frame, end_frame=end_frame
                )
                # Filter to our range and adjust to relative indices
                filtered = []
                for b in raw_boundaries:
                    rel_b = b - start_frame
                    if 0 < rel_b < analysis_frame_count:
                        filtered.append(rel_b)
                if filtered:
                    scene_boundaries = []
                    last = 0
                    for b in filtered:
                        scene_boundaries.append((last, b))
                        last = b
                    scene_boundaries.append((last, analysis_frame_count))
            except Exception as e:
                logger.warning(f"Scene detection failed during analysis: {e}")

        # Step 3: Process each scene to get crop windows
        all_rel_windows = []  # Will hold relative windows for ALL frames
        scene_infos = []
        dominant_camera_mode = "STATIONARY"
        is_talking_head = False

        if self.detection_method == "saliency":
            all_rel_windows, scene_infos, dominant_camera_mode, is_talking_head, mode = (
                self._analyze_scenes_saliency(
                    video_reader, scene_boundaries, start_frame, analysis_frame_count
                )
            )
        else:
            all_rel_windows, scene_infos, dominant_camera_mode, is_talking_head, mode = (
                self._analyze_scenes_detection(
                    video_reader, scene_boundaries, start_frame, analysis_frame_count
                )
            )

        # Step 5: Extract keyframes at scene boundaries + within long scenes
        keyframes = self._extract_keyframes(
            all_rel_windows, fps, start_frame, analysis_frame_count, scene_infos
        )

        # Step 6: Sample dense windows at the requested fps
        dense_windows = self._sample_dense_windows(all_rel_windows, fps, sample_fps)

        video_reader.cap.release()

        elapsed = time.time() - total_start
        logger.info(f"Analysis completed in {elapsed:.2f}s — {len(keyframes)} keyframes, {len(dense_windows)} samples")

        return CropAnalysis(
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            total_frames=analysis_frame_count,
            target_aspect_ratio=self.target_aspect_ratio_str,
            mode=mode,
            keyframes=keyframes,
            crop_windows=dense_windows,
            crop_windows_fps=sample_fps,
            camera_mode=dominant_camera_mode,
            is_talking_head=is_talking_head,
            scenes=scene_infos,
        )

    def _analyze_scenes_detection(
        self,
        video_reader: VideoReader,
        scene_boundaries: List[Tuple[int, int]],
        global_start_frame: int,
        total_analysis_frames: int,
    ) -> Tuple[List, List[SceneInfo], str, bool, str]:
        """Analyze scenes using detection pipeline, return multi-region crop windows."""
        all_rel_windows = [None] * total_analysis_frames
        scene_infos = []
        camera_modes = []
        any_talking_head = False

        for scene_idx, (scene_start, scene_end) in enumerate(scene_boundaries):
            scene_length = scene_end - scene_start
            abs_start = global_start_frame + scene_start

            # Get key frame data (detections)
            key_frame_data = self._process_key_frames(
                video_reader, abs_start, scene_length
            )
            if not key_frame_data:
                # Fill with center crop
                center_window = (0.25, 0.0, 0.5, 1.0)  # rough center
                for i in range(scene_start, scene_end):
                    all_rel_windows[i] = center_window
                scene_infos.append(SceneInfo(scene_start, scene_end, "STATIONARY"))
                continue

            # Create scene cropper and process
            cropper = SceneCropper(
                target_aspect_ratio=self.target_aspect_ratio,
                motion_threshold=self.motion_threshold,
                padding_method=self.padding_method,
            )
            rel_crop_windows = cropper.process_scene(
                key_frame_data["key_frames"],
                key_frame_data["face_detections"],
                key_frame_data["object_detections"],
                scene_length,
            )

            # Determine camera mode for this scene
            key_frame_indices = sorted(key_frame_data["key_frames"].keys())
            frame_height, frame_width = list(key_frame_data["key_frames"].values())[0].shape[:2]
            target_width, target_height = cropper._calculate_target_dimensions(
                frame_width, frame_height, self.target_aspect_ratio
            )
            key_crop_regions = cropper._compute_key_crop_regions(
                key_frame_indices,
                cropper._process_scene_detections(
                    key_frame_data["face_detections"],
                    key_frame_data["object_detections"],
                    key_frame_indices,
                ),
                frame_width, frame_height, target_width, target_height,
            )
            cam_mode = cropper.camera_motion_handler.select_camera_motion_mode(key_crop_regions)
            camera_modes.append(cam_mode.name)

            is_th = cropper.detection_processor.identify_talking_head(
                key_frame_data["face_detections"], key_frame_indices
            )
            if is_th:
                any_talking_head = True

            # Store windows — wrap each single tuple in a list for multi-region format
            if rel_crop_windows and len(rel_crop_windows) == scene_length:
                for i in range(scene_length):
                    all_rel_windows[scene_start + i] = [rel_crop_windows[i]]
            else:
                logger.warning(
                    f"Scene {scene_idx}: expected {scene_length} windows, got {len(rel_crop_windows) if rel_crop_windows else 0}"
                )
                fallback = [rel_crop_windows[-1] if rel_crop_windows else (0.25, 0.0, 0.5, 1.0)]
                for i in range(scene_start, scene_end):
                    if all_rel_windows[i] is None:
                        all_rel_windows[i] = fallback

            scene_infos.append(SceneInfo(scene_start, scene_end, cam_mode.name))

        # Fill any remaining None entries
        last_valid = [(0.25, 0.0, 0.5, 1.0)]
        for i in range(len(all_rel_windows)):
            if all_rel_windows[i] is not None:
                last_valid = all_rel_windows[i]
            else:
                all_rel_windows[i] = last_valid

        # Dominant camera mode
        if camera_modes:
            from collections import Counter
            dominant = Counter(camera_modes).most_common(1)[0][0]
        else:
            dominant = "STATIONARY"

        return all_rel_windows, scene_infos, dominant, any_talking_head, "full"

    def _analyze_scenes_saliency(
        self,
        video_reader: VideoReader,
        scene_boundaries: List[Tuple[int, int]],
        global_start_frame: int,
        total_analysis_frames: int,
    ) -> Tuple[List, List[SceneInfo], str, bool]:
        """Analyze scenes using saliency pipeline.

        Returns per-frame multi-region windows:
          all_rel_windows[i] = [(x,y,w,h)] for single-region frames
          all_rel_windows[i] = [(x1,y1,w1,h1), (x2,y2,w2,h2)] for split-screen frames
        """
        from pyautoflip.cropping.saliency_cropper import find_split_faces

        all_rel_windows = [None] * total_analysis_frames
        scene_infos = []

        saliency_cropper = SaliencyCropper(
            target_aspect_ratio=self.target_aspect_ratio,
            motion_threshold=self.motion_threshold,
            padding_method=self.padding_method,
        )

        for scene_idx, (scene_start, scene_end) in enumerate(scene_boundaries):
            scene_length = scene_end - scene_start
            abs_start = global_start_frame + scene_start

            # Sample key frames
            n_samples = min(scene_length, saliency_cropper.max_frames_per_scene)
            sample_indices = sorted(
                [int(i) for i in np.linspace(0, scene_length - 1, max(2, n_samples))]
            )

            key_frames = []
            for idx in sample_indices:
                video_reader.cap.set(cv2.CAP_PROP_POS_FRAMES, abs_start + idx)
                ret, frame = video_reader.cap.read()
                if ret:
                    key_frames.append(frame)

            if not key_frames:
                fallback = [(0.25, 0.0, 0.5, 1.0)]
                for i in range(scene_start, scene_end):
                    all_rel_windows[i] = fallback
                scene_infos.append(SceneInfo(scene_start, scene_end, "STATIONARY"))
                continue

            # Get single-region crop windows from saliency
            rel_crop_windows = saliency_cropper.process_scene(key_frames, scene_length)

            # Also detect per-sampled-frame split faces for stacked regions
            small_frames = saliency_cropper._downscale_frames(key_frames)
            sh, sw = small_frames[0].shape[:2]
            all_faces = saliency_cropper._detect_faces(small_frames)
            target_ar_tuple = saliency_cropper._aspect_ratio_to_tuple()

            # Build per-sample split info: sample_idx -> face centers or None
            # Panel geometry mirrors render_split_screen_from_centers: each
            # panel is a face-centered box (both axes) whose width equals the
            # full-target-AR crop width and whose aspect is panel_ratio
            # (target_w : target_h/2, e.g. 9:8) — the shape that fills half of
            # the stacked output exactly, no padding.
            target_ratio = target_ar_tuple[0] / target_ar_tuple[1]           # e.g. 9/16
            panel_ratio = target_ar_tuple[0] / (target_ar_tuple[1] / 2.0)    # e.g. 9/8
            crop_w_px = min(sh * target_ratio, sw)
            crop_h_px = min(crop_w_px / panel_ratio, sh)
            panel_crop_w = crop_w_px / sw
            panel_crop_h = crop_h_px / sh

            per_sample_split = {}
            for si, faces in enumerate(all_faces):
                split_result = find_split_faces(faces, sw, sh, target_ar_tuple)
                if split_result is not None:
                    regions = []
                    for cx_norm, cy_norm in split_result:
                        rx = max(0.0, min(cx_norm - panel_crop_w / 2, 1.0 - panel_crop_w))
                        ry = max(0.0, min(cy_norm - panel_crop_h / 2, 1.0 - panel_crop_h))
                        regions.append((rx, ry, panel_crop_w, panel_crop_h))
                    per_sample_split[si] = regions

            # Map sample indices to frame indices for split info propagation
            # For frames between samples, inherit the nearest sample's split state
            split_for_frame = [None] * scene_length
            if per_sample_split:
                for fi in range(scene_length):
                    # Find nearest sample
                    best_si = 0
                    best_dist = abs(fi - sample_indices[0]) if sample_indices else float('inf')
                    for si_idx, si_frame in enumerate(sample_indices):
                        d = abs(fi - si_frame)
                        if d < best_dist:
                            best_dist = d
                            best_si = si_idx
                    split_for_frame[fi] = per_sample_split.get(best_si)

            # Combine: single-region windows + per-frame split detection
            if rel_crop_windows and len(rel_crop_windows) == scene_length:
                for i in range(scene_length):
                    single = rel_crop_windows[i]
                    split = split_for_frame[i]
                    if split and len(split) == 2:
                        all_rel_windows[scene_start + i] = [split[0], split[1]]
                    else:
                        all_rel_windows[scene_start + i] = [single]
            else:
                fallback = [(rel_crop_windows[-1] if rel_crop_windows else (0.25, 0.0, 0.5, 1.0))]
                for i in range(scene_start, scene_end):
                    if all_rel_windows[i] is None:
                        all_rel_windows[i] = fallback

            scene_infos.append(SceneInfo(scene_start, scene_end, "TRACKING"))

        # Fill gaps
        last_valid = [(0.25, 0.0, 0.5, 1.0)]
        for i in range(len(all_rel_windows)):
            if all_rel_windows[i] is not None:
                last_valid = all_rel_windows[i]
            else:
                all_rel_windows[i] = last_valid

        # Determine dominant mode
        stacked_count = sum(1 for w in all_rel_windows if w and len(w) > 1)
        mode = "stacked" if stacked_count > len(all_rel_windows) / 2 else "full"

        return all_rel_windows, scene_infos, "TRACKING", False, mode

    @staticmethod
    def _tuples_to_regions(region_list):
        """Convert a list of (x,y,w,h) tuples to CropRegion objects."""
        return [CropRegion(x=t[0], y=t[1], w=t[2], h=t[3]) for t in region_list]

    def _extract_keyframes(
        self,
        all_rel_windows: List,
        fps: float,
        start_frame: int,
        total_frames: int,
        scene_infos: Optional[List[SceneInfo]] = None,
    ) -> List[CropKeyframe]:
        """Extract keyframes at scene boundaries only.

        One keyframe per scene start + first/last frame.
        The user can add more manually in the editor.
        """
        if not all_rel_windows:
            return []

        keyframe_indices = set()
        keyframe_indices.add(0)
        keyframe_indices.add(total_frames - 1)

        if scene_infos:
            for scene in scene_infos:
                keyframe_indices.add(scene.start_frame)

        sorted_indices = sorted(idx for idx in keyframe_indices if 0 <= idx < len(all_rel_windows))

        keyframes = []
        for idx in sorted_indices:
            regions = self._tuples_to_regions(all_rel_windows[idx])
            keyframes.append(CropKeyframe(
                frame=idx,
                time=idx / fps,
                regions=regions,
            ))
        return keyframes

    def _sample_dense_windows(
        self,
        all_rel_windows: List,
        source_fps: float,
        target_fps: float,
    ) -> List[CropWindow]:
        """Downsample per-frame multi-region windows to target FPS.

        All times are segment-relative (0-based), matching keyframe times.
        Consumers apply the windows to the trimmed segment, never the source.
        """
        if not all_rel_windows:
            return []

        total_frames = len(all_rel_windows)
        frame_step = max(1, int(source_fps / target_fps))

        windows = []
        for frame_idx in range(0, total_frames, frame_step):
            regions = self._tuples_to_regions(all_rel_windows[frame_idx])
            windows.append(CropWindow(
                time=frame_idx / source_fps,
                regions=regions,
            ))
        return windows

    def apply_precomputed_windows(
        self,
        input_path: str,
        output_path: str,
        crop_windows_data: dict,
    ) -> str:
        """
        Apply pre-computed crop windows to a video without re-running detection.

        Handles both single-region (full) and multi-region (stacked) frames.
        Region count can vary per-frame.
        """
        from pyautoflip.cropping.saliency_cropper import render_split_screen_from_centers

        analysis = CropAnalysis.from_dict(crop_windows_data)

        video_reader, metadata = self._initialize_video(input_path)
        video_writer = self._initialize_writer(output_path, video_reader)

        total_frames = metadata["frame_count"]
        fps = metadata["fps"]

        per_frame_windows = self._interpolate_to_per_frame(
            analysis.crop_windows, analysis.crop_windows_fps, fps, total_frames
        )

        cropper = SceneCropper(
            target_aspect_ratio=self.target_aspect_ratio,
            motion_threshold=self.motion_threshold,
            padding_method=self.padding_method,
        )

        target_ar_tuple = None
        ratio_map = {0.5625: (9, 16), 1.0: (1, 1), 0.75: (3, 4), 1.7778: (16, 9)}
        for r, t in ratio_map.items():
            if abs(self.target_aspect_ratio - r) < 0.01:
                target_ar_tuple = t
                break
        if target_ar_tuple is None:
            target_ar_tuple = (int(self.target_aspect_ratio * 16), 16)

        batch_size = 60
        frames_processed = 0
        max_workers = os.cpu_count() or 4

        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_frames = []

            for i in range(batch_start, batch_end):
                ret, frame = video_reader.cap.read()
                if not ret:
                    break
                batch_frames.append((frame, per_frame_windows[i]))

            if not batch_frames:
                break

            def apply_window(args):
                frame, window = args
                if len(window) >= 2:
                    # Stacked: two regions → split-screen rendering
                    # Convert region tuples to face centers for render_split_screen_from_centers
                    centers = []
                    for (rx, ry, rw, rh) in window:
                        cx = rx + rw / 2
                        cy = ry + rh / 2
                        centers.append((cx, cy))
                    return render_split_screen_from_centers(frame, centers, target_ar_tuple)
                else:
                    # Single region → normal crop + padding
                    return cropper.apply_crop_window(frame, window[0])

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                cropped_batch = list(executor.map(apply_window, batch_frames))

            for cropped_frame in cropped_batch:
                if cropped_frame is not None:
                    video_writer.write_frame(cropped_frame)
                    frames_processed += 1

        output_path = video_writer.finalize()
        logger.info(f"Applied precomputed windows to {frames_processed} frames → {output_path}")
        return output_path

    def _interpolate_to_per_frame(
        self,
        sampled_windows: List[CropWindow],
        sample_fps: float,
        video_fps: float,
        total_frames: int,
    ) -> List[List[Tuple[float, float, float, float]]]:
        """Interpolate sampled crop windows back to per-frame resolution.

        Returns per-frame list of region tuples. Each entry is a list of 1 or 2
        (x,y,w,h) tuples. Handles mixed region counts by duplicating single regions
        when interpolating toward 2-region samples (panels start identical, then diverge).
        """
        if not sampled_windows:
            return [[(0.25, 0.0, 0.5, 1.0)]] * total_frames

        from scipy.interpolate import interp1d

        sample_times = np.array([w.time for w in sampled_windows])
        frame_times = np.array([i / video_fps for i in range(total_frames)])
        frame_times = np.clip(frame_times, sample_times[0], sample_times[-1])

        if len(sample_times) == 1:
            regions = [(r.x, r.y, r.w, r.h) for r in sampled_windows[0].regions]
            return [regions] * total_frames

        # Group consecutive samples by region count for independent interpolation.
        # Between groups of different counts → hard cut (snap to nearest sample).
        sample_region_counts = [len(w.regions) for w in sampled_windows]

        # For each frame, find the two surrounding samples
        nearest_indices = np.searchsorted(sample_times, frame_times, side='right') - 1
        nearest_indices = np.clip(nearest_indices, 0, len(sampled_windows) - 1)
        next_indices = np.clip(nearest_indices + 1, 0, len(sampled_windows) - 1)

        result = []
        for fi in range(total_frames):
            si = int(nearest_indices[fi])
            si_next = int(next_indices[fi])
            w_a = sampled_windows[si]
            w_b = sampled_windows[si_next]

            # Different region counts → hold previous sample's regions until next sample
            if len(w_a.regions) != len(w_b.regions):
                result.append([(float(r.x), float(r.y), float(r.w), float(r.h)) for r in w_a.regions])
                continue

            # Same region count → interpolate each region
            if si == si_next:
                result.append([(float(r.x), float(r.y), float(r.w), float(r.h)) for r in w_a.regions])
                continue

            t_a = sample_times[si]
            t_b = sample_times[si_next]
            t = (frame_times[fi] - t_a) / (t_b - t_a) if t_b > t_a else 0
            frame_regions = []
            for ri in range(len(w_a.regions)):
                ra = w_a.regions[ri]
                rb = w_b.regions[ri]
                frame_regions.append((
                    float(ra.x + (rb.x - ra.x) * t),
                    float(ra.y + (rb.y - ra.y) * t),
                    float(ra.w + (rb.w - ra.w) * t),
                    float(ra.h + (rb.h - ra.h) * t),
                ))
            result.append(frame_regions)

        return result

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
