import os
import logging
import tempfile
import subprocess
from typing import Dict, Any, Optional

import cv2
import numpy as np

logger = logging.getLogger("autoflip.utils.video")


class VideoReader:
    """
    Utility for reading video files.

    VideoReader provides functionality for loading video frames,
    extracting metadata, and saving the audio stream for later use.
    """

    def __init__(self, video_path: str):
        """
        Initialize the video reader.

        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path

        # Open the video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"Could not open video: {video_path}", exc_info=True)
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize temp file for audio
        self.audio_temp_path = None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get video metadata.

        Returns:
            Dictionary containing video metadata: width, height, fps, frame_count, aspect_ratio, duration
        """
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "aspect_ratio": self.width / self.height,
            "duration": self.frame_count / self.fps if self.fps > 0 else 0,
        }

    def extract_audio(self) -> Optional[str]:
        """
        Extract audio stream from the video.

        Returns:
            Path to the extracted audio file, or None if extraction fails
        """
        if self.audio_temp_path:
            return self.audio_temp_path

        try:
            # Create a temporary file for the audio
            fd, self.audio_temp_path = tempfile.mkstemp(suffix=".aac")
            os.close(fd)

            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i",
                self.video_path,
                "-vn",  # Skip video
                "-acodec",
                "copy",  # Copy audio codec
                "-y",  # Overwrite output file
                self.audio_temp_path,
            ]

            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            logger.debug(f"Successfully extracted audio to {self.audio_temp_path}")
            return self.audio_temp_path
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning(f"Could not extract audio from video: {str(e)}")
            return None

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

        # Remove temporary audio file
        if self.audio_temp_path and os.path.exists(self.audio_temp_path):
            try:
                os.remove(self.audio_temp_path)
            except OSError:
                pass


class VideoWriter:
    """
    Utility for writing video files.

    This class provides functionality for saving video frames
    and combining them with an audio stream.
    """

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        audio_path: Optional[str] = None,
        codec: str = "mp4v",
        frame_size: Optional[tuple] = None,
    ):
        """
        Initialize the video writer.

        Args:
            output_path: Path to save the output video
            fps: Frames per second for the output video
            audio_path: Path to an audio file to combine with the video
            codec: FourCC codec to use for the video
            frame_size: Optional (width, height) tuple. If None, will be determined
                       from the first frame passed to write_frame
        """
        self.output_path = output_path
        self.fps = fps
        self.audio_path = audio_path
        self.codec = codec
        self.frame_size = frame_size
        self.temp_path = None
        self.writer = None
        self.frame_count = 0
        self.total_expected_frames = None
        self.input_frame_count = None  # Used to verify consistency
        self.input_duration = None  # Used to verify output duration

    def set_input_metadata(self, frame_count: int, duration: float):
        """Set metadata from input video for verification purposes."""
        self.input_frame_count = frame_count
        self.input_duration = duration

    def _init_writer(self, frame: np.ndarray) -> None:
        """Initialize the video writer with the frame size."""
        if self.writer is not None:
            return

        # Get frame dimensions if not provided
        if self.frame_size is None:
            height, width = frame.shape[:2]
            self.frame_size = (width, height)

        # Create temporary file path for video without audio
        if self.audio_path:
            fd, self.temp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            video_path = self.temp_path
        else:
            video_path = self.output_path

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.frame_size)

        if not self.writer.isOpened():
            raise RuntimeError(
                f"Failed to open VideoWriter with codec {self.codec} and fps {self.fps}"
            )

        logger.debug(
            f"Initialized video writer with dimensions {self.frame_size} and fps {self.fps}"
        )

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a single frame to the video.

        Args:
            frame: Video frame to write
        """
        if self.writer is None:
            self._init_writer(frame)

        # Ensure the frame has the correct data type
        if frame.dtype != np.uint8:
            logger.warning(
                f"Converting frame from {frame.dtype} to uint8 for VideoWriter compatibility"
            )
            if np.issubdtype(frame.dtype, np.floating):
                # Scale float values to 0-255 range
                frame = cv2.convertScaleAbs(frame, alpha=255.0)
            else:
                # For other types, use convertScaleAbs for proper conversion
                frame = cv2.convertScaleAbs(frame)

        # Ensure the frame has the correct size
        current_height, current_width = frame.shape[:2]
        target_width, target_height = self.frame_size

        if current_width != target_width or current_height != target_height:
            logger.warning(
                f"Resizing frame from {current_width}x{current_height} to {target_width}x{target_height}"
            )
            frame = cv2.resize(frame, self.frame_size)

        # Ensure the frame has 3 channels in BGR format (OpenCV's default)
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] != 3:  # Not RGB/BGR
            raise ValueError(f"Unsupported number of channels: {frame.shape[2]}")

        # Note: We assume 3-channel images are already in BGR format (OpenCV's default)
        # and don't convert them to avoid potentially double-converting

        self.writer.write(frame)
        self.frame_count += 1

        # Show progress at intervals if total frames is known
        if self.total_expected_frames:
            if self.frame_count % max(1, self.total_expected_frames // 100) == 0:
                logger.debug(
                    f"\rWriting frame {self.frame_count}/{self.total_expected_frames} "
                    f"({self.frame_count / self.total_expected_frames * 100:.1f}%)",
                    end="",
                    flush=True,
                )

    def finalize(self) -> str:
        """
        Finalize the video file and return the path.

        This method should be called after all frames have been written.

        Returns:
            Path to the output video file
        """
        if self.writer is None:
            raise ValueError("No frames have been written")

        # Release writer
        self.writer.release()
        self.writer = None

        # Show final count
        logger.debug(f"\rWrote {self.frame_count} frames (100%)" + " " * 20)

        # Combine with audio if available
        if self.audio_path and os.path.exists(self.audio_path):
            self._combine_audio_video()

        return self.output_path

    def _combine_audio_video(self) -> None:
        """Combine the video with the audio stream."""
        try:
            logger.debug("Combining video with audio...")

            # First, get input video stream info to verify frame rate consistency
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                self.temp_path,
            ]

            probe_result = subprocess.run(
                probe_cmd, check=True, capture_output=True, text=True
            )
            frame_rate_str = probe_result.stdout.strip()
            logger.debug(f"Detected video frame rate: {frame_rate_str}")

            # Use more accurate ffmpeg command for proper audio sync
            cmd = [
                "ffmpeg",
                "-i", self.temp_path,  # Video input
                "-i", self.audio_path,  # Audio input
                "-c:v", "libx264",  # Use libx264 encoder (very widely supported)
                "-crf", "18",      # Quality setting
                # "-preset", "fast", # Speed/quality tradeoff
                "-c:a", "aac",     # AAC audio codec
                "-vsync", "1",     # Ensure video frames are preserved
                "-map", "0:v:0",   # Use video from first input
                "-map", "1:a:0",   # Use audio from second input
                "-shortest",       # Match shortest input
                "-y",              # Overwrite output file
                self.output_path,
            ]

            # Run the command
            result = subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            logger.debug("Audio/video combination complete")

            # Verify output duration matches original
            try:
                # Get durations
                orig_duration_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    self.temp_path,
                ]

                output_duration_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    self.output_path,
                ]

                orig_result = subprocess.run(
                    orig_duration_cmd, check=True, capture_output=True, text=True
                )
                output_result = subprocess.run(
                    output_duration_cmd, check=True, capture_output=True, text=True
                )

                orig_duration = float(orig_result.stdout.strip())
                output_duration = float(output_result.stdout.strip())

                logger.debug(f"Original video duration: {orig_duration:.2f}s")
                logger.debug(f"Output video duration: {output_duration:.2f}s")

                # Check for significant mismatch (more than 1 second or 5%)
                if abs(orig_duration - output_duration) > max(
                    1.0, orig_duration * 0.05
                ):
                    logger.warning(
                        f"Output video duration ({output_duration:.2f}s) differs significantly from original ({orig_duration:.2f}s)"
                    )
                    logger.warning(
                        "This might indicate an issue with audio/video synchronization"
                    )

            except Exception as e:
                logger.warning(f"Could not verify output duration: {e}")

            # Remove temporary video file
            if os.path.exists(self.temp_path):
                os.remove(self.temp_path)

        except (subprocess.SubprocessError, OSError) as e:
            logger.warning(f"Could not combine audio and video: {e}")
            # If combining fails, use the video-only file
            if os.path.exists(self.temp_path):
                os.rename(self.temp_path, self.output_path)

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.release()

        # Remove temporary video file
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except OSError:
                pass
