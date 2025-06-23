"""
Camera motion detection and trajectory smoothing for AutoFlip.

This module handles camera motion decisions and smoothing of crop windows.
"""

import logging
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from pyautoflip.cropping.types import CameraMotionMode

# Create module-level logger
logger = logging.getLogger("autoflip.cropping.camera_motion")


class CameraMotionHandler:
    """
    Handles camera motion detection and trajectory smoothing.

    This class is responsible for determining the optimal camera motion mode
    for a scene and applying appropriate smoothing to crop window trajectories.
    """

    def __init__(self, motion_threshold: float = 0.5, smoothing_window: int = 30):
        """
        Initialize the camera motion handler.

        Args:
            motion_threshold: Threshold for camera motion (0.0-1.0)
            smoothing_window: Window size for trajectory smoothing
        """
        self.motion_threshold = motion_threshold
        self.smoothing_window = smoothing_window

    def select_camera_motion_mode(
        self, key_crop_windows: List[Tuple[Tuple[int, int, int, int], float]]
    ) -> CameraMotionMode:
        """
        Select the best camera motion mode for a scene.

        Args:
            key_crop_windows: List of tuples containing (crop_window, score) for each key frame

        Returns:
            CameraMotionMode enum value
        """
        if len(key_crop_windows) <= 1:
            logger.debug("Only one key frame, defaulting to STATIONARY mode")
            return CameraMotionMode.STATIONARY

        # Extract just the windows (not scores)
        windows = [window for window, _ in key_crop_windows]

        # Detect if this is likely a talking head video
        # (For later use in decision logic)
        # We consider it a talking head if multiple faces are consistently detected
        face_count = 0
        for i, (window, score) in enumerate(key_crop_windows):
            # The windows are positioned to include faces when faces are detected
            # So we can use the positioning as a proxy for face detection
            # Windows near the center of the frame likely contain centered faces
            x, y, w, h = window

            # Check if this window is in the center third of the frame horizontally
            # This is a simple heuristic for detecting talking head videos
            if 1 / 3 <= (x + w / 2) / max(key_crop_windows[0][0][2], 1) <= 2 / 3:
                face_count += 1

        is_talking_head = (
            face_count >= len(key_crop_windows) * 0.7
        )  # 70% of frames have centered windows

        # Calculate movement between key frames
        total_movement = 0
        max_movement = 0
        consistent_direction = True
        last_direction = 0  # 0=no movement, 1=right, -1=left

        for i in range(1, len(windows)):
            prev_x = windows[i - 1][0]
            curr_x = windows[i][0]
            movement = curr_x - prev_x

            # Track total and maximum movement
            total_movement += abs(movement)
            max_movement = max(max_movement, abs(movement))

            # Check if movement direction is consistent
            curr_direction = 0 if movement == 0 else (1 if movement > 0 else -1)
            if (
                i > 1
                and last_direction != 0
                and curr_direction != 0
                and curr_direction != last_direction
            ):
                consistent_direction = False
            if curr_direction != 0:
                last_direction = curr_direction

        # Normalize by number of transitions
        avg_movement = total_movement / (len(windows) - 1) if len(windows) > 1 else 0

        # Adjusted motion threshold for better stability
        # Use a higher threshold for talking head videos
        motion_threshold_multiplier = 25 if is_talking_head else 15

        logger.debug(
            f"Scene analysis: avg_movement={avg_movement:.2f}, max_movement={max_movement}, "
            + f"consistent_direction={consistent_direction}, is_talking_head={is_talking_head}"
        )

        # Decision logic for camera mode
        if max_movement < self.motion_threshold * motion_threshold_multiplier:
            logger.debug(
                f"Selected STATIONARY mode - max movement {max_movement} < threshold "
                + f"{self.motion_threshold * motion_threshold_multiplier}"
            )
            return CameraMotionMode.STATIONARY

        elif consistent_direction and avg_movement < max_movement * 0.7:
            logger.debug(
                f"Selected PANNING mode - consistent direction with avg movement {avg_movement} "
                + f"< {max_movement * 0.7}"
            )
            return CameraMotionMode.PANNING

        else:
            logger.debug(
                f"Selected TRACKING mode - inconsistent direction or high movement"
            )
            return CameraMotionMode.TRACKING

    def interpolate_crop_windows(
        self,
        key_crop_windows: List[Tuple[Tuple[int, int, int, int], float]],
        key_frame_indices: List[int],
        total_frames: int,
        camera_mode: CameraMotionMode,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Interpolate crop windows for all frames.

        Args:
            key_crop_windows: List of tuples containing (crop_window, score) for each key frame
            key_frame_indices: List of key frame indices
            total_frames: Total number of frames
            camera_mode: Camera motion mode to use

        Returns:
            List of crop windows (x, y, width, height) for all frames
        """
        # If only one key frame or empty list, use same crop for all frames
        if len(key_crop_windows) <= 1:
            # Get crop window from first key frame, or use a default center crop
            if key_crop_windows:
                crop_window = key_crop_windows[0][
                    0
                ]  # Extract the crop window, not the score
            else:
                # Default to a center crop (will be adjusted later if needed)
                crop_window = (0, 0, 100, 100)

            logger.warning(
                f"Only one key crop window, using same crop for all {total_frames} frames: {crop_window}"
            )
            return [crop_window] * total_frames

        # Extract x and y coordinates from key crop windows
        key_xs = []
        key_ys = []

        for i, (window, _) in enumerate(key_crop_windows):
            key_xs.append(window[0])
            key_ys.append(window[1])

            logger.debug(
                f"Key frame {key_frame_indices[i]}: x={window[0]}, y={window[1]}"
            )

        # Get crop dimensions from first window
        crop_width = key_crop_windows[0][0][2]
        crop_height = key_crop_windows[0][0][3]

        logger.debug(f"Key crop window dimensions: {crop_width}x{crop_height}")

        # Handle different camera motion modes
        if camera_mode == CameraMotionMode.STATIONARY:
            # For stationary mode, use a fixed position for the entire scene
            # Find the position that best covers all salient regions
            # (simple approach: average all key positions)
            avg_x = int(sum(key_xs) / len(key_xs))
            avg_y = int(sum(key_ys) / len(key_ys))

            fixed_window = (avg_x, avg_y, crop_width, crop_height)

            logger.debug(
                f"STATIONARY mode: using fixed position {fixed_window} for all frames"
            )
            return [fixed_window] * total_frames

        elif camera_mode == CameraMotionMode.PANNING:
            # For panning, create a linear motion from first to last position
            # This creates a smooth constant-velocity pan
            start_x, start_y = key_xs[0], key_ys[0]
            end_x, end_y = key_xs[-1], key_ys[-1]

            # Create linearly spaced values from start to end
            x_values = np.linspace(start_x, end_x, total_frames).astype(int)
            y_values = np.linspace(start_y, end_y, total_frames).astype(int)

            logger.debug(
                f"PANNING mode: interpolating from ({start_x}, {start_y}) to ({end_x}, {end_y}) across {total_frames} frames"
            )

            # Log some interpolation samples
            sample_indices = [
                0,
                total_frames // 4,
                total_frames // 2,
                3 * total_frames // 4,
                total_frames - 1,
            ]
            sample_indices = [i for i in sample_indices if i < total_frames]

            for i in sample_indices:
                logger.debug(
                    f"Frame {i}: interpolated position ({x_values[i]}, {y_values[i]})"
                )

            return [(x, y, crop_width, crop_height) for x, y in zip(x_values, y_values)]

        else:  # TRACKING mode - use interpolation for smooth tracking
            from scipy.interpolate import interp1d

            # Create interpolation functions for x and y
            x_interp = interp1d(
                key_frame_indices,
                key_xs,
                kind="linear",
                bounds_error=False,
                fill_value=(key_xs[0], key_xs[-1]),
            )
            y_interp = interp1d(
                key_frame_indices,
                key_ys,
                kind="linear",
                bounds_error=False,
                fill_value=(key_ys[0], key_ys[-1]),
            )

            # Interpolate for all frames
            frame_indices = np.arange(total_frames)
            interp_xs = x_interp(frame_indices).astype(int)
            interp_ys = y_interp(frame_indices).astype(int)

            logger.debug(
                f"TRACKING mode: interpolating between {len(key_frame_indices)} key positions across {total_frames} frames"
            )

            # Log some interpolation samples
            sample_indices = [
                0,
                total_frames // 4,
                total_frames // 2,
                3 * total_frames // 4,
                total_frames - 1,
            ]
            sample_indices = [i for i in sample_indices if i < total_frames]

            for i in sample_indices:
                logger.debug(
                    f"Frame {i}: interpolated position ({interp_xs[i]}, {interp_ys[i]})"
                )

            # Create crop windows for all frames
            return [
                (x, y, crop_width, crop_height) for x, y in zip(interp_xs, interp_ys)
            ]

    def smooth_trajectory(
        self,
        crop_windows: List[Tuple[int, int, int, int]],
        camera_mode: CameraMotionMode,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Smooth the crop window trajectory.

        Args:
            crop_windows: List of crop windows
            camera_mode: Camera motion mode to use

        Returns:
            List of smoothed crop windows
        """
        if len(crop_windows) <= 1:
            logger.debug("Only one crop window, no smoothing needed")
            return crop_windows

        # Extract x and y coordinates
        xs = np.array([cw[0] for cw in crop_windows])
        ys = np.array([cw[1] for cw in crop_windows])

        # For stationary camera, no need to smooth
        if camera_mode == CameraMotionMode.STATIONARY:
            logger.debug("STATIONARY camera mode, no smoothing applied")
            return crop_windows

        # For panning camera, linear motion is already smooth
        if camera_mode == CameraMotionMode.PANNING:
            logger.debug("PANNING camera mode, no smoothing needed (already linear)")
            return crop_windows

        # Apply different smoothing based on camera mode
        # For tracking, apply stronger smoothing
        sigma = self.smoothing_window / 6.0  # Heuristic for sigma value

        logger.debug(
            f"TRACKING mode: applying Gaussian smoothing with sigma={sigma:.2f} (window={self.smoothing_window})"
        )

        # Calculate statistics before smoothing
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = x_max - x_min
        y_range = y_max - y_min

        logger.debug(
            f"Before smoothing - X range: {x_min} to {x_max} (span {x_range}), Y range: {y_min} to {y_max} (span {y_range})"
        )

        # Apply smoothing
        smoothed_xs = gaussian_filter1d(xs, sigma=sigma)
        smoothed_ys = gaussian_filter1d(ys, sigma=sigma)

        # Calculate statistics after smoothing
        sx_min, sx_max = min(smoothed_xs), max(smoothed_xs)
        sy_min, sy_max = min(smoothed_ys), max(smoothed_ys)
        sx_range = sx_max - sx_min
        sy_range = sy_max - sy_min

        logger.debug(
            f"After smoothing - X range: {sx_min:.1f} to {sx_max:.1f} (span {sx_range:.1f}), Y range: {sy_min:.1f} to {sy_max:.1f} (span {sy_range:.1f})"
        )

        # Log a few sample points before and after smoothing
        num_frames = len(crop_windows)
        sample_indices = [
            0,
            num_frames // 4,
            num_frames // 2,
            3 * num_frames // 4,
            num_frames - 1,
        ]
        sample_indices = [i for i in sample_indices if i < num_frames]

        for i in sample_indices:
            logger.debug(
                f"Frame {i}: original ({xs[i]}, {ys[i]}) → smoothed ({smoothed_xs[i]:.1f}, {smoothed_ys[i]:.1f})"
            )

        # Create new crop windows with smoothed coordinates
        width = crop_windows[0][2]
        height = crop_windows[0][3]

        return [
            (int(x), int(y), width, height) for x, y in zip(smoothed_xs, smoothed_ys)
        ]
