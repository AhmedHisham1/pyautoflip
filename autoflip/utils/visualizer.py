"""
Visualization utilities for autoflip.
"""

import cv2
import numpy as np
import os
import logging
import time

# Create logger
logger = logging.getLogger(__name__)


class Visualizer:
    """
    Visualization utilities for autoflip.

    This class provides a simple generic visualization interface
    for reporting debug frames from different components.
    """

    # Class variables for visualization control
    _debug_window_name = "AutoFlip Debug"
    _debug_window_created = False
    _debug_dir = None

    # Visualization control state
    _paused = False
    _step_mode = True  # start in step mode
    _step_ready = False
    _last_key = -1

    # Visualization layout
    _frames = {}  # Dictionary to store the latest frame from each component
    _layout = None  # Current layout (rows, cols)
    _viz_frame = None  # Combined visualization frame
    _latest_frame_title = None  # Title of the most recently reported frame

    # Window size configuration
    _initial_window_width = 1600
    _initial_window_height = 900

    @classmethod
    def report_frame(cls, frame: np.ndarray, title: str) -> None:
        """
        Report a debug frame from a component.

        This method will update the component's latest frame and refresh the
        visualization layout if needed.

        Args:
            frame: Debug frame to display
            title: Title/identifier for this frame
        """
        if frame is None:
            logger.warning(f"Received None frame for '{title}'")
            return

        # Store the frame in the dictionary
        cls._frames[title] = frame.copy()

        # Update the latest frame title
        cls._latest_frame_title = title

        # Update the visualization
        cls._update_visualization()

        # If in step mode or paused, wait for user action
        cls.wait_for_step()

    @classmethod
    def _update_visualization(cls) -> None:
        """
        Update the visualization layout and display the debug frames.
        """
        if not cls._frames:
            return

        # Determine layout
        num_frames = len(cls._frames)
        if num_frames <= 1:
            rows, cols = 1, 1
        elif num_frames <= 2:
            rows, cols = 1, 2
        elif num_frames <= 4:
            rows, cols = 2, 2
        elif num_frames <= 6:
            rows, cols = 2, 3
        elif num_frames <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 3, 4

        # Create a visualization grid
        # First determine the size of each cell
        max_height = 0
        max_width = 0

        for frame in cls._frames.values():
            if frame is not None:
                h, w = frame.shape[:2]
                max_height = max(max_height, h)
                max_width = max(max_width, w)

        # Ensure we have valid dimensions
        if max_height == 0 or max_width == 0:
            return

        # Limit cell size for very large frames, but use larger dimensions than before
        cell_max_height = 480  # Increased from 360
        cell_max_width = 800  # Increased from 640

        # Calculate aspect ratio
        aspect_ratio = max_width / max_height

        # Adjust cell dimensions to maintain aspect ratio but limit size
        if max_height > cell_max_height:
            cell_height = cell_max_height
            cell_width = int(cell_height * aspect_ratio)
        elif max_width > cell_max_width:
            cell_width = cell_max_width
            cell_height = int(cell_width / aspect_ratio)
        else:
            cell_height = max_height
            cell_width = max_width

        # Add padding
        padding = 15  # Increased padding between cells

        # Create canvas
        canvas_height = rows * cell_height + (rows + 1) * padding
        canvas_width = cols * cell_width + (cols + 1) * padding
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Track position of the latest frame
        latest_frame_pos = None

        # Place frames in the grid
        frame_index = 0
        titles = list(cls._frames.keys())

        for r in range(rows):
            for c in range(cols):
                if frame_index >= len(titles):
                    break

                title = titles[frame_index]
                frame = cls._frames[title]

                # Skip if frame is None
                if frame is None:
                    frame_index += 1
                    continue

                # Calculate position
                y = r * (cell_height + padding) + padding
                x = c * (cell_width + padding) + padding

                # Resize frame to fit cell
                resized_frame = cv2.resize(frame, (cell_width, cell_height))

                # Place frame in canvas
                canvas[y : y + cell_height, x : x + cell_width] = resized_frame

                # Add title
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(canvas, title, (x + 10, y + 30), font, 0.8, (0, 255, 0), 2)

                # Check if this is the latest reported frame
                if title == cls._latest_frame_title:
                    latest_frame_pos = (x, y, cell_width, cell_height)

                frame_index += 1

        # Draw a yellow highlight around the latest frame if applicable
        if latest_frame_pos is not None:
            x, y, w, h = latest_frame_pos
            # Draw a thick yellow rectangle around the latest frame
            highlight_color = (0, 255, 255)  # Yellow
            highlight_thickness = 5
            cv2.rectangle(
                canvas,
                (x - 2, y - 2),
                (x + w + 2, y + h + 2),
                highlight_color,
                highlight_thickness,
            )

            # Add a "LATEST" indicator
            cv2.putText(
                canvas, "LATEST", (x + w - 80, y + 20), font, 0.7, highlight_color, 2
            )

        # Add status info at the bottom
        status_height = 50  # Increased height for status bar
        status_canvas = np.zeros((status_height, canvas_width, 3), dtype=np.uint8)
        status_y = 30

        # Display status
        if cls._paused:
            status = "PAUSED (Press 'p' to resume)"
            status_color = (0, 0, 255)  # Red
        elif cls._step_mode:
            status = "STEP MODE (Press 's' for next step, 'p' to exit step mode)"
            status_color = (0, 255, 255)  # Yellow
        else:
            status = "RUNNING (Press 'p' to pause, 's' for step mode, 'ESC' to exit)"
            status_color = (0, 255, 0)  # Green

        cv2.putText(
            status_canvas,
            status,
            (10, status_y),
            font,
            0.7,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # Add frame count and latest frame info
        frame_count_str = f"Frames: {len(cls._frames)}"
        latest_str = (
            f"Latest: {cls._latest_frame_title}" if cls._latest_frame_title else ""
        )
        cv2.putText(
            status_canvas,
            frame_count_str,
            (canvas_width - 200, status_y),
            font,
            0.7,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Combine main canvas with status bar
        full_canvas = np.zeros(
            (canvas_height + status_height, canvas_width, 3), dtype=np.uint8
        )
        full_canvas[:canvas_height, :] = canvas
        full_canvas[canvas_height:, :] = status_canvas

        # Store the visualization frame
        cls._viz_frame = full_canvas

        # Create window if needed
        if not cls._debug_window_created:
            cv2.namedWindow(cls._debug_window_name, cv2.WINDOW_NORMAL)

            # Set initial window size to be larger
            window_width = max(cls._initial_window_width, canvas_width)
            window_height = max(
                cls._initial_window_height, canvas_height + status_height
            )
            cv2.resizeWindow(cls._debug_window_name, window_width, window_height)

            cls._debug_window_created = True

        # Display the frame
        cv2.imshow(cls._debug_window_name, full_canvas)

        # Handle keyboard input
        cls._check_key_presses()

        # Save frame if debug directory is set
        if cls._debug_dir is not None:
            timestamp = int(time.time() * 1000)
            filename = f"debug_{cls._latest_frame_title}_{timestamp}.jpg"
            filepath = os.path.join(cls._debug_dir, filename)
            cv2.imwrite(filepath, full_canvas)

    @classmethod
    def _check_key_presses(cls, wait_time: int = 1) -> None:
        """
        Check for keyboard input to control visualization.

        Args:
            wait_time: Time to wait for key press in milliseconds
        """
        key = cv2.waitKey(wait_time)
        if key != -1:
            cls._last_key = key

            # ESC key to exit
            if key == 27:  # ESC key
                logger.debug("Visualization stopped by user")
                cls.end()

            # 'p' key to toggle pause
            elif key == ord("p"):
                cls._paused = not cls._paused
                if cls._paused:
                    logger.debug("Visualization paused")
                    # When pausing, turn off step mode
                    cls._step_mode = False
                else:
                    logger.debug("Visualization resumed")

            # 's' key for step mode
            elif key == ord("s"):
                # If we're in step mode, indicate ready for next frame
                if cls._step_mode:
                    cls._step_ready = True
                    logger.debug("Step: Next frame")
                else:
                    # Enter step mode
                    cls._step_mode = True
                    cls._paused = False
                    logger.debug("Entered step mode")

            # 'f' key to toggle fullscreen
            elif key == ord("f"):
                # Get the window property
                fullscreen_prop = cv2.getWindowProperty(
                    cls._debug_window_name, cv2.WND_PROP_FULLSCREEN
                )
                # Toggle fullscreen
                if fullscreen_prop == 0:  # Not fullscreen
                    cv2.setWindowProperty(
                        cls._debug_window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                    logger.debug("Entered fullscreen mode")
                else:
                    cv2.setWindowProperty(
                        cls._debug_window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL,
                    )
                    logger.debug("Exited fullscreen mode")

    @classmethod
    def wait_for_step(cls) -> None:
        """
        In step mode or pause mode, wait for user action before continuing.

        This method should be called at appropriate points in the processing
        pipeline to ensure the visualization respects step mode.
        """
        # If paused, wait indefinitely for a key press
        while cls._paused:
            cls._check_key_presses(100)  # Check every 100ms

            # Exit the loop if not paused anymore
            if not cls._paused:
                break

        # If in step mode, wait until step_ready is true
        while cls._step_mode and not cls._step_ready:
            cls._check_key_presses(100)  # Check every 100ms

            # Exit if step_ready is set or we're no longer in step mode
            if cls._step_ready or not cls._step_mode:
                break

        # Reset step_ready flag
        if cls._step_ready:
            cls._step_ready = False

    @classmethod
    def is_paused(cls) -> bool:
        """Return whether visualization is paused."""
        return cls._paused

    @classmethod
    def is_step_mode(cls) -> bool:
        """Return whether step mode is enabled."""
        return cls._step_mode

    @classmethod
    def is_step_ready(cls) -> bool:
        """Return whether ready to advance to next frame in step mode."""
        # In step mode, we need the step_ready flag to be true
        if cls._step_mode:
            if cls._step_ready:
                # Reset the flag and return true
                cls._step_ready = False
                return True
            return False

        # If not in step mode, always ready
        return True

    @classmethod
    def set_debug_directory(cls, debug_dir: str) -> None:
        """
        Set the directory for saving debug frames.

        Args:
            debug_dir: Path to directory for saving debug frames
        """
        cls._debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)
        logger.debug(f"Set debug directory to {debug_dir}")

    @classmethod
    def clear(cls) -> None:
        """
        Clear all frames from the visualization.
        """
        cls._frames = {}
        cls._latest_frame_title = None
        logger.debug("Cleared visualization frames")

    @classmethod
    def clear_frames_with_prefix(cls, prefix: str) -> None:
        """
        Safely remove frames with a specific prefix from the visualization.

        Args:
            prefix: String prefix to match against frame titles
        """
        # Create a list of keys to delete to avoid modifying dict during iteration
        keys_to_delete = [key for key in cls._frames if key.startswith(prefix)]

        # Delete the keys
        for key in keys_to_delete:
            del cls._frames[key]

        # Update the latest frame title if it was deleted
        if cls._latest_frame_title is not None and cls._latest_frame_title.startswith(
            prefix
        ):
            cls._latest_frame_title = None

        logger.debug(f"Cleared {len(keys_to_delete)} frames with prefix '{prefix}'")

    @classmethod
    def end(cls) -> None:
        """
        Clean up visualization resources.
        """
        if cls._debug_window_created:
            cv2.destroyWindow(cls._debug_window_name)
            cls._debug_window_created = False
            logger.debug("Closed debug window")

        # Clear all frames
        cls._frames = {}
        cls._latest_frame_title = None
