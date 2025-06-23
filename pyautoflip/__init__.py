"""
AutoFlip: Intelligent Video Reframing

A Python library for automatic video reframing that intelligently crops videos 
to different aspect ratios while preserving salient content.
"""

__version__ = "0.1.0"

import logging
from typing import Optional, Union

from pyautoflip.core.processor import AutoFlipProcessor

# Configure root logger
root_logger = logging.getLogger()
if not root_logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

# Create main autoflip logger
logger = logging.getLogger("autoflip")


def configure_logging(level: Union[int, str] = "INFO", log_file: Optional[str] = None):
    """
    Configure the global logging level and optionally add file logging.

    Args:
        level: Logging level (int, str, or LogLevel enum)
        log_file: Optional path to a log file
    """
    # Handle string level names
    if isinstance(level, str):
        level = level.upper()
        try:
            level = getattr(logging, level)
        except AttributeError:
            logger.warning(f"Invalid log level: {level}, using INFO")
            level = logging.INFO
    else:
        try:
            level = int(level)  # Convert to int if possible
        except (TypeError, ValueError):
            logger.warning(f"Invalid log level: {level}, using INFO")
            level = logging.INFO

    # Set the level on the root logger
    root_logger.setLevel(level)

    # Configure file logging if requested
    if log_file:
        # Create a file handler
        file_handler = logging.FileHandler(log_file, mode="w")  # mode='w' to overwrite
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Add the handler to the root logger
        root_logger.addHandler(file_handler)

    logger.debug(f"Logging configured with level: {level}")


def reframe_video(
    input_path: str,
    output_path: str,
    target_aspect_ratio: str = "9:16",
    motion_threshold: float = 0.5,
    padding_method: str = "blur",
    debug_mode: bool = False,
    log_level: Union[int, str] = "INFO",
    log_file: Optional[str] = None,
) -> str:
    """
    Reframe a video to a target aspect ratio while preserving important content.

    Args:
        input_path: Path to the input video file
        output_path: Path to save the output video
        target_aspect_ratio: Target aspect ratio as "width:height" (e.g., "9:16")
        motion_threshold: Threshold for camera motion (0.0-1.0)
        padding_method: Method for padding ("blur" or "solid_color")
        debug_mode: If True, enables visualization, debug frames, and debug mode processing
        log_level: Logging level (use LogLevel enum values or string names like 'INFO', 'DEBUG')
        log_file: Path to save logs to a file (None to log to console only)

    Returns:
        Path to the reframed video file
    """
    # Configure logging globally
    configure_logging(log_level, log_file)

    processor = AutoFlipProcessor(
        target_aspect_ratio=target_aspect_ratio,
        motion_threshold=motion_threshold,
        padding_method=padding_method,
        debug_mode=debug_mode,
    )

    return processor.process_video(
        input_path=input_path,
        output_path=output_path,
    )
