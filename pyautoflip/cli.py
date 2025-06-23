"""
Command-line interface for AutoFlip.
"""

import os
import argparse
import sys
from typing import List, Optional

from pyautoflip import reframe_video, __version__


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="AutoFlip: Intelligent Video Reframing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add version command
    parser.add_argument(
        "-v", "--version", action="version", version=f"AutoFlip {__version__}"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Reframe command
    reframe_parser = subparsers.add_parser(
        "reframe", help="Reframe a video to a target aspect ratio"
    )
    reframe_parser.add_argument(
        "-i", "--input", required=True, help="Path to the input video file"
    )
    reframe_parser.add_argument(
        "-o", "--output", required=True, help="Path to save the output video"
    )
    reframe_parser.add_argument(
        "-a",
        "--aspect-ratio",
        default="9:16",
        help="Target aspect ratio (width:height)",
    )
    reframe_parser.add_argument(
        "-m",
        "--motion-threshold",
        type=float,
        default=0.5,
        help="Threshold for camera motion (0.0-1.0)",
    )
    reframe_parser.add_argument(
        "-p",
        "--padding-method",
        choices=["blur", "solid_color"],
        default="blur",
        help="Method for padding when content cannot be fully included",
    )
    reframe_parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode: shows visualization, saves debug frames, provides comprehensive debug information, sets log level to DEBUG, and saves logs to autoflip.log",
    )

    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        sys.exit(0)

    return parsed_args


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code
    """
    parsed_args = parse_args(args)

    try:
        if parsed_args.command == "reframe":
            # Ensure output directory exists
            output_dir = os.path.dirname(parsed_args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Set log level and log file based on debug mode
            log_level = "DEBUG" if parsed_args.debug else "INFO"
            log_file = "autoflip.log" if parsed_args.debug else None

            # Process the video
            output_path = reframe_video(
                input_path=parsed_args.input,
                output_path=parsed_args.output,
                target_aspect_ratio=parsed_args.aspect_ratio,
                motion_threshold=parsed_args.motion_threshold,
                padding_method=parsed_args.padding_method,
                debug_mode=parsed_args.debug,
                log_level=log_level,
                log_file=log_file,
            )

            print(f"✅ Video reframing completed: {output_path}")
            return 0
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
