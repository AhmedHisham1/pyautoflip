"""
Visualization utilities for autoflip.

This module provides functions for visualizing crop regions, detections, and
other debugging information for the autoflip video reframing system.
"""

import logging
import os
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

from autoflip.cropping.types import CameraMotionMode

# Create module-level logger
logger = logging.getLogger("autoflip.cropping.visualization")


class VisualizationUtils:
    """
    Utilities for creating debug visualizations.
    
    This class provides methods for drawing crop regions, face and object
    detections, and other debug information on frames.
    """
    
    # Class variables for visualization control
    _debug_window_name = "AutoFlip Debug"
    _debug_window_created = False
    _debug_dir = None  # Directory for saving debug frames
    
    # Visualization control state
    _paused = False  # Whether visualization is paused
    _step_mode = False  # Whether to advance one frame at a time
    _step_ready = False  # Whether the system is ready for the next step
    _last_key = -1  # Last key pressed
    
    # Current visualization frame
    _current_vis_frame = None
    
    @staticmethod
    def is_paused() -> bool:
        """Return whether visualization is paused."""
        return VisualizationUtils._paused
        
    @staticmethod
    def is_step_mode() -> bool:
        """Return whether step mode is enabled."""
        return VisualizationUtils._step_mode
        
    @staticmethod
    def is_step_ready() -> bool:
        """Return whether ready to advance to next frame in step mode."""
        # In step mode, we need the step_ready flag to be true
        if VisualizationUtils._step_mode:
            if VisualizationUtils._step_ready:
                # Reset the flag and return true
                VisualizationUtils._step_ready = False
                return True
            return False
        
        # If not in step mode, always ready
        return True
        
    @staticmethod
    def check_for_key_press(wait_time: int = 1) -> int:
        """
        Check for and handle key presses for controlling visualization flow.
        
        Args:
            wait_time: Time to wait for key press (milliseconds)
            
        Returns:
            The key that was pressed, or -1 if no key was pressed
        """
        key = cv2.waitKey(wait_time)
        if key != -1:
            VisualizationUtils._last_key = key
            
            # ESC key to exit
            if key == 27:  # ESC key
                logger.info("Visualization stopped by user")
                VisualizationUtils.close_debug_window()
                
            # 'p' key to toggle pause
            elif key == ord('p'):
                VisualizationUtils._paused = not VisualizationUtils._paused
                if VisualizationUtils._paused:
                    logger.info("Visualization paused")
                    # When pausing, turn off step mode
                    VisualizationUtils._step_mode = False
                else:
                    logger.info("Visualization resumed")
                
            # 'e' key for step mode
            elif key == ord('e'):
                # If we're in step mode, indicate ready for next frame
                if VisualizationUtils._step_mode:
                    VisualizationUtils._step_ready = True
                    logger.info("Step: Next frame")
                else:
                    # Enter step mode
                    VisualizationUtils._step_mode = True
                    VisualizationUtils._paused = False
                    logger.info("Entered step mode")
                    
        return key
    
    @staticmethod
    def set_debug_directory(debug_dir: str) -> None:
        """Set the directory for saving debug frames."""
        VisualizationUtils._debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)
    
    @staticmethod
    def close_debug_window() -> None:
        """Close the debug window."""
        if VisualizationUtils._debug_window_created:
            cv2.destroyWindow(VisualizationUtils._debug_window_name)
            VisualizationUtils._debug_window_created = False
    
    @staticmethod
    def create_debug_frame(
        frame: np.ndarray,
        x: int,
        y: int,
        crop_width: int,
        crop_height: int,
        face_dets: Optional[List[Dict[str, Any]]] = None,
        object_dets: Optional[List[Dict[str, Any]]] = None,
        saliency_score: Optional[float] = None,
        camera_mode: Optional[CameraMotionMode] = None
    ) -> np.ndarray:
        """
        Create a debug visualization frame showing crop rectangle and detections.
        
        Args:
            frame: Original video frame
            x, y: Crop position
            crop_width, crop_height: Crop dimensions
            face_dets: Face detections
            object_dets: Object detections
            saliency_score: Saliency score
            camera_mode: Camera motion mode
            
        Returns:
            Debug visualization frame
        """
        height, width = frame.shape[:2]
        debug_frame = frame.copy()
        
        # Draw target crop rectangle
        crop_color = (0, 255, 0)  # Green
        thickness = 3
        cv2.rectangle(debug_frame, (x, y), (x + crop_width, y + crop_height), crop_color, thickness)
        
        # Draw aspect ratio text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        aspect_ratio = crop_width / crop_height
        cv2.putText(debug_frame, f"AR: {aspect_ratio:.2f}", (x+10, y+30), 
                    font, font_scale, crop_color, 2, cv2.LINE_AA)
        
        # Draw camera mode if provided
        if camera_mode is not None:
            mode_str = f"Mode: {camera_mode.name}"
            cv2.putText(debug_frame, mode_str, (x+10, y+60), 
                        font, font_scale, crop_color, 2, cv2.LINE_AA)
        
        # Draw dimensions
        cv2.putText(debug_frame, f"{crop_width}x{crop_height}", (x+10, y+90), 
                    font, font_scale, crop_color, 2, cv2.LINE_AA)
        
        # Draw saliency score if provided
        if saliency_score is not None:
            cv2.putText(debug_frame, f"Score: {saliency_score:.2f}", (x+10, y+120),
                        font, font_scale, crop_color, 2, cv2.LINE_AA)
        
        # Draw face detections if provided
        if face_dets is not None:
            for face in face_dets:
                face_x = int(face.get('x', 0) * width)
                face_y = int(face.get('y', 0) * height)
                face_w = int(face.get('width', 0) * width)
                face_h = int(face.get('height', 0) * height)
                
                # Check if face is marked as required
                is_required = face.get('is_required', False)
                
                # Use different color for required and non-required faces
                face_color = (0, 0, 255)  # Default: Red for faces
                if is_required:
                    face_color = (255, 0, 255)  # Magenta for required faces
                    thickness = 3
                else:
                    thickness = 2
                
                # Draw face rectangle
                cv2.rectangle(debug_frame, (face_x, face_y), (face_x + face_w, face_y + face_h), 
                             face_color, thickness)
                
                # Draw face label with confidence and priority
                conf = face.get('confidence', 0)
                priority = face.get('priority', 0)
                label_text = f"Face: {conf:.2f} P:{priority:.1f}"
                
                # Add 'REQ' tag for required detections
                if is_required:
                    label_text += " [REQ]"
                    
                cv2.putText(debug_frame, label_text, 
                           (face_x, face_y - 10), font, 0.6, face_color, 1, cv2.LINE_AA)
        
        # Draw object detections if provided
        if object_dets is not None:
            for obj in object_dets:
                obj_x = int(obj.get('x', 0) * width)
                obj_y = int(obj.get('y', 0) * height)
                obj_w = int(obj.get('width', 0) * width)
                obj_h = int(obj.get('height', 0) * height)
                
                # Check if object is marked as required
                is_required = obj.get('is_required', False)
                
                # Use different color for required and non-required objects
                obj_color = (255, 0, 0)  # Default: Blue for objects
                if is_required:
                    obj_color = (255, 255, 0)  # Cyan for required objects
                    thickness = 3
                else:
                    thickness = 2
                
                # Draw object rectangle
                cv2.rectangle(debug_frame, (obj_x, obj_y), (obj_x + obj_w, obj_y + obj_h), 
                             obj_color, thickness)
                
                # Draw object label with confidence, class, and priority
                label = obj.get('class', 'unknown')
                conf = obj.get('confidence', 0)
                priority = obj.get('priority', 0)
                label_text = f"{label}: {conf:.2f} P:{priority:.1f}"
                
                # Add 'REQ' tag for required detections
                if is_required:
                    label_text += " [REQ]"
                    
                cv2.putText(debug_frame, label_text, 
                           (obj_x, obj_y - 10), font, 0.6, obj_color, 1, cv2.LINE_AA)
        
        # Draw "rule of thirds" grid
        third_h, third_w = height // 3, width // 3
        grid_color = (200, 200, 200)  # Light gray
        
        # Vertical lines
        cv2.line(debug_frame, (third_w, 0), (third_w, height), grid_color, 1)
        cv2.line(debug_frame, (2 * third_w, 0), (2 * third_w, height), grid_color, 1)
        
        # Horizontal lines
        cv2.line(debug_frame, (0, third_h), (width, third_h), grid_color, 1)
        cv2.line(debug_frame, (0, 2 * third_h), (width, 2 * third_h), grid_color, 1)
        
        return debug_frame
        
    @staticmethod
    def create_trajectory_vis(
        frame: np.ndarray,
        crop_windows: List[Tuple[int, int, int, int]],
        current_idx: int,
        max_points: int = 20
    ) -> np.ndarray:
        """
        Create a visualization of the crop window trajectory.
        
        Args:
            frame: Original video frame
            crop_windows: List of crop windows (x, y, width, height)
            current_idx: Index of the current frame
            max_points: Maximum number of points to show
            
        Returns:
            Visualization frame
        """
        vis_frame = frame.copy()
        
        # Determine range of windows to show
        start_idx = max(0, current_idx - max_points // 2)
        end_idx = min(len(crop_windows), current_idx + max_points // 2)
        
        # Draw past windows in blue (fading), current in green, future in red (fading)
        for i in range(start_idx, end_idx):
            x, y, width, height = crop_windows[i]
            center_x = x + width // 2
            center_y = y + height // 2
            
            if i < current_idx:
                # Past point - blue, fading with distance
                alpha = 0.5 * (1 - (current_idx - i) / (max_points // 2))
                color = (int(255 * alpha), 0, 0)  # Blue with alpha
            elif i == current_idx:
                # Current point - green
                color = (0, 255, 0)
            else:
                # Future point - red, fading with distance
                alpha = 0.5 * (1 - (i - current_idx) / (max_points // 2))
                color = (0, 0, int(255 * alpha))  # Red with alpha
            
            # Draw center point
            cv2.circle(vis_frame, (center_x, center_y), 5, color, -1)
            
            # Connect with line if not the first point
            if i > start_idx:
                prev_x, prev_y, prev_w, prev_h = crop_windows[i-1]
                prev_center_x = prev_x + prev_w // 2
                prev_center_y = prev_y + prev_h // 2
                cv2.line(vis_frame, (prev_center_x, prev_center_y), (center_x, center_y), color, 2)
            
            # Draw crop window rectangle with lower opacity
            if i == current_idx:
                # Current window in semi-transparent green
                window_color = (0, 200, 0)
                cv2.rectangle(vis_frame, (x, y), (x + width, y + height), window_color, 2)
        
        # Add title
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_frame, "Camera Motion Trajectory", (20, 40),
                   font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        return vis_frame
    
    @staticmethod
    def create_simple_comparison(
        original_frame: np.ndarray, 
        cropped_frame: np.ndarray,
        crop_coords: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Create a simple side-by-side comparison of original and cropped frames.
        
        Args:
            original_frame: Original input frame
            cropped_frame: Cropped output frame
            crop_coords: Optional crop coordinates to highlight on original frame
            
        Returns:
            Comparison visualization
        """
        # Calculate display dimensions
        orig_h, orig_w = original_frame.shape[:2]
        crop_h, crop_w = cropped_frame.shape[:2]
        
        # Set a fixed display height
        display_height = 400
        
        # Calculate display widths to maintain aspect ratios
        display_width_orig = int(orig_w * display_height / orig_h)
        display_width_crop = int(crop_w * display_height / crop_h)
        
        # Resize frames
        orig_resized = cv2.resize(original_frame, (display_width_orig, display_height))
        crop_resized = cv2.resize(cropped_frame, (display_width_crop, display_height))
        
        # Create side-by-side comparison
        padding = 10
        total_width = display_width_orig + display_width_crop + padding
        
        vis = np.zeros((display_height, total_width, 3), dtype=np.uint8)
        
        # Place frames
        vis[:, :display_width_orig] = orig_resized
        vis[:, display_width_orig + padding:] = crop_resized
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, "Original", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, "Reframed", (display_width_orig + padding + 10, 30), font, 0.8, (0, 255, 0), 2)
        
        # If crop coordinates provided, draw crop rectangle on original frame
        if crop_coords is not None:
            x, y, w, h = crop_coords
            # Scale coordinates to fit display size
            scaled_x = int(x * display_width_orig / orig_w)
            scaled_y = int(y * display_height / orig_h)
            scaled_w = int(w * display_width_orig / orig_w)
            scaled_h = int(h * display_height / orig_h)
            
            # Draw rectangle
            cv2.rectangle(vis[:, :display_width_orig], (scaled_x, scaled_y), 
                         (scaled_x + scaled_w, scaled_y + scaled_h), (0, 255, 0), 2)
        
        return vis
    
    @staticmethod
    def display_processing_view(
        original_frame: np.ndarray,
        detection_frame: Optional[np.ndarray] = None,
        cropped_frame: Optional[np.ndarray] = None,
        crop_window: Optional[Tuple[int, int, int, int]] = None,
        trajectory_frame: Optional[np.ndarray] = None,
        frame_info: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Display a simplified processing view showing essential steps in the pipeline.
        
        This is the main visualization function that shows key frames from each step
        of the processing pipeline in a simple, fixed layout.
        
        Args:
            original_frame: Original input frame
            detection_frame: Frame with face/object detection visualization
            cropped_frame: Final cropped output frame
            crop_window: Crop window coordinates (x, y, width, height)
            trajectory_frame: Optional trajectory visualization frame
            frame_info: Optional text information to display
            save_path: Optional path to save the visualization
        """
        if original_frame is None:
            logger.error("Cannot create processing view: original frame is None")
            return
            
        # Create the window if it doesn't exist
        if not VisualizationUtils._debug_window_created:
            cv2.namedWindow(VisualizationUtils._debug_window_name, cv2.WINDOW_NORMAL)
            # Set initial window size
            cv2.resizeWindow(VisualizationUtils._debug_window_name, 1600, 900)
            VisualizationUtils._debug_window_created = True
            
        # Setup display dimensions
        display_width = 1600
        display_height = 900
            
        # Header and footer heights
        header_height = 50
        footer_height = 50
        
        # Create a blank canvas
        final_display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        # Create header with title
        header = np.zeros((header_height, display_width, 3), dtype=np.uint8)
        header_color = (50, 50, 120)  # Dark blue-ish
        cv2.rectangle(header, (0, 0), (display_width, header_height), header_color, -1)
        
        # Add title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(header, "AutoFlip Processing Pipeline", (20, 35), 
                   font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add header to final display
        final_display[:header_height, :] = header
        
        # Calculate usable area
        usable_height = display_height - header_height - footer_height
        
        # Define layout - fixed 2x2 grid for the main view components
        grid_padding = 10
        cell_width = (display_width - 3 * grid_padding) // 2
        cell_height = (usable_height - 3 * grid_padding) // 2
        
        # Function to resize and place a frame in the grid
        def place_frame(frame, row, col, title):
            if frame is None:
                # Create a blank frame with text if None
                blank = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                cv2.putText(blank, f"No {title} Available", 
                           (cell_width//4, cell_height//2), 
                           font, 0.8, (150, 150, 150), 2)
                frame = blank
            else:
                # Resize frame to fit cell while maintaining aspect ratio
                h, w = frame.shape[:2]
                aspect = w / max(1, h)
                
                if aspect > cell_width / cell_height:  # Width limited
                    new_w = cell_width
                    new_h = int(new_w / aspect)
                else:  # Height limited
                    new_h = cell_height
                    new_w = int(new_h * aspect)
                
                frame = cv2.resize(frame, (new_w, new_h))
                
                # If frame is smaller than cell, create a blank frame and center
                if new_h < cell_height or new_w < cell_width:
                    blank = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                    y_offset = (cell_height - new_h) // 2
                    x_offset = (cell_width - new_w) // 2
                    blank[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame
                    frame = blank
            
            # Add title to frame
            title_height = 30
            title_y = 5 + title_height // 2
            cv2.putText(frame, title, (10, title_y), 
                       font, 0.7, (0, 255, 0), 2)
                       
            # Calculate position in the grid
            y = header_height + grid_padding + row * (cell_height + grid_padding)
            x = grid_padding + col * (cell_width + grid_padding)
            
            # Place in the final display
            if frame.shape[:2] == (cell_height, cell_width):
                final_display[y:y+cell_height, x:x+cell_width] = frame
            else:
                # Handle case where frame might not exactly match cell size
                h, w = frame.shape[:2]
                final_display[y:y+h, x:x+w] = frame
                
        # Place the main view components
        # Row 0, Col 0: Original Frame
        place_frame(original_frame, 0, 0, "Original Frame")
        
        # Row 1, Col 0: Detection Frame (Face/Object)
        # If detection_frame is None but we have crop_window, create a basic detection visualization
        if detection_frame is None and crop_window is not None and original_frame is not None:
            detection_frame = VisualizationUtils.create_debug_frame(
                original_frame, crop_window[0], crop_window[1], 
                crop_window[2], crop_window[3]
            )
        place_frame(detection_frame, 1, 0, "Face & Object Detection")
        
        # Row 1, Col 1: Comparison or Trajectory
        if trajectory_frame is not None:
            place_frame(trajectory_frame, 1, 1, "Camera Motion Trajectory")
        elif cropped_frame is not None:
            # Create a simple comparison if we have both frames
            comparison = VisualizationUtils.create_simple_comparison(
                original_frame, cropped_frame, crop_window
            )
            place_frame(comparison, 1, 1, "Original vs Reframed")
        else:
            place_frame(None, 1, 1, "Reframed Output")
        
        # Create footer with controls and frame info
        footer = np.zeros((footer_height, display_width, 3), dtype=np.uint8)
        footer_color = (40, 40, 40)  # Dark gray
        cv2.rectangle(footer, (0, 0), (display_width, footer_height), footer_color, -1)
        
        # Add controls text
        controls_text = "Controls: [P] Pause/Resume | [E] Step Mode | [ESC] Exit"
        cv2.putText(footer, controls_text, (20, 35), 
                   font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Add frame info if provided
        if frame_info:
            text_size = cv2.getTextSize(frame_info, font, 0.7, 1)[0]
            info_x = display_width - text_size[0] - 20
            cv2.putText(footer, frame_info, (info_x, 35), 
                       font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Add playback status
        if VisualizationUtils._paused:
            status = "PAUSED"
            status_color = (0, 0, 255)  # Red
        elif VisualizationUtils._step_mode:
            status = "STEP MODE"
            status_color = (0, 255, 255)  # Yellow
        else:
            status = "PLAYING"
            status_color = (0, 255, 0)  # Green
            
        status_x = display_width // 2 - 50
        cv2.putText(footer, status, (status_x, 35), 
                   font, 0.7, status_color, 2, cv2.LINE_AA)
        
        # Add footer to final display
        final_display[display_height-footer_height:, :] = footer
        
        # Save to file if requested
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            cv2.imwrite(save_path, final_display)
            logger.debug(f"Saved debug visualization to {save_path}")
        
        # Show the display
        cv2.imshow(VisualizationUtils._debug_window_name, final_display)
        VisualizationUtils._current_vis_frame = final_display
        
        # Handle keyboard input (non-blocking)
        VisualizationUtils.check_for_key_press(1)
    
    @staticmethod
    def get_standard_debug_path(
        debug_dir: str = "debug_frames",
        scene_idx: int = 0,
        frame_idx: int = 0,
        frame_type: str = "pipeline",
        ext: str = "jpg"
    ) -> str:
        """
        Generate a standardized path for debug frames.
        
        Args:
            debug_dir: Directory for debug frames
            scene_idx: Scene index
            frame_idx: Frame index
            frame_type: Type of frame
            ext: File extension
            
        Returns:
            Standard path for the debug frame
        """
        os.makedirs(debug_dir, exist_ok=True)
        return os.path.join(
            debug_dir, 
            f"scene_{scene_idx+1:03d}_frame_{frame_idx:05d}_{frame_type}.{ext}"
        ) 