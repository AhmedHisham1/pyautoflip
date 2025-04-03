from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import logging

# Create and configure module-level logger
logger = logging.getLogger("autoflip.detection.border_detector")


class Color:
    """Simple color class to match MediaPipe's implementation."""
    
    def __init__(self, r: int = 0, g: int = 0, b: int = 0):
        self.r = r
        self.g = g
        self.b = b


class BorderDetector:
    """
    Detector for identifying black borders and static content in videos.
    
    This detector analyzes frames to find top and bottom borders
    that can be safely cropped without losing content.
    """
    
    MIN_BORDER_DISTANCE = 5
    K_MEANS_CLUSTER_COUNT = 4
    MAX_PIXELS_TO_PROCESS = 300000
    
    def __init__(
        self,
        color_tolerance: int = 25,
        border_color_pixel_percentage: float = 0.6,
        vertical_search_distance: float = 0.3,
        default_padding_px: int = 0,
        border_object_padding_px: int = 0,
        solid_background_tolerance_percentage: float = 0.8,
        max_frames_to_process: int = 10,
        debug_mode: bool = False,
    ):
        """
        Initialize the border detector.
        
        Args:
            color_tolerance: Tolerance for color matching
            border_color_pixel_percentage: Minimum percentage of pixels in a row that must match border color
            vertical_search_distance: Fraction of frame height to search for borders
            default_padding_px: Default padding in pixels
            border_object_padding_px: Additional padding for detected borders
            solid_background_tolerance_percentage: Threshold for determining if background is solid color
            max_frames_to_process: Maximum number of frames to process for border detection
            debug_mode: If True, draw visualizations of detected borders
        """
        self.color_tolerance = color_tolerance
        self.border_color_pixel_percentage = border_color_pixel_percentage
        self.vertical_search_distance = vertical_search_distance
        self.default_padding_px = default_padding_px
        self.border_object_padding_px = border_object_padding_px
        self.solid_background_tolerance_percentage = solid_background_tolerance_percentage
        self.max_frames_to_process = max_frames_to_process
        self.debug_mode = debug_mode
        
    def detect(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Detect borders in a sequence of frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary containing border information and background color
        """
        if not frames:
            raise ValueError("No frames provided")
            
        # Limit the number of frames to process for performance
        if len(frames) > self.max_frames_to_process:
            # Choose frames evenly distributed throughout the video
            step = max(1, len(frames) // self.max_frames_to_process)
            frames_to_process = frames[::step][:self.max_frames_to_process]
            logger.debug(f"Border detection: Processing {len(frames_to_process)} of {len(frames)} frames")
        else:
            frames_to_process = frames
            
        # Process the first frame
        frame = frames_to_process[0]
        frame_height, frame_width = frame.shape[:2]
        
        # Initialize output with default values
        features = {
            "top_border": 0,
            "bottom_border": 0,
            "background_color": (0, 0, 0),
            "non_border_area": {
                "x": 0,
                "y": self.default_padding_px,
                "width": frame_width,
                "height": max(0, frame_height - self.default_padding_px * 2)
            }
        }
        
        # Check for border at the top of the frame
        seed_color_top = Color()
        self._find_dominant_color(frame[0:1, :, :], seed_color_top)
        self._detect_border(frame, seed_color_top, "top", features)
        
        # Check for border at the bottom of the frame
        seed_color_bottom = Color()
        self._find_dominant_color(frame[frame_height-1:frame_height, :, :], seed_color_bottom)
        self._detect_border(frame, seed_color_bottom, "bottom", features)
        
        # Fallback: If no borders were detected but it looks like a letterboxed video,
        # try dedicated letterbox detection
        if features["top_border"] == 0 and features["bottom_border"] == 0:
            is_letterboxed, letterbox_features = self._detect_letterbox(frame)
            if is_letterboxed:
                logger.debug("Detected letterboxed video using histogram analysis")
                features.update(letterbox_features)
        
        # Check the non-border area for a dominant color
        non_static_area = features["non_border_area"]
        non_static_frame = frame[
            non_static_area["y"]:non_static_area["y"]+non_static_area["height"],
            non_static_area["x"]:non_static_area["x"]+non_static_area["width"]
        ]
        
        dominant_color_nonborder = Color()
        dominant_color_percent = self._find_dominant_color(non_static_frame, dominant_color_nonborder)
        
        if dominant_color_percent > self.solid_background_tolerance_percentage:
            features["background_color"] = (
                dominant_color_nonborder.b,
                dominant_color_nonborder.g,
                dominant_color_nonborder.r
            )
            
        return features
        
    def _find_dominant_color(self, image_raw: np.ndarray, dominant_color: Color) -> float:
        """
        Find the dominant color within an image using k-means clustering.
        
        Args:
            image_raw: Input image
            dominant_color: Color object to be filled with dominant color
            
        Returns:
            Percentage of pixels that match the dominant color
        """
        # Check if image is empty
        if image_raw.size == 0:
            return 0.0
            
        # Resize if needed for performance
        if image_raw.size > self.MAX_PIXELS_TO_PROCESS:
            resize = np.sqrt(self.MAX_PIXELS_TO_PROCESS / image_raw.size)
            image = cv2.resize(image_raw, None, fx=resize, fy=resize)
        else:
            image = image_raw
            
        # Reshape the image for K-means
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, 
            self.K_MEANS_CLUSTER_COUNT, 
            None, 
            criteria, 
            10, 
            cv2.KMEANS_PP_CENTERS
        )
        
        # Count occurrences of each cluster
        count = np.zeros(self.K_MEANS_CLUSTER_COUNT, dtype=int)
        for i in range(len(labels)):
            count[labels[i][0]] += 1
            
        # Find the most common cluster
        max_cluster_idx = np.argmax(count)
        max_cluster_perc = count[max_cluster_idx] / float(len(labels))
        
        # Set the dominant color
        dominant_color.b = int(centers[max_cluster_idx][0])
        dominant_color.g = int(centers[max_cluster_idx][1])
        dominant_color.r = int(centers[max_cluster_idx][2])
        
        return max_cluster_perc
        
    def _color_count(self, mask_color: Color, image: np.ndarray) -> float:
        """
        Count the percentage of pixels that match a given color within tolerance.
        
        Args:
            mask_color: Color to match
            image: Input image
            
        Returns:
            Percentage of pixels that match the color
        """
        # Create color ranges
        lower_bound = np.array([
            max(0, mask_color.b - self.color_tolerance),
            max(0, mask_color.g - self.color_tolerance),
            max(0, mask_color.r - self.color_tolerance)
        ], dtype=np.uint8)
        
        upper_bound = np.array([
            min(255, mask_color.b + self.color_tolerance),
            min(255, mask_color.g + self.color_tolerance),
            min(255, mask_color.r + self.color_tolerance)
        ], dtype=np.uint8)
        
        # Create mask where pixels are within range
        mask = cv2.inRange(image, lower_bound, upper_bound)
        
        # Count pixels
        matching_pixels = cv2.countNonZero(mask)
        total_pixels = image.shape[0] * image.shape[1]
        
        return matching_pixels / float(total_pixels)
        
    def _detect_border(self, frame: np.ndarray, color: Color, direction: str, features: Dict[str, Any]) -> None:
        """
        Detect borders in the given direction.
        
        Args:
            frame: Input image frame
            color: Dominant color to search for
            direction: Direction to search ("top" or "bottom")
            features: Features dictionary to update
        """
        height, width = frame.shape[:2]
        
        # Determine search distance
        search_distance = int(height * self.vertical_search_distance)
        
        # Check if the dominant color is very dark (near black)
        is_dark_border = (color.r < 30 and color.g < 30 and color.b < 30)
        
        # If we have a very dark border, try using a brightness-based detection approach first
        if is_dark_border:
            # Convert to grayscale to detect brightness changes
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness values for each row
            brightness_values = []
            for i in range(search_distance):
                if direction == "top":
                    row_idx = i
                else:  # bottom
                    row_idx = height - i - 1
                
                row_brightness = np.mean(gray[row_idx:row_idx+1, :])
                brightness_values.append(row_brightness)
            
            # Look for significant brightness jump (transition from black letterbox to content)
            threshold_jump = 30  # Minimum brightness increase to consider a transition
            last_border = -1
            
            for i in range(1, len(brightness_values)):
                # Check for significant brightness increase
                if brightness_values[i] - brightness_values[i-1] > threshold_jump:
                    last_border = i - 1
                    logger.debug(f"Detected {direction} letterbox border via brightness jump at row {i}")
                    
                    # Apply defined padding
                    last_border += self.border_object_padding_px
                    
                    if direction == "top":
                        features["top_border"] = last_border
                        features["non_border_area"]["y"] = last_border + self.default_padding_px
                        features["non_border_area"]["height"] = max(
                            0, 
                            height - (features["non_border_area"]["y"] + self.default_padding_px)
                        )
                    else:  # bottom
                        features["bottom_border"] = last_border
                        features["non_border_area"]["height"] = max(
                            0,
                            height - (features["non_border_area"]["y"] + last_border + self.default_padding_px)
                        )
                    return
        
        # Use more lenient threshold for dark borders
        border_threshold = self.border_color_pixel_percentage
        if is_dark_border:
            border_threshold *= 0.8  # 20% more lenient for black borders
            
        logger.debug(f"Detecting {direction} border: Color R:{color.r} G:{color.g} B:{color.b} | Threshold: {border_threshold:.2f} | Dark border: {is_dark_border}")
        
        # Check if each next line has a dominant color that matches the given border color
        last_border = -1
        
        # Log match percentages for the first few rows to help debug
        debug_rows = min(10, search_distance)
        
        for i in range(search_distance):
            if direction == "top":
                current_row = frame[i:i+1, :, :]
            else:  # bottom
                current_row = frame[height-i-1:height-i, :, :]
            
            match_percentage = self._color_count(color, current_row)
            
            # Log debug info for first few rows
            if i < debug_rows:
                logger.debug(f"{direction} row {i}: match={match_percentage:.2f}, threshold={border_threshold:.2f}")
                
            if match_percentage < border_threshold:
                break
                
            last_border = i
            
        # Special case for letterboxed videos with large black areas
        # If we have a very dark border that spans nearly the entire search distance,
        # it's likely a letterbox and should not be rejected
        is_letterbox = is_dark_border and last_border > self.MIN_BORDER_DISTANCE and last_border > (search_distance * 0.5)
        
        # Reject results that are too small or exactly at the search boundary (which suggests no actual border)
        if last_border <= self.MIN_BORDER_DISTANCE or (last_border == search_distance - 1 and not is_letterbox):
            logger.debug(f"Rejected {direction} border: size={last_border}, min={self.MIN_BORDER_DISTANCE}")
            return
            
        # Apply defined padding
        last_border += self.border_object_padding_px
        
        logger.debug(f"Detected {direction} border: {last_border} pixels")
        
        if direction == "top":
            features["top_border"] = last_border
            features["non_border_area"]["y"] = last_border + self.default_padding_px
            features["non_border_area"]["height"] = max(
                0, 
                height - (features["non_border_area"]["y"] + self.default_padding_px)
            )
        else:  # bottom
            features["bottom_border"] = last_border
            features["non_border_area"]["height"] = max(
                0,
                height - (features["non_border_area"]["y"] + last_border + self.default_padding_px)
            )
            
    def _create_debug_visualization(self, frame: np.ndarray, features: Dict[str, Any]) -> np.ndarray:
        """
        Create a visualization frame that shows detected borders.
        
        Args:
            frame: Original input frame
            features: Detected border features
            
        Returns:
            Visualization frame with borders highlighted
        """
        debug_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Colors for visualization
        top_color = (0, 255, 0)     # Green
        bottom_color = (0, 255, 255) # Yellow
        content_color = (255, 0, 0)  # Blue
        
        # Define font once at the beginning to avoid the reference error
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        
        # Draw top border
        if features["top_border"] > 0:
            top_border = features["top_border"]
            # Draw line at the border
            cv2.line(debug_frame, (0, top_border), (width, top_border), top_color, 2)
            
            # Label the border area
            cv2.putText(debug_frame, f"Top Border: {top_border}px", 
                       (10, max(20, top_border // 2)), 
                       font, font_scale, top_color, 2, cv2.LINE_AA)
        
        # Draw bottom border
        if features["bottom_border"] > 0:
            bottom_border = height - features["bottom_border"]
            # Draw line at the border
            cv2.line(debug_frame, (0, bottom_border), (width, bottom_border), bottom_color, 2)
            
            # Label the border area
            cv2.putText(debug_frame, f"Bottom Border: {features['bottom_border']}px", 
                       (10, min(height - 10, bottom_border + 20)), 
                       font, font_scale, bottom_color, 2, cv2.LINE_AA)
        
        # Draw content area rectangle
        non_border = features["non_border_area"]
        x, y = non_border["x"], non_border["y"]
        w, h = non_border["width"], non_border["height"]
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), content_color, 2)
        
        # Label content area
        cv2.putText(debug_frame, "Content Area", 
                   (x + 10, y + 25), 
                   font, font_scale, content_color, 2, cv2.LINE_AA)
        
        # Add background color information if detected
        bg_color = features["background_color"]
        if bg_color != (0, 0, 0):  # if not default black
            color_text = f"BG Color: R:{bg_color[2]} G:{bg_color[1]} B:{bg_color[0]}"
            cv2.putText(debug_frame, color_text, 
                       (10, height - 10), 
                       font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        
        return debug_frame

    def _detect_letterbox(self, frame: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Specialized method to detect letterboxed videos using histogram analysis.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (is_letterboxed, features) where features contains border info
        """
        height, width = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate row-wise mean intensity (for each horizontal line)
        row_means = np.mean(gray, axis=1)
        
        # A typical letterboxed video will have very low intensity at top and bottom
        # Calculate the derivative to find sharp transitions
        row_diff = np.diff(row_means)
        
        # Find potential transitions (from black to content)
        transitions = []
        threshold = 10  # Minimum intensity change to consider a transition
        
        for i in range(len(row_diff)):
            if abs(row_diff[i]) > threshold:
                transitions.append((i, row_diff[i]))
        
        # No significant transitions found
        if len(transitions) < 2:
            return False, {}
        
        # Sort transitions by absolute magnitude
        transitions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get the largest positive and negative transitions
        positive_transitions = [t for t in transitions if t[1] > 0]
        negative_transitions = [t for t in transitions if t[1] < 0]
        
        if not positive_transitions or not negative_transitions:
            return False, {}
            
        # Top border is the position of the largest positive transition near the top
        top_candidates = [t for t in positive_transitions if t[0] < height // 3]
        if top_candidates:
            top_border = top_candidates[0][0]
        else:
            top_border = 0
            
        # Bottom border is the position of the largest negative transition near the bottom
        bottom_candidates = [t for t in negative_transitions if t[0] > 2 * height // 3]
        if bottom_candidates:
            bottom_border = height - bottom_candidates[0][0]
        else:
            bottom_border = 0
        
        # Check if the detected borders make sense for letterboxing
        # Letterboxed videos typically have symmetric borders
        border_ratio = top_border / bottom_border if bottom_border else float('inf')
        is_symmetric = 0.5 < border_ratio < 2.0 if bottom_border and top_border else False
        
        # Both borders should be significant
        has_significant_borders = top_border > 20 and bottom_border > 20
        
        # The content area should be a reasonable portion of the frame
        content_height = height - top_border - bottom_border
        reasonable_content = 0.3 < (content_height / height) < 0.95
        
        is_letterboxed = has_significant_borders and reasonable_content
        
        if is_letterboxed:
            logger.debug(f"Letterbox detection: top={top_border}, bottom={bottom_border}, content={content_height}/{height}")
            
            features = {
                "top_border": top_border,
                "bottom_border": bottom_border,
                "background_color": (0, 0, 0),
                "non_border_area": {
                    "x": 0,
                    "y": top_border,
                    "width": width,
                    "height": content_height
                }
            }
            return True, features
        
        return False, {}