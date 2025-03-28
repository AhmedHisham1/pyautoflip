"""
Padding effects for autoflip video reframing.

This module provides the PaddingEffectGenerator class for applying padding effects
to maintain target aspect ratios when cropping video frames.
"""

import logging
from typing import Tuple, Optional

import cv2
import numpy as np

# Create module-level logger
logger = logging.getLogger("autoflip.cropping.padding_effects")


class PaddingEffectGenerator:
    """
    Generates padding effects to maintain target aspect ratio.
    
    This class handles various padding effects similar to MediaPipe's implementation,
    including blurred backgrounds and solid color padding.
    """
    
    def __init__(self):
        """Initialize the padding effect generator."""
        pass
    
    def apply_padding(
        self,
        frame: np.ndarray,
        crop_region: np.ndarray,
        x: int,
        y: int,
        crop_width: int,
        crop_height: int,
        target_width: int,
        target_height: int,
        padding_method: str = "blur",
        background_color: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        Apply MediaPipe-style padding to fit the target aspect ratio.
        
        Args:
            frame: Original frame
            crop_region: Cropped region
            x, y: Crop position
            crop_width, crop_height: Crop dimensions
            target_width, target_height: Target dimensions
            padding_method: Method for padding ("blur" or "solid_color")
            background_color: Optional background color for solid_color padding
            
        Returns:
            Padded frame
        """
        # Ensure input frames are uint8 - the expected format for OpenCV operations
        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)
        if crop_region.dtype != np.uint8:
            crop_region = cv2.convertScaleAbs(crop_region)
        
        # OpenCV uses BGR color order - we'll maintain this throughout
        
        # Determine if we need vertical or horizontal padding
        input_aspect_ratio = crop_width / crop_height
        target_aspect_ratio = target_width / target_height
        is_vertical_padding = input_aspect_ratio > target_aspect_ratio
        
        # Compute foreground dimensions (maintaining original content aspect ratio)
        if is_vertical_padding:
            # Vertical padding (bars on top/bottom)
            foreground_width = target_width
            foreground_height = int(foreground_width * crop_height / crop_width)
            padding_y = (target_height - foreground_height) // 2
            padding_x = 0
        else:
            # Horizontal padding (bars on left/right)
            foreground_height = target_height
            foreground_width = int(foreground_height * crop_width / crop_height)
            padding_x = (target_width - foreground_width) // 2
            padding_y = 0
            
        logger.debug(f"Padding: {'vertical' if is_vertical_padding else 'horizontal'}, "
                     f"foreground: {foreground_width}x{foreground_height}, "
                     f"padding: x={padding_x}, y={padding_y}")
        
        # Create a canvas for the output - explicitly using uint8 for consistency
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        if padding_method == "blur":
            # Extract background region from original frame for blurring
            # This follows MediaPipe's approach:
            
            # For vertical padding, take a center crop of the original frame
            if is_vertical_padding:
                bg_x = max(0, x + (crop_width - target_width) // 2)
                bg_y = y
                bg_width = min(target_width, frame.shape[1] - bg_x)
                bg_height = min(target_height, frame.shape[0] - bg_y)
            else:
                bg_x = x
                bg_y = max(0, y + (crop_height - target_height) // 2)
                bg_width = min(target_width, frame.shape[1] - bg_x)
                bg_height = min(target_height, frame.shape[0] - bg_y)
            
            # Extract background from original frame
            bg_region = frame[
                bg_y:bg_y+bg_height, 
                bg_x:bg_x+bg_width
            ]
            
            # Resize to target dimensions if needed
            if bg_region.shape[:2] != (target_height, target_width):
                bg_region = cv2.resize(bg_region, (target_width, target_height))
            
            # Copy background to canvas
            canvas = bg_region.copy()
            
            # Apply blur to the padding areas
            if is_vertical_padding:
                # Blur top area
                blur_kernel_size = 101 if target_height > 720 else 51
                blur_kernel_size = blur_kernel_size + (0 if blur_kernel_size % 2 == 1 else 1)  # Ensure odd
                
                # Calculate blur regions
                top_region = canvas[:padding_y + blur_kernel_size//2, :]
                bottom_region = canvas[padding_y + foreground_height - blur_kernel_size//2:, :]
                
                # Apply blur if regions exist
                if top_region.size > 0:
                    canvas[:padding_y + blur_kernel_size//2, :] = cv2.GaussianBlur(
                        top_region, (blur_kernel_size, blur_kernel_size), 0)
                
                if bottom_region.size > 0:
                    canvas[padding_y + foreground_height - blur_kernel_size//2:, :] = cv2.GaussianBlur(
                        bottom_region, (blur_kernel_size, blur_kernel_size), 0)
            else:
                # Blur left and right areas
                blur_kernel_size = 101 if target_width > 720 else 51
                blur_kernel_size = blur_kernel_size + (0 if blur_kernel_size % 2 == 1 else 1)  # Ensure odd
                
                # Calculate blur regions
                left_region = canvas[:, :padding_x + blur_kernel_size//2]
                right_region = canvas[:, padding_x + foreground_width - blur_kernel_size//2:]
                
                # Apply blur if regions exist
                if left_region.size > 0:
                    canvas[:, :padding_x + blur_kernel_size//2] = cv2.GaussianBlur(
                        left_region, (blur_kernel_size, blur_kernel_size), 0)
                
                if right_region.size > 0:
                    canvas[:, padding_x + foreground_width - blur_kernel_size//2:] = cv2.GaussianBlur(
                        right_region, (blur_kernel_size, blur_kernel_size), 0)
            
            # Apply contrast adjustment to background (0.6 is a good value)
            background_contrast = 0.8
            
            # Convert to float32 for multiplication, then back to uint8
            canvas = cv2.convertScaleAbs(canvas, alpha=background_contrast, beta=0)
            
            # Apply translucent black overlay for dimming (opacity 0.3)
            overlay_opacity = 0.3
            overlay = np.zeros_like(canvas)
            canvas = cv2.addWeighted(overlay, overlay_opacity, canvas, 1 - overlay_opacity, 0)
            
        elif padding_method == "solid_color":
            # Use solid color background
            if background_color:
                canvas[:] = background_color
            else:
                # Default to black
                canvas[:] = (0, 0, 0)
        
        # Resize cropped region to fit the foreground area
        resized_crop = cv2.resize(crop_region, (foreground_width, foreground_height))
        
        # Ensure resized_crop is also uint8
        if resized_crop.dtype != np.uint8:
            resized_crop = cv2.convertScaleAbs(resized_crop)
        
        # Embed the foreground into the canvas
        canvas[padding_y:padding_y+foreground_height, padding_x:padding_x+foreground_width] = resized_crop
        
        # Final check to ensure the output is uint8 for OpenCV compatibility
        if canvas.dtype != np.uint8:
            logger.warning(f"Converting canvas from {canvas.dtype} to uint8 for VideoWriter compatibility")
            if np.issubdtype(canvas.dtype, np.floating):
                # Scale float values to 0-255 range and convert using proper OpenCV function
                canvas = cv2.convertScaleAbs(canvas, alpha=255.0)
            else:
                # For other types, convert using OpenCV function
                canvas = cv2.convertScaleAbs(canvas)
        
        return canvas 