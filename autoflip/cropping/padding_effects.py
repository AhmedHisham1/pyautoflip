"""
Padding effects for autoflip video reframing.

This module provides the PaddingEffectGenerator class for applying padding effects
to maintain target aspect ratios when cropping video frames.
"""

import logging
from typing import Tuple

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
    
    def _fast_gaussian_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Apply a faster Gaussian blur for large kernel sizes using multi-stage blur.
        
        For large kernels, this performs multiple smaller blurs which is much faster
        than a single large kernel blur operation.
        
        Args:
            image: Input image
            kernel_size: Original kernel size
            
        Returns:
            Blurred image
        """
        # For small kernels, use standard Gaussian blur
        if kernel_size <= 75:
            # Ensure kernel size is odd
            kernel_size = kernel_size + (0 if kernel_size % 2 == 1 else 1)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
        # For large kernels, use multi-stage blur
        # Calculate sigma based on the original kernel size
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        
        # Apply multiple blurs with smaller kernels
        # The effect is nearly identical but much faster
        result = image.copy()
        
        # Apply pyramid downsampling for very large kernels (> 150)
        if kernel_size > 150:
            # Downsample
            downscale_factor = 2
            downsampled = cv2.resize(image, (image.shape[1] // downscale_factor, 
                                            image.shape[0] // downscale_factor))
            
            # Blur downsampled image (smaller kernel)
            downsampled_blur = cv2.GaussianBlur(
                downsampled, (51, 51), sigma / downscale_factor)
                
            # Upsample back to original size
            result = cv2.resize(downsampled_blur, (image.shape[1], image.shape[0]))
            
            # Apply a final small blur to smooth any artifacts
            result = cv2.GaussianBlur(result, (31, 31), sigma / 3)
        else:
            # For medium-large kernels, apply sequential blurs with smaller kernels
            # 3 blurs with kernel size ~= original kernel / 3 approximates the original blur
            small_kernel = 51  # Fixed size for medium blur
            for _ in range(3):
                result = cv2.GaussianBlur(result, (small_kernel, small_kernel), sigma / 3)
        
        return result
    
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
        background_color: Tuple[int, int, int] = None,
        blur_cv_size: int = 200,
        overlay_opacity: float = 0.6,
        background_contrast: float = 0.6
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
            blur_cv_size: Size of the blur kernel (default: 200, matching MediaPipe)
            overlay_opacity: Opacity of the darkening overlay (default: 0.6, matching MediaPipe)
            background_contrast: Contrast adjustment for background (default: 0.6, matching MediaPipe)
            
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
        
        if padding_method == "solid_color":
            # Fast path: Use solid color background with direct creation
            if background_color is None:
                background_color = (0, 0, 0)  # Default to black
            
            # Create canvas with solid color (faster than np.zeros + fill)
            canvas = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)
            
            # Resize cropped region to fit the foreground area (single operation)
            resized_crop = cv2.resize(crop_region, (foreground_width, foreground_height))
            
            # Embed the foreground into the canvas
            canvas[padding_y:padding_y+foreground_height, padding_x:padding_x+foreground_width] = resized_crop
            
            return canvas
        
        # Process blur padding - more complex case
        # Extract background region from original frame for blurring
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
        bg_region = frame[bg_y:bg_y+bg_height, bg_x:bg_x+bg_width]
        
        # Resize to target dimensions if needed (single operation)
        if bg_region.shape[:2] != (target_height, target_width):
            canvas = cv2.resize(bg_region, (target_width, target_height))
        else:
            canvas = bg_region.copy()
        
        # Apply blur to the padding areas using optimized method
        # First, resize cropped region (will need it later)
        resized_crop = cv2.resize(crop_region, (foreground_width, foreground_height))
        
        # Create a mask for the padding areas (all 255s)
        mask = np.ones((target_height, target_width), dtype=np.uint8) * 255
        
        # Set foreground area in mask to 0
        mask[padding_y:padding_y+foreground_height, padding_x:padding_x+foreground_width] = 0
        
        # Find contours of the mask to identify padding regions
        if is_vertical_padding:
            # For vertical padding, we only need top and bottom regions
            # Directly define regions without expensive contour operations
            top_mask = None
            bottom_mask = None
            
            if padding_y > 0:
                # Create top blur region
                top_region = canvas[:padding_y + blur_cv_size//4, :]
                if top_region.size > 0:
                    # Blur entire top region at once
                    blurred_top = self._fast_gaussian_blur(top_region, blur_cv_size)
                    canvas[:padding_y + blur_cv_size//4, :] = blurred_top
            
            if padding_y + foreground_height < target_height:
                # Create bottom blur region
                bottom_region = canvas[padding_y + foreground_height - blur_cv_size//4:, :]
                if bottom_region.size > 0:
                    # Blur entire bottom region at once
                    blurred_bottom = self._fast_gaussian_blur(bottom_region, blur_cv_size)
                    canvas[padding_y + foreground_height - blur_cv_size//4:, :] = blurred_bottom
        else:
            # For horizontal padding, we only need left and right regions
            if padding_x > 0:
                # Create left blur region
                left_region = canvas[:, :padding_x + blur_cv_size//4]
                if left_region.size > 0:
                    # Blur entire left region at once
                    blurred_left = self._fast_gaussian_blur(left_region, blur_cv_size)
                    canvas[:, :padding_x + blur_cv_size//4] = blurred_left
            
            if padding_x + foreground_width < target_width:
                # Create right blur region
                right_region = canvas[:, padding_x + foreground_width - blur_cv_size//4:]
                if right_region.size > 0:
                    # Blur entire right region at once
                    blurred_right = self._fast_gaussian_blur(right_region, blur_cv_size)
                    canvas[:, padding_x + foreground_width - blur_cv_size//4:] = blurred_right
        
        # Apply contrast adjustment and overlay in a single operation if possible
        if overlay_opacity > 0:
            # Vectorized operations for contrast and overlay
            # We combine contrast adjustment and overlay into a single formula:
            # result = background_contrast * canvas * (1-overlay_opacity)
            # This is mathematically equivalent to applying them separately
            combined_factor = background_contrast * (1 - overlay_opacity)
            canvas = cv2.convertScaleAbs(canvas, alpha=combined_factor, beta=0)
        else:
            # Just apply contrast adjustment
            canvas = cv2.convertScaleAbs(canvas, alpha=background_contrast, beta=0)
        
        # Embed the foreground into the canvas (already resized earlier)
        canvas[padding_y:padding_y+foreground_height, padding_x:padding_x+foreground_width] = resized_crop
        
        return canvas 