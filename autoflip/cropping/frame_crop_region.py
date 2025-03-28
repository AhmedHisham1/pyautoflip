import logging
import math
from typing import Dict, List, Tuple, Any

from autoflip.cropping.types import CoverageType

logger = logging.getLogger("autoflip.cropping.frame_crop_region")


class FrameCropRegionComputer:
    """
    Computes optimal crop regions based on salient detections.
    
    Similar to MediaPipe's FrameCropRegionComputer, this class determines the
    best crop window that includes required regions and tries to cover as many
    non-required regions as possible.
    """
    
    def __init__(
        self,
        target_width: int,
        target_height: int,
        non_required_region_min_coverage_fraction: float = 0.7
    ):
        """
        Initialize the frame crop region computer.
        
        Args:
            target_width: Target width for the crop window
            target_height: Target height for the crop window
            non_required_region_min_coverage_fraction: Minimum fraction of non-required
                regions that must be covered (0.0-1.0)
        """
        self.target_width = target_width
        self.target_height = target_height
        self.non_required_region_min_coverage_fraction = non_required_region_min_coverage_fraction
        
        logger.debug(f"Initialized FrameCropRegionComputer with target dimensions: "
                    f"{target_width}x{target_height}")
        
    def expand_segment_under_constraint(
        self,
        segment_to_add: Tuple[int, int],
        base_segment: Tuple[int, int],
        max_length: int,
        non_required_region_min_coverage_fraction: float = None
    ) -> Tuple[Tuple[int, int], CoverageType]:
        """
        Expand a segment under constraints, similar to MediaPipe's ExpandSegmentUnderConstraint.
        
        Args:
            segment_to_add: (start, end) of segment to add
            base_segment: (start, end) of base segment
            max_length: Maximum length constraint
            non_required_region_min_coverage_fraction: Minimum coverage fraction for non-required regions
            
        Returns:
            Tuple containing:
            - combined_segment: (start, end) of resulting segment
            - cover_type: Coverage type enum value
        """
        if non_required_region_min_coverage_fraction is None:
            non_required_region_min_coverage_fraction = self.non_required_region_min_coverage_fraction
            
        segment_to_add_start, segment_to_add_end = segment_to_add
        base_segment_start, base_segment_end = base_segment
        
        # Validate inputs
        if segment_to_add_end < segment_to_add_start:
            logger.error("Invalid segment to add")
            return base_segment, CoverageType.NOT_COVERED
            
        if base_segment_end < base_segment_start:
            logger.error("Invalid base segment")
            return base_segment, CoverageType.NOT_COVERED
            
        base_length = base_segment_end - base_segment_start
        if base_length > max_length:
            logger.error("Base segment length exceeds max length")
            return base_segment, CoverageType.NOT_COVERED
            
        # Calculate maximum amount that can be left out while maintaining minimum coverage
        segment_to_add_length = segment_to_add_end - segment_to_add_start
        max_leftout_amount = int(math.ceil((1.0 - non_required_region_min_coverage_fraction) * 
                                segment_to_add_length / 2))
        
        min_coverage_segment_to_add_start = segment_to_add_start + max_leftout_amount
        min_coverage_segment_to_add_end = segment_to_add_end - max_leftout_amount
        
        # Calculate combined segments
        combined_segment_start = min(segment_to_add_start, base_segment_start)
        combined_segment_end = max(segment_to_add_end, base_segment_end)
        
        min_coverage_combined_segment_start = min(min_coverage_segment_to_add_start, base_segment_start)
        min_coverage_combined_segment_end = max(min_coverage_segment_to_add_end, base_segment_end)
        
        # Determine coverage type
        cover_type = CoverageType.NOT_COVERED
        
        if (combined_segment_end - combined_segment_start) <= max_length:
            # Can fully cover the segment
            cover_type = CoverageType.FULLY_COVERED
            combined_segment = (combined_segment_start, combined_segment_end)
        elif (min_coverage_combined_segment_end - min_coverage_combined_segment_start) <= max_length:
            # Can partially cover the segment (meeting min coverage requirements)
            cover_type = CoverageType.PARTIALLY_COVERED
            combined_segment = (min_coverage_combined_segment_start, min_coverage_combined_segment_end)
        else:
            # Cannot cover even the minimum required portion
            cover_type = CoverageType.NOT_COVERED
            combined_segment = base_segment
            
        return combined_segment, cover_type
        
    def expand_rect_under_constraints(
        self,
        rect_to_add: Tuple[int, int, int, int],
        max_width: int,
        max_height: int,
        base_rect: Tuple[int, int, int, int],
        non_required_region_min_coverage_fraction: float = None
    ) -> Tuple[Tuple[int, int, int, int], CoverageType]:
        """
        Expand a rectangle under constraints, similar to MediaPipe's ExpandRectUnderConstraints.
        
        Args:
            rect_to_add: (x, y, width, height) of rectangle to add
            max_width: Maximum width constraint
            max_height: Maximum height constraint
            base_rect: (x, y, width, height) of base rectangle
            non_required_region_min_coverage_fraction: Minimum coverage fraction for non-required regions
            
        Returns:
            Tuple containing:
            - combined_rect: (x, y, width, height) of resulting rectangle
            - cover_type: Coverage type enum value
        """
        if non_required_region_min_coverage_fraction is None:
            non_required_region_min_coverage_fraction = self.non_required_region_min_coverage_fraction
            
        # Validate base rect
        base_x, base_y, base_width, base_height = base_rect
        if base_width > max_width or base_height > max_height:
            logger.error("Base rect already exceeds target size")
            return base_rect, CoverageType.NOT_COVERED
            
        # Extract coordinates
        rect_to_add_x, rect_to_add_y, rect_to_add_width, rect_to_add_height = rect_to_add
        rect_to_add_right = rect_to_add_x + rect_to_add_width
        rect_to_add_bottom = rect_to_add_y + rect_to_add_height
        
        base_rect_right = base_x + base_width
        base_rect_bottom = base_y + base_height
        
        # Expand in horizontal and vertical directions separately
        horizontal_segment = (rect_to_add_x, rect_to_add_right)
        base_horizontal_segment = (base_x, base_rect_right)
        horizontal_combined_segment, horizontal_cover_type = self.expand_segment_under_constraint(
            horizontal_segment, base_horizontal_segment, max_width, non_required_region_min_coverage_fraction
        )
        
        vertical_segment = (rect_to_add_y, rect_to_add_bottom)
        base_vertical_segment = (base_y, base_rect_bottom)
        vertical_combined_segment, vertical_cover_type = self.expand_segment_under_constraint(
            vertical_segment, base_vertical_segment, max_height, non_required_region_min_coverage_fraction
        )
        
        # Combine results
        if horizontal_cover_type == CoverageType.NOT_COVERED or vertical_cover_type == CoverageType.NOT_COVERED:
            # Can't cover the rectangle
            return base_rect, CoverageType.NOT_COVERED
        else:
            # Create new rectangle from expanded segments
            new_x, new_right = horizontal_combined_segment
            new_y, new_bottom = vertical_combined_segment
            new_width = new_right - new_x
            new_height = new_bottom - new_y
            new_rect = (new_x, new_y, new_width, new_height)
            
            if (horizontal_cover_type == CoverageType.FULLY_COVERED and 
                vertical_cover_type == CoverageType.FULLY_COVERED):
                return new_rect, CoverageType.FULLY_COVERED
            else:
                return new_rect, CoverageType.PARTIALLY_COVERED
                
    def compute_frame_crop_region(
        self,
        fused_detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int
    ) -> Tuple[Tuple[int, int, int, int], float, bool]:
        """
        Compute optimal crop region for a frame based on detections.
        
        Similar to MediaPipe's ComputeFrameCropRegion.
        
        Args:
            fused_detections: List of fused detections with required/non-required flags
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Tuple containing:
            - crop_region: (x, y, width, height) of resulting crop region
            - crop_score: Score for the crop region
            - required_regions_covered: Whether all required regions are covered
        """
        # Separate required and non-required detections
        required_detections = []
        non_required_detections = []
        
        for detection in fused_detections:
            if detection.get('is_required', False):
                required_detections.append(detection)
            else:
                non_required_detections.append(detection)
        
        # Log detection info for debugging
        logger.debug(f"Processing {len(fused_detections)} detections: {len(required_detections)} required, {len(non_required_detections)} non-required")
                
        crop_region_is_empty = True
        crop_region = (0, 0, 0, 0)  # x, y, width, height
        crop_region_score = 0.0
        
        # First handle required detections - union all required regions
        for detection in required_detections:
            detection_x = int(detection['x'] * frame_width)
            detection_y = int(detection['y'] * frame_height)
            detection_width = int(detection['width'] * frame_width)
            detection_height = int(detection['height'] * frame_height)
            detection_rect = (detection_x, detection_y, detection_width, detection_height)
            
            # Log detection details for debugging
            logger.debug(f"Required detection: x={detection_x}, y={detection_y}, w={detection_width}, h={detection_height}, " +
                        f"class={detection.get('class', 'unknown')}, priority={detection.get('priority', 0)}")
            
            if crop_region_is_empty:
                crop_region = detection_rect
                crop_region_is_empty = False
            else:
                # Calculate union
                x1 = min(crop_region[0], detection_rect[0])
                y1 = min(crop_region[1], detection_rect[1])
                x2 = max(crop_region[0] + crop_region[2], detection_rect[0] + detection_rect[2])
                y2 = max(crop_region[1] + crop_region[3], detection_rect[1] + detection_rect[3])
                crop_region = (x1, y1, x2 - x1, y2 - y1)
                
            # Update score
            detection_score = detection.get('priority', detection.get('confidence', 1.0))
            crop_region_score += detection_score
        
        # Log initial crop region based on required detections
        if not crop_region_is_empty:
            logger.debug(f"Initial crop region from required detections: x={crop_region[0]}, y={crop_region[1]}, " +
                        f"w={crop_region[2]}, h={crop_region[3]}, score={crop_region_score:.2f}")
                
        # Store required region info
        required_region = crop_region if not crop_region_is_empty else (0, 0, 0, 0)
        required_regions_covered = (crop_region_is_empty or 
                                  (crop_region[2] <= self.target_width and 
                                   crop_region[3] <= self.target_height))
                                   
        # Update target dimensions if required regions are larger
        # This is the key change to match MediaPipe's behavior
        target_width = self.target_width
        target_height = self.target_height
        if not crop_region_is_empty:
            original_target_width = target_width
            original_target_height = target_height
            target_width = max(target_width, crop_region[2])
            target_height = max(target_height, crop_region[3])
            
            if target_width != original_target_width or target_height != original_target_height:
                logger.info(f"Adjusted target dimensions from {original_target_width}x{original_target_height} to {target_width}x{target_height} to fit required regions")
                
        # Now try to fit non-required regions
        num_covered = 0
        
        for detection in non_required_detections:
            detection_x = int(detection['x'] * frame_width)
            detection_y = int(detection['y'] * frame_height)
            detection_width = int(detection['width'] * frame_width)
            detection_height = int(detection['height'] * frame_height)
            detection_rect = (detection_x, detection_y, detection_width, detection_height)
            
            if crop_region_is_empty:
                # Start with a point at the center of this region
                center_x = detection_x + detection_width // 2
                center_y = detection_y + detection_height // 2
                empty_rect = (center_x, center_y, 0, 0)
                
                new_rect, cover_type = self.expand_rect_under_constraints(
                    detection_rect, target_width, target_height, empty_rect
                )
                
                if cover_type != CoverageType.NOT_COVERED:
                    crop_region = new_rect
                    crop_region_is_empty = False
            else:
                # Try to expand existing crop region
                new_rect, cover_type = self.expand_rect_under_constraints(
                    detection_rect, target_width, target_height, crop_region
                )
                
                crop_region = new_rect
                
            # Update score and coverage count
            if cover_type == CoverageType.FULLY_COVERED:
                num_covered += 1
                detection_score = detection.get('priority', detection.get('confidence', 1.0))
                crop_region_score += detection_score
                
        # Calculate fraction of non-required regions covered
        fraction_covered = (0.0 if len(non_required_detections) == 0 
                          else num_covered / len(non_required_detections))
                          
        # If region is still empty, use a center crop
        if crop_region_is_empty:
            center_x = frame_width // 2 - target_width // 2
            center_y = frame_height // 2 - target_height // 2
            crop_region = (center_x, center_y, target_width, target_height)
            
        # Ensure crop region stays within frame boundaries
        x, y, width, height = crop_region
        x = max(0, min(x, frame_width - width))
        y = max(0, min(y, frame_height - height))
        crop_region = (x, y, width, height)
            
        logger.debug(f"Final crop region: {crop_region}, score: {crop_region_score:.2f}, " +
                    f"required covered: {required_regions_covered}, " + 
                    f"non-required covered: {fraction_covered:.2f}")
            
        return crop_region, crop_region_score, required_regions_covered 