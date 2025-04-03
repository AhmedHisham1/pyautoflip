import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger("autoflip.cropping.frame_crop_region")


class FrameCropRegionComputer:
    """
    Computes optimal crop regions based on salient detections.
    """

    def __init__(
        self,
        target_width: int,
        target_height: int,
        non_required_region_min_coverage_fraction: float = 0.7,
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
        self.non_required_region_min_coverage_fraction = (
            non_required_region_min_coverage_fraction
        )

        logger.debug(
            f"Initialized FrameCropRegionComputer with target dimensions: "
            f"{target_width}x{target_height}"
        )

    def compute_frame_crop_region(
        self,
        fused_detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[Tuple[int, int, int, int], float, bool]:
        """
        Compute optimal crop region for a frame based on detections.

        Args:
            fused_detections: List of fused detections with required/non-required flags
            frame_width: Width of the frame
            frame_height: Height of the frame

        Returns:
            Tuple containing:
            - crop_region: (x, y, width, height) of resulting crop region
            - crop_score: Score for the crop region
        """
        required_detections = [
            d for d in fused_detections if d.get("is_required", False)
        ]

        # Process required detections first
        (
            crop_region,
            crop_region_score,
            crop_region_is_empty,
        ) = self._process_required_detections(
            required_detections, frame_width, frame_height
        )

        # Handle empty region case and ensure boundaries
        if crop_region_is_empty:
            crop_region = self._create_center_crop(
                frame_width, frame_height, self.target_width, self.target_height
            )
        else:
            self._adjust_target_dimensions(crop_region)

        # Ensure crop region stays within frame boundaries
        crop_region = self._ensure_crop_within_frame(
            crop_region, frame_width, frame_height
        )

        logger.debug(
            f"Final crop region: {crop_region}, score: {crop_region_score:.2f}"
        )

        return crop_region, crop_region_score

    def _process_required_detections(
        self,
        required_detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[Tuple[int, int, int, int], float, bool]:
        """
        Process required detections and compute initial crop region.

        Args:
            required_detections: List of required detections
            frame_width: Width of the frame
            frame_height: Height of the frame

        Returns:
            Tuple containing:
            - crop_region: (x, y, width, height) of resulting crop region
            - crop_region_score: Score for the crop region
            - crop_region_is_empty: Whether the crop region is empty (no required detections)
        """
        crop_region_is_empty = True
        crop_region = (0, 0, 0, 0)  # x, y, width, height
        crop_region_score = 0.0

        # First handle required detections - union all required regions
        for detection in required_detections:
            detection_x = int(detection["x"] * frame_width)
            detection_y = int(detection["y"] * frame_height)
            detection_width = int(detection["width"] * frame_width)
            detection_height = int(detection["height"] * frame_height)
            detection_rect = (
                detection_x,
                detection_y,
                detection_width,
                detection_height,
            )

            # Log detection details for debugging
            logger.debug(
                f"Required detection: x={detection_x}, y={detection_y}, w={detection_width}, h={detection_height}, "
                + f"class={detection.get('class', 'unknown')}, priority={detection.get('priority', 0)}"
            )

            if crop_region_is_empty:
                crop_region = detection_rect
                crop_region_is_empty = False
            else:
                # Calculate union
                crop_region = self._compute_union_rect(crop_region, detection_rect)

            # Update score
            detection_score = detection.get(
                "priority", detection.get("confidence", 1.0)
            )
            crop_region_score += detection_score

        # Log initial crop region based on required detections
        if not crop_region_is_empty:
            logger.debug(
                f"Initial crop region from required detections: x={crop_region[0]}, y={crop_region[1]}, "
                + f"w={crop_region[2]}, h={crop_region[3]}, score={crop_region_score:.2f}"
            )

        return crop_region, crop_region_score, crop_region_is_empty

    def _compute_union_rect(
        self, rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Compute the union of two rectangles."""
        x1 = min(rect1[0], rect2[0])
        y1 = min(rect1[1], rect2[1])
        x2 = max(rect1[0] + rect1[2], rect2[0] + rect2[2])
        y2 = max(rect1[1] + rect1[3], rect2[1] + rect2[3])
        return (x1, y1, x2 - x1, y2 - y1)

    def _adjust_target_dimensions(
        self,
        crop_region: Tuple[int, int, int, int],
    ) -> Tuple[int, int]:
        """Adjust target dimensions if required regions are larger."""
        target_width = max(self.target_width, crop_region[2])
        target_height = max(self.target_height, crop_region[3])

        if (target_width != self.target_width) or (target_height != self.target_height):
            logger.debug(
                f"Adjusted target from {self.target_width}x{self.target_height} to {target_width}x{target_height} to fit required regions",
            )
            self.target_width = target_width
            self.target_height = target_height

    def _create_center_crop(
        self, frame_width: int, frame_height: int, target_width: int, target_height: int
    ) -> Tuple[int, int, int, int]:
        """Create a center crop when no detections are covered."""
        center_x = frame_width // 2 - target_width // 2
        center_y = frame_height // 2 - target_height // 2
        return (center_x, center_y, target_width, target_height)

    def _ensure_crop_within_frame(
        self,
        crop_region: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int, int, int]:
        """Ensure the crop region stays within frame boundaries."""
        x, y, width, height = crop_region

        # expand the crop region horizontally if it's smaller than the target width
        if width < self.target_width:
            logger.debug(f"crop w < target w: {width} < {self.target_width}")
            # expand the crop region horizontally to the target width
            x = max(0, x - (self.target_width - width) // 2)
            width = self.target_width
            # edge case: if the crop ends at the right edge of the frame, we need to move it to the left by the difference
            if (x + width) > frame_width:
                x = frame_width - width

        # expand the crop region vertically if it's smaller than the target height
        if height < self.target_height:
            logger.debug(f"crop h < target h: {height} < {self.target_height}")
            # expand the crop region vertically to the target height
            y = max(0, y - (self.target_height - height) // 2)
            height = self.target_height
            # edge case: if the crop ends at the bottom edge of the frame, we need to move it up by the difference
            if (y + height) > frame_height:
                y = frame_height - height

        # ensure the crop region stays within frame boundaries
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if (x + width) > frame_width:
            width = frame_width - x
        if (y + height) > frame_height:
            height = frame_height - y

        return (x, y, width, height)
