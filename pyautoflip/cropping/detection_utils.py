import logging
from typing import Dict, List, Any

logger = logging.getLogger("autoflip.cropping.detection_utils")


class DetectionProcessor:
    """
    Processes and prioritizes object detections.

    This class handles processing of face and object detections to assign
    priorities based on object types, sizes, and confidence levels.
    """

    def __init__(self, object_weights: Dict[str, float] = None):
        """
        Initialize the detection processor.

        Args:
            object_weights: Dictionary mapping object classes to importance weights
        """
        # Default object weights if none provided
        self.object_weights = object_weights or {
            "face": 5.0,  # Highest priority - faces
            "person": 2.5,  # High priority - people
            "animal": 1.8,  # Medium-high priority - animals
            "text": 1.5,  # Medium-high priority - text overlays
            "default": 1.0,  # Default priority for other objects
        }

        logger.debug(
            f"Initialized DetectionProcessor with weights: {self.object_weights}"
        )

    def process_detections(
        self,
        face_detections: Dict[int, List[Dict[str, Any]]],
        object_detections: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Process and assign priorities to all face and object detections.

        Args:
            face_detections: Dictionary of face detections per frame
            object_detections: Dictionary of object detections per frame

        Returns:
            Dictionary of processed detections with priorities per frame
        """
        processed_detections = {}

        # Process all frames with detections
        frame_indices = set(face_detections.keys()) | set(object_detections.keys())

        for frame_idx in frame_indices:
            frame_processed = []

            # Process face detections for this frame
            for face in face_detections.get(frame_idx, []):
                processed_face = face.copy()

                # Base priority from object weights
                detection_class = "face"
                confidence = face.get("confidence", 0)

                # Calculate priority
                priority = self.object_weights.get(detection_class, 1.0)

                # Adjust priority based on size (if dimensions are available)
                if all(k in face for k in ["width", "height"]):
                    w = face["width"]
                    h = face["height"]
                    if w * h > 0.05:  # Detection is >5% of frame
                        priority *= 1.5

                # Adjust by confidence
                priority *= max(0.5, confidence)

                # Add priority to the detection dictionary
                processed_face["priority"] = priority
                processed_face["class"] = detection_class

                # Mark all faces as required content - this is the key fix
                processed_face["is_required"] = True

                # Add to frame detections
                frame_processed.append(processed_face)

            # Process object detections for this frame
            for obj in object_detections.get(frame_idx, []):
                processed_obj = obj.copy()

                # Get object class
                detection_class = obj.get("class", "").lower()
                confidence = obj.get("confidence", 0)

                # Calculate priority
                priority = self.object_weights.get(
                    detection_class, self.object_weights.get("default", 1.0)
                )

                # Adjust priority based on size (if dimensions are available)
                if all(k in obj for k in ["width", "height"]):
                    w = obj["width"]
                    h = obj["height"]
                    if w * h > 0.05:  # Detection is >5% of frame
                        priority *= 1.5

                # Adjust by confidence
                priority *= max(0.5, confidence)

                # Add priority to the detection dictionary
                processed_obj["priority"] = priority

                # Also mark person/human detections as required
                if detection_class == "person" or detection_class == "human":
                    processed_obj["is_required"] = True

                # Add to frame detections
                frame_processed.append(processed_obj)

            # Store processed detections for this frame
            processed_detections[frame_idx] = frame_processed

        return processed_detections

    def identify_talking_head(
        self,
        face_detections: Dict[int, List[Dict[str, Any]]],
        key_frame_indices: List[int],
    ) -> bool:
        """
        Identify if this is likely a talking head video.

        Args:
            face_detections: Dictionary of face detections per frame
            key_frame_indices: List of key frame indices

        Returns:
            Boolean indicating if this is likely a talking head video
        """
        # Count frames with faces near the center of the frame
        talking_head_count = 0
        for idx in key_frame_indices:
            faces = face_detections.get(idx, [])
            if faces:
                center_faces = 0
                for face in faces:
                    # Check if face is in center region of frame
                    face_center_x = face.get("x", 0) + face.get("width", 0) / 2
                    if 0.3 <= face_center_x <= 0.7:  # Center 40% of frame width
                        center_faces += 1

                if center_faces > 0:
                    talking_head_count += 1

        # Consider it a talking head if 60% of frames have centered faces
        is_talking_head = talking_head_count >= len(key_frame_indices) * 0.6

        if is_talking_head:
            logger.debug("Detected talking head video - optimizing for face framing")

        return is_talking_head

    def boost_talking_head_priorities(
        self,
        processed_detections: Dict[int, List[Dict[str, Any]]],
        boost_factor: float = 1.5,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Boost face priorities for talking head videos.

        Args:
            processed_detections: Dictionary of processed detections per frame
            boost_factor: Factor to multiply face priorities by

        Returns:
            Updated detections dictionary
        """
        boosted_detections = {}

        for frame_idx, detections in processed_detections.items():
            boosted_frame_detections = []

            for detection in detections:
                boosted_detection = detection.copy()

                # Boost priority for face detections
                if detection.get("class", "").lower() == "face":
                    boosted_detection["priority"] = (
                        detection.get("priority", 3.0) * boost_factor
                    )
                    # Also ensure the face is marked as required
                    boosted_detection["is_required"] = True

                boosted_frame_detections.append(boosted_detection)

            boosted_detections[frame_idx] = boosted_frame_detections

        return boosted_detections
