import logging
import time
from typing import List, Dict, Any, Optional

import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

from .onnx_device import on_gpu, onnx_providers

logger = logging.getLogger("autoflip.detection.face_detector")

# 2d106det layout: the 20 mouth points, corners at 52 and 61 (upper lip
# 62-71, lower lip 53-60)
MOUTH_POINTS = slice(52, 72)
MOUTH_CORNERS = (52, 61)


def mouth_openness(landmarks) -> Optional[float]:
    """How open a mouth is: the lips' height at their middle over the mouth's width.

    From 106-point landmarks. Lip thickness counts, so a closed mouth isn't 0;
    talking shows up as this value moving over time.
    """
    if landmarks is None:
        return None
    points = np.asarray(landmarks, dtype=float)
    if points.shape[0] < MOUTH_POINTS.stop:
        return None
    left, right = points[MOUTH_CORNERS[0]], points[MOUTH_CORNERS[1]]
    width = float(np.linalg.norm(right[:2] - left[:2]))
    if width <= 0:
        return None
    mouth = points[MOUTH_POINTS]
    middle = mouth[np.abs(mouth[:, 0] - (left[0] + right[0]) / 2) <= 0.2 * width]
    if len(middle) < 2:
        return None
    return float(np.ptp(middle[:, 1]) / width)


class FaceDetector:
    """
    Face detector using InsightFace.
    """
    
    # Class-level variable to store the loaded model instance
    _app = None
    
    @classmethod
    def get_face_analyzer(cls, model_name: str = "buffalo_s"):
        """Get or initialize the FaceAnalysis instance.
        
        Args:
            model_name: Name of InsightFace model to use
            
        Returns:
            FaceAnalysis instance
        """
        if cls._app is None:
            logger.info(f"Initializing InsightFace FaceAnalysis ({model_name}) for the first time...")
            providers = list(onnx_providers())
            # Detection, plus 106-point landmarks for mouth motion (who is
            # talking); packs without the landmark model just skip it
            cls._app = FaceAnalysis(
                name=model_name, providers=providers, allowed_modules=["detection", "landmark_2d_106"]
            )
            # A negative ctx_id puts every model back on the CPU
            cls._app.prepare(ctx_id=0 if on_gpu() else -1, det_size=(640, 640))
            detection = cls._app.models.get("detection")
            device = detection.session.get_providers()[0] if detection is not None else providers[0]
            logger.info(f"FaceAnalysis model loaded successfully on {device}")
        return cls._app

    def __init__(
        self,
        model_name: str = "buffalo_s",
        min_confidence: float = 0.4,
    ):
        """
        Initialize the face detector.

        Args:
            model_name: Name of InsightFace model to use (e.g. 'buffalo_l', 'buffalo_s')
            min_confidence: Minimum confidence threshold for detections
        """
        self.model_name = model_name
        self.min_confidence = min_confidence
        
        # Use class method to get or initialize the shared model
        self.app = self.get_face_analyzer(model_name)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a frame.

        Args:
            frame: Input image frame

        Returns:
            List of face detections, each containing:
            - x, y, width, height: Normalized coordinates (0-1)
            - confidence: Detection confidence score
            - mouth_open: mouth_openness() from the landmarks, or None
        """
        time_start = time.time()
        # Get frame dimensions for normalization
        height, width = frame.shape[:2]

        try:
            # Detect faces using InsightFace
            faces = self.app.get(frame)

            # Convert to our detection format with normalized coordinates
            detections = []
            for face in faces:
                # Get bounding box and confidence
                bbox = face.bbox
                confidence = face.det_score

                if confidence < self.min_confidence:
                    continue

                # Extract coordinates (x1, y1, x2, y2) from bbox
                x1, y1, x2, y2 = bbox

                # Convert to normalized coordinates (0-1) in (x, y, width, height) format
                detection = {
                    "x": x1 / width,
                    "y": y1 / height,
                    "width": (x2 - x1) / width,
                    "height": (y2 - y1) / height,
                    "confidence": confidence,
                    "mouth_open": mouth_openness(face.get("landmark_2d_106")),
                }
                detections.append(detection)

            logger.info(f"Detected {len(detections)} faces in {time.time() - time_start} seconds")
            return detections

        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []

    def mouth_openness_at(self, frame: np.ndarray, boxes) -> List[Optional[float]]:
        """mouth_openness() for faces already located in `frame`.

        boxes: (x, y, w, h) in the frame's pixels, e.g. a face's box from a
        nearby frame. Runs only the landmark model, no detection. None for a
        box without a result (or when the model pack has no landmarks).
        """
        model = self.app.models.get("landmark_2d_106")
        if model is None:
            return [None] * len(boxes)
        results = []
        for x, y, w, h in boxes:
            face = Face(bbox=np.array([x, y, x + w, y + h], dtype=np.float32))
            try:
                results.append(mouth_openness(model.get(frame, face)))
            except Exception as e:
                logger.debug(f"Landmarks failed for box {(x, y, w, h)}: {e}")
                results.append(None)
        return results


if __name__ == "__main__":
    # Simple test code to benchmark face detection
    import cv2
    import requests
    
    # Download test image
    url = "https://img.zeit.de/sport/2017-04/ancelotti-zidane/wide__1000x562"
    response = requests.get(url)
    # Convert to numpy array
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    # Resize for testing
    image = cv2.resize(image, (1280, 720))
    
    # Create detector and run multiple detections to test caching
    detector = FaceDetector()
    for i in range(10):
        time_start = time.time()
        faces = detector.detect(image)
        print(f"Time taken to detect faces: {time.time() - time_start} seconds")
    
    print(faces)
