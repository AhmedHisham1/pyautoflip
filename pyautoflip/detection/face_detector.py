import logging
import time
from typing import List, Dict, Any

import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger("autoflip.detection.face_detector")


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
            providers = ["CPUExecutionProvider"]
            # Initialize the face analyzer
            cls._app = FaceAnalysis(
                name=model_name, providers=providers, allowed_modules=["detection"]
            )
            cls._app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("FaceAnalysis model loaded successfully")
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
                }
                detections.append(detection)

            logger.info(f"Detected {len(detections)} faces in {time.time() - time_start} seconds")
            return detections

        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []


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
