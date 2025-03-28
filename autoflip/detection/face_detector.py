import logging
from typing import List, Dict, Any

import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger("autoflip.detection.face_detector")

class FaceDetector:
    """
    Face detector using InsightFace.    
    """
    
    def __init__(
        self,
        model_name: str = "buffalo_s",
        min_confidence: float = 0.5,
        use_gpu: bool = True,
    ):
        """
        Initialize the face detector.
        
        Args:
            model_name: Name of InsightFace model to use (e.g. 'buffalo_l', 'buffalo_s')
            min_confidence: Minimum confidence threshold for detections
            use_gpu: Whether to use GPU for inference (if available)
        """
        self.model_name = model_name
        self.min_confidence = min_confidence
        
        # Configure providers based on GPU availability
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        # Initialize the face analyzer
        self.app = FaceAnalysis(name=model_name, providers=providers, allowed_modules=['detection'])
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
        
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
                
            return detections
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []
