import os
import logging
import tempfile
from typing import List, Dict, Any, Tuple

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("autoflip.detection.object_detector")

class ObjectDetector:
    """
    Object detector using Ultralytics YOLO.
    """
    
    # Map classes to more general categories
    CLASS_MAPPING = {
        "person": "person",
        "cat": "animal",
        "dog": "animal",
        "horse": "animal",
        "sheep": "animal",
        "cow": "animal",
        "elephant": "animal",
        "bear": "animal",
        "zebra": "animal",
        "giraffe": "animal",
        "bird": "animal",
        "car": "vehicle",
        "bicycle": "vehicle",
        "motorcycle": "vehicle",
        "airplane": "vehicle",
        "bus": "vehicle",
        "train": "vehicle",
        "truck": "vehicle",
        "boat": "vehicle"
    }
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        model_name: str = "yolo11n.pt",
    ):
        """
        Initialize the YOLO object detector.
        
        Args:
            confidence_threshold: Minimum confidence threshold for detections
            model_name: Name of the YOLO model to use
        """
        self.confidence_threshold = confidence_threshold
        
        # Create cache directory for models
        model_dir = os.path.join(tempfile.gettempdir(), "autoflip_models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Load YOLO model
        logger.info(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of object detections, each containing:
            - x, y, width, height: Normalized coordinates (0-1)
            - class: Object class
            - confidence: Detection confidence score
        """
        # Get frame dimensions
        (h, w) = frame.shape[:2]
        
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Process results
        detections = []
        if len(results) > 0:
            result = results[0]  # Get first result
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get confidence
                confidence = float(boxes.conf[i].item())
                
                # Get class name
                class_id = int(boxes.cls[i].item())
                class_name = result.names[class_id]
                
                # Map to category
                category = self.CLASS_MAPPING.get(class_name, "object")
                
                # Get normalized xywh coordinates
                x, y, width, height = boxes.xywhn[i].tolist()
                
                # Add detection to results
                detection = {
                    "x": x - width/2,  # YOLO returns center x, convert to top-left x
                    "y": y - height/2,  # YOLO returns center y, convert to top-left y
                    "width": width,
                    "height": height,
                    "class": category,
                    "confidence": confidence
                }
                detections.append(detection)
                
        return detections
