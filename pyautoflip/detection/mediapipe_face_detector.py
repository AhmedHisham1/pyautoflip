"""
Face detection using MediaPipe.
"""

import logging
import time
from typing import List, Dict, Any
import os

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger("autoflip.detection.mediapipe_face_detector")


class FaceDetector:
    """
    Face detection using MediaPipe.
    Uses the MediaPipe Face Detector task.
    https://developers.google.com/mediapipe/solutions/vision/face_detector
    """
    
    _MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
    _DEFAULT_MODEL_NAME = "blaze_face_short_range.tflite"

    @classmethod
    def get_model(cls, model_asset_path: str = _DEFAULT_MODEL_NAME):
        """Get or initialize the MediaPipe Face Detector model instance.
        
        Args:
            model_asset_path: Path or name of the MediaPipe model asset.
                              If relative, assumed to be in the same directory as this script.
                              Defaults to 'blaze_face_short_range.tflite'.
        """
        if not os.path.isabs(model_asset_path):
            model_asset_path = os.path.join(cls._MODEL_DIR, model_asset_path)

        if not os.path.exists(model_asset_path):
            raise FileNotFoundError(f"Unable to find model file at {model_asset_path}")

        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
        return vision.FaceDetector.create_from_options(options)
    
    def __init__(self, model_asset_path: str = _DEFAULT_MODEL_NAME):
        """Initialize the MediaPipe Face Detector instance.
        
        Args:
            model_asset_path: Path to the MediaPipe face detector model asset.
        """
        self.model = self.get_model(model_asset_path=model_asset_path)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in the frame.
        
        Args:
            frame: The frame to detect faces in (expects BGR format)

        Returns:
            List of detected faces with the following fields:
            - x, y, width, height: Normalized coordinates (0-1), top-left corner
            - confidence: Detection score
        """
        start_time = time.time()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.model.detect(mp_image)
        
        detections_list = []
        img_height, img_width, _ = frame.shape

        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                
                norm_x = bbox.origin_x / img_width
                norm_y = bbox.origin_y / img_height
                norm_width = bbox.width / img_width
                norm_height = bbox.height / img_height
                
                score = detection.categories[0].score if detection.categories else 0.0

                detections_list.append({
                    "x": norm_x,
                    "y": norm_y,
                    "width": norm_width,
                    "height": norm_height,
                    "confidence": score,
                })
            
        logger.info(f"Detected {len(detections_list)} faces in {time.time() - start_time:.4f} seconds")
        return detections_list

if __name__ == "__main__":
    import requests

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_MODEL_NAME = "blaze_face_short_range.tflite"
    MODEL_PATH = os.path.join(SCRIPT_DIR, DEFAULT_MODEL_NAME)

    # Download test image
    url = "https://cdn.vox-cdn.com/thumbor/68fLzaKp_0nzejMR-snix_kPm28=/0x0:3000x1917/1200x800/filters:focal(1260x719:1740x1199)/cdn.vox-cdn.com/uploads/chorus_image/image/70379346/174553165.0.jpg"
    try:
        response = requests.get(url, timeout=10) # Add timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading test image from {url}: {e}")
        exit()

    image_data = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        print("Error: Could not load image.")
    else:
        try:
            detector = FaceDetector(model_asset_path=DEFAULT_MODEL_NAME)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Please download the model using:\nwget -q https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite -O {MODEL_PATH}")
            exit()

        detections = detector.detect(image)
        print(f"Detected {len(detections)} faces:")
        for det in detections:
            print(det)

            x = int(det["x"] * image.shape[1])
            y = int(det["y"] * image.shape[0])
            width = int(det["width"] * image.shape[1])
            height = int(det["height"] * image.shape[0])
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            label = f"Face: {det['confidence']:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite("mediapipe_face_detections.jpg", image)
        print("Output image saved to mediapipe_face_detections.jpg")