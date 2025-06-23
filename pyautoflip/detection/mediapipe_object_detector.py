"""
Object detection using MediaPipe.
"""

import logging
from typing import List, Dict, Any
import os
import time
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Object detection using MediaPipe.
    """

    _MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def get_model(cls, model_asset_path: str = "efficientdet_lite0.tflite"):
        """Get or initialize the MediaPipe model instance.
        
        Args:
            model_asset_path: Path or name of the MediaPipe model asset. 
                              If relative, assumed to be in the same directory as this script.
        """
        if not os.path.isabs(model_asset_path):
            model_asset_path = os.path.join(cls._MODEL_DIR, model_asset_path)

        if not os.path.exists(model_asset_path):
            raise FileNotFoundError(f"Unable to find model file at {model_asset_path}")

        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.3)
        return vision.ObjectDetector.create_from_options(options)
    
    def __init__(self, model_asset_path: str = "efficientdet_lite0.tflite"):
        """Initialize the MediaPipe model instance.
        
        Args:
            model_asset_path: Path to the MediaPipe model asset
        """
        self.model = self.get_model(model_asset_path=model_asset_path)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in the frame.
        
        Args:
            frame: The frame to detect objects in (expects BGR format)

        Returns:
            List of detected objects with the following fields:
            - x, y, width, height: Normalized coordinates (0-1), top-left corner
            - class: Object class name
            - score: Detection score
        """
        start_time = time.time()
        # Convert the frame to MediaPipe Image format (assuming BGR input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe expects RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Perform object detection
        detection_result = self.model.detect(mp_image)
        
        detections_list = []
        img_height, img_width, _ = frame.shape

        for detection in detection_result.detections:
            # Each detection has bounding_box, categories, keypoints
            bbox = detection.bounding_box
            
            # Normalize coordinates if they aren't already (MediaPipe usually gives pixel values)
            # Convert origin from top-left to make coordinates relative to image dimensions
            norm_x = bbox.origin_x / img_width
            norm_y = bbox.origin_y / img_height
            norm_width = bbox.width / img_width
            norm_height = bbox.height / img_height

            # Get the top category (highest score)
            top_category = detection.categories[0]
            class_name = top_category.category_name
            score = top_category.score

            detections_list.append({
                "x": norm_x,
                "y": norm_y,
                "width": norm_width,
                "height": norm_height,
                "class": class_name, # Use category_name
                "score": score,      # Use score
            })
        
        logger.info(f"Detected {len(detections_list)} objects in {time.time() - start_time:.2f} seconds")
        return detections_list

# Example usage (optional, for testing)
if __name__ == "__main__":
    import requests

    # Define the expected model path relative to this script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_MODEL_NAME = "efficientdet_lite0.tflite"
    MODEL_PATH = os.path.join(SCRIPT_DIR, DEFAULT_MODEL_NAME)

    # Download test image
    url = "https://storage.googleapis.com/mediapipe-tasks/object_detector/cat_and_dog.jpg"
    response = requests.get(url)
    image_data = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        print("Error: Could not load image.")
    else:
        # Initialize detector
        # Make sure 'efficientdet_lite0.tflite' is downloaded or provide the correct path
        # You might need to download it to the script's directory:
        # wget -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/efficientdet_lite0.tflite -O {MODEL_PATH}
        
        try:
            detector = ObjectDetector(model_asset_path=DEFAULT_MODEL_NAME) # Use default name, path resolved internally
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Please download the model using:\nwget -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/efficientdet_lite0.tflite -O {MODEL_PATH}")
            exit() # Exit if model not found

        # Detect objects
        detections = detector.detect(image)
        print(f"Detected {len(detections)} objects:")
        for det in detections:
            print(det)

            # Draw bounding box for visualization (optional)
            x = int(det["x"] * image.shape[1])
            y = int(det["y"] * image.shape[0])
            width = int(det["width"] * image.shape[1])
            height = int(det["height"] * image.shape[0])
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            label = f"{det['class']}: {det['score']:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show image (optional)
        # cv2.imshow("Detections", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Save image (optional)
        cv2.imwrite("mediapipe_detections.jpg", image)
        print("Output image saved to mediapipe_detections.jpg")