# from .object_detector import ObjectDetector
from .mediapipe_object_detector import ObjectDetector
from .face_detector import FaceDetector
from .shot_boundary import ShotBoundaryDetector

__all__ = ["ObjectDetector", "FaceDetector", "ShotBoundaryDetector"]
