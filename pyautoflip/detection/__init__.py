# ObjectDetector needs mediapipe, which ships no linux/arm64 wheels.
# Import it lazily so saliency-mode users (and arm64 dev machines) work
# without it; instantiating the stub explains what is missing.
try:
    from .mediapipe_object_detector import ObjectDetector
except ImportError as _mediapipe_err:
    _err = _mediapipe_err

    class ObjectDetector:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ObjectDetector requires mediapipe, which is not installed on "
                "this platform (no linux/arm64 wheels). Use detection_method="
                "'saliency', or run on linux/amd64 or macOS."
            ) from _err

from .face_detector import FaceDetector
from .shot_boundary import ShotBoundaryDetector

__all__ = ["ObjectDetector", "FaceDetector", "ShotBoundaryDetector"]
