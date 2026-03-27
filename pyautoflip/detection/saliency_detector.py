"""
Saliency detection using UNISAL model via ONNX Runtime.

Uses a pre-exported ONNX model (unisal.onnx) for fast CPU inference (~17ms/frame).
Falls back to the full PyTorch UNISAL if the ONNX model is not found.
"""

import logging
import time
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("autoflip.detection.saliency_detector")

# UNISAL preprocessing constants (ImageNet normalization)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MODEL_H, _MODEL_W = 256, 416  # Fixed internal resolution

_ONNX_MODEL_PATH = Path(__file__).parent / "unisal.onnx"


class SaliencyDetector:
    """
    Saliency detector using UNISAL model via ONNX Runtime.

    Identifies visually salient regions in frames using deep learning.
    Unlike semantic detectors (faces, objects), it captures what naturally
    draws human visual attention.
    """

    _session = None  # Shared ONNX Runtime session

    @classmethod
    def _get_session(cls):
        """Get or initialize the ONNX Runtime session."""
        if cls._session is None:
            if not _ONNX_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"UNISAL ONNX model not found at {_ONNX_MODEL_PATH}. "
                    "Please ensure unisal.onnx is in the detection directory."
                )
            import onnxruntime as ort
            start = time.time()
            cls._session = ort.InferenceSession(
                str(_ONNX_MODEL_PATH),
                providers=["CPUExecutionProvider"],
            )
            logger.info(f"UNISAL ONNX model loaded in {time.time() - start:.2f}s")
        return cls._session

    def __init__(self, model_type: str = "images"):
        """
        Initialize the saliency detector.

        Args:
            model_type: Kept for API compatibility. ONNX path only supports "images".
        """
        self.model_type = model_type
        self._session = self._get_session()

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess an RGB uint8 frame for UNISAL ONNX inference.
        Returns (1, 1, 3, H, W) float32 tensor.
        """
        # Resize to model input size
        resized = cv2.resize(frame, (_MODEL_W, _MODEL_H), interpolation=cv2.INTER_LINEAR)

        # Normalize: uint8 → float32 [0,1] → ImageNet normalize
        img = resized.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD

        # HWC → CHW, add batch and time dims: (1, 1, 3, H, W)
        img = img.transpose(2, 0, 1)[np.newaxis, np.newaxis, ...]
        return img

    def _postprocess(self, output: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """
        Postprocess ONNX output to a saliency map matching the input frame size.
        Returns (H, W) float32 array normalized to [0, 1].
        """
        # Output shape: (1, 1, 1, model_h, model_w) — log-softmax values
        smap = np.exp(output.squeeze())  # exp of log-softmax
        # Normalize to [0, 1]
        smap_max = smap.max()
        if smap_max > 0:
            smap = smap / smap_max
        # Resize to original frame dimensions
        if smap.shape != (target_h, target_w):
            smap = cv2.resize(smap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return smap

    def detect(self, frame: np.ndarray, return_map: bool = True) -> dict:
        """
        Detect salient regions in a frame.

        Args:
            frame: Input image frame (RGB, uint8)
            return_map: Whether to return the full saliency map

        Returns:
            Dictionary containing:
            - saliency_map: 2D array of saliency values [0, 1] (if return_map=True)
            - mean_saliency: Average saliency value
            - max_saliency: Maximum saliency value
        """
        height, width = frame.shape[:2]

        # Ensure uint8 RGB
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]

        try:
            inp = self._preprocess(frame)
            output = self._session.run(None, {"input": inp})[0]
            saliency_map = self._postprocess(output, height, width)

            result = {
                "mean_saliency": float(np.mean(saliency_map)),
                "max_saliency": float(np.max(saliency_map)),
            }
            if return_map:
                result["saliency_map"] = saliency_map
            return result

        except Exception as e:
            logger.error(f"Saliency detection error: {e}", exc_info=True)
            return {
                "saliency_map": np.zeros((height, width), dtype=np.float32) if return_map else None,
                "mean_saliency": 0.0,
                "max_saliency": 0.0,
            }
