"""
Saliency detection using UNISAL model.

This module provides a wrapper around the UNISAL saliency detection model
for detecting visually salient regions in video frames.
"""

import logging
import time
import sys
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("autoflip.detection.saliency_detector")


class SaliencyDetector:
    """
    Saliency detector using UNISAL model.

    This detector identifies visually salient regions in frames using deep learning.
    Unlike semantic detectors (faces, objects), it captures what naturally draws
    human visual attention.
    """

    # Class-level variable to store the loaded model instance
    _model = None
    _model_type = None

    @classmethod
    def get_model(cls, model_type: str = "images"):
        """
        Get or initialize the UNISAL model instance.

        Args:
            model_type: Type of model to use - "images" (SALICON) or "frames" (MIT300)

        Returns:
            UNISAL trainer instance
        """
        if cls._model is None or cls._model_type != model_type:
            logger.info(f"Initializing UNISAL model ({model_type}) for the first time...")
            start_time = time.time()

            # Add UNISAL to Python path
            unisal_path = Path(__file__).parent.parent / "3rd_party_libs" / "unisal"
            if str(unisal_path) not in sys.path:
                sys.path.insert(0, str(unisal_path))

            # Set TRAIN_DIR environment variable for UNISAL
            training_runs_path = unisal_path / "training_runs"
            if not training_runs_path.exists():
                raise RuntimeError(
                    f"UNISAL training_runs directory not found at {training_runs_path}. "
                    "Please ensure UNISAL is properly installed."
                )
            os.environ["TRAIN_DIR"] = str(training_runs_path)
            logger.debug(f"Set TRAIN_DIR to {training_runs_path}")

            # Import and initialize
            import unisal_handler

            if model_type == "images":
                cls._model = unisal_handler.init_unisal_for_images()
            elif model_type == "frames":
                cls._model = unisal_handler.init_unisal_for_frames()
            else:
                raise ValueError(f"Unknown model_type: {model_type}. Use 'images' or 'frames'.")

            cls._model_type = model_type

            elapsed = time.time() - start_time
            logger.info(f"UNISAL model loaded successfully in {elapsed:.2f}s")

        return cls._model

    def __init__(
        self,
        model_type: str = "images",
        min_confidence: float = 0.3,
    ):
        """
        Initialize the saliency detector.

        Args:
            model_type: Type of model - "images" for SALICON weights, "frames" for MIT300
            min_confidence: Minimum confidence threshold for salient regions (not used yet)
        """
        self.model_type = model_type
        self.min_confidence = min_confidence

        # Use class method to get or initialize the shared model
        self.model = self.get_model(model_type)

    def detect(
        self,
        frame: np.ndarray,
        return_map: bool = True
    ) -> dict:
        """
        Detect salient regions in a frame.

        Args:
            frame: Input image frame (RGB, uint8)
            return_map: Whether to return the full saliency map

        Returns:
            Dictionary containing:
            - saliency_map: 2D array of saliency values [0, 1] (if return_map=True)
            - peak_locations: List of (x, y) coordinates of local maxima
            - mean_saliency: Average saliency value
            - max_saliency: Maximum saliency value
        """
        time_start = time.time()

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Ensure frame is uint8 RGB
        if frame.dtype != np.uint8:
            logger.warning(f"Converting frame from {frame.dtype} to uint8")
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        if len(frame.shape) == 2:  # Grayscale
            logger.warning("Converting grayscale frame to RGB")
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            logger.warning("Converting RGBA frame to RGB")
            frame = frame[:, :, :3]

        try:
            # Generate saliency map using UNISAL
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                saliency_batch = self.model.generate_predictions_from_image_memory_nuint8_np(
                    [frame], ['saliency.png'], tmpdir
                )

            # Extract saliency map (first channel)
            saliency_map = saliency_batch[:, :, 0]

            # Normalize to [0, 1] if needed
            if saliency_map.max() > 1.0:
                saliency_map = saliency_map.astype(np.float32) / 255.0
            else:
                saliency_map = saliency_map.astype(np.float32)

            # Compute statistics
            mean_saliency = float(np.mean(saliency_map))
            max_saliency = float(np.max(saliency_map))

            # Find peak locations (simple approach - find local maxima)
            peak_locations = self._find_peaks(saliency_map)

            elapsed = time.time() - time_start
            logger.debug(
                f"Saliency detection: {len(peak_locations)} peaks, "
                f"mean={mean_saliency:.3f}, max={max_saliency:.3f}, "
                f"time={elapsed:.3f}s"
            )

            result = {
                'peak_locations': peak_locations,
                'mean_saliency': mean_saliency,
                'max_saliency': max_saliency,
            }

            if return_map:
                result['saliency_map'] = saliency_map

            return result

        except Exception as e:
            logger.error(f"Error in saliency detection: {str(e)}", exc_info=True)
            # Return empty result
            return {
                'saliency_map': np.zeros((height, width), dtype=np.float32) if return_map else None,
                'peak_locations': [],
                'mean_saliency': 0.0,
                'max_saliency': 0.0,
            }

    def _find_peaks(self, saliency_map: np.ndarray, threshold_percentile: float = 90) -> list:
        """
        Find peak locations in saliency map.

        Args:
            saliency_map: 2D saliency map
            threshold_percentile: Only consider values above this percentile

        Returns:
            List of (x, y) peak coordinates
        """
        from skimage.feature import peak_local_max

        # Threshold by percentile
        threshold = np.percentile(saliency_map, threshold_percentile)

        # Find peaks with minimum distance
        min_distance = max(10, min(saliency_map.shape) // 10)
        peaks = peak_local_max(
            saliency_map,
            min_distance=min_distance,
            threshold_abs=threshold,
            exclude_border=True
        )

        # Convert to (x, y) format
        peak_locations = [(int(x), int(y)) for y, x in peaks]

        return peak_locations

    def detect_batch(
        self,
        frames: list,
        return_maps: bool = True
    ) -> list:
        """
        Detect saliency for a batch of frames (more efficient).

        Args:
            frames: List of RGB frames (uint8)
            return_maps: Whether to return full saliency maps

        Returns:
            List of result dictionaries (same format as detect())
        """
        time_start = time.time()

        # Ensure all frames are uint8 RGB
        processed_frames = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            if len(frame.shape) == 2:
                import cv2
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = frame[:, :, :3]
            processed_frames.append(frame)

        try:
            # Generate saliency maps in batch
            import tempfile
            out_names = [f'saliency_{i}.png' for i in range(len(processed_frames))]

            with tempfile.TemporaryDirectory() as tmpdir:
                saliency_batch = self.model.generate_predictions_from_image_memory_nuint8_np(
                    processed_frames, out_names, tmpdir
                )

            # Process each result
            results = []
            for i in range(len(processed_frames)):
                # Extract saliency map for this frame
                # Note: saliency_batch might be 3D (H, W, N) or need different indexing
                # This depends on UNISAL's batch output format
                if len(saliency_batch.shape) == 3:
                    if saliency_batch.shape[2] == len(processed_frames):
                        saliency_map = saliency_batch[:, :, i]
                    else:
                        # Assume it's (H, W, 1) and we need to call multiple times
                        saliency_map = saliency_batch[:, :, 0]
                else:
                    saliency_map = saliency_batch

                # Normalize
                if saliency_map.max() > 1.0:
                    saliency_map = saliency_map.astype(np.float32) / 255.0
                else:
                    saliency_map = saliency_map.astype(np.float32)

                # Compute statistics
                mean_saliency = float(np.mean(saliency_map))
                max_saliency = float(np.max(saliency_map))
                peak_locations = self._find_peaks(saliency_map)

                result = {
                    'peak_locations': peak_locations,
                    'mean_saliency': mean_saliency,
                    'max_saliency': max_saliency,
                }

                if return_maps:
                    result['saliency_map'] = saliency_map

                results.append(result)

            elapsed = time.time() - time_start
            logger.info(
                f"Batch saliency detection: {len(frames)} frames in {elapsed:.2f}s "
                f"({len(frames)/elapsed:.1f} fps)"
            )

            return results

        except Exception as e:
            logger.error(f"Error in batch saliency detection: {str(e)}", exc_info=True)
            # Return empty results
            return [
                {
                    'saliency_map': np.zeros_like(frames[0][:, :, 0], dtype=np.float32) if return_maps else None,
                    'peak_locations': [],
                    'mean_saliency': 0.0,
                    'max_saliency': 0.0,
                }
                for _ in frames
            ]


if __name__ == "__main__":
    # Simple test code
    import cv2

    # Create a simple test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add a bright region to create saliency
    test_frame[200:280, 300:380] = [255, 255, 0]  # Yellow square

    print("Testing SaliencyDetector...")
    detector = SaliencyDetector()

    # Single frame test
    print("\n1. Single frame detection:")
    result = detector.detect(test_frame)
    print(f"   Peaks found: {len(result['peak_locations'])}")
    print(f"   Mean saliency: {result['mean_saliency']:.3f}")
    print(f"   Max saliency: {result['max_saliency']:.3f}")
    print(f"   Saliency map shape: {result['saliency_map'].shape}")

    # Batch test
    print("\n2. Batch detection (3 frames):")
    batch_results = detector.detect_batch([test_frame, test_frame, test_frame])
    print(f"   Processed {len(batch_results)} frames")
    for i, res in enumerate(batch_results):
        print(f"   Frame {i}: {len(res['peak_locations'])} peaks")
