"""
Saliency-based scene cropper for video reframing.

Uses UNISAL saliency maps + InsightFace face detection to determine
optimal crop windows. Supports split-screen for multi-face scenes.
"""

import logging
from typing import List, Dict, Tuple, Any, Optional

import cv2
import numpy as np

from pyautoflip.cropping.camera_motion import CameraMotionHandler

logger = logging.getLogger("autoflip.cropping.saliency_cropper")

FACE_WEIGHT = 2.0       # Saliency boost for face regions
WIDE_CROP_FACTOR = 1.30  # 30% wider crop when saliency exceeds 9:16 strip
PROCESSING_WIDTH = 640    # Downscale width for saliency/face processing


class SaliencyCropper:
    """
    Coordinates saliency-based video scene cropping.

    Pipeline per scene:
    1. Compute saliency maps (UNISAL) on downscaled frames
    2. Detect faces (InsightFace) with size filtering
    3. Build composite saliency (saliency + face regions)
    4. Extract bbox + center of mass per frame
    5. Determine per-scene crop width (narrow or wide)
    6. Compute fixed-width crop windows centered on CoM
    7. Stabilize with CameraMotionHandler (STATIONARY/PANNING/TRACKING)
    8. Apply padding when crop is wider than target AR
    9. Split-screen fallback when faces can't fit in one crop
    """

    def __init__(
        self,
        target_aspect_ratio: float,
        motion_threshold: float = 0.5,
        padding_method: str = "blur",
        max_frames_per_scene: int = 5,
        min_face_fraction: float = 0.03,
    ):
        self.target_aspect_ratio = target_aspect_ratio
        self.motion_threshold = motion_threshold
        self.padding_method = padding_method
        self.max_frames_per_scene = max_frames_per_scene
        self.min_face_fraction = min_face_fraction

        self.camera_motion_handler = CameraMotionHandler(
            motion_threshold=motion_threshold, smoothing_window=30
        )

        # Lazy-loaded detectors
        self._saliency_detector = None
        self._face_detector = None

    @property
    def saliency_detector(self):
        if self._saliency_detector is None:
            from pyautoflip.detection.saliency_detector import SaliencyDetector
            self._saliency_detector = SaliencyDetector(model_type="images")
        return self._saliency_detector

    @property
    def face_detector(self):
        if self._face_detector is None:
            from pyautoflip.detection.face_detector import FaceDetector
            self._face_detector = FaceDetector()
        return self._face_detector

    # ─── Core Processing ──────────────────────────────────────────────────

    def process_scene(
        self,
        frames: List[np.ndarray],
        frame_count: int,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Process a scene's sampled frames and return relative crop windows
        for all frames in the scene.

        Args:
            frames: Sampled key frames (BGR, full resolution)
            frame_count: Total number of frames in the scene

        Returns:
            List of (x_rel, y_rel, w_rel, h_rel) crop windows for all frames
        """
        if not frames:
            return []

        frame_h, frame_w = frames[0].shape[:2]

        # Downscale for processing
        small_frames = self._downscale_frames(frames)
        sh, sw = small_frames[0].shape[:2]

        # Step 1: Compute saliency maps
        saliency_maps = self._compute_saliency_maps(small_frames)

        # Step 2: Detect faces (size-filtered)
        all_faces = self._detect_faces(small_frames)

        # Step 3: Composite saliency + faces → bbox + CoM per frame
        raw_bboxes, raw_coms = self._compute_saliency_data(
            saliency_maps, all_faces, sh, sw
        )

        # Step 4: Per-scene crop width
        target_aspect_tuple = self._aspect_ratio_to_tuple()
        crop_w = compute_scene_crop_width(raw_bboxes, sw, sh, target_aspect_tuple)

        # Step 5: Raw crop windows (on small frame coords)
        raw_crops = [
            compute_crop_window(com, sw, sh, crop_w) for com in raw_coms
        ]

        # Step 6: Stabilize with CameraMotionHandler
        smoothed_crops = self._stabilize_crops(raw_crops)

        # Step 7: Interpolate to all frames and convert to relative coords
        key_indices = list(range(len(smoothed_crops)))
        interpolated = self.camera_motion_handler.interpolate_crop_windows(
            [(c, 1.0) for c in smoothed_crops], key_indices, frame_count,
            self.camera_motion_handler.select_camera_motion_mode(
                [(c, 1.0) for c in smoothed_crops]
            )
        )

        # Convert to relative coordinates (normalized 0-1)
        rel_windows = []
        for x, y, w, h in interpolated:
            rel_windows.append((x / sw, y / sh, w / sw, h / sh))

        # Check for split-screen need (on small frame coords)
        self._split_faces = self._check_split_screen(all_faces, sw, sh, target_aspect_tuple)

        return rel_windows

    def needs_split_screen(self) -> bool:
        """Check if the last processed scene needs split-screen rendering."""
        return self._split_faces is not None

    def apply_crop_window(
        self,
        frame: np.ndarray,
        crop_window: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """
        Apply a crop window to a frame with padding if needed.

        Args:
            frame: Full-resolution BGR frame
            crop_window: (x_rel, y_rel, w_rel, h_rel) normalized 0-1

        Returns:
            Cropped and padded frame at target aspect ratio
        """
        frame_h, frame_w = frame.shape[:2]
        x = int(crop_window[0] * frame_w)
        y = int(crop_window[1] * frame_h)
        w = int(crop_window[2] * frame_w)
        h = int(crop_window[3] * frame_h)

        # Ensure even dimensions
        w -= w % 2
        h -= h % 2

        # Clamp
        x = max(0, min(x, frame_w - w))
        y = max(0, min(y, frame_h - h))

        crop = (x, y, w, h)
        target_tuple = self._aspect_ratio_to_tuple()

        return apply_padding_to_crop(frame, crop, target_tuple, self.padding_method)

    def apply_split_screen(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply split-screen rendering if faces are far apart.
        Returns the split-screen frame or None if not applicable.
        """
        if self._split_faces is None:
            return None

        frame_h, frame_w = frame.shape[:2]
        target_tuple = self._aspect_ratio_to_tuple()

        # Scale face centers from normalized to pixel coords
        face_rects_px = []
        for cx_n, cy_n in self._split_faces:
            # Create approximate face rects from centers for render_split_screen
            # The render function only needs centers, but find_split_faces returns them
            pass

        return render_split_screen_from_centers(
            frame, self._split_faces, target_tuple
        )

    # ─── Internal Methods ─────────────────────────────────────────────────

    def _aspect_ratio_to_tuple(self):
        """Convert float aspect ratio to (w, h) integer tuple."""
        # Common ratios
        ratio_map = {
            0.5625: (9, 16),
            1.0: (1, 1),
            0.75: (3, 4),
            1.7778: (16, 9),
        }
        for r, t in ratio_map.items():
            if abs(self.target_aspect_ratio - r) < 0.01:
                return t
        # Fallback: approximate
        return (int(self.target_aspect_ratio * 16), 16)

    def _downscale_frames(self, frames):
        """Downscale frames to processing width."""
        small = []
        for f in frames:
            h, w = f.shape[:2]
            if w > PROCESSING_WIDTH:
                scale = PROCESSING_WIDTH / w
                small.append(cv2.resize(f, (PROCESSING_WIDTH, int(h * scale))))
            else:
                small.append(f)
        return small

    def _compute_saliency_maps(self, frames):
        """Run UNISAL on each frame."""
        maps = []
        for frame in frames:
            # UNISAL expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.saliency_detector.detect(rgb, return_map=True)
            sal_map = result["saliency_map"]
            h, w = frame.shape[:2]
            if sal_map.shape != (h, w):
                sal_map = cv2.resize(sal_map, (w, h), interpolation=cv2.INTER_LINEAR)
            maps.append(sal_map)
        return maps

    def _detect_faces(self, frames):
        """Detect faces on each frame with size filtering."""
        all_faces = []
        for frame in frames:
            h, w = frame.shape[:2]
            frame_area = h * w
            detections = self.face_detector.detect(frame)

            candidates = []
            for d in detections:
                fx = int(d["x"] * w)
                fy = int(d["y"] * h)
                fw = int(d["width"] * w)
                fh = int(d["height"] * h)
                candidates.append((fx, fy, fw, fh, fw * fh))

            if not candidates:
                all_faces.append([])
                continue

            max_area = max(c[4] for c in candidates)
            rects = []
            for fx, fy, fw, fh, area in candidates:
                if area < frame_area * self.min_face_fraction and area < max_area * 0.3:
                    continue
                rects.append((fx, fy, fw, fh))
            all_faces.append(rects)

        return all_faces

    def _compute_saliency_data(self, saliency_maps, all_faces, h, w):
        """Compute composite saliency → bbox + CoM for each frame."""
        bboxes, coms = [], []
        for i in range(len(saliency_maps)):
            composite = get_composite_mask(saliency_maps[i], all_faces[i])
            bbox, com = saliency_to_bbox(composite)
            bboxes.append(bbox)
            coms.append(com)
        return bboxes, coms

    def _stabilize_crops(self, raw_crops):
        """Apply camera motion stabilization."""
        if len(raw_crops) <= 1:
            return raw_crops

        scene_windows = [(c, 1.0) for c in raw_crops]
        mode = self.camera_motion_handler.select_camera_motion_mode(scene_windows)

        key_indices = list(range(len(raw_crops)))
        interpolated = self.camera_motion_handler.interpolate_crop_windows(
            scene_windows, key_indices, len(raw_crops), mode
        )
        smoothed = self.camera_motion_handler.smooth_trajectory(interpolated, mode)

        logger.debug(f"Scene stabilization: {mode.name}")
        return smoothed

    def _check_split_screen(self, all_faces, frame_w, frame_h, target_aspect):
        """Check if any frame in the scene has faces too far apart for one crop."""
        for faces in all_faces:
            result = find_split_faces(faces, frame_w, frame_h, target_aspect)
            if result is not None:
                return result
        return None


# ─── Standalone Functions ─────────────────────────────────────────────────────


def get_composite_mask(saliency_map, face_rects=None):
    """Combine saliency map with face detection regions."""
    composite = saliency_map.copy()
    if face_rects and len(face_rects) > 0:
        for (fx, fy, fw, fh) in face_rects:
            composite[fy:fy+fh, fx:fx+fw] = np.maximum(
                composite[fy:fy+fh, fx:fx+fw], FACE_WEIGHT
            )
    return composite


def saliency_to_bbox(composite_mask):
    """
    Convert saliency mask to a core bounding box + weighted center of mass.
    Uses weighted 10th-90th percentiles to avoid outlier patches.
    Returns (x, y, w, h) normalized 0-1, plus (cx, cy) center of mass.
    """
    h, w = composite_mask.shape
    binary = (composite_mask > 0.5).astype(np.uint8)

    if binary.sum() == 0:
        return (0.25, 0.25, 0.5, 0.5), (0.5, 0.5)

    ys, xs = np.where(binary > 0)
    weights = composite_mask[ys, xs]

    cx = np.average(xs, weights=weights) / w
    cy = np.average(ys, weights=weights) / h

    # Weighted percentiles for robust bbox
    sorted_x_idx = np.argsort(xs)
    sorted_xs = xs[sorted_x_idx]
    cum_w = np.cumsum(weights[sorted_x_idx])
    total_w = cum_w[-1]
    x_min = sorted_xs[np.searchsorted(cum_w, total_w * 0.10)] / w
    x_max = sorted_xs[np.searchsorted(cum_w, total_w * 0.90)] / w

    sorted_y_idx = np.argsort(ys)
    sorted_ys = ys[sorted_y_idx]
    cum_wy = np.cumsum(weights[sorted_y_idx])
    y_min = sorted_ys[np.searchsorted(cum_wy, cum_wy[-1] * 0.10)] / h
    y_max = sorted_ys[np.searchsorted(cum_wy, cum_wy[-1] * 0.90)] / h

    return (x_min, y_min, x_max - x_min, y_max - y_min), (cx, cy)


def compute_crop_window(center_of_mass, frame_w, frame_h, crop_w):
    """Compute a fixed-width crop window centered on the CoM."""
    cx_px = center_of_mass[0] * frame_w
    crop_x = int(cx_px - crop_w / 2)
    crop_x = max(0, min(crop_x, frame_w - crop_w))
    return (crop_x, 0, crop_w, frame_h)


def _make_even(n):
    """Ensure a dimension is even (for video encoding)."""
    return n - (n % 2)


def compute_scene_crop_width(scene_bboxes, frame_w, frame_h, target_aspect):
    """
    Compute fixed crop width for a scene (always even).
    Narrow (exact AR) when saliency fits, wide (+30%) when it doesn't.
    """
    aspect_w, aspect_h = target_aspect
    narrow_w = _make_even(int(frame_h * aspect_w / aspect_h))
    wide_w = _make_even(min(int(narrow_w * WIDE_CROP_FACTOR), frame_w))

    if not scene_bboxes:
        return min(narrow_w, frame_w)

    max_sal_w = max(bbox[2] * frame_w for bbox in scene_bboxes)
    return min(narrow_w, frame_w) if max_sal_w <= narrow_w else wide_w


def apply_padding_to_crop(frame_bgr, crop, target_aspect, method="blur"):
    """
    Extract crop region and apply padding to fit target aspect ratio.
    Optimized: stretches the crop to fill the output and darkens the padding
    bars in-place — no extra resizes needed.
    """
    cx, cy, cw, ch = crop
    crop_region = frame_bgr[cy:cy+ch, cx:cx+cw]

    aspect_w, aspect_h = target_aspect
    target_ratio = aspect_w / aspect_h
    content_ratio = cw / ch

    if abs(content_ratio - target_ratio) < 0.01:
        eh, ew = _make_even(crop_region.shape[0]), _make_even(crop_region.shape[1])
        return crop_region[:eh, :ew]

    out_w = _make_even(cw)
    out_h = _make_even(int(out_w / target_ratio))

    # Stretch the crop to fill the full output — fast single resize
    canvas = cv2.resize(crop_region, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    if content_ratio > target_ratio:
        # Wider than target → darken top/bottom padding bars
        fg_h = int(out_w * ch / cw)
        pad_y = (out_h - fg_h) // 2
        if pad_y > 0:
            canvas[:pad_y] = (canvas[:pad_y] * 0.15).astype(np.uint8)
            canvas[pad_y + fg_h:] = (canvas[pad_y + fg_h:] * 0.15).astype(np.uint8)
    else:
        # Taller than target → darken left/right padding bars
        fg_w = int(out_h * cw / ch)
        pad_x = (out_w - fg_w) // 2
        if pad_x > 0:
            canvas[:, :pad_x] = (canvas[:, :pad_x] * 0.15).astype(np.uint8)
            canvas[:, pad_x + fg_w:] = (canvas[:, pad_x + fg_w:] * 0.15).astype(np.uint8)

    return canvas


def find_split_faces(face_rects, frame_w, frame_h, target_aspect):
    """
    Check if 2+ faces are too far apart to fit in one crop.
    Returns [(cx, cy), (cx, cy)] normalized 0-1, or None.
    """
    if not face_rects or len(face_rects) < 2:
        return None

    crop_w = int(frame_h * target_aspect[0] / target_aspect[1])
    crop_w_norm = crop_w / frame_w

    face_centers = sorted(
        [((fx + fw / 2) / frame_w, (fy + fh / 2) / frame_h)
         for (fx, fy, fw, fh) in face_rects],
        key=lambda c: c[0]
    )
    left, right = face_centers[0], face_centers[-1]

    if right[0] - left[0] > crop_w_norm:
        return [left, right]
    return None


def render_split_screen_from_centers(frame, face_centers, target_aspect):
    """
    Render a 2-panel split-screen frame, each panel centered on a face.
    face_centers: [(cx_norm, cy_norm), (cx_norm, cy_norm)]
    """
    h, w = frame.shape[:2]
    aspect_w, aspect_h = target_aspect
    target_ratio = aspect_w / aspect_h

    out_w = int(h * target_ratio)
    out_h = h
    divider_h = 4
    panel_h = (out_h - divider_h) // 2
    panel_ratio = out_w / panel_h

    src_crop_w = min(int(h * target_ratio), w)
    src_crop_h = min(int(src_crop_w / panel_ratio), h)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y_cursor = 0

    for i, (cx_norm, cy_norm) in enumerate(face_centers):
        cx_px = int(cx_norm * w)
        cy_px = int(cy_norm * h)
        crop_x = max(0, min(cx_px - src_crop_w // 2, w - src_crop_w))
        crop_y = max(0, min(cy_px - src_crop_h // 2, h - src_crop_h))

        strip = frame[crop_y:crop_y + src_crop_h, crop_x:crop_x + src_crop_w]
        panel = cv2.resize(strip, (out_w, panel_h), interpolation=cv2.INTER_LANCZOS4)
        canvas[y_cursor:y_cursor + panel_h, :] = panel
        y_cursor += panel_h

        if i == 0:
            canvas[y_cursor:y_cursor + divider_h, :] = (40, 40, 40)
            y_cursor += divider_h

    return canvas
