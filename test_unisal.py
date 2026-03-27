"""
Test script to run UNISAL saliency on test.mp4 with scene-aware chunking.
Visualizes step-by-step: saliency → salient region bbox → crop window.
Same pipeline as test_autogaze.py but swaps AutoGaze for UNISAL saliency maps.
"""

import logging
logging.getLogger("autoflip").setLevel(logging.WARNING)

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from pyautoflip.detection.shot_boundary import ShotBoundaryDetector
from pyautoflip.detection.saliency_detector import SaliencyDetector


TARGET_ASPECT = (9, 16)  # width:height for the crop


# ─── Scene Detection ──────────────────────────────────────────────────────────


def detect_scenes(video_path: str):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    boundaries = ShotBoundaryDetector().detect(video_path)

    scenes = []
    prev = 0
    for b in boundaries:
        scenes.append((prev, b))
        prev = b
    scenes.append((prev, total_frames))

    print(f"Video: {total_frames} frames @ {fps:.1f}fps ({total_frames/fps:.1f}s)")
    print(f"Detected {len(scenes)} scenes")
    return scenes, total_frames, fps


# ─── Frame Loading ────────────────────────────────────────────────────────────


def load_scene_frames(video_path: str, scenes: list, max_per_scene: int = 16):
    cap = cv2.VideoCapture(video_path)
    all_frames, all_indices, scene_labels = [], [], []

    for scene_idx, (start, end) in enumerate(scenes):
        scene_len = end - start
        if scene_len <= 0:
            continue
        n = min(scene_len, max_per_scene)
        for idx in np.linspace(start, end - 1, n, dtype=int).tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                all_indices.append(idx)
                scene_labels.append(scene_idx)

    cap.release()
    print(f"Loaded {len(all_frames)} frames across {len(scenes)} scenes")
    return all_frames, all_indices, scene_labels


# ─── UNISAL Saliency Processing ──────────────────────────────────────────────


def compute_saliency_maps(frames):
    """Run UNISAL saliency detection on all frames. Returns list of (H, W) float32 maps."""
    print("Computing UNISAL saliency maps...")
    detector = SaliencyDetector(model_type="images")

    saliency_maps = []
    for frame in tqdm(frames, desc="UNISAL"):
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        result = detector.detect(frame, return_map=True)
        sal_map = result["saliency_map"]
        # Resize to frame size if needed
        h, w = frame.shape[:2]
        if sal_map.shape != (h, w):
            sal_map = cv2.resize(sal_map, (w, h), interpolation=cv2.INTER_LINEAR)
        saliency_maps.append(sal_map)

    return saliency_maps


# ─── Face Detection (MediaPipe BlazeFace) ─────────────────────────────────────

from pyautoflip.detection.mediapipe_face_detector import FaceDetector as MPFaceDetector

_face_detector = None
FACE_WEIGHT = 2.0


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = MPFaceDetector()
    return _face_detector


def detect_faces_mp(frame_bgr):
    """Detect faces using MediaPipe BlazeFace. Returns list of (x,y,w,h) in pixels."""
    detector = _get_face_detector()
    h, w = frame_bgr.shape[:2]
    detections = detector.detect(frame_bgr)
    rects = []
    for d in detections:
        rects.append((
            int(d["x"] * w),
            int(d["y"] * h),
            int(d["width"] * w),
            int(d["height"] * h),
        ))
    return rects


# ─── Saliency → BBox → Crop ──────────────────────────────────────────────────


def get_composite_mask(saliency_map, face_rects=None):
    """Combine UNISAL saliency map with face detections."""
    composite = saliency_map.copy()

    if face_rects is not None and len(face_rects) > 0:
        for (fx, fy, fw, fh) in face_rects:
            composite[fy:fy+fh, fx:fx+fw] = np.maximum(
                composite[fy:fy+fh, fx:fx+fw], FACE_WEIGHT
            )

    return composite


def saliency_to_bbox(composite_mask):
    """
    Convert a saliency mask to a core bounding box + center of mass.
    Uses weighted percentiles (10th-90th) instead of min/max to avoid
    the bbox spanning the entire frame from sparse outlier patches.
    Returns (x, y, w, h) normalized to 0-1, plus center of mass (cx, cy).
    """
    h, w = composite_mask.shape
    binary = (composite_mask > 0.5).astype(np.uint8)

    if binary.sum() == 0:
        return (0.25, 0.25, 0.5, 0.5), (0.5, 0.5)

    ys, xs = np.where(binary > 0)
    weights = composite_mask[ys, xs]

    cx = np.average(xs, weights=weights) / w
    cy = np.average(ys, weights=weights) / h

    sorted_x_idx = np.argsort(xs)
    sorted_xs = xs[sorted_x_idx]
    sorted_ws = weights[sorted_x_idx]
    cum_w = np.cumsum(sorted_ws)
    total_w = cum_w[-1]
    x_min = sorted_xs[np.searchsorted(cum_w, total_w * 0.10)] / w
    x_max = sorted_xs[np.searchsorted(cum_w, total_w * 0.90)] / w

    sorted_y_idx = np.argsort(ys)
    sorted_ys = ys[sorted_y_idx]
    sorted_wy = weights[sorted_y_idx]
    cum_wy = np.cumsum(sorted_wy)
    total_wy = cum_wy[-1]
    y_min = sorted_ys[np.searchsorted(cum_wy, total_wy * 0.10)] / h
    y_max = sorted_ys[np.searchsorted(cum_wy, total_wy * 0.90)] / h

    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    return bbox, (cx, cy)


def compute_crop_window(center_of_mass, frame_w, frame_h, crop_w):
    """
    Compute a crop window with a pre-determined fixed width, centered on the CoM.
    Only position varies — padding handles AR mismatch.
    Returns (x, y, w, h) in pixels.
    """
    cx_px = center_of_mass[0] * frame_w

    crop_x = int(cx_px - crop_w / 2)
    crop_y = 0
    crop_h = frame_h

    crop_x = max(0, min(crop_x, frame_w - crop_w))

    return (crop_x, crop_y, crop_w, crop_h)


WIDE_CROP_FACTOR = 1.30


def compute_scene_crop_width(scene_bboxes, frame_w, frame_h, target_aspect):
    """
    Compute a fixed crop width for a scene.
    - If saliency fits within 9:16 strip → use exact 9:16 width (no padding)
    - If saliency is wider → use 9:16 width * 1.30 (padding fills top/bottom)
    """
    aspect_w, aspect_h = target_aspect
    narrow_w = int(frame_h * aspect_w / aspect_h)
    wide_w = min(int(narrow_w * WIDE_CROP_FACTOR), frame_w)

    if not scene_bboxes:
        return min(narrow_w, frame_w)

    max_sal_w = max(bbox[2] * frame_w for bbox in scene_bboxes)

    if max_sal_w <= narrow_w:
        return min(narrow_w, frame_w)
    else:
        return wide_w


def apply_padding_to_crop(frame_bgr, crop, target_aspect, method="blur"):
    """
    Extract the crop region from the frame and apply padding to fit the target
    aspect ratio. Returns the final padded frame at the target aspect ratio.
    """
    cx, cy, cw, ch = crop
    crop_region = frame_bgr[cy:cy+ch, cx:cx+cw]

    aspect_w, aspect_h = target_aspect
    target_ratio = aspect_w / aspect_h

    content_ratio = cw / ch

    if abs(content_ratio - target_ratio) < 0.01:
        return crop_region

    out_w = cw
    out_h = int(out_w / target_ratio)

    if content_ratio > target_ratio:
        fg_w = out_w
        fg_h = int(fg_w * ch / cw)
        pad_y = (out_h - fg_h) // 2

        if method == "blur":
            tiny = cv2.resize(crop_region, (16, 16), interpolation=cv2.INTER_AREA)
            bg = cv2.resize(tiny, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            bg = cv2.convertScaleAbs(bg, alpha=0.2, beta=0)
            fg = cv2.resize(crop_region, (fg_w, fg_h))
            bg[pad_y:pad_y+fg_h, :fg_w] = fg
            return bg
        else:
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            fg = cv2.resize(crop_region, (fg_w, fg_h))
            canvas[pad_y:pad_y+fg_h, :fg_w] = fg
            return canvas
    else:
        fg_h = out_h
        fg_w = int(fg_h * cw / ch)
        pad_x = (out_w - fg_w) // 2

        if method == "blur":
            tiny = cv2.resize(crop_region, (16, 16), interpolation=cv2.INTER_AREA)
            bg = cv2.resize(tiny, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            bg = cv2.convertScaleAbs(bg, alpha=0.2, beta=0)
            fg = cv2.resize(crop_region, (fg_w, fg_h))
            bg[:fg_h, pad_x:pad_x+fg_w] = fg
            return bg
        else:
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            fg = cv2.resize(crop_region, (fg_w, fg_h))
            canvas[:fg_h, pad_x:pad_x+fg_w] = fg
            return canvas


# ─── Temporal Smoothing ───────────────────────────────────────────────────────


def detect_all_faces(frames):
    """Run face detection on all frames upfront. Returns list of face_rects per frame."""
    print("Detecting faces (MediaPipe BlazeFace)...")
    all_faces = []
    for frame in tqdm(frames, desc="Face detection"):
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        faces = detect_faces_mp(bgr)
        all_faces.append(faces)
    n_with_faces = sum(1 for f in all_faces if len(f) > 0)
    print(f"Faces found in {n_with_faces}/{len(frames)} frames")
    return all_faces


def smooth_scene_crops(frames, saliency_maps, scene_labels, target_aspect,
                       all_faces, motion_threshold=0.5):
    """
    Pre-compute per-frame crop data with camera-motion-aware stabilization per scene.
    """
    from pyautoflip.cropping.camera_motion import CameraMotionHandler

    num_frames = len(frames)

    # Step 1: Compute raw per-frame saliency data (unisal + faces)
    raw_coms = []
    raw_bboxes = []
    for i in range(num_frames):
        frame = frames[i]
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        composite = get_composite_mask(saliency_maps[i], face_rects=all_faces[i])
        bbox, com = saliency_to_bbox(composite)
        raw_bboxes.append(bbox)
        raw_coms.append(com)

    # Step 2: Compute per-scene fixed crop width, then raw crop windows
    motion_handler = CameraMotionHandler(motion_threshold=motion_threshold, smoothing_window=30)
    scene_ids = sorted(set(scene_labels))

    scene_crop_widths = {}
    for sid in scene_ids:
        idxs = [i for i, s in enumerate(scene_labels) if s == sid]
        frame = frames[idxs[0]]
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        h, w = frame.shape[:2]
        scene_bboxes = [raw_bboxes[i] for i in idxs]
        scene_crop_widths[sid] = compute_scene_crop_width(scene_bboxes, w, h, target_aspect)

    raw_crops = []
    for i in range(num_frames):
        frame = frames[i]
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        h, w = frame.shape[:2]
        crop_w = scene_crop_widths[scene_labels[i]]
        raw_crops.append(compute_crop_window(raw_coms[i], w, h, crop_w))

    # Step 3: Stabilize per scene using CameraMotionHandler
    smoothed_crops = list(raw_crops)
    smoothed_coms = list(raw_coms)

    for sid in scene_ids:
        idxs = [i for i, s in enumerate(scene_labels) if s == sid]
        if len(idxs) < 2:
            continue

        scene_windows = [(raw_crops[i], 1.0) for i in idxs]

        mode = motion_handler.select_camera_motion_mode(scene_windows)
        crop_w = scene_crop_widths[sid]
        frame = frames[idxs[0]]
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        fh, fw = frame.shape[:2]
        min_w = int(fh * target_aspect[0] / target_aspect[1])
        needs_pad = "+ padding" if crop_w > min_w else ""
        print(f"  Scene {sid}: {len(idxs)} frames -> {mode.name}, crop_w={crop_w} (min={min_w}) {needs_pad}")

        scene_key_indices = list(range(len(idxs)))
        interpolated = motion_handler.interpolate_crop_windows(
            scene_windows, scene_key_indices, len(idxs), mode
        )

        smoothed = motion_handler.smooth_trajectory(interpolated, mode)

        for j, idx in enumerate(idxs):
            smoothed_crops[idx] = smoothed[j]
            frame = frames[idx]
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            fh, fw = frame.shape[:2]
            cx = (smoothed[j][0] + smoothed[j][2] / 2) / fw
            cy = (smoothed[j][1] + smoothed[j][3] / 2) / fh
            smoothed_coms[idx] = (cx, cy)

    return raw_bboxes, raw_coms, raw_crops, smoothed_coms, smoothed_crops


# ─── Visualization ────────────────────────────────────────────────────────────

VIEW_SALIENCY = 0
VIEW_BBOX = 1
VIEW_CROP_RAW = 2
VIEW_CROP_SMOOTH = 3
VIEW_RESULT_CROP = 4
VIEW_RESULT_PADDED = 5
VIEW_NAMES = ["Saliency", "Saliency + BBox", "Crop (raw)", "Crop (smoothed)", "Cropped (no pad)", "Cropped (blur pad)"]


def letterbox(img, canvas_w, canvas_h):
    """Fit img into canvas_w x canvas_h preserving aspect ratio, black bars on sides."""
    ih, iw = img.shape[:2]
    scale = min(canvas_w / iw, canvas_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    x_off = (canvas_w - new_w) // 2
    y_off = (canvas_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas


def draw_crop_overlay(display, composite, bbox, com, crop, h, w, color, face_rects=None):
    """Draw saliency + faces + bbox + crop window overlay."""
    display_composite = np.clip(composite, 0, 1)
    alpha = 0.3 + 0.7 * display_composite
    overlay = (display * alpha[:, :, None]).astype(np.uint8)

    if face_rects is not None:
        for (fx, fy, fw, fh) in face_rects:
            cv2.rectangle(overlay, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)

    bx, by, bw, bh = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
    cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), (0, 255, 255), 2)

    cx_px, cy_px = int(com[0]*w), int(com[1]*h)
    cv2.circle(overlay, (cx_px, cy_px), 6, (0, 0, 255), -1)

    cx, cy, cw, ch = crop
    cv2.rectangle(overlay, (cx, cy), (cx+cw, cy+ch), color, 3)

    mask = np.zeros((h, w), dtype=np.float32)
    mask[cy:cy+ch, cx:cx+cw] = 1.0
    overlay = (overlay * (0.4 + 0.6 * mask)[:, :, None]).astype(np.uint8)
    cv2.rectangle(overlay, (cx, cy), (cx+cw, cy+ch), color, 3)

    return overlay


def visualize(frames, saliency_maps, sample_indices, scene_labels, total_frames, fps):
    num_frames = len(frames)
    num_scenes = len(set(scene_labels))

    print(f"\nFrames: {num_frames}, Scenes: {num_scenes}")
    print(f"Target aspect ratio: {TARGET_ASPECT[0]}:{TARGET_ASPECT[1]}")

    # Detect faces on all frames
    all_faces = detect_all_faces(frames)

    # Pre-compute smoothed crops
    print("Smoothing crop trajectories...")
    raw_bboxes, raw_coms, raw_crops, smoothed_coms, smoothed_crops = smooth_scene_crops(
        frames, saliency_maps, scene_labels, TARGET_ASPECT, all_faces
    )

    print("\nControls: [Space] pause | [N/P] next/prev | [V] cycle view | [Q] quit")
    cv2.namedWindow("UNISAL", cv2.WINDOW_NORMAL)

    frame_idx = 0
    view_mode = VIEW_SALIENCY
    paused = True

    while True:
        frame = frames[frame_idx]
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        h, w = frame.shape[:2]
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face_rects = all_faces[frame_idx]
        composite = get_composite_mask(saliency_maps[frame_idx], face_rects)
        display_composite = np.clip(composite, 0, 1)
        bbox = raw_bboxes[frame_idx]
        com_raw = raw_coms[frame_idx]
        com_smooth = smoothed_coms[frame_idx]
        crop_raw = raw_crops[frame_idx]
        crop_smooth = smoothed_crops[frame_idx]
        n_faces = len(face_rects) if face_rects is not None and len(face_rects) > 0 else 0

        # ── Render based on view mode ──
        if view_mode == VIEW_SALIENCY:
            alpha = 0.3 + 0.7 * display_composite
            overlay = (display * alpha[:, :, None]).astype(np.uint8)
            for (fx, fy, fw, fh) in (face_rects if n_faces else []):
                cv2.rectangle(overlay, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)

        elif view_mode == VIEW_BBOX:
            alpha = 0.3 + 0.7 * display_composite
            overlay = (display * alpha[:, :, None]).astype(np.uint8)
            for (fx, fy, fw, fh) in (face_rects if n_faces else []):
                cv2.rectangle(overlay, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
            bx, by, bw, bh = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
            cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), (0, 255, 255), 2)
            cx_px, cy_px = int(com_raw[0]*w), int(com_raw[1]*h)
            cv2.circle(overlay, (cx_px, cy_px), 6, (0, 0, 255), -1)
            cv2.circle(overlay, (cx_px, cy_px), 8, (255, 255, 255), 1)

        elif view_mode == VIEW_CROP_RAW:
            overlay = draw_crop_overlay(display, composite, bbox, com_raw, crop_raw, h, w,
                                        (0, 0, 255), face_rects)

        elif view_mode == VIEW_CROP_SMOOTH:
            overlay = draw_crop_overlay(display, composite, bbox, com_smooth, crop_smooth, h, w,
                                        (0, 255, 0), face_rects)

        elif view_mode == VIEW_RESULT_CROP:
            cx, cy, cw, ch = crop_smooth
            cropped = display[cy:cy+ch, cx:cx+cw]
            overlay = letterbox(cropped, w, h)

        elif view_mode == VIEW_RESULT_PADDED:
            padded = apply_padding_to_crop(display, crop_smooth, TARGET_ASPECT, method="blur")
            overlay = letterbox(padded, w, h)

        # ── HUD ──
        vid_frame = sample_indices[frame_idx]
        timestamp = vid_frame / fps
        scene_id = scene_labels[frame_idx]
        active_com = com_smooth if view_mode >= VIEW_CROP_SMOOTH else com_raw

        hud_bg_h = 70
        overlay[:hud_bg_h, :] = (overlay[:hud_bg_h, :] * 0.5).astype(np.uint8)
        cv2.putText(
            overlay,
            f"Frame {vid_frame}/{total_frames} ({timestamp:.1f}s)  |  Scene {scene_id+1}/{num_scenes}  |  Faces: {n_faces}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        cv2.putText(
            overlay,
            f"View: {VIEW_NAMES[view_mode]}  |  CoM: ({active_com[0]:.2f},{active_com[1]:.2f})",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )
        cv2.putText(
            overlay,
            "[Space] play  [N/P] next/prev  [V] view  [Q] quit",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
        )

        cv2.imshow("UNISAL", overlay)

        key = cv2.waitKey(0 if paused else 33) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("v"):
            view_mode = (view_mode + 1) % len(VIEW_NAMES)
        elif key == ord("n"):
            frame_idx = (frame_idx + 1) % num_frames
        elif key == ord("p"):
            frame_idx = (frame_idx - 1) % num_frames
        elif not paused:
            frame_idx = (frame_idx + 1) % num_frames

    cv2.destroyAllWindows()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    video_path = "test.mp4"
    assert Path(video_path).exists(), f"Video not found: {video_path}"

    # Detect scenes
    print(f"Detecting scenes in: {video_path}")
    scenes, total_frames, fps = detect_scenes(video_path)

    # Load sampled frames per scene
    frames, indices, scene_labels = load_scene_frames(
        video_path, scenes, max_per_scene=16
    )

    # Compute saliency maps with UNISAL
    saliency_maps = compute_saliency_maps(frames)

    # Visualize step by step
    visualize(frames, saliency_maps, indices, scene_labels, total_frames, fps)


if __name__ == "__main__":
    main()
