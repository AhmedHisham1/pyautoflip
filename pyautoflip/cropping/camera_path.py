"""Camera path planning for saliency-based cropping.

Turns per-sample subject positions into a crop position for every frame of
a scene: a fixed camera when the subject barely moves, otherwise a smoothed
track. Smoothing and speed limits are expressed in seconds, so behaviour
doesn't depend on how densely a scene was sampled.
"""
from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

STATIONARY = "STATIONARY"
TRACKING = "TRACKING"


def sample_indices_for_scene(
    scene_length: int,
    fps: float,
    analysis_fps: float = 3.0,
    max_samples: int = 300,
) -> List[int]:
    """Scene-relative frame indices to analyze.

    `analysis_fps` samples per second, always including the scene's first and
    last frame, and at most `max_samples` (long scenes are sampled sparser).
    """
    if scene_length <= 0:
        return []
    if scene_length == 1:
        return [0]
    step = max(1, int(round((fps or 30.0) / max(analysis_fps, 1e-6))))
    indices = list(range(0, scene_length, step))
    if indices[-1] != scene_length - 1:
        indices.append(scene_length - 1)
    if len(indices) > max_samples:
        indices = sorted({int(round(i)) for i in np.linspace(0, scene_length - 1, max_samples)})
    return indices


def plan_camera_path(
    centers: Sequence[float],
    sample_indices: Sequence[int],
    frame_count: int,
    fps: float,
    crop_w: float,
    frame_w: float,
    stationary_fraction: float = 0.1,
    smoothing_s: float = 0.5,
    max_speed: float = 0.6,
) -> Tuple[np.ndarray, str]:
    """Crop left edge for every frame of a scene.

    Args:
        centers: desired crop centre x per sample, in pixels
        sample_indices: scene-relative frame index of each sample (ascending)
        frame_count: frames in the scene
        fps: frame rate
        crop_w: crop width in pixels
        frame_w: frame width in pixels
        stationary_fraction: when the desired positions span at most this
            fraction of the crop width, the camera holds still (at the median)
        smoothing_s: Gaussian smoothing sigma, in seconds, while tracking
        max_speed: pan speed limit, in crop widths per second

    Returns:
        (lefts, mode): per-frame crop left edge (px, float) and
        "STATIONARY" or "TRACKING"
    """
    if frame_count <= 0:
        return np.zeros(0), STATIONARY
    fps = fps or 30.0
    max_left = max(0.0, float(frame_w) - float(crop_w))
    lefts = np.clip(np.asarray(centers, dtype=float) - crop_w / 2.0, 0.0, max_left)
    if lefts.size == 0:
        return np.full(frame_count, max_left / 2.0), STATIONARY

    if lefts.size == 1 or np.ptp(lefts) <= stationary_fraction * crop_w:
        return np.full(frame_count, float(np.median(lefts))), STATIONARY

    # Interpolate between samples at their real frame positions
    path = np.interp(np.arange(frame_count), np.asarray(sample_indices, dtype=float), lefts)
    sigma = smoothing_s * fps
    if sigma > 0:
        path = gaussian_filter1d(path, sigma=sigma, mode="nearest")
    path = _limit_speed(path, max_speed * crop_w / fps)
    return np.clip(path, 0.0, max_left), TRACKING


def _limit_speed(path: np.ndarray, max_step: float) -> np.ndarray:
    """Cap per-frame movement.

    Forward and backward passes are averaged so the limit doesn't make the
    camera lag behind (or lead) the subject.
    """
    if max_step <= 0 or len(path) < 2:
        return path
    forward = path.copy()
    for i in range(1, len(forward)):
        forward[i] = min(max(forward[i], forward[i - 1] - max_step), forward[i - 1] + max_step)
    backward = path.copy()
    for i in range(len(backward) - 2, -1, -1):
        backward[i] = min(max(backward[i], backward[i + 1] - max_step), backward[i + 1] + max_step)
    return (forward + backward) / 2.0


def stable_states(flags: Sequence[bool], min_run: int) -> List[bool]:
    """Debounce a per-sample on/off signal.

    A change of state sticks only when the new state holds for `min_run`
    consecutive samples; shorter blips keep the surrounding state. The opening
    state is the majority of the first `min_run` samples.
    """
    flags = [bool(f) for f in flags]
    if min_run <= 1 or not flags:
        return flags
    head = flags[:min_run]
    state = sum(head) * 2 > len(head)
    out = []
    for i, flag in enumerate(flags):
        window = flags[i:i + min_run]
        if flag != state and len(window) == min_run and all(f == flag for f in window):
            state = flag
        out.append(state)
    return out
