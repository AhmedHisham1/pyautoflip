"""Shared fixtures: small synthetic clips generated on the fly.

Nothing binary lives in git — every test clip is drawn with OpenCV: a bright
disc (the salient subject) over a dark, lightly noisy background. Motion and
hard cuts are scripted so tests can assert where the crop should be.
"""
import itertools

import cv2
import numpy as np
import pytest

# BGR backgrounds different enough in hue, saturation and value that
# PySceneDetect's ContentDetector reliably reports the switch as a cut
BG_BEFORE_CUT = (90, 30, 10)   # dark blue
BG_AFTER_CUT = (20, 140, 230)  # orange


def write_clip(path, *, seconds=3.0, fps=30, size=(640, 360), subject_x=None, cut_at=None, seed=0):
    """Write a synthetic clip and return its path.

    Args:
        seconds: clip duration
        fps: frame rate
        size: (width, height)
        subject_x: callable t -> subject centre x in [0, 1]; default static 0.5
        cut_at: time (s) of a hard cut: the background colour switches there
        seed: noise seed (clips are deterministic)
    """
    w, h = size
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    radius = h // 6
    for i in range(int(round(seconds * fps))):
        t = i / fps
        bg = BG_AFTER_CUT if cut_at is not None and t >= cut_at else BG_BEFORE_CUT
        frame = np.empty((h, w, 3), np.uint8)
        frame[:] = bg
        noise = rng.integers(0, 10, (h, w, 1), dtype=np.uint8)
        frame = cv2.add(frame, np.repeat(noise, 3, axis=2))
        cx = subject_x(t) if subject_x else 0.5
        cv2.circle(frame, (int(cx * w), h // 2), radius, (245, 245, 245), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="session")
def make_clip(tmp_path_factory):
    """Factory fixture: make_clip(**write_clip_kwargs) -> path to a new clip."""
    base = tmp_path_factory.mktemp("clips")
    counter = itertools.count()

    def _make(**kwargs):
        return write_clip(base / f"clip_{next(counter)}.mp4", **kwargs)

    return _make
