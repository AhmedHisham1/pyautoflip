"""Who is talking: faces linked into tracks across a scene's samples, each
scored by how much its mouth moves.

A talking mouth opens and closes several times a second; a listening one
barely moves. A track's activity at a sample is the spread (standard
deviation) of its mouth openness over the surrounding second, measured at
about SPEAKER_FPS. The speaker is the clear leader, and a new one has to
lead for HOLD_S before the camera follows them.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SPEAKER_FPS = 7.0         # mouth measurements per second in multi-face scenes
ACTIVITY_WINDOW_S = 1.0   # mouth motion is measured over this span
MIN_ACTIVITY = 0.02       # below this nobody is clearly talking (in mouth widths)
LEAD_RATIO = 1.4          # the speaker's mouth moves this much more than anyone else's
HOLD_S = 1.5              # a new speaker must lead this long to take over
MIN_IOU = 0.3             # box overlap that continues a track

Box = Tuple[float, float, float, float]  # x, y, w, h in pixels


def build_tracks(all_faces: Sequence[Sequence[Box]], max_gap: int = 2) -> List[List[int]]:
    """Link faces across samples into tracks.

    Returns, for each sample, the track id of each of its faces (same order).
    A face continues the track whose latest box, seen at most `max_gap`
    samples earlier, it overlaps best (or whose centre is within a face width
    of its own); otherwise it starts a new track.
    """
    track_ids, latest, next_id = [], {}, 0
    for si, faces in enumerate(all_faces):
        matches = sorted(
            (
                (score, fi, tid)
                for fi, box in enumerate(faces)
                for tid, (seen_at, prev) in latest.items()
                if si - seen_at <= max_gap and (score := _continuation(prev, box)) > 0
            ),
            reverse=True,
        )
        ids, taken = [None] * len(faces), set()
        for _, fi, tid in matches:
            if ids[fi] is None and tid not in taken:
                ids[fi] = tid
                taken.add(tid)
        for fi, box in enumerate(faces):
            if ids[fi] is None:
                ids[fi], next_id = next_id, next_id + 1
            latest[ids[fi]] = (si, box)
        track_ids.append(ids)
    return track_ids


def track_spans(all_faces, track_ids, sample_indices) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Per track: the frames it was seen at (ascending) and its box at each."""
    frames, boxes = {}, {}
    for index, faces, ids in zip(sample_indices, all_faces, track_ids):
        for box, tid in zip(faces, ids):
            frames.setdefault(tid, []).append(index)
            boxes.setdefault(tid, []).append(box)
    return {tid: (np.asarray(frames[tid], float), np.asarray(boxes[tid], float)) for tid in frames}


def box_at(span, frame_index: float, reach: float) -> Optional[Box]:
    """A track's box at `frame_index`, interpolated between sightings; None
    more than `reach` frames before or after the frames it was seen at."""
    frames, boxes = span
    if frame_index < frames[0] - reach or frame_index > frames[-1] + reach:
        return None
    return tuple(float(np.interp(frame_index, frames, boxes[:, k])) for k in range(4))


def track_activity(series, sample_indices, fps: float, window_s: float = ACTIVITY_WINDOW_S) -> Dict[int, np.ndarray]:
    """Each track's mouth activity at each sample.

    series: track id -> (frame indices, mouth openness). Activity is the
    standard deviation of openness within ±window_s/2 of the sample; NaN
    where the track has fewer than 3 measurements there.
    """
    half = window_s * (fps or 30.0) / 2.0
    samples = np.asarray(sample_indices, dtype=float)
    out = {}
    for tid, (frames, values) in series.items():
        frames = np.asarray(frames, dtype=float)
        values = np.asarray(values, dtype=float)
        order = np.argsort(frames, kind="stable")
        frames, values = frames[order], values[order]
        lo = np.searchsorted(frames, samples - half, side="left")
        hi = np.searchsorted(frames, samples + half, side="right")
        out[tid] = np.array([np.std(values[a:b]) if b - a >= 3 else np.nan for a, b in zip(lo, hi)])
    return out


def active_track_per_sample(activity: Dict[int, np.ndarray], n_samples: int, hold_samples: int) -> List[Optional[int]]:
    """The track clearly talking at each sample (None while nobody is), debounced."""
    raw = []
    for si in range(n_samples):
        scores = sorted(
            ((float(a[si]), tid) for tid, a in activity.items() if np.isfinite(a[si])),
            reverse=True,
        )
        if not scores or scores[0][0] < MIN_ACTIVITY:
            raw.append(None)
        elif len(scores) > 1 and scores[0][0] < LEAD_RATIO * scores[1][0]:
            raw.append(None)
        else:
            raw.append(scores[0][1])
    return stable_labels(raw, hold_samples)


def stable_labels(labels: Sequence[Optional[int]], min_run: int) -> List[Optional[int]]:
    """Debounce a per-sample label (None: no clear label at that sample).

    The label in force changes only when a different label leads a run of
    `min_run` samples: no other label in it, and at least half of it set. The
    change applies from the start of that run (the analysis can look ahead),
    samples before the first such run take the first label, and None samples
    keep the label in force.
    """
    n, need = len(labels), min_run // 2 + 1
    out, state = [], None
    for i, label in enumerate(labels):
        if label is not None and label != state and i + min_run <= n:
            window = labels[i:i + min_run]
            if all(other is None or other == label for other in window) and sum(
                other is not None for other in window
            ) >= need:
                state = label
        out.append(state)
    first = next((label for label in out if label is not None), None)
    return [first if label is None else label for label in out]


def _iou(a: Box, b: Box) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _continuation(prev: Box, box: Box) -> float:
    """How well `box` continues a track last seen at `prev` (0: it doesn't)."""
    overlap = _iou(prev, box)
    if overlap >= MIN_IOU:
        return 1.0 + overlap
    size = max(prev[2], box[2], 1.0)
    dist = float(np.hypot(
        (prev[0] + prev[2] / 2) - (box[0] + box[2] / 2),
        (prev[1] + prev[3] / 2) - (box[1] + box[3] / 2),
    ))
    return max(0.0, 1.0 - dist / size)
