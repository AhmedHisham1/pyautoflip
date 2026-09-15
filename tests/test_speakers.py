"""Active speaker: face tracks, mouth motion, and the debounced speaker."""
import math

import numpy as np
import pytest

from pyautoflip.cropping.saliency_cropper import SaliencyCropper
from pyautoflip.cropping.speakers import (
    active_track_per_sample,
    box_at,
    build_tracks,
    stable_labels,
    track_activity,
    track_spans,
)
from pyautoflip.detection.face_detector import mouth_openness

LEFT, RIGHT = (100, 80, 60, 60), (440, 80, 60, 60)


# ─── Tracks ──────────────────────────────────────────────────────────────────


def test_faces_keep_their_track_as_they_move():
    moved = (110, 82, 60, 60)

    ids = build_tracks([[LEFT, RIGHT], [RIGHT, moved], [moved]])

    assert ids[1] == [ids[0][1], ids[0][0]]
    assert ids[2] == [ids[0][0]]


def test_a_face_elsewhere_starts_a_new_track():
    ids = build_tracks([[LEFT], [RIGHT]])

    assert ids[0] != ids[1]


def test_a_track_survives_a_short_gap():
    ids = build_tracks([[LEFT], [], [LEFT]])

    assert ids[2] == ids[0]


def test_a_tracks_box_is_interpolated_between_sightings():
    faces = [[LEFT], [(120, 80, 60, 60)]]
    span = track_spans(faces, build_tracks(faces), [0, 10])[0]

    assert box_at(span, 5, reach=2)[0] == pytest.approx(110)
    assert box_at(span, 11, reach=2)[0] == pytest.approx(120)  # held just past the last sighting
    assert box_at(span, 20, reach=2) is None


# ─── Mouth motion ────────────────────────────────────────────────────────────


def test_mouth_openness_is_the_lips_height_over_the_mouth_width():
    landmarks = np.zeros((106, 2))
    landmarks[52], landmarks[61] = (0, 0), (40, 0)  # corners
    for i in list(range(53, 61)) + list(range(62, 72)):
        landmarks[i] = (5 if i % 2 else 35, 0)       # points off the middle
    landmarks[62], landmarks[60] = (20, -4), (20, 6)  # the lips at the middle

    assert mouth_openness(landmarks) == pytest.approx(10 / 40)
    assert mouth_openness(None) is None


def test_a_talking_mouth_is_more_active_than_a_still_one():
    frames = np.arange(0, 60, 4)  # ~7 fps at 30 fps
    talking = 0.5 + 0.15 * np.sin(frames / 30 * 2 * np.pi * 2)
    still = np.full(len(frames), 0.45)

    activity = track_activity({0: (frames, talking), 1: (frames, still)}, [30], fps=30)

    assert activity[0][0] > 0.05 > activity[1][0]


def test_activity_needs_a_few_measurements():
    activity = track_activity({0: ([28, 31], [0.4, 0.7])}, [30], fps=30)

    assert np.isnan(activity[0][0])


# ─── The speaker ─────────────────────────────────────────────────────────────


def test_the_clear_leader_is_the_speaker():
    activity = {0: np.full(6, 0.10), 1: np.full(6, 0.01)}

    assert active_track_per_sample(activity, 6, hold_samples=2) == [0] * 6


def test_nobody_speaks_when_mouths_move_alike():
    activity = {0: np.full(4, 0.10), 1: np.full(4, 0.09)}

    assert active_track_per_sample(activity, 4, hold_samples=2) == [None] * 4


def test_the_speaker_changes_only_once_the_new_one_holds():
    assert stable_labels([0, 0, 0, 1, 0, 0, 1, 1, 1, 1], 3) == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_unclear_samples_keep_the_current_speaker():
    assert stable_labels([0, 0, 0, None, None, 0], 3) == [0] * 6


def test_the_first_speaker_is_framed_from_the_start():
    assert stable_labels([None, None, 1, 1, 1], 3) == [1] * 5


# ─── The crop ────────────────────────────────────────────────────────────────


class TalkingOnTheLeft:
    """Landmarks stand-in: the left face's mouth moves, the right one's doesn't.
    Frames carry their index in pixel (0, 0)."""

    def mouth_openness_at(self, frame, boxes):
        index = int(frame[0, 0, 0])
        return [0.5 + 0.2 * math.sin(index) if box[0] < 320 else 0.45 for box in boxes]


def frames_at(indices):
    frames = []
    for index in indices:
        frame = np.zeros((360, 640, 3), np.uint8)
        frame[0, 0, 0] = index
        frames.append(frame)
    return frames, list(indices)


@pytest.fixture
def two_faces(monkeypatch):
    """A cropper seeing two faces in every sample (no saliency elsewhere)."""
    cropper = SaliencyCropper(target_aspect_ratio=9 / 16)
    left, right = (150, 100, 60, 60), (390, 100, 60, 60)
    monkeypatch.setattr(
        cropper, "_compute_saliency_maps", lambda frames: [np.zeros((360, 640), np.float32) for _ in frames]
    )
    monkeypatch.setattr(
        cropper, "_detect_faces", lambda frames: ([[left, right] for _ in frames], [[0.5, 0.45] for _ in frames])
    )
    cropper._face_detector = TalkingOnTheLeft()
    return cropper


def test_the_crop_leans_toward_whoever_is_talking(two_faces):
    indices = two_faces.sample_indices(90, 30.0)
    frames, _ = frames_at(indices)

    neutral = two_faces.process_scene(frames, 90, indices, 30.0)
    leaning = two_faces.process_scene(frames, 90, indices, 30.0, read_frames=frames_at)

    def centre(window):
        return window[0] + window[2] / 2

    assert centre(leaning[45]) < centre(neutral[45]) - 0.03
    assert two_faces.last_active_tracks[0] == two_faces.last_track_ids[0][0]


def test_no_speaker_pass_without_mouth_measurements(two_faces, monkeypatch):
    monkeypatch.setattr(
        two_faces, "_detect_faces", lambda frames: ([[LEFT, RIGHT] for _ in frames], [[None, None] for _ in frames])
    )

    def unexpected(indices):
        raise AssertionError("read extra frames without landmarks")

    indices = two_faces.sample_indices(90, 30.0)
    two_faces.process_scene(frames_at(indices)[0], 90, indices, 30.0, read_frames=unexpected)

    assert two_faces.last_face_activity is None
