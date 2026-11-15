"""analyze_video(scene_workers=N): a clip's scenes analysed at once give exactly what one after the other gives."""
import pytest

import pyautoflip.detection.face_detector as face_detector
from pyautoflip import analyze_video
from pyautoflip.core.processor import AutoFlipProcessor

W, H, FPS = 640, 360, 30


def _analyze(path, **kwargs):
    return analyze_video(str(path), target_aspect_ratio="9:16", log_level="WARNING", **kwargs).to_dict()


@pytest.fixture(scope="module")
def four_scenes(tmp_path_factory):
    from conftest import write_clip

    path = tmp_path_factory.mktemp("scenes") / "four.mp4"
    return write_clip(path, seconds=8.0, fps=FPS, size=(W, H), cuts=(2.0, 4.0, 6.0),
                      subject_x=lambda t: 0.2 + 0.08 * t)


class TwoTalkingFaces:
    """Two faces far apart (too far for one 9:16 crop), whose mouths move with each frame's own pixels: the speaker
    pass and the stacked layout run, the same on any thread for the same frame."""

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _open(frame, k):
        return float((frame[k::7, k::5].mean() * (3 + k)) % 1.0) * 0.2

    def detect(self, frame):
        return [
            {"x": 0.05, "y": 0.2, "width": 0.15, "height": 0.35, "confidence": 0.9, "mouth_open": self._open(frame, 0)},
            {"x": 0.78, "y": 0.2, "width": 0.15, "height": 0.35, "confidence": 0.9, "mouth_open": self._open(frame, 1)},
        ]

    def mouth_openness_at(self, frame, boxes):
        return [self._open(frame, k % 2) for k, _ in enumerate(boxes)]


@pytest.mark.parametrize("workers", [2, 4])
def test_scenes_at_once_give_the_same_analysis(four_scenes, workers):
    one_by_one = _analyze(four_scenes)
    assert len(one_by_one["scenes"]) == 4
    assert _analyze(four_scenes, scene_workers=workers) == one_by_one


def test_a_two_speaker_stacked_clip_is_the_same_at_once(four_scenes, monkeypatch):
    monkeypatch.setattr(face_detector, "FaceDetector", TwoTalkingFaces)
    monkeypatch.setattr("pyautoflip.core.processor.FaceDetector", TwoTalkingFaces)
    one_by_one = _analyze(four_scenes)
    assert any(len(w["regions"]) == 2 for w in one_by_one["crop_windows"]), "the clip should stack its two speakers"
    assert _analyze(four_scenes, scene_workers=4) == one_by_one


def test_one_scene_and_more_workers_than_scenes(make_clip):
    single = make_clip(seconds=2.0, fps=FPS, size=(W, H))
    assert _analyze(single, scene_workers=4) == _analyze(single)
    two = make_clip(seconds=3.0, fps=FPS, size=(W, H), cut_at=1.5)
    assert _analyze(two, scene_workers=8) == _analyze(two)


def test_a_scene_with_no_frames_read_is_filled_the_same_way(four_scenes, monkeypatch):
    plain = AutoFlipProcessor._read_samples
    unread = {}

    def read(self, reader, abs_start, indices):
        if abs_start == unread.get("start"):
            return [], []
        return plain(self, reader, abs_start, indices)

    scenes = _analyze(four_scenes)["scenes"]
    unread["start"] = scenes[2]["start_frame"]
    monkeypatch.setattr(AutoFlipProcessor, "_read_samples", read)
    one_by_one = _analyze(four_scenes)
    assert one_by_one["scenes"][2]["camera_mode"] == "STATIONARY"
    third = [w for w in one_by_one["crop_windows"] if scenes[2]["start_frame"] / FPS + 0.1 < w["time"] < scenes[2]["end_frame"] / FPS - 0.1]
    assert third and all(w["regions"] == [{"x": 0.25, "y": 0.0, "w": 0.5, "h": 1.0}] for w in third)
    assert _analyze(four_scenes, scene_workers=3) == one_by_one
