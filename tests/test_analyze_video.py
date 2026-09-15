"""analyze_video contract tests (the analyze-only mode the worker uses)."""
import pytest

from pyautoflip import analyze_video

W, H, FPS = 640, 360, 30


def _analyze(path, **kwargs):
    kwargs.setdefault("target_aspect_ratio", "9:16")
    return analyze_video(str(path), log_level="WARNING", **kwargs)


def test_dense_windows_are_segment_relative(make_clip):
    analysis = _analyze(make_clip(seconds=3.0, fps=FPS, size=(W, H)))

    assert analysis.total_frames == 3 * FPS
    times = [w.time for w in analysis.crop_windows]
    assert times[0] == 0
    assert times == sorted(times)
    assert times[-1] < 3.0
    assert analysis.keyframes[0].time == 0


def test_regions_stay_inside_the_frame_at_the_target_aspect(make_clip):
    analysis = _analyze(make_clip(seconds=2.0, fps=FPS, size=(W, H)))

    for window in analysis.crop_windows:
        assert len(window.regions) == 1
        r = window.regions[0]
        assert 0 <= r.x and r.x + r.w <= 1 + 1e-6
        assert 0 <= r.y and r.y + r.h <= 1 + 1e-6
        # full-height 9:16 strip of a 16:9 frame
        assert (r.w * W) / (r.h * H) == pytest.approx(9 / 16, rel=0.02)


def test_output_is_compact(make_clip):
    """Stored per clip and sent to browsers: coords to 4 decimals, times to 3."""
    analysis = _analyze(make_clip(seconds=2.0, fps=FPS, size=(W, H)))

    for window in analysis.crop_windows + analysis.keyframes:
        assert window.time == round(window.time, 3)
        for r in window.regions:
            for v in (r.x, r.y, r.w, r.h):
                assert v == round(v, 4)
            assert r.x + r.w <= 1.0
