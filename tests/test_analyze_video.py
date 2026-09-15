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


def _xs(analysis):
    return [w.regions[0].x for w in analysis.crop_windows if len(w.regions) == 1]


def test_crop_follows_a_moving_subject(make_clip):
    # The subject crosses from 25% to 75% of the frame width over 4 s
    clip = make_clip(seconds=4.0, fps=FPS, size=(W, H), subject_x=lambda t: 0.25 + 0.5 * t / 4.0)

    analysis = _analyze(clip)

    xs = _xs(analysis)
    assert analysis.camera_mode == "TRACKING"
    assert xs[-1] - xs[0] > 0.3
    assert all(b >= a - 1e-3 for a, b in zip(xs, xs[1:]))  # never swings back


def test_crop_holds_still_on_a_static_subject(make_clip):
    analysis = _analyze(make_clip(seconds=3.0, fps=FPS, size=(W, H), subject_x=lambda t: 0.3))

    xs = _xs(analysis)
    assert analysis.camera_mode == "STATIONARY"
    assert max(xs) - min(xs) < 1e-6
    region = analysis.crop_windows[0].regions[0]
    assert region.x + region.w / 2 == pytest.approx(0.3, abs=0.05)  # centred on the subject


def test_crop_jumps_at_a_hard_cut(make_clip):
    cut = 2.0
    clip = make_clip(
        seconds=4.0, fps=FPS, size=(W, H), cut_at=cut,
        subject_x=lambda t: 0.25 if t < cut else 0.75,
    )

    analysis = _analyze(clip)

    assert [s.start_frame for s in analysis.scenes] == [0, int(cut * FPS)]
    x_at = {w.time: w.regions[0].x for w in analysis.crop_windows}
    # Samples on both sides of the cut: consumers can't slide across it
    before, after = x_at[round(cut - 1 / FPS, 3)], x_at[round(cut, 3)]
    assert after - before > 0.3


def test_four_by_five_regions_keep_their_shape(make_clip):
    analysis = _analyze(make_clip(seconds=2.0, fps=FPS, size=(W, H)), target_aspect_ratio="4:5")

    r = analysis.crop_windows[0].regions[0]
    assert (r.w * W) / (r.h * H) == pytest.approx(4 / 5, rel=0.02)


def test_dense_windows_report_their_real_rate(make_clip):
    analysis = _analyze(make_clip(seconds=2.0, fps=FPS, size=(W, H)))

    assert analysis.crop_windows_fps == pytest.approx(5.0)
