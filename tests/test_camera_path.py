"""camera_path: sampling, camera behaviour, and layout debouncing."""
import numpy as np
import pytest

from pyautoflip.cropping.camera_path import (
    STATIONARY,
    TRACKING,
    plan_camera_path,
    sample_indices_for_scene,
    stable_states,
)


def test_samples_by_rate_and_keeps_both_ends():
    assert sample_indices_for_scene(90, fps=30, analysis_fps=3) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 89]


def test_short_and_empty_scenes():
    assert sample_indices_for_scene(0, fps=30) == []
    assert sample_indices_for_scene(1, fps=30) == [0]
    assert sample_indices_for_scene(5, fps=30) == [0, 4]


def test_long_scenes_are_capped():
    indices = sample_indices_for_scene(30 * 600, fps=30, analysis_fps=3, max_samples=300)
    assert len(indices) <= 300
    assert indices[0] == 0 and indices[-1] == 30 * 600 - 1
    assert indices == sorted(set(indices))


def test_small_movement_holds_the_camera_still():
    lefts, mode = plan_camera_path([500, 505, 498], [0, 30, 60], 61, 30, crop_w=200, frame_w=1000)

    assert mode == STATIONARY
    assert np.all(lefts == lefts[0])
    assert lefts[0] == pytest.approx(400)  # median centre 500, minus half the crop


def test_tracking_interpolates_at_real_frame_positions():
    lefts, mode = plan_camera_path(
        [300, 500, 700], [0, 60, 120], 121, 30, crop_w=200, frame_w=1000, smoothing_s=0, max_speed=10
    )

    assert mode == TRACKING
    assert lefts[30] == pytest.approx(300)  # half-way between the first two samples
    assert lefts[90] == pytest.approx(500)
    # The old interpolation treated samples as frames 0..n-1 and froze here
    assert lefts[10] < lefts[60] < lefts[110]


def test_pan_speed_is_limited():
    fps, crop_w, max_speed = 30, 100, 0.6
    lefts, _ = plan_camera_path(
        [100, 900], [0, 1], 60, fps, crop_w=crop_w, frame_w=1000, smoothing_s=0, max_speed=max_speed
    )

    assert np.abs(np.diff(lefts)).max() <= max_speed * crop_w / fps + 1e-9


def test_path_stays_inside_the_frame():
    lefts, _ = plan_camera_path([0, 1000], [0, 30], 31, 30, crop_w=200, frame_w=1000)

    assert lefts.min() >= 0
    assert lefts.max() <= 800


def test_layout_blips_are_ignored():
    assert stable_states([False, False, True, False, False], 2) == [False] * 5


def test_layout_changes_that_last_are_kept():
    assert stable_states([False, True, True, True, False, False], 2) == [False, True, True, True, False, False]
