"""Scene-level crop decisions in the saliency cropper."""
import numpy as np
import pytest

from pyautoflip.core.processor import AutoFlipProcessor
from pyautoflip.cropping.saliency_cropper import (
    compute_scene_crop_width,
    find_split_faces,
    get_composite_mask,
    saliency_to_bbox,
)

NARROW = (0.4, 0.2, 0.2, 0.5)  # salient box fits a 9:16 strip of 640x360
WIDE = (0.0, 0.0, 0.9, 1.0)    # salient box spans most of the frame


def test_one_wide_sample_does_not_widen_the_scene():
    assert compute_scene_crop_width([NARROW] * 9 + [WIDE], 640, 360, (9, 16)) == 202


def test_mostly_wide_scenes_get_the_wide_crop():
    assert compute_scene_crop_width([WIDE] * 7 + [NARROW] * 3, 640, 360, (9, 16)) == 262


def test_no_samples_default_to_the_narrow_crop():
    assert compute_scene_crop_width([], 640, 360, (9, 16)) == 202


def face(cx, size=40):
    """A face box centred at x = cx (of 640) in a 640x360 frame."""
    return (int(cx * 640) - size // 2, 150, size, size)


def test_the_speakers_face_pulls_the_centre_of_attention():
    blank = np.zeros((360, 640), np.float32)
    faces = [face(0.3), face(0.6)]

    _, even = saliency_to_bbox(get_composite_mask(blank, faces))
    _, leaning = saliency_to_bbox(get_composite_mask(blank, faces, [2.0, 1.0]))

    assert even[0] == pytest.approx(0.45, abs=0.01)
    assert leaning[0] < 0.41


def test_three_faces_split_on_the_two_most_active():
    faces = [face(0.1), face(0.5), face(0.9)]

    split = find_split_faces(faces, 640, 360, (9, 16), activity=[0.1, 0.2, 0.0])

    assert [round(c[0], 1) for c in split] == [0.1, 0.5]


def test_without_activity_the_outermost_faces_split():
    faces = [face(0.1), face(0.5), face(0.9)]

    split = find_split_faces(faces, 640, 360, (9, 16))

    assert [round(c[0], 1) for c in split] == [0.1, 0.9]


def test_stacked_panels_leave_headroom_above_each_face():
    splits = [[(0.2, 0.4), (0.8, 0.4)]] * 3

    regions = AutoFlipProcessor._stacked_panel_regions(splits, [True] * 3, 640, 360, (9, 16))

    x, y, w, h = regions[1][0]
    assert (0.4 - y) / h == pytest.approx(1 / 3)  # face a third of the way down
