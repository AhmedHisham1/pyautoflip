"""Scene-level crop decisions in the saliency cropper."""
from pyautoflip.cropping.saliency_cropper import compute_scene_crop_width

NARROW = (0.4, 0.2, 0.2, 0.5)  # salient box fits a 9:16 strip of 640x360
WIDE = (0.0, 0.0, 0.9, 1.0)    # salient box spans most of the frame


def test_one_wide_sample_does_not_widen_the_scene():
    assert compute_scene_crop_width([NARROW] * 9 + [WIDE], 640, 360, (9, 16)) == 202


def test_mostly_wide_scenes_get_the_wide_crop():
    assert compute_scene_crop_width([WIDE] * 7 + [NARROW] * 3, 640, 360, (9, 16)) == 262


def test_no_samples_default_to_the_narrow_crop():
    assert compute_scene_crop_width([], 640, 360, (9, 16)) == 202
