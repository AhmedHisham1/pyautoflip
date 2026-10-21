"""The final encode of a reframed video: NVENC where the box has it, libx264 at a fast preset otherwise. With libx264 at
its default preset (medium) a 27-minute video took 36 silent minutes after the scenes were done (2026-10-21)."""
from pyautoflip.utils.video import final_encoders


def test_nvenc_is_tried_first_then_the_fast_cpu_preset(monkeypatch):
    monkeypatch.delenv("PYAUTOFLIP_FINAL_ENCODER", raising=False)
    monkeypatch.delenv("PYAUTOFLIP_X264_PRESET", raising=False)
    encoders = final_encoders()
    assert [e[1] for e in encoders] == ["h264_nvenc", "libx264"]
    cpu = encoders[1]
    assert cpu[cpu.index("-preset") + 1] == "veryfast" and cpu[cpu.index("-crf") + 1] == "18"


def test_cpu_only_and_a_chosen_preset(monkeypatch):
    monkeypatch.setenv("PYAUTOFLIP_FINAL_ENCODER", "cpu")
    monkeypatch.setenv("PYAUTOFLIP_X264_PRESET", "fast")
    [cpu] = final_encoders()
    assert cpu[1] == "libx264" and cpu[cpu.index("-preset") + 1] == "fast"
