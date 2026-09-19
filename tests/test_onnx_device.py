"""Which device the ONNX models (UNISAL saliency, InsightFace) are put on.

onnxruntime is faked: what matters is the providers each build lists, and
the two places that would quietly put the models back on the CPU.
"""

import sys
import types

import pytest

from pyautoflip.detection import face_detector, onnx_device, saliency_detector

CPU_BUILD = ["AzureExecutionProvider", "CPUExecutionProvider"]
GPU_BUILD = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
ON_CUDA = ("CUDAExecutionProvider", "CPUExecutionProvider")
ON_CPU = ("CPUExecutionProvider",)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.delenv("PYAUTOFLIP_ONNX_DEVICE", raising=False)
    onnx_device.onnx_providers.cache_clear()
    yield
    onnx_device.onnx_providers.cache_clear()


class FakeSession:
    def __init__(self, path=None, providers=None):
        self.providers = list(providers or ON_CPU)

    def get_providers(self):
        return self.providers


def fake_ort(monkeypatch, available, preload=None):
    ort = types.SimpleNamespace(get_available_providers=lambda: list(available), InferenceSession=FakeSession)
    if preload is not None:
        ort.preload_dlls = preload
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    return ort


def test_the_cpu_build_stays_on_the_cpu(monkeypatch):
    fake_ort(monkeypatch, CPU_BUILD)
    assert onnx_device.onnx_providers() == ON_CPU
    assert not onnx_device.on_gpu()


def test_the_gpu_build_runs_on_cuda_after_loading_its_libraries(monkeypatch):
    loaded = []
    fake_ort(monkeypatch, GPU_BUILD, preload=lambda: loaded.append(True))
    assert onnx_device.onnx_providers() == ON_CUDA
    assert onnx_device.on_gpu()
    assert loaded == [True]


def test_cpu_keeps_a_gpu_box_on_the_cpu(monkeypatch):
    monkeypatch.setenv("PYAUTOFLIP_ONNX_DEVICE", "cpu")
    fake_ort(monkeypatch, GPU_BUILD, preload=lambda: None)
    assert onnx_device.onnx_providers() == ON_CPU


def test_an_unknown_setting_is_auto(monkeypatch):
    monkeypatch.setenv("PYAUTOFLIP_ONNX_DEVICE", "gpu")
    fake_ort(monkeypatch, GPU_BUILD, preload=lambda: None)
    assert onnx_device.onnx_providers() == ON_CUDA


def test_cuda_libraries_that_do_not_load_mean_the_cpu(monkeypatch):
    def missing():
        raise OSError("libcudnn.so.9: cannot open shared object file")

    fake_ort(monkeypatch, GPU_BUILD, preload=missing)
    assert onnx_device.onnx_providers() == ON_CPU


@pytest.mark.parametrize("available, providers, ctx_id", [(GPU_BUILD, list(ON_CUDA), 0), (CPU_BUILD, list(ON_CPU), -1)])
def test_insightface_keeps_the_device_it_was_given(monkeypatch, available, providers, ctx_id):
    # prepare(ctx_id=-1) resets every model to the CPU, whatever the providers
    seen = {}

    class FakeFaceAnalysis:
        def __init__(self, name, providers, allowed_modules):
            seen["providers"] = providers
            self.models = {"detection": types.SimpleNamespace(session=FakeSession(providers=providers))}

        def prepare(self, ctx_id, det_size):
            seen["ctx_id"] = ctx_id

    fake_ort(monkeypatch, available, preload=lambda: None)
    monkeypatch.setattr(face_detector, "FaceAnalysis", FakeFaceAnalysis)
    monkeypatch.setattr(face_detector.FaceDetector, "_app", None)
    face_detector.FaceDetector.get_face_analyzer()
    assert seen == {"providers": providers, "ctx_id": ctx_id}


@pytest.mark.parametrize("available, providers", [(GPU_BUILD, list(ON_CUDA)), (CPU_BUILD, list(ON_CPU))])
def test_unisal_runs_where_the_box_can(monkeypatch, tmp_path, available, providers):
    model = tmp_path / "unisal.onnx"
    model.write_bytes(b"")
    fake_ort(monkeypatch, available, preload=lambda: None)
    monkeypatch.setattr(saliency_detector, "_ONNX_MODEL_PATH", model)
    monkeypatch.setattr(saliency_detector.SaliencyDetector, "_session", None)
    assert saliency_detector.SaliencyDetector._get_session().get_providers() == providers
