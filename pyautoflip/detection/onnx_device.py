"""Where pyautoflip's ONNX models run: the GPU when the box has one, else the CPU.

UNISAL saliency and InsightFace (detection, 106-point landmarks) are ONNX
models. With onnxruntime's GPU build installed (a worker image built with
ONNXRUNTIME_GPU=1) they run on CUDA: the same crop windows to the last digit,
a clip analyzed in two thirds of the time with a third of the CPU, which the
renders on the same box get back. The CPU build, in every other image, lists
no CUDA provider, so nothing changes there.

PYAUTOFLIP_ONNX_DEVICE: `auto` (default) uses CUDA when it is there; `cpu`
keeps a GPU box on the CPU without a rebuild.
"""

import logging
import os
from functools import lru_cache
from typing import Tuple

logger = logging.getLogger("autoflip.detection.onnx_device")

CPU = ("CPUExecutionProvider",)
CUDA = ("CUDAExecutionProvider", "CPUExecutionProvider")


@lru_cache(maxsize=1)
def onnx_providers() -> Tuple[str, ...]:
    """The execution providers for every ONNX session, decided once per process."""
    setting = os.environ.get("PYAUTOFLIP_ONNX_DEVICE", "auto").strip().lower()
    if setting == "cpu":
        return CPU
    if setting != "auto":
        logger.warning(f"PYAUTOFLIP_ONNX_DEVICE={setting!r} is not auto or cpu; using auto")

    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return CPU
    # The GPU build finds CUDA and cuDNN in their pip wheels only once they
    # are loaded; without them the session would quietly fall back to the CPU
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        try:
            preload()
        except Exception as e:
            logger.warning(f"Could not load the CUDA libraries, running ONNX models on the CPU: {e}")
            return CPU
    return CUDA


def on_gpu() -> bool:
    return onnx_providers()[0] == "CUDAExecutionProvider"
