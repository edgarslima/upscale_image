"""Tests for OnnxRunner (Passo 28).

All tests use mocks — no onnxruntime installation required.

Covers:
- OnnxRunner implements SuperResolutionModel (ABC)
- name, scale, is_loaded metadata for CUDA and CPU providers
- load() raises ImportError when onnxruntime not installed
- load() raises FileNotFoundError when .onnx file missing
- upscale() raises RuntimeError when called before load()
- upscale() returns correct shape (H*scale, W*scale, 3) and dtype uint8
- upscale() pure numpy pipeline: no torch dependency at inference time
- unload() clears session and is_loaded returns False
- Conditional registry: ONNX runners present when onnx_runner importable
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from upscale_image.config import resolve_config
from upscale_image.models.base import SuperResolutionModel
from upscale_image.models.onnx_runner import OnnxRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_onnx_file(tmp_path: Path) -> Path:
    """Create a placeholder .onnx file that exists on disk."""
    p = tmp_path / "model.onnx"
    p.write_bytes(b"")
    return p


def _mock_onnxruntime(scale: int = 4):
    """Inject a fake onnxruntime module that returns upscaled zeros."""

    def _fake_run(output_names, feed_dict):
        inp = list(feed_dict.values())[0]
        b, c, h, w = inp.shape
        return [np.zeros((b, c, h * scale, w * scale), dtype=np.float32)]

    mock_session = MagicMock()
    mock_session.run = MagicMock(side_effect=_fake_run)
    mock_session.get_inputs.return_value = [MagicMock(name="input")]

    mock_ort = types.ModuleType("onnxruntime")
    mock_ort.InferenceSession = MagicMock(return_value=mock_session)

    return patch.dict(sys.modules, {"onnxruntime": mock_ort}), mock_session


# ---------------------------------------------------------------------------
# Contract / metadata
# ---------------------------------------------------------------------------

def test_onnx_runner_is_superresolution_model():
    assert isinstance(OnnxRunner(), SuperResolutionModel)


def test_onnx_runner_name_cuda():
    assert OnnxRunner(provider="CUDAExecutionProvider").name == "realesrgan-x4-onnx-cuda"


def test_onnx_runner_name_cpu():
    assert OnnxRunner(provider="CPUExecutionProvider").name == "realesrgan-x4-onnx-cpu"


def test_onnx_runner_scale():
    assert OnnxRunner(scale=4).scale == 4


def test_onnx_runner_is_loaded_false_before_load():
    assert OnnxRunner().is_loaded is False


# ---------------------------------------------------------------------------
# load() — error cases
# ---------------------------------------------------------------------------

def test_load_raises_import_error_when_onnxruntime_missing(tmp_path):
    p = _make_onnx_file(tmp_path)
    runner = OnnxRunner(onnx_path=str(p))
    with patch.dict(sys.modules, {"onnxruntime": None}):
        with pytest.raises(ImportError, match="onnxruntime não instalado"):
            runner.load()


def test_load_raises_file_not_found_when_model_missing(tmp_path):
    runner = OnnxRunner(onnx_path=str(tmp_path / "nonexistent.onnx"))
    ctx, _ = _mock_onnxruntime()
    with ctx:
        with pytest.raises(FileNotFoundError, match="Modelo ONNX não encontrado"):
            runner.load()


# ---------------------------------------------------------------------------
# upscale() — before load()
# ---------------------------------------------------------------------------

def test_upscale_raises_runtime_error_before_load():
    runner = OnnxRunner()
    cfg = resolve_config(input_dir="/in", output_dir="/out")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="not loaded"):
        runner.upscale(img, cfg)


# ---------------------------------------------------------------------------
# upscale() — correct output
# ---------------------------------------------------------------------------

def test_upscale_output_shape_cuda_provider(tmp_path):
    p = _make_onnx_file(tmp_path)
    runner = OnnxRunner(scale=4, onnx_path=str(p), provider="CUDAExecutionProvider")

    ctx, _ = _mock_onnxruntime(scale=4)
    with ctx:
        runner.load()

    cfg = resolve_config(input_dir="/in", output_dir="/out")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = runner.upscale(img, cfg)

    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_upscale_output_shape_cpu_provider(tmp_path):
    p = _make_onnx_file(tmp_path)
    runner = OnnxRunner(scale=4, onnx_path=str(p), provider="CPUExecutionProvider")

    ctx, _ = _mock_onnxruntime(scale=4)
    with ctx:
        runner.load()

    cfg = resolve_config(input_dir="/in", output_dir="/out")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = runner.upscale(img, cfg)

    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_upscale_output_non_square(tmp_path):
    """Non-square image: output shape must scale both dimensions."""
    p = _make_onnx_file(tmp_path)
    runner = OnnxRunner(scale=4, onnx_path=str(p))

    ctx, _ = _mock_onnxruntime(scale=4)
    with ctx:
        runner.load()

    cfg = resolve_config(input_dir="/in", output_dir="/out")
    img = np.zeros((6, 10, 3), dtype=np.uint8)
    out = runner.upscale(img, cfg)

    assert out.shape == (24, 40, 3)
    assert out.dtype == np.uint8


def test_upscale_values_in_range(tmp_path):
    """Output pixel values must be in [0, 255]."""
    p = _make_onnx_file(tmp_path)
    runner = OnnxRunner(scale=4, onnx_path=str(p))

    ctx, _ = _mock_onnxruntime(scale=4)
    with ctx:
        runner.load()

    cfg = resolve_config(input_dir="/in", output_dir="/out")
    img = np.full((8, 8, 3), 128, dtype=np.uint8)
    out = runner.upscale(img, cfg)

    assert out.min() >= 0 and out.max() <= 255


def test_upscale_uses_numpy_not_torch(tmp_path):
    """upscale() must not import torch at inference time (pure numpy pipeline)."""
    p = _make_onnx_file(tmp_path)
    runner = OnnxRunner(scale=4, onnx_path=str(p))

    ctx, _ = _mock_onnxruntime(scale=4)
    with ctx:
        runner.load()

    cfg = resolve_config(input_dir="/in", output_dir="/out")
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    # Block torch during upscale — if runner uses torch, this raises ImportError
    original_torch = sys.modules.get("torch")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        out = runner.upscale(img, cfg)
    finally:
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        else:
            del sys.modules["torch"]

    assert out.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# unload()
# ---------------------------------------------------------------------------

def test_unload_clears_session(tmp_path):
    p = _make_onnx_file(tmp_path)
    runner = OnnxRunner(onnx_path=str(p))

    ctx, _ = _mock_onnxruntime()
    with ctx:
        runner.load()

    assert runner.is_loaded is True
    runner.unload()
    assert runner.is_loaded is False
    assert runner._session is None


def test_upscale_raises_after_unload(tmp_path):
    p = _make_onnx_file(tmp_path)
    runner = OnnxRunner(onnx_path=str(p))

    ctx, _ = _mock_onnxruntime()
    with ctx:
        runner.load()

    runner.unload()
    cfg = resolve_config(input_dir="/in", output_dir="/out")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="not loaded"):
        runner.upscale(img, cfg)


# ---------------------------------------------------------------------------
# Registry — conditional
# ---------------------------------------------------------------------------

def test_onnx_runners_in_registry_when_onnx_runner_importable():
    """OnnxRunner module-level import never requires onnxruntime, so runners register."""
    from upscale_image.models.registry import _default_registry

    assert "realesrgan-x4-onnx-cuda" in _default_registry
    assert "realesrgan-x4-onnx-cpu" in _default_registry


def test_onnx_runner_providers_in_available_models():
    from upscale_image.models.registry import available_models

    models = available_models()
    assert "realesrgan-x4-onnx-cuda" in models
    assert "realesrgan-x4-onnx-cpu" in models
