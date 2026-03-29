"""Tests for RealESRGANRunner: weights loading, device, precision, output dims."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from upscale_image.config import resolve_config
from upscale_image.models.realesrgan import RealESRGANRunner

# Tiny arch params — fast init, same code path as production
_TINY = dict(num_feat=4, num_block=1, num_grow_ch=4)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_runner(weights_path: str, scale: int = 4) -> RealESRGANRunner:
    return RealESRGANRunner(scale=scale, weights_path=weights_path, **_TINY)


def _save_tiny_weights(path: Path, scale: int = 4) -> None:
    """Create a tiny randomly-initialised RRDBNet and save its state dict."""
    import upscale_image.models._compat  # noqa: F401
    from basicsr.archs.rrdbnet_arch import RRDBNet

    net = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=_TINY["num_feat"],
        num_block=_TINY["num_block"],
        num_grow_ch=_TINY["num_grow_ch"],
        scale=scale,
    )
    torch.save(net.state_dict(), str(path))


def _cfg(device: str = "cpu", precision: str = "fp32", scale: int = 4):
    return resolve_config(
        input_dir="/in", output_dir="/out",
        device=device, scale=scale,
    )


# ---------------------------------------------------------------------------
# Contract / metadata
# ---------------------------------------------------------------------------

def test_name():
    runner = RealESRGANRunner(scale=4, weights_path="weights/any.pth", **_TINY)
    assert runner.name == "realesrgan-x4"


def test_scale():
    runner = RealESRGANRunner(scale=4, weights_path="weights/any.pth", **_TINY)
    assert runner.scale == 4


def test_not_loaded_initially():
    runner = RealESRGANRunner(scale=4, weights_path="weights/any.pth", **_TINY)
    assert runner.is_loaded is False


# ---------------------------------------------------------------------------
# load() — weights handling
# ---------------------------------------------------------------------------

def test_load_missing_weights_raises(tmp_path):
    runner = _tiny_runner(str(tmp_path / "missing.pth"))
    with pytest.raises(FileNotFoundError, match="Weights file not found"):
        runner.load()


def test_load_sets_is_loaded(tmp_path):
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))
    runner.load()
    assert runner.is_loaded is True


def test_load_raw_state_dict(tmp_path):
    """Weights saved directly as state dict (no 'params'/'params_ema' key)."""
    w = tmp_path / "raw.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))
    runner.load()
    assert runner.is_loaded


def test_load_params_ema_layout(tmp_path):
    """Weights nested under 'params_ema' key (standard Real-ESRGAN layout)."""
    import upscale_image.models._compat  # noqa: F401
    from basicsr.archs.rrdbnet_arch import RRDBNet

    net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, **_TINY)
    w = tmp_path / "ema.pth"
    torch.save({"params_ema": net.state_dict()}, str(w))

    runner = _tiny_runner(str(w))
    runner.load()
    assert runner.is_loaded


def test_load_params_layout(tmp_path):
    """Weights nested under 'params' key."""
    import upscale_image.models._compat  # noqa: F401
    from basicsr.archs.rrdbnet_arch import RRDBNet

    net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, **_TINY)
    w = tmp_path / "params.pth"
    torch.save({"params": net.state_dict()}, str(w))

    runner = _tiny_runner(str(w))
    runner.load()
    assert runner.is_loaded


# ---------------------------------------------------------------------------
# upscale() — output correctness
# ---------------------------------------------------------------------------

def test_upscale_requires_load(tmp_path):
    runner = _tiny_runner(str(tmp_path / "x.pth"))
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="not loaded"):
        runner.upscale(img, _cfg())


def test_upscale_output_shape_4x(tmp_path):
    w = tmp_path / "model.pth"
    _save_tiny_weights(w, scale=4)
    runner = _tiny_runner(str(w), scale=4)
    runner.load()
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = runner.upscale(img, _cfg(scale=4))
    assert out.shape == (32, 32, 3)


def test_upscale_output_dtype_uint8(tmp_path):
    w = tmp_path / "model.pth"
    _save_tiny_weights(w, scale=4)
    runner = _tiny_runner(str(w))
    runner.load()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    out = runner.upscale(img, _cfg())
    assert out.dtype == np.uint8


def test_upscale_output_values_in_range(tmp_path):
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))
    runner.load()
    img = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)
    out = runner.upscale(img, _cfg())
    assert out.min() >= 0
    assert out.max() <= 255


# ---------------------------------------------------------------------------
# unload()
# ---------------------------------------------------------------------------

def test_unload_clears_loaded(tmp_path):
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))
    runner.load()
    runner.unload()
    assert runner.is_loaded is False


def test_upscale_after_unload_raises(tmp_path):
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))
    runner.load()
    runner.unload()
    with pytest.raises(RuntimeError, match="not loaded"):
        runner.upscale(np.zeros((4, 4, 3), dtype=np.uint8), _cfg())


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------

def test_cuda_unavailable_raises_runtime_error(tmp_path, monkeypatch):
    """Requesting CUDA on a CPU-only system raises RuntimeError."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))
    runner.load()
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        runner.upscale(np.zeros((4, 4, 3), dtype=np.uint8), _cfg(device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_upscale_on_cuda(tmp_path):
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))
    runner.load()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    out = runner.upscale(img, _cfg(device="cuda"))
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# Default registry integration
# ---------------------------------------------------------------------------

def test_realesrgan_x4_registered_in_default_registry():
    from upscale_image.models.registry import available_models
    assert "realesrgan-x4" in available_models()
