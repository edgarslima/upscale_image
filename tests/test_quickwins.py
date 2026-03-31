"""Tests for Passo 24 quick-win optimizations.

Covers:
- tile_pad default = 32
- cudnn.benchmark activation on CUDA
- torch.compile activation on CUDA
- torch.autocast replacing manual AMP
- _hann_window shape, dtype and value properties
- _upscale_tiled output shape with Hann blending
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from upscale_image.config import resolve_config
from upscale_image.config.schema import RuntimeConfig
from upscale_image.models.realesrgan import RealESRGANRunner

_TINY = dict(num_feat=4, num_block=1, num_grow_ch=4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_tiny_weights(path: Path, scale: int = 4) -> None:
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


def _tiny_runner(weights_path: str, scale: int = 4) -> RealESRGANRunner:
    return RealESRGANRunner(scale=scale, weights_path=weights_path, **_TINY)


def _cfg(device: str = "cpu", scale: int = 4):
    return resolve_config(input_dir="/in", output_dir="/out", device=device, scale=scale)


class _ScaleNet(nn.Module):
    """Mock nn.Module: returns zero-filled tensor scaled by `scale`."""

    def __init__(self, scale: int, channels: int = 3) -> None:
        super().__init__()
        self._scale = scale
        self._channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        return x.new_zeros(b, self._channels, h * self._scale, w * self._scale)


# ---------------------------------------------------------------------------
# tile_pad default
# ---------------------------------------------------------------------------

def test_tile_pad_default_is_32():
    assert RuntimeConfig().tile_pad == 32


def test_tile_pad_default_via_resolve_config():
    cfg = resolve_config(input_dir="/in", output_dir="/out")
    assert cfg.runtime.tile_pad == 32


# ---------------------------------------------------------------------------
# cudnn.benchmark
# ---------------------------------------------------------------------------

def test_cudnn_benchmark_set_when_cuda_available(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))

    with patch("torch.compile", side_effect=lambda m, **kw: m):
        runner.load()

    assert torch.backends.cudnn.benchmark is True


def test_cudnn_benchmark_not_explicitly_set_without_cuda(tmp_path, monkeypatch):
    # When CUDA is not available the branch is never entered; benchmark stays
    # at its prior value (could be True from another test, so just verify no crash).
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))
    runner.load()  # must not raise
    assert runner.is_loaded


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------

def test_torch_compile_called_with_reduce_overhead(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))

    with patch("torch.compile", side_effect=lambda m, **kw: m) as mock_compile:
        runner.load()

    mock_compile.assert_called_once()
    _, kwargs = mock_compile.call_args
    assert kwargs.get("mode") == "reduce-overhead"


def test_torch_compile_not_called_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))

    with patch("torch.compile") as mock_compile:
        runner.load()

    mock_compile.assert_not_called()


# ---------------------------------------------------------------------------
# torch.autocast (AMP)
# ---------------------------------------------------------------------------

def test_autocast_called_with_enabled_false_for_fp32_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))
    runner.load()

    with patch("torch.autocast") as mock_ac:
        mock_ac.return_value.__enter__ = MagicMock(return_value=None)
        mock_ac.return_value.__exit__ = MagicMock(return_value=False)
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        runner.upscale(img, _cfg(device="cpu"))

    mock_ac.assert_called_once()
    assert mock_ac.call_args.kwargs["enabled"] is False


def test_upscale_fp32_cpu_produces_correct_output(tmp_path, monkeypatch):
    """After autocast refactor, FP32 CPU path must still produce correct results."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))
    runner.load()
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = runner.upscale(img, _cfg())
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


# ---------------------------------------------------------------------------
# _hann_window
# ---------------------------------------------------------------------------

def test_hann_window_shape():
    w = RealESRGANRunner._hann_window(8, 16, device=torch.device("cpu"))
    assert w.shape == (1, 1, 8, 16)


def test_hann_window_dtype_float32():
    w = RealESRGANRunner._hann_window(4, 4, device=torch.device("cpu"))
    assert w.dtype == torch.float32


def test_hann_window_values_between_zero_and_one():
    w = RealESRGANRunner._hann_window(16, 16, device=torch.device("cpu"))
    assert w.min().item() >= 0.0
    assert w.max().item() <= 1.0 + 1e-6


def test_hann_window_centre_near_one():
    h, w_size = 16, 16
    win = RealESRGANRunner._hann_window(h, w_size, device=torch.device("cpu"))
    centre = win[0, 0, h // 2, w_size // 2].item()
    assert centre > 0.9, f"Expected centre near 1.0, got {centre:.4f}"


def test_hann_window_corners_are_zero():
    win = RealESRGANRunner._hann_window(8, 8, device=torch.device("cpu"))
    assert win[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert win[0, 0, 0, -1].item() == pytest.approx(0.0, abs=1e-6)
    assert win[0, 0, -1, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert win[0, 0, -1, -1].item() == pytest.approx(0.0, abs=1e-6)


def test_hann_window_asymmetric_dimensions():
    """Non-square window must have correct shape."""
    w = RealESRGANRunner._hann_window(6, 10, device=torch.device("cpu"))
    assert w.shape == (1, 1, 6, 10)


# ---------------------------------------------------------------------------
# _upscale_tiled with Hann blending
# ---------------------------------------------------------------------------

def _tiled_cfg(tile_size: int, tile_pad: int = 2, scale: int = 4):
    cfg = resolve_config(input_dir="/in", output_dir="/out", scale=scale)
    cfg.runtime.tile_size = tile_size
    cfg.runtime.tile_pad = tile_pad
    return cfg


def test_tiled_output_shape_single_tile():
    """Image fits in one tile: output is (1, C, H*scale, W*scale)."""
    runner = RealESRGANRunner(scale=4, weights_path="weights/x.pth")
    net = _ScaleNet(scale=4)
    tensor = torch.zeros(1, 3, 8, 8)
    cfg = _tiled_cfg(tile_size=16, tile_pad=2)
    out = runner._upscale_tiled(tensor, net, cfg)
    assert out.shape == (1, 3, 32, 32)


def test_tiled_output_shape_four_tiles():
    """2×2 tile grid: output shape must match full scaled image."""
    runner = RealESRGANRunner(scale=4, weights_path="weights/x.pth")
    net = _ScaleNet(scale=4)
    tensor = torch.zeros(1, 3, 16, 16)
    cfg = _tiled_cfg(tile_size=8, tile_pad=2)
    out = runner._upscale_tiled(tensor, net, cfg)
    assert out.shape == (1, 3, 64, 64)


def test_tiled_output_shape_non_square():
    """Non-square image: output shape must scale both dimensions correctly."""
    runner = RealESRGANRunner(scale=4, weights_path="weights/x.pth")
    net = _ScaleNet(scale=4)
    tensor = torch.zeros(1, 3, 12, 20)
    cfg = _tiled_cfg(tile_size=8, tile_pad=2)
    out = runner._upscale_tiled(tensor, net, cfg)
    assert out.shape == (1, 3, 48, 80)


def test_tiled_output_no_nan_or_inf():
    """Hann weighting must never produce NaN or Inf."""
    runner = RealESRGANRunner(scale=4, weights_path="weights/x.pth")
    net = _ScaleNet(scale=4)
    tensor = torch.zeros(1, 3, 8, 8)
    cfg = _tiled_cfg(tile_size=4, tile_pad=1)
    out = runner._upscale_tiled(tensor, net, cfg)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_upscale_with_tiling_end_to_end(tmp_path, monkeypatch):
    """Full upscale with tile_size > 0 must return correct shape and dtype."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)
    runner = _tiny_runner(str(w))
    runner.load()

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    cfg = _cfg(device="cpu")
    cfg.runtime.tile_size = 8
    cfg.runtime.tile_pad = 4

    out = runner.upscale(img, cfg)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
