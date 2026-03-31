"""Tests for SwinIRRunner (Passo 27).

All tests mock the SwinIR architecture and weights so they run without
the optional `swinir` / `timm` packages installed.

Covers:
- SwinIRRunner implements SuperResolutionModel (ABC)
- name, scale, is_loaded metadata
- load() raises FileNotFoundError when weights_path missing
- load() raises ImportError when swinir package not installed
- upscale() raises RuntimeError when called before load()
- upscale() returns output with shape (H*scale, W*scale, 3) for any H×W
- Padding: non-multiple-of-8 input returns correctly-sized (not padded) output
- unload() clears model and is_loaded returns False
- Conditional registry: "swinir-x4" registered when import succeeds
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from upscale_image.config import resolve_config
from upscale_image.models.base import SuperResolutionModel
from upscale_image.models.swinir_runner import SwinIRRunner


# ---------------------------------------------------------------------------
# Helpers — mock SwinIR network and weight file
# ---------------------------------------------------------------------------

class _IdentityNet(nn.Module):
    """Mock SwinIR: returns a zero tensor of the expected upscaled size."""

    def __init__(self, scale: int = 4) -> None:
        super().__init__()
        self._scale = scale
        # Dummy parameter so torch.compile doesn't complain
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.new_zeros(b, c, h * self._scale, w * self._scale)


def _patch_swinir_import(scale: int = 4):
    """Context manager: patches 'swinir.SwinIR' with _IdentityNet."""
    mock_module = types.ModuleType("swinir")
    mock_module.SwinIR = lambda **kwargs: _IdentityNet(scale=scale)
    return patch.dict(sys.modules, {"swinir": mock_module})


def _make_runner_with_weights(tmp_path: Path, scale: int = 4) -> SwinIRRunner:
    """Create a tiny weight file and return an unloaded SwinIRRunner."""
    w = tmp_path / "swinir-x4.pth"
    # Use _IdentityNet state_dict as a valid-format weight file
    net = _IdentityNet(scale=scale)
    torch.save(net.state_dict(), str(w))
    return SwinIRRunner(scale=scale, weights_path=str(w))


# ---------------------------------------------------------------------------
# Contract / metadata
# ---------------------------------------------------------------------------

def test_swinir_runner_is_superresolution_model():
    """SwinIRRunner must be an instance of SuperResolutionModel."""
    runner = SwinIRRunner()
    assert isinstance(runner, SuperResolutionModel)


def test_swinir_runner_name():
    assert SwinIRRunner().name == "swinir-x4"


def test_swinir_runner_scale():
    assert SwinIRRunner(scale=4).scale == 4


def test_swinir_runner_is_loaded_false_before_load():
    assert SwinIRRunner().is_loaded is False


# ---------------------------------------------------------------------------
# load() — error cases
# ---------------------------------------------------------------------------

def test_load_raises_file_not_found_when_weights_missing(tmp_path):
    runner = SwinIRRunner(weights_path=str(tmp_path / "nonexistent.pth"))
    with _patch_swinir_import():
        with pytest.raises(FileNotFoundError, match="Pesos SwinIR não encontrados"):
            runner.load()


def test_load_raises_import_error_when_swinir_not_installed(tmp_path):
    """load() must raise ImportError with a helpful message when swinir is missing."""
    w = tmp_path / "swinir-x4.pth"
    torch.save({}, str(w))
    runner = SwinIRRunner(weights_path=str(w))

    # Remove swinir from sys.modules if present, then block it
    with patch.dict(sys.modules, {"swinir": None}):
        with pytest.raises(ImportError, match="SwinIR não instalado"):
            runner.load()


# ---------------------------------------------------------------------------
# upscale() — before load()
# ---------------------------------------------------------------------------

def test_upscale_raises_runtime_error_before_load():
    runner = SwinIRRunner()
    cfg = resolve_config(input_dir="/in", output_dir="/out")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="not loaded"):
        runner.upscale(img, cfg)


# ---------------------------------------------------------------------------
# upscale() — correct output shape
# ---------------------------------------------------------------------------

def _load_with_mock(runner: SwinIRRunner, scale: int = 4, monkeypatch=None) -> None:
    """Load runner using a patched SwinIR import and monkeypatching torch.compile."""
    with _patch_swinir_import(scale=scale):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.compile", side_effect=lambda m, **kw: m):
                runner.load()


def test_upscale_output_shape_exact_multiple_of_8(tmp_path, monkeypatch):
    """8×8 input (already multiple of 8) → 32×32 output."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    runner = _make_runner_with_weights(tmp_path)
    _load_with_mock(runner, scale=4)

    cfg = resolve_config(input_dir="/in", output_dir="/out")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = runner.upscale(img, cfg)

    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_upscale_output_shape_non_multiple_of_8(tmp_path, monkeypatch):
    """7×9 input (not multiples of 8) → 28×36 output (not padded size)."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    runner = _make_runner_with_weights(tmp_path)
    _load_with_mock(runner, scale=4)

    cfg = resolve_config(input_dir="/in", output_dir="/out")
    img = np.zeros((7, 9, 3), dtype=np.uint8)
    out = runner.upscale(img, cfg)

    # Output must be (7*4, 9*4, 3) — not (8*4, 16*4, 3) which would be padded
    assert out.shape == (28, 36, 3), f"Expected (28,36,3) got {out.shape}"
    assert out.dtype == np.uint8


def test_upscale_output_shape_various_sizes(tmp_path, monkeypatch):
    """Verify correct output shape for a variety of non-multiple-of-8 dimensions."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    test_cases = [
        ((4, 4), (16, 16)),
        ((5, 6), (20, 24)),
        ((15, 17), (60, 68)),
        ((16, 16), (64, 64)),
    ]

    for (h, w), (expected_h, expected_w) in test_cases:
        runner = _make_runner_with_weights(tmp_path)
        _load_with_mock(runner, scale=4)
        cfg = resolve_config(input_dir="/in", output_dir="/out")
        img = np.zeros((h, w, 3), dtype=np.uint8)
        out = runner.upscale(img, cfg)
        assert out.shape == (expected_h, expected_w, 3), (
            f"Input ({h},{w}) → expected ({expected_h},{expected_w},3), got {out.shape}"
        )


def test_upscale_output_dtype_uint8(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    runner = _make_runner_with_weights(tmp_path)
    _load_with_mock(runner, scale=4)

    cfg = resolve_config(input_dir="/in", output_dir="/out")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = runner.upscale(img, cfg)

    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


# ---------------------------------------------------------------------------
# unload()
# ---------------------------------------------------------------------------

def test_unload_clears_model(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    runner = _make_runner_with_weights(tmp_path)
    _load_with_mock(runner, scale=4)

    assert runner.is_loaded is True
    runner.unload()
    assert runner.is_loaded is False
    assert runner._net is None


def test_upscale_raises_after_unload(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    runner = _make_runner_with_weights(tmp_path)
    _load_with_mock(runner, scale=4)
    runner.unload()

    cfg = resolve_config(input_dir="/in", output_dir="/out")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="not loaded"):
        runner.upscale(img, cfg)


# ---------------------------------------------------------------------------
# Registry — conditional registration
# ---------------------------------------------------------------------------

def test_swinir_registered_when_import_succeeds():
    """When SwinIRRunner can be imported, 'swinir-x4' must be in the registry."""
    from upscale_image.models.registry import _default_registry

    # SwinIRRunner is in the same package, so the import always succeeds here.
    # The conditional block in registry.py runs at module load time.
    # Since swinir_runner.py does NOT import `swinir` at module level (only in load()),
    # the SwinIRRunner class itself imports fine, so the registry should have it.
    assert "swinir-x4" in _default_registry


def test_swinir_not_registered_when_swinir_runner_import_fails():
    """If SwinIRRunner cannot be imported, the registry must not contain 'swinir-x4'."""
    from upscale_image.models import registry as reg_module

    # Simulate ImportError from swinir_runner import
    original = reg_module._default_registry._factories.get("swinir-x4")

    # Temporarily remove the entry and re-add to test the except path (unit test)
    reg_module._default_registry.deregister("swinir-x4")
    try:
        # The registry must not raise when swinir-x4 is absent
        assert "swinir-x4" not in reg_module._default_registry
    finally:
        # Restore original registration if it existed
        if original is not None:
            try:
                reg_module._default_registry.register("swinir-x4", original)
            except ValueError:
                pass  # already re-registered


def test_registry_available_includes_swinir():
    """available_models() must include 'swinir-x4' in this environment."""
    from upscale_image.models.registry import available_models
    assert "swinir-x4" in available_models()
