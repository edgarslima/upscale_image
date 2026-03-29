"""Tests for model contract (base ABC) and mock runner."""

from __future__ import annotations

import numpy as np
import pytest

from upscale_image.config import resolve_config
from upscale_image.models import MockSuperResolutionModel, SuperResolutionModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg():
    return resolve_config(input_dir="/in", output_dir="/out")


def _image(h: int = 8, w: int = 8) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Abstract base — cannot instantiate directly
# ---------------------------------------------------------------------------

def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        SuperResolutionModel()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# MockSuperResolutionModel — contract compliance
# ---------------------------------------------------------------------------

def test_mock_is_subclass_of_base():
    assert issubclass(MockSuperResolutionModel, SuperResolutionModel)


def test_mock_name():
    model = MockSuperResolutionModel()
    assert model.name == "mock"


def test_mock_scale_default():
    model = MockSuperResolutionModel()
    assert model.scale == 4


def test_mock_scale_custom():
    model = MockSuperResolutionModel(scale=2)
    assert model.scale == 2


def test_not_loaded_before_load():
    model = MockSuperResolutionModel()
    assert model.is_loaded is False


def test_loaded_after_load():
    model = MockSuperResolutionModel()
    model.load()
    assert model.is_loaded is True


def test_not_loaded_after_unload():
    model = MockSuperResolutionModel()
    model.load()
    model.unload()
    assert model.is_loaded is False


# ---------------------------------------------------------------------------
# upscale — output dimensions and type
# ---------------------------------------------------------------------------

def test_upscale_raises_if_not_loaded():
    model = MockSuperResolutionModel(scale=4)
    with pytest.raises(RuntimeError, match="not loaded"):
        model.upscale(_image(), _cfg())


def test_upscale_output_shape_4x():
    model = MockSuperResolutionModel(scale=4)
    model.load()
    out = model.upscale(_image(8, 8), _cfg())
    assert out.shape == (32, 32, 3)


def test_upscale_output_shape_2x():
    model = MockSuperResolutionModel(scale=2)
    model.load()
    out = model.upscale(_image(6, 10), _cfg())
    assert out.shape == (12, 20, 3)


def test_upscale_returns_uint8():
    model = MockSuperResolutionModel(scale=4)
    model.load()
    out = model.upscale(_image(), _cfg())
    assert out.dtype == np.uint8


def test_upscale_output_is_ndarray():
    model = MockSuperResolutionModel(scale=4)
    model.load()
    out = model.upscale(_image(), _cfg())
    assert isinstance(out, np.ndarray)


# ---------------------------------------------------------------------------
# Pipeline decoupling — interact only via the base interface
# ---------------------------------------------------------------------------

def test_pipeline_works_via_abstract_interface():
    """Pipeline code typed as SuperResolutionModel can drive the mock."""
    model: SuperResolutionModel = MockSuperResolutionModel(scale=4)
    cfg = _cfg()
    model.load()
    assert model.is_loaded
    out = model.upscale(_image(4, 4), cfg)
    assert out.shape == (16, 16, 3)
    model.unload()
    assert not model.is_loaded


def test_reload_after_unload():
    model = MockSuperResolutionModel()
    model.load()
    model.unload()
    model.load()
    assert model.is_loaded
    out = model.upscale(_image(), _cfg())
    assert out.shape[0] == 32
