"""Tests for ModelRegistry: registration, resolution, error handling."""

from __future__ import annotations

import pytest

from upscale_image.config import resolve_config
from upscale_image.models import MockSuperResolutionModel, SuperResolutionModel
from upscale_image.models.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(model: str = "mock", scale: int = 4):
    return resolve_config(input_dir="/in", output_dir="/out", model=model, scale=scale)


def _mock_factory(scale: int) -> SuperResolutionModel:
    return MockSuperResolutionModel(scale=scale)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_register_and_contains():
    reg = ModelRegistry()
    reg.register("mymodel", _mock_factory)
    assert "mymodel" in reg


def test_available_returns_sorted_names():
    reg = ModelRegistry()
    reg.register("zebra", _mock_factory)
    reg.register("alpha", _mock_factory)
    assert reg.available() == ["alpha", "zebra"]


def test_register_duplicate_raises():
    reg = ModelRegistry()
    reg.register("dup", _mock_factory)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("dup", _mock_factory)


def test_deregister_removes_name():
    reg = ModelRegistry()
    reg.register("temp", _mock_factory)
    reg.deregister("temp")
    assert "temp" not in reg


def test_deregister_nonexistent_is_silent():
    reg = ModelRegistry()
    reg.deregister("nonexistent")  # must not raise


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_resolve_returns_correct_type():
    reg = ModelRegistry()
    reg.register("mock", _mock_factory)
    cfg = _cfg(model="mock", scale=4)
    model = reg.resolve(cfg)
    assert isinstance(model, SuperResolutionModel)


def test_resolve_passes_scale():
    reg = ModelRegistry()
    reg.register("mock", _mock_factory)
    cfg = _cfg(model="mock", scale=2)
    model = reg.resolve(cfg)
    assert model.scale == 2


def test_resolve_returns_unloaded_instance():
    reg = ModelRegistry()
    reg.register("mock", _mock_factory)
    model = reg.resolve(_cfg())
    assert model.is_loaded is False


def test_resolve_unknown_model_raises():
    reg = ModelRegistry()
    with pytest.raises(ValueError, match="Unknown model"):
        reg.resolve(_cfg(model="nonexistent"))


def test_resolve_error_lists_available_models():
    reg = ModelRegistry()
    reg.register("real-model", _mock_factory)
    with pytest.raises(ValueError, match="real-model"):
        reg.resolve(_cfg(model="wrong"))


def test_resolve_produces_independent_instances():
    reg = ModelRegistry()
    reg.register("mock", _mock_factory)
    cfg = _cfg()
    m1 = reg.resolve(cfg)
    m2 = reg.resolve(cfg)
    m1.load()
    assert m1.is_loaded
    assert not m2.is_loaded  # independent instances


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------

def test_default_registry_has_mock():
    from upscale_image.models.registry import available_models
    assert "mock" in available_models()


def test_resolve_model_default_registry():
    from upscale_image.models.registry import resolve_model
    cfg = _cfg(model="mock", scale=4)
    model = resolve_model(cfg)
    assert isinstance(model, MockSuperResolutionModel)
    assert model.scale == 4


def test_resolve_model_unknown_raises_from_default():
    from upscale_image.models.registry import resolve_model
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model(_cfg(model="does-not-exist"))


# ---------------------------------------------------------------------------
# No if/else in pipeline — pipeline works via base interface only
# ---------------------------------------------------------------------------

def test_pipeline_resolves_and_runs_without_concrete_import():
    """Pipeline code only needs ModelRegistry + SuperResolutionModel."""
    import numpy as np

    reg = ModelRegistry()
    reg.register("pipeline-model", _mock_factory)

    cfg = _cfg(model="pipeline-model", scale=4)
    model: SuperResolutionModel = reg.resolve(cfg)
    model.load()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    out = model.upscale(img, cfg)
    model.unload()

    assert out.shape == (16, 16, 3)
    assert not model.is_loaded
