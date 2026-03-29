"""Tests for BicubicRunner (step 17 — second real model).

Validates:
  - Contract compliance (load/upscale/unload/metadata)
  - Output correctness (scale, dtype, value range)
  - Registry integration (resolved by name via default registry)
  - Pipeline compatibility (full run produces correct artefacts)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from upscale_image.config import resolve_config
from upscale_image.models import BicubicRunner, available_models, resolve_model
from upscale_image.models.registry import ModelRegistry
from upscale_image.pipeline import BatchResult, create_run, run_batch, setup_run_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path, model="bicubic", scale=4, device="cpu"):
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    return resolve_config(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "output"),
        model=model,
        scale=scale,
        device=device,
    )


def _write_image(path: Path, h: int = 16, w: int = 16) -> None:
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Contract compliance
# ---------------------------------------------------------------------------

class TestBicubicContract:
    def test_name_property(self):
        r = BicubicRunner(scale=4)
        assert r.name == "bicubic"

    def test_scale_property(self):
        r = BicubicRunner(scale=2)
        assert r.scale == 2

    def test_is_loaded_false_before_load(self):
        r = BicubicRunner()
        assert r.is_loaded is False

    def test_is_loaded_true_after_load(self):
        r = BicubicRunner()
        r.load()
        assert r.is_loaded is True

    def test_is_loaded_false_after_unload(self):
        r = BicubicRunner()
        r.load()
        r.unload()
        assert r.is_loaded is False

    def test_upscale_raises_before_load(self):
        r = BicubicRunner()
        cfg = resolve_config(input_dir="/tmp/in", output_dir="/tmp/out")
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            r.upscale(img, cfg)

    def test_upscale_works_after_load(self):
        r = BicubicRunner(scale=4)
        r.load()
        cfg = resolve_config(input_dir="/tmp/in", output_dir="/tmp/out")
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        out = r.upscale(img, cfg)
        assert out is not None
        r.unload()


# ---------------------------------------------------------------------------
# Output correctness
# ---------------------------------------------------------------------------

class TestBicubicOutput:
    def _run(self, scale: int, h: int = 8, w: int = 12) -> np.ndarray:
        r = BicubicRunner(scale=scale)
        r.load()
        cfg = resolve_config(input_dir="/tmp/in", output_dir="/tmp/out")
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        out = r.upscale(img, cfg)
        r.unload()
        return out

    def test_output_shape_scale_4(self):
        out = self._run(scale=4, h=8, w=12)
        assert out.shape == (32, 48, 3)

    def test_output_shape_scale_2(self):
        out = self._run(scale=2, h=10, w=15)
        assert out.shape == (20, 30, 3)

    def test_output_dtype_is_uint8(self):
        out = self._run(scale=4)
        assert out.dtype == np.uint8

    def test_output_values_in_valid_range(self):
        out = self._run(scale=4)
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_identical_inputs_produce_identical_outputs(self):
        r = BicubicRunner(scale=4)
        r.load()
        cfg = resolve_config(input_dir="/tmp/in", output_dir="/tmp/out")
        img = np.full((8, 8, 3), 128, dtype=np.uint8)
        out1 = r.upscale(img, cfg)
        out2 = r.upscale(img, cfg)
        r.unload()
        np.testing.assert_array_equal(out1, out2)

    def test_uniform_input_produces_uniform_output(self):
        """A flat colour image should remain flat after bicubic upscale."""
        r = BicubicRunner(scale=4)
        r.load()
        cfg = resolve_config(input_dir="/tmp/in", output_dir="/tmp/out")
        img = np.full((16, 16, 3), 100, dtype=np.uint8)
        out = r.upscale(img, cfg)
        r.unload()
        # All pixels should be the same value (bicubic is exact on flat areas)
        assert out.min() == out.max()


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestBicubicRegistry:
    def test_bicubic_in_available_models(self):
        assert "bicubic" in available_models()

    def test_resolve_returns_bicubic_runner(self, tmp_path):
        cfg = _make_config(tmp_path, model="bicubic", scale=4)
        model = resolve_model(cfg)
        assert isinstance(model, BicubicRunner)

    def test_resolved_instance_has_correct_scale(self, tmp_path):
        cfg = _make_config(tmp_path, model="bicubic", scale=2)
        model = resolve_model(cfg)
        assert model.scale == 2

    def test_each_resolve_returns_fresh_instance(self, tmp_path):
        cfg = _make_config(tmp_path, model="bicubic", scale=4)
        m1 = resolve_model(cfg)
        m2 = resolve_model(cfg)
        assert m1 is not m2

    def test_bicubic_distinct_from_mock(self, tmp_path):
        cfg = _make_config(tmp_path, model="bicubic")
        model = resolve_model(cfg)
        assert model.name != "mock"

    def test_registry_has_three_models(self):
        models = available_models()
        assert "bicubic" in models
        assert "mock" in models
        assert "realesrgan-x4" in models


# ---------------------------------------------------------------------------
# Pipeline compatibility
# ---------------------------------------------------------------------------

class TestBicubicPipelineCompatibility:
    def test_full_batch_run_with_bicubic(self, tmp_path):
        """A complete run using bicubic must produce the correct artefacts."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for i in range(2):
            _write_image(input_dir / f"img{i}.png")

        cfg = resolve_config(
            input_dir=str(input_dir),
            output_dir=str(tmp_path / "output"),
            model="bicubic",
            scale=4,
        )
        ts = datetime(2026, 3, 28, 12, 0, 0)
        ctx = create_run(cfg, base_dir=tmp_path / "runs", now=ts)
        logger = setup_run_logger(ctx)

        model = resolve_model(cfg)
        model.load()
        batch = run_batch(cfg, ctx, model, logger)
        model.unload()
        logger.close()

        assert batch.total == 2
        assert batch.done == 2
        assert batch.failed == 0

    def test_outputs_have_correct_scale(self, tmp_path):
        h, w = 16, 16
        scale = 4
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "img.png"), img)

        cfg = resolve_config(
            input_dir=str(input_dir),
            output_dir=str(tmp_path / "output"),
            model="bicubic",
            scale=scale,
        )
        ts = datetime(2026, 3, 28, 12, 0, 1)
        ctx = create_run(cfg, base_dir=tmp_path / "runs", now=ts)
        logger = setup_run_logger(ctx)

        model = resolve_model(cfg)
        model.load()
        batch = run_batch(cfg, ctx, model, logger)
        model.unload()
        logger.close()

        out_path = Path(batch.results[0].task.output_path)
        assert out_path.exists()
        result_img = cv2.imread(str(out_path))
        assert result_img.shape == (h * scale, w * scale, 3)

    def test_item_result_has_dimensions(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "img.png", h=8, w=8)

        cfg = resolve_config(
            input_dir=str(input_dir),
            output_dir=str(tmp_path / "output"),
            model="bicubic",
            scale=4,
        )
        ts = datetime(2026, 3, 28, 12, 0, 2)
        ctx = create_run(cfg, base_dir=tmp_path / "runs", now=ts)
        logger = setup_run_logger(ctx)

        model = resolve_model(cfg)
        model.load()
        batch = run_batch(cfg, ctx, model, logger)
        model.unload()
        logger.close()

        r = batch.results[0]
        assert r.input_width == 8
        assert r.input_height == 8
        assert r.output_width == 32
        assert r.output_height == 32

    def test_manifest_contains_bicubic_model_name(self, tmp_path):
        """Manifest written after run must reflect the bicubic model."""
        from upscale_image.pipeline import write_manifest
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "img.png")

        cfg = resolve_config(
            input_dir=str(input_dir),
            output_dir=str(tmp_path / "output"),
            model="bicubic",
            scale=4,
        )
        ts = datetime(2026, 3, 28, 12, 0, 3)
        ctx = create_run(cfg, base_dir=tmp_path / "runs", now=ts)
        logger = setup_run_logger(ctx)

        model = resolve_model(cfg)
        model.load()
        batch = run_batch(cfg, ctx, model, logger)
        model.unload()
        logger.close()

        manifest = write_manifest(ctx, cfg, batch)
        assert manifest["model"]["name"] == "bicubic"
