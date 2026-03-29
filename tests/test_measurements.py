"""Tests for per-item measurements and run-level aggregation (step 10)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from upscale_image.config import resolve_config
from upscale_image.models import MockSuperResolutionModel
from upscale_image.pipeline import BatchResult, ItemResult, RunStats, create_run, run_batch, setup_run_logger


_SCALE = 4
_TS_BASE = datetime(2026, 3, 28, 16, 0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_image(path: Path, h: int = 8, w: int = 8) -> None:
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))


def _setup(tmp_path, suffix: int = 0):
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    cfg = resolve_config(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "output"),
        scale=_SCALE,
    )
    ts = datetime(2026, 3, 28, 16, 0, suffix)
    ctx = create_run(cfg, base_dir=tmp_path / "runs", now=ts)
    model = MockSuperResolutionModel(scale=_SCALE)
    model.load()
    logger = setup_run_logger(ctx)
    return cfg, ctx, model, logger, input_dir


# ---------------------------------------------------------------------------
# ItemResult — dimensions on success
# ---------------------------------------------------------------------------

def test_input_dimensions_recorded(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "img.png", h=6, w=10)
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    item = result.results[0]
    assert item.input_height == 6
    assert item.input_width == 10


def test_output_dimensions_recorded(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "img.png", h=8, w=8)
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    item = result.results[0]
    assert item.output_height == 8 * _SCALE
    assert item.output_width == 8 * _SCALE


def test_dimensions_match_actual_output_file(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "img.png", h=5, w=7)
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    item = result.results[0]
    saved = cv2.imread(str(ctx.outputs_dir / "img.png"))
    assert saved.shape[0] == item.output_height
    assert saved.shape[1] == item.output_width


# ---------------------------------------------------------------------------
# ItemResult — timing
# ---------------------------------------------------------------------------

def test_inference_time_ms_recorded(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "img.png")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    item = result.results[0]
    assert item.inference_time_ms is not None
    assert item.inference_time_ms >= 0.0


def test_elapsed_covers_full_item(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "img.png")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    item = result.results[0]
    # elapsed (total) >= inference (subset)
    assert item.elapsed >= item.inference_time_ms / 1000.0


def test_total_elapsed_in_batch_result(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "a.png")
    _write_image(input_dir / "b.png")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    assert result.total_elapsed_s >= 0.0


# ---------------------------------------------------------------------------
# ItemResult — uniform structure on failure
# ---------------------------------------------------------------------------

def test_failed_item_has_none_inference_time(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "img.png")

    original = model.upscale

    def fail_always(image, config):
        raise RuntimeError("injected failure")

    model.upscale = fail_always
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    item = result.results[0]
    assert item.status == "failed"
    assert item.inference_time_ms is None


def test_failed_item_input_dims_recorded_if_read_succeeded(tmp_path):
    """Dimensions are captured even when inference fails."""
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "img.png", h=4, w=6)

    original = model.upscale

    def fail_in_upscale(image, config):
        raise RuntimeError("upscale failed")

    model.upscale = fail_in_upscale
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    item = result.results[0]
    assert item.input_height == 4
    assert item.input_width == 6
    assert item.output_height is None
    assert item.output_width is None


def test_failed_item_output_dims_none(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "img.png")
    model.upscale = lambda img, cfg: (_ for _ in ()).throw(RuntimeError("fail"))
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    item = result.results[0]
    assert item.output_width is None
    assert item.output_height is None


def test_failed_item_has_elapsed(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "img.png")
    model.upscale = lambda img, cfg: (_ for _ in ()).throw(RuntimeError("fail"))
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    assert result.results[0].elapsed >= 0.0


# ---------------------------------------------------------------------------
# RunStats aggregation
# ---------------------------------------------------------------------------

def test_stats_returns_run_stats(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "a.png")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    assert isinstance(result.stats(), RunStats)


def test_stats_counts_match(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "a.png")
    _write_image(input_dir / "b.png")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    s = result.stats()
    assert s.total == 2
    assert s.done == 2
    assert s.failed == 0


def test_stats_success_rate_all_pass(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "a.png")
    _write_image(input_dir / "b.png")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    assert result.stats().success_rate == 1.0


def test_stats_success_rate_partial(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "a.png")
    _write_image(input_dir / "b.png")
    call_count = [0]
    original = model.upscale

    def fail_first(img, cfg):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("fail")
        return original(img, cfg)

    model.upscale = fail_first
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    s = result.stats()
    assert s.success_rate == pytest.approx(0.5)
    assert s.done == 1
    assert s.failed == 1


def test_stats_avg_inference_ms_computed(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "a.png")
    _write_image(input_dir / "b.png")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    s = result.stats()
    assert s.avg_inference_ms is not None
    assert s.avg_inference_ms >= 0.0


def test_stats_min_max_inference_ms(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "a.png")
    _write_image(input_dir / "b.png")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    s = result.stats()
    assert s.min_inference_ms <= s.avg_inference_ms <= s.max_inference_ms


def test_stats_inference_none_when_all_failed(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "a.png")
    model.upscale = lambda img, cfg: (_ for _ in ()).throw(RuntimeError("fail"))
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    s = result.stats()
    assert s.avg_inference_ms is None
    assert s.min_inference_ms is None
    assert s.max_inference_ms is None


def test_stats_empty_run(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    s = result.stats()
    assert s.total == 0
    assert s.success_rate == 1.0
    assert s.avg_inference_ms is None
