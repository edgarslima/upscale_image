"""Tests for the batch inference pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from upscale_image.config import resolve_config
from upscale_image.models import MockSuperResolutionModel
from upscale_image.pipeline import BatchResult, ItemResult, create_run, run_batch, setup_run_logger

_FIXED_TS = datetime(2026, 3, 28, 15, 0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_image(path: Path, h: int = 8, w: int = 8) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_corrupted(path: Path) -> None:
    path.write_bytes(b"\x00\x01\x02bad")


def _setup(tmp_path, suffix=0):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    cfg = resolve_config(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "output"),
    )
    ts = datetime(2026, 3, 28, 15, 0, suffix)
    ctx = create_run(cfg, base_dir=tmp_path / "runs", now=ts)
    model = MockSuperResolutionModel(scale=4)
    model.load()
    logger = setup_run_logger(ctx)
    return cfg, ctx, model, logger, input_dir


# ---------------------------------------------------------------------------
# Empty directory
# ---------------------------------------------------------------------------

def test_empty_directory_returns_empty_batch(tmp_path):
    cfg, ctx, model, logger, _ = _setup(tmp_path)
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    assert result.total == 0
    assert result.done == 0
    assert result.failed == 0


# ---------------------------------------------------------------------------
# All valid images
# ---------------------------------------------------------------------------

def test_all_valid_images_processed(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "a.png")
    _write_image(input_dir / "b.png")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    assert result.total == 2
    assert result.done == 2
    assert result.failed == 0


def test_outputs_written_to_run_outputs_dir(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "photo.jpg")
    run_batch(cfg, ctx, model, logger)
    logger.close()
    assert (ctx.outputs_dir / "photo.png").exists()


def test_output_dimensions_match_scale(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "img.png", h=8, w=8)
    run_batch(cfg, ctx, model, logger)
    logger.close()
    out = cv2.imread(str(ctx.outputs_dir / "img.png"))
    assert out.shape == (32, 32, 3)  # 8 * scale(4)


def test_item_status_done_on_success(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "ok.png")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    assert result.results[0].status == "done"
    assert result.results[0].error is None


def test_item_elapsed_is_positive(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "ok.png")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    assert result.results[0].elapsed >= 0.0


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------

def test_corrupted_file_marks_item_failed(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "good.png")
    # Force a runtime error by making model.upscale raise for one file.
    # We can't easily corrupt post-discovery, so we patch model.upscale.
    original_upscale = model.upscale
    call_count = [0]

    def failing_upscale(image, config):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("simulated inference failure")
        return original_upscale(image, config)

    model.upscale = failing_upscale
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    assert result.failed == 1
    assert result.results[0].status == "failed"
    assert "simulated inference failure" in result.results[0].error


def test_one_failure_does_not_stop_remaining(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "a.png")
    _write_image(input_dir / "b.png")
    _write_image(input_dir / "c.png")

    call_count = [0]
    original = model.upscale

    def fail_second(image, config):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("second item fails")
        return original(image, config)

    model.upscale = fail_second
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    assert result.total == 3
    assert result.done == 2
    assert result.failed == 1


def test_failed_item_has_error_message(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "img.png")
    original = model.upscale

    def always_fail(image, config):
        raise RuntimeError("hard failure")

    model.upscale = always_fail
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    item = result.results[0]
    assert item.error == "hard failure"


# ---------------------------------------------------------------------------
# BatchResult aggregation
# ---------------------------------------------------------------------------

def test_batch_result_counts(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "x.png")
    _write_image(input_dir / "y.png")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    assert isinstance(result, BatchResult)
    assert result.total == result.done + result.failed


def test_batch_result_skipped_populated(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "ok.png")
    (input_dir / "ignore.txt").write_text("not an image")
    result = run_batch(cfg, ctx, model, logger)
    logger.close()
    assert len(result.skipped) == 1


# ---------------------------------------------------------------------------
# Output naming convention
# ---------------------------------------------------------------------------

def test_output_always_png_extension(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    _write_image(input_dir / "photo.bmp")
    run_batch(cfg, ctx, model, logger)
    logger.close()
    assert (ctx.outputs_dir / "photo.png").exists()


def test_multiple_outputs_all_present(tmp_path):
    cfg, ctx, model, logger, input_dir = _setup(tmp_path)
    for name in ["a.png", "b.jpg", "c.bmp"]:
        _write_image(input_dir / name)
    run_batch(cfg, ctx, model, logger)
    logger.close()
    outputs = list(ctx.outputs_dir.glob("*.png"))
    assert len(outputs) == 3
