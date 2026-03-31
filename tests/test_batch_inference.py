"""Tests for Passo 26 — Batch Inference Real (batch_size > 1).

Covers:
- upscale_batch() default in base calls upscale() N times and returns N results
- MockSuperResolutionModel.upscale_batch() works (inherits default)
- RealESRGANRunner.upscale_batch() returns correct shapes (CPU, tiny weights)
- Padding: batch with images of different sizes returns unpadded outputs
- group_tasks_by_size() groups correctly with batch_size tolerance
- group_tasks_by_size() with batch_size=1 returns N groups of 1
- estimate_safe_batch_size() returns 1 when CUDA not available
- run_batch() with batch_size=1 has no regression
- run_batch_async() with batch_size=2 preserves order
- batch failure in upscale_batch() marks all items in the batch as failed
- RuntimeConfig has batch_size field
- CLI exposes --batch-size flag
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from upscale_image.config import resolve_config
from upscale_image.io.task import ImageTask
from upscale_image.models import MockSuperResolutionModel
from upscale_image.models.base import SuperResolutionModel
from upscale_image.pipeline import create_run, run_batch, setup_run_logger
from upscale_image.pipeline.async_worker import run_batch_async
from upscale_image.pipeline.batch import estimate_safe_batch_size, group_tasks_by_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_image(path: Path, h: int = 8, w: int = 8) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _setup_run(tmp_path, suffix: int = 0):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
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


def _make_tasks(input_dir: Path, output_dir: Path, n: int, h: int = 8, w: int = 8) -> list[ImageTask]:
    tasks = []
    for i in range(n):
        src = input_dir / f"img_{i:03d}.png"
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imwrite(str(src), img)
        tasks.append(ImageTask(
            input_path=str(src),
            output_path=str(output_dir / f"img_{i:03d}.png"),
            filename=src.name,
        ))
    return tasks


# ---------------------------------------------------------------------------
# upscale_batch() default in base
# ---------------------------------------------------------------------------

def test_upscale_batch_default_calls_upscale_n_times():
    """Default upscale_batch() must call upscale() for each image."""
    model = MockSuperResolutionModel(scale=4)
    model.load()
    cfg = resolve_config(input_dir="/in", output_dir="/out")

    images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    results = model.upscale_batch(images, cfg)

    assert len(results) == 3
    for r in results:
        assert r.shape == (32, 32, 3)
        assert r.dtype == np.uint8


def test_upscale_batch_default_single_image():
    """upscale_batch() with one image must return a list of one result."""
    model = MockSuperResolutionModel(scale=4)
    model.load()
    cfg = resolve_config(input_dir="/in", output_dir="/out")

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    results = model.upscale_batch([image], cfg)

    assert len(results) == 1
    assert results[0].shape == (32, 32, 3)


def test_upscale_batch_default_empty_list():
    """upscale_batch() with empty list must return empty list."""
    model = MockSuperResolutionModel(scale=4)
    model.load()
    cfg = resolve_config(input_dir="/in", output_dir="/out")

    results = model.upscale_batch([], cfg)
    assert results == []


# ---------------------------------------------------------------------------
# RealESRGANRunner.upscale_batch() — CPU path with tiny weights
# ---------------------------------------------------------------------------

def _save_tiny_weights(path: Path, scale: int = 4) -> None:
    import upscale_image.models._compat  # noqa: F401
    from basicsr.archs.rrdbnet_arch import RRDBNet

    net = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=4, num_block=1, num_grow_ch=4,
        scale=scale,
    )
    import torch
    torch.save(net.state_dict(), str(path))


def test_realesrgan_upscale_batch_returns_correct_shapes(tmp_path, monkeypatch):
    """upscale_batch() must return N outputs with (orig_h*scale, orig_w*scale, 3)."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)

    from upscale_image.models.realesrgan import RealESRGANRunner
    runner = RealESRGANRunner(scale=4, weights_path=str(w), num_feat=4, num_block=1, num_grow_ch=4)
    runner.load()
    cfg = resolve_config(input_dir="/in", output_dir="/out")

    images = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ]
    results = runner.upscale_batch(images, cfg)

    assert len(results) == 2
    for r in results:
        assert r.shape == (32, 32, 3)
        assert r.dtype == np.uint8


def test_realesrgan_upscale_batch_mixed_sizes_unpadded(tmp_path, monkeypatch):
    """Outputs must NOT have padding — each output matches orig_h*scale × orig_w*scale."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)

    from upscale_image.models.realesrgan import RealESRGANRunner
    runner = RealESRGANRunner(scale=4, weights_path=str(w), num_feat=4, num_block=1, num_grow_ch=4)
    runner.load()
    cfg = resolve_config(input_dir="/in", output_dir="/out")

    img_a = np.zeros((6, 8, 3), dtype=np.uint8)   # 6×8
    img_b = np.zeros((8, 12, 3), dtype=np.uint8)  # 8×12

    results = runner.upscale_batch([img_a, img_b], cfg)

    assert results[0].shape == (24, 32, 3), f"Expected (24,32,3) got {results[0].shape}"
    assert results[1].shape == (32, 48, 3), f"Expected (32,48,3) got {results[1].shape}"


def test_realesrgan_upscale_batch_falls_back_to_serial_when_tiling(tmp_path, monkeypatch):
    """When tile_size > 0, upscale_batch() must fall back to serial upscale()."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    w = tmp_path / "model.pth"
    _save_tiny_weights(w)

    from upscale_image.models.realesrgan import RealESRGANRunner
    runner = RealESRGANRunner(scale=4, weights_path=str(w), num_feat=4, num_block=1, num_grow_ch=4)
    runner.load()
    cfg = resolve_config(input_dir="/in", output_dir="/out")
    cfg.runtime.tile_size = 4
    cfg.runtime.tile_pad = 2

    images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    results = runner.upscale_batch(images, cfg)

    assert len(results) == 2
    for r in results:
        assert r.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# group_tasks_by_size()
# ---------------------------------------------------------------------------

def test_group_tasks_batch_size_1_returns_n_singleton_groups(tmp_path):
    """batch_size=1 must return one group per task."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    tasks = _make_tasks(input_dir, output_dir, n=5)
    groups = group_tasks_by_size(tasks, batch_size=1)

    assert len(groups) == 5
    for g in groups:
        assert len(g) == 1


def test_group_tasks_groups_similar_sizes(tmp_path):
    """Images of identical size must be grouped together up to batch_size."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # All 4 images are the same size — should form at most ceil(4/2)=2 groups of 2
    tasks = _make_tasks(input_dir, output_dir, n=4, h=8, w=8)
    groups = group_tasks_by_size(tasks, batch_size=2)

    total_tasks = sum(len(g) for g in groups)
    assert total_tasks == 4

    for g in groups:
        assert len(g) <= 2


def test_group_tasks_empty_list():
    """Empty task list must return empty groups list."""
    groups = group_tasks_by_size([], batch_size=4)
    assert groups == []


def test_group_tasks_preserves_all_tasks(tmp_path):
    """All tasks must appear in exactly one group."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    tasks = _make_tasks(input_dir, output_dir, n=7)
    groups = group_tasks_by_size(tasks, batch_size=3)

    all_tasks_in_groups = [t for g in groups for t in g]
    assert len(all_tasks_in_groups) == 7
    # Every original task appears once
    for t in tasks:
        assert t in all_tasks_in_groups


# ---------------------------------------------------------------------------
# estimate_safe_batch_size()
# ---------------------------------------------------------------------------

def test_estimate_safe_batch_size_returns_1_without_cuda(monkeypatch):
    """Without CUDA, estimate_safe_batch_size must return 1."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    model = MockSuperResolutionModel(scale=4)
    model.load()
    cfg = resolve_config(input_dir="/in", output_dir="/out")
    sample = np.zeros((8, 8, 3), dtype=np.uint8)

    result = estimate_safe_batch_size(sample, model, cfg)
    assert result == 1


def test_estimate_safe_batch_size_at_least_1():
    """Result must always be >= 1."""
    model = MockSuperResolutionModel(scale=4)
    model.load()
    cfg = resolve_config(input_dir="/in", output_dir="/out")
    sample = np.zeros((8, 8, 3), dtype=np.uint8)

    result = estimate_safe_batch_size(sample, model, cfg)
    assert result >= 1


# ---------------------------------------------------------------------------
# run_batch_async() with batch_size > 1
# ---------------------------------------------------------------------------

def test_run_batch_async_batch_size_2_preserves_order(tmp_path):
    """run_batch_async with batch_size=2 must still return results in task order."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    cfg, ctx, model, logger, _ = _setup_run(tmp_path, suffix=20)
    tasks = _make_tasks(input_dir, output_dir, n=5)

    results = run_batch_async(cfg, tasks, model, logger,
                               prefetch_size=4, write_workers=1, batch_size=2)
    logger.close()

    assert len(results) == 5
    for i, (task, result) in enumerate(zip(tasks, results)):
        assert result.task is task, f"Position {i}: task order broken"
        assert result.status == "done"


def test_run_batch_async_batch_failure_marks_all_items_failed(tmp_path):
    """If upscale_batch() raises, every item in the batch must be failed."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    cfg, ctx, model, logger, _ = _setup_run(tmp_path, suffix=21)
    tasks = _make_tasks(input_dir, output_dir, n=3)

    original_upscale_batch = model.upscale_batch

    call_count = {"n": 0}

    def _failing_batch(images, config):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Simulated batch failure")
        return original_upscale_batch(images, config)

    model.upscale_batch = _failing_batch

    # batch_size=3 means all 3 tasks form one batch → one call → all fail
    results = run_batch_async(cfg, tasks, model, logger,
                               prefetch_size=4, write_workers=1, batch_size=3)
    logger.close()

    assert len(results) == 3
    # All items in the failed batch should be failed
    assert all(r.status == "failed" for r in results)


def test_run_batch_async_partial_batch_at_end(tmp_path):
    """With 5 tasks and batch_size=3, the partial batch of 2 must also complete."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    cfg, ctx, model, logger, _ = _setup_run(tmp_path, suffix=22)
    tasks = _make_tasks(input_dir, output_dir, n=5)

    results = run_batch_async(cfg, tasks, model, logger,
                               prefetch_size=4, write_workers=1, batch_size=3)
    logger.close()

    assert len(results) == 5
    assert all(r.status == "done" for r in results)


# ---------------------------------------------------------------------------
# RuntimeConfig and CLI
# ---------------------------------------------------------------------------

def test_runtime_config_has_batch_size_field():
    from upscale_image.config.schema import RuntimeConfig
    cfg = RuntimeConfig()
    assert cfg.batch_size == 1


def test_cli_has_batch_size_flag():
    from typer.testing import CliRunner
    from upscale_image.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["upscale", "--help"])
    assert "--batch-size" in result.output


# ---------------------------------------------------------------------------
# Regression — serial run_batch() unchanged
# ---------------------------------------------------------------------------

def test_run_batch_serial_batch_size_1_no_regression(tmp_path):
    """run_batch() with default batch_size=1 must behave exactly as before."""
    cfg, ctx, model, logger, input_dir = _setup_run(tmp_path, suffix=23)
    _write_image(input_dir / "a.png")
    _write_image(input_dir / "b.png")

    batch = run_batch(cfg, ctx, model, logger, async_io=False, batch_size=1)
    logger.close()

    assert batch.total == 2
    assert batch.done == 2
    assert batch.failed == 0
