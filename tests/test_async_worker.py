"""Tests for the async producer-consumer pipeline (Passo 25).

Covers:
- Results returned in the same order as tasks
- Read failure in one item does not interrupt others
- Inference failure in one item does not interrupt others
- Write failure in one item does not interrupt others
- Clean shutdown: no thread hangs after batch completion
- run_batch() with async_io=True produces equivalent results to serial mode
- run_batch() with async_io=False (default) has no regression
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from upscale_image.config import AppConfig, resolve_config
from upscale_image.io.task import ImageTask
from upscale_image.models import MockSuperResolutionModel
from upscale_image.pipeline import BatchResult, ItemResult, create_run, run_batch, setup_run_logger
from upscale_image.pipeline.async_worker import run_batch_async


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_image(path: Path, h: int = 8, w: int = 8) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_corrupted(path: Path) -> None:
    path.write_bytes(b"\x00\x01\x02bad")


def _cfg(tmp_path) -> AppConfig:
    return resolve_config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
    )


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


def _make_tasks(input_dir: Path, output_dir: Path, n: int) -> list[ImageTask]:
    """Create n valid image tasks with synthetic images written to disk."""
    tasks = []
    for i in range(n):
        src = input_dir / f"img_{i:03d}.png"
        _write_image(src)
        dst = str(output_dir / f"img_{i:03d}.png")
        tasks.append(ImageTask(input_path=str(src), output_path=dst, filename=src.name))
    return tasks


# ---------------------------------------------------------------------------
# Order preservation
# ---------------------------------------------------------------------------

def test_results_in_same_order_as_tasks(tmp_path):
    """run_batch_async must return results indexed by original task order."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    cfg, ctx, model, logger, _ = _setup_run(tmp_path, suffix=1)
    tasks = _make_tasks(input_dir, output_dir, n=5)

    results = run_batch_async(cfg, tasks, model, logger, prefetch_size=2, write_workers=1)
    logger.close()

    assert len(results) == len(tasks)
    for i, (task, result) in enumerate(zip(tasks, results)):
        assert result.task is task, f"Position {i}: result.task mismatch"


def test_results_order_with_many_images(tmp_path):
    """Order must be preserved with a large batch (more than prefetch_size)."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    cfg, ctx, model, logger, _ = _setup_run(tmp_path, suffix=2)
    tasks = _make_tasks(input_dir, output_dir, n=12)

    results = run_batch_async(cfg, tasks, model, logger, prefetch_size=3, write_workers=2)
    logger.close()

    assert len(results) == 12
    for i, (task, result) in enumerate(zip(tasks, results)):
        assert result.task is task, f"Position {i}: task order broken"


# ---------------------------------------------------------------------------
# Failure isolation — read stage
# ---------------------------------------------------------------------------

def test_read_failure_does_not_abort_run(tmp_path):
    """A corrupted file at position 1 must not prevent positions 0 and 2 from succeeding."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    cfg, ctx, model, logger, _ = _setup_run(tmp_path, suffix=3)

    src0 = input_dir / "img_000.png"
    src1 = input_dir / "img_001.png"  # will be corrupted
    src2 = input_dir / "img_002.png"
    _write_image(src0)
    _write_corrupted(src1)
    _write_image(src2)

    tasks = [
        ImageTask(input_path=str(src0), output_path=str(output_dir / "img_000.png"), filename=src0.name),
        ImageTask(input_path=str(src1), output_path=str(output_dir / "img_001.png"), filename=src1.name),
        ImageTask(input_path=str(src2), output_path=str(output_dir / "img_002.png"), filename=src2.name),
    ]

    results = run_batch_async(cfg, tasks, model, logger, prefetch_size=4, write_workers=1)
    logger.close()

    assert len(results) == 3
    assert results[0].status == "done"
    assert results[1].status == "failed"
    assert results[2].status == "done"


# ---------------------------------------------------------------------------
# Failure isolation — inference stage
# ---------------------------------------------------------------------------

def test_inference_failure_does_not_abort_run(tmp_path):
    """An inference error on one task must produce failed result without stopping others."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    cfg, ctx, model, logger, _ = _setup_run(tmp_path, suffix=4)
    tasks = _make_tasks(input_dir, output_dir, n=3)

    call_count = {"n": 0}

    original_upscale = model.upscale

    def _failing_upscale(image, config):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("Simulated inference error")
        return original_upscale(image, config)

    model.upscale = _failing_upscale

    results = run_batch_async(cfg, tasks, model, logger, prefetch_size=4, write_workers=1)
    logger.close()

    assert len(results) == 3
    assert results[0].status == "done"
    assert results[1].status == "failed"
    assert results[2].status == "done"


# ---------------------------------------------------------------------------
# Failure isolation — write stage
# ---------------------------------------------------------------------------

def test_write_failure_does_not_abort_run(tmp_path):
    """A write error on one item must produce failed result without stopping others."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    cfg, ctx, model, logger, _ = _setup_run(tmp_path, suffix=5)
    tasks = _make_tasks(input_dir, output_dir, n=3)

    write_count = {"n": 0}

    import upscale_image.pipeline.async_worker as _aw_module

    original_save = _aw_module._save_output

    def _failing_save(image, output_path):
        write_count["n"] += 1
        if write_count["n"] == 2:
            raise RuntimeError("Simulated write error")
        original_save(image, output_path)

    with patch.object(_aw_module, "_save_output", _failing_save):
        results = run_batch_async(cfg, tasks, model, logger, prefetch_size=4, write_workers=1)
    logger.close()

    assert len(results) == 3
    assert results[0].status == "done"
    assert results[1].status == "failed"
    assert results[2].status == "done"


# ---------------------------------------------------------------------------
# Clean shutdown
# ---------------------------------------------------------------------------

def test_clean_shutdown_no_threads_left(tmp_path):
    """After run_batch_async returns, no daemon threads from the worker remain active."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    cfg, ctx, model, logger, _ = _setup_run(tmp_path, suffix=6)
    tasks = _make_tasks(input_dir, output_dir, n=4)

    threads_before = set(threading.enumerate())

    results = run_batch_async(cfg, tasks, model, logger, prefetch_size=2, write_workers=2)
    logger.close()

    threads_after = set(threading.enumerate())
    new_threads = threads_after - threads_before

    # All worker threads must have exited (joined) before the function returned
    worker_threads = [t for t in new_threads if "async-" in t.name]
    assert worker_threads == [], f"Leftover worker threads: {worker_threads}"


def test_empty_task_list_returns_empty_list(tmp_path):
    """run_batch_async with empty task list returns empty list without errors."""
    cfg, ctx, model, logger, _ = _setup_run(tmp_path, suffix=7)
    results = run_batch_async(cfg, [], model, logger)
    logger.close()
    assert results == []


# ---------------------------------------------------------------------------
# run_batch() integration — async_io=True vs serial
# ---------------------------------------------------------------------------

def test_run_batch_async_io_produces_same_status_as_serial(tmp_path):
    """run_batch(async_io=True) must produce equivalent status list to serial mode."""
    # Serial run
    cfg_s, ctx_s, model_s, logger_s, input_dir = _setup_run(tmp_path / "serial", suffix=8)
    for i in range(3):
        _write_image(input_dir / f"img_{i:03d}.png")
    model_s.load()
    batch_serial = run_batch(cfg_s, ctx_s, model_s, logger_s, async_io=False)
    logger_s.close()

    # Async run (separate tmp dirs)
    cfg_a, ctx_a, model_a, logger_a, input_dir_a = _setup_run(tmp_path / "async", suffix=9)
    for i in range(3):
        _write_image(input_dir_a / f"img_{i:03d}.png")
    model_a.load()
    batch_async = run_batch(cfg_a, ctx_a, model_a, logger_a, async_io=True, prefetch_size=2, write_workers=1)
    logger_a.close()

    assert batch_serial.total == batch_async.total
    assert batch_serial.done == batch_async.done
    assert batch_serial.failed == batch_async.failed


def test_run_batch_serial_default_no_regression(tmp_path):
    """run_batch() default (async_io=False) must work identically to before."""
    cfg, ctx, model, logger, input_dir = _setup_run(tmp_path, suffix=10)
    _write_image(input_dir / "a.png")
    _write_image(input_dir / "b.png")

    batch = run_batch(cfg, ctx, model, logger)
    logger.close()

    assert batch.total == 2
    assert batch.done == 2
    assert batch.failed == 0


# ---------------------------------------------------------------------------
# RuntimeConfig fields
# ---------------------------------------------------------------------------

def test_runtime_config_has_async_io_field():
    from upscale_image.config.schema import RuntimeConfig
    cfg = RuntimeConfig()
    assert cfg.async_io is False
    assert cfg.prefetch_size == 4
    assert cfg.write_workers == 2


def test_cli_has_async_io_flag():
    """CLI must expose --async-io and --prefetch flags."""
    from typer.testing import CliRunner
    from upscale_image.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["upscale", "--help"])
    assert "--async-io" in result.output
    assert "--prefetch" in result.output
