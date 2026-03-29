"""Tests for RunLogger: dual output, structured events, file persistence."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from upscale_image.config import resolve_config
from upscale_image.io.task import ImageTask, SkippedFile
from upscale_image.pipeline import RunLogger, create_run, setup_run_logger

_FIXED_TS = datetime(2026, 3, 28, 12, 0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(tmp_path, suffix=""):
    cfg = resolve_config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
    )
    ts = datetime(2026, 3, 28, 12, 0, int(suffix) if suffix else 0)
    return create_run(cfg, base_dir=tmp_path / "runs", now=ts), cfg


def _read_log(ctx) -> str:
    return ctx.logs_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

def test_logs_file_created_after_logger_init(tmp_path):
    ctx, _ = _make_ctx(tmp_path)
    logger = setup_run_logger(ctx)
    logger.info("hello")
    logger.close()
    assert ctx.logs_path.exists()


def test_logs_file_in_run_dir(tmp_path):
    ctx, _ = _make_ctx(tmp_path)
    assert ctx.logs_path.parent == ctx.run_dir


# ---------------------------------------------------------------------------
# Content written to file
# ---------------------------------------------------------------------------

def test_info_written_to_file(tmp_path):
    ctx, _ = _make_ctx(tmp_path)
    logger = setup_run_logger(ctx)
    logger.info("test info message")
    logger.close()
    assert "test info message" in _read_log(ctx)


def test_warning_written_to_file(tmp_path):
    ctx, _ = _make_ctx(tmp_path)
    logger = setup_run_logger(ctx)
    logger.warning("test warning")
    logger.close()
    assert "test warning" in _read_log(ctx)
    assert "WARNING" in _read_log(ctx)


def test_error_written_to_file(tmp_path):
    ctx, _ = _make_ctx(tmp_path)
    logger = setup_run_logger(ctx)
    logger.error("test error")
    logger.close()
    assert "test error" in _read_log(ctx)
    assert "ERROR" in _read_log(ctx)


def test_debug_written_to_file(tmp_path):
    ctx, _ = _make_ctx(tmp_path)
    logger = setup_run_logger(ctx)
    logger.debug("trace detail")
    logger.close()
    assert "trace detail" in _read_log(ctx)


# ---------------------------------------------------------------------------
# Structured events
# ---------------------------------------------------------------------------

def test_log_run_start_writes_key_fields(tmp_path):
    ctx, cfg = _make_ctx(tmp_path)
    logger = setup_run_logger(ctx)
    logger.log_run_start(ctx, cfg, task_count=5, skipped_count=2)
    logger.close()
    content = _read_log(ctx)
    assert ctx.run_id in content
    assert cfg.model.name in content
    assert cfg.runtime.device in content
    assert cfg.input_dir in content
    assert "5" in content
    assert "2" in content


def test_log_skipped_files_writes_reasons(tmp_path):
    ctx, _ = _make_ctx(tmp_path)
    logger = setup_run_logger(ctx)
    skipped = [
        SkippedFile(path="/a/file.txt", reason="unsupported extension '.txt'"),
        SkippedFile(path="/a/bad.png", reason="file is unreadable or corrupted"),
    ]
    logger.log_skipped_files(skipped)
    logger.close()
    content = _read_log(ctx)
    assert "file.txt" in content
    assert "unsupported extension" in content
    assert "bad.png" in content
    assert "corrupted" in content


def test_log_item_done_contains_filename_and_time(tmp_path):
    ctx, _ = _make_ctx(tmp_path)
    logger = setup_run_logger(ctx)
    task = ImageTask(input_path="/in/img.jpg", output_path="/out/img.png", filename="img.jpg")
    logger.log_item_done(task, elapsed=1.23)
    logger.close()
    content = _read_log(ctx)
    assert "img.jpg" in content
    assert "1.23" in content


def test_log_item_error_contains_exception_message(tmp_path):
    ctx, _ = _make_ctx(tmp_path)
    logger = setup_run_logger(ctx)
    task = ImageTask(input_path="/in/bad.jpg", output_path="/out/bad.png", filename="bad.jpg")
    logger.log_item_error(task, exc=RuntimeError("inference failed"))
    logger.close()
    content = _read_log(ctx)
    assert "bad.jpg" in content
    assert "inference failed" in content
    assert "ERROR" in content


def test_log_run_summary_writes_counts(tmp_path):
    ctx, _ = _make_ctx(tmp_path)
    logger = setup_run_logger(ctx)
    logger.log_run_summary(total=10, done=8, failed=2, elapsed=45.6)
    logger.close()
    content = _read_log(ctx)
    assert "10" in content
    assert "8" in content
    assert "2" in content
    assert "45.6" in content


# ---------------------------------------------------------------------------
# Independence between runs
# ---------------------------------------------------------------------------

def test_two_loggers_write_to_separate_files(tmp_path):
    ctx1, _ = _make_ctx(tmp_path, suffix="0")
    ctx2, _ = _make_ctx(tmp_path, suffix="1")
    log1 = setup_run_logger(ctx1)
    log2 = setup_run_logger(ctx2)
    log1.info("only in run1")
    log2.info("only in run2")
    log1.close()
    log2.close()
    assert "only in run1" in _read_log(ctx1)
    assert "only in run2" not in _read_log(ctx1)
    assert "only in run2" in _read_log(ctx2)
    assert "only in run1" not in _read_log(ctx2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_setup_run_logger_returns_run_logger(tmp_path):
    ctx, _ = _make_ctx(tmp_path)
    logger = setup_run_logger(ctx)
    assert isinstance(logger, RunLogger)
    logger.close()
