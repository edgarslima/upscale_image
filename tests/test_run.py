"""Tests for run management: ID generation, directory tree, no-overwrite, config persistence."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
import pytest

from upscale_image.config import resolve_config
from upscale_image.pipeline import RunContext, create_run, generate_run_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path, model="realesrgan-x4", scale=4):
    return resolve_config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        model=model,
        scale=scale,
    )


_FIXED_TS = datetime(2026, 3, 28, 12, 0, 0)


# ---------------------------------------------------------------------------
# generate_run_id
# ---------------------------------------------------------------------------

def test_run_id_format():
    run_id = generate_run_id("realesrgan-x4", 4, now=_FIXED_TS)
    assert run_id == "run_20260328_120000_realesrgan-x4_4x"


def test_run_id_sanitizes_special_chars():
    run_id = generate_run_id("model/v2+fast", 4, now=_FIXED_TS)
    assert "/" not in run_id
    assert "+" not in run_id
    assert run_id.startswith("run_")


def test_run_id_scale_embedded():
    run_id = generate_run_id("mymodel", 2, now=_FIXED_TS)
    assert run_id.endswith("_2x")


def test_run_id_different_timestamps_are_unique():
    ts1 = datetime(2026, 1, 1, 0, 0, 0)
    ts2 = datetime(2026, 1, 1, 0, 0, 1)
    assert generate_run_id("m", 4, now=ts1) != generate_run_id("m", 4, now=ts2)


# ---------------------------------------------------------------------------
# create_run — directory structure
# ---------------------------------------------------------------------------

def test_creates_run_directory(tmp_path):
    cfg = _cfg(tmp_path)
    ctx = create_run(cfg, base_dir=tmp_path / "runs", now=_FIXED_TS)
    assert ctx.run_dir.is_dir()


def test_creates_outputs_subdir(tmp_path):
    cfg = _cfg(tmp_path)
    ctx = create_run(cfg, base_dir=tmp_path / "runs", now=_FIXED_TS)
    assert ctx.outputs_dir.is_dir()
    assert ctx.outputs_dir == ctx.run_dir / "outputs"


def test_creates_metrics_subdir(tmp_path):
    cfg = _cfg(tmp_path)
    ctx = create_run(cfg, base_dir=tmp_path / "runs", now=_FIXED_TS)
    assert ctx.metrics_dir.is_dir()
    assert ctx.metrics_dir == ctx.run_dir / "metrics"


def test_run_context_paths_are_consistent(tmp_path):
    cfg = _cfg(tmp_path)
    ctx = create_run(cfg, base_dir=tmp_path / "runs", now=_FIXED_TS)
    assert ctx.logs_path == ctx.run_dir / "logs.txt"
    assert ctx.manifest_path == ctx.run_dir / "manifest.json"
    assert ctx.effective_config_path == ctx.run_dir / "effective_config.yaml"


def test_run_id_is_embedded_in_dir(tmp_path):
    cfg = _cfg(tmp_path)
    ctx = create_run(cfg, base_dir=tmp_path / "runs", now=_FIXED_TS)
    assert ctx.run_id in str(ctx.run_dir)


# ---------------------------------------------------------------------------
# create_run — effective_config.yaml
# ---------------------------------------------------------------------------

def test_effective_config_is_saved(tmp_path):
    cfg = _cfg(tmp_path)
    ctx = create_run(cfg, base_dir=tmp_path / "runs", now=_FIXED_TS)
    assert ctx.effective_config_path.exists()


def test_effective_config_reflects_resolved_values(tmp_path):
    cfg = _cfg(tmp_path, model="realesrgan-x4", scale=4)
    ctx = create_run(cfg, base_dir=tmp_path / "runs", now=_FIXED_TS)
    saved = yaml.safe_load(ctx.effective_config_path.read_text())
    assert saved["model"]["name"] == "realesrgan-x4"
    assert saved["model"]["scale"] == 4
    assert saved["runtime"]["device"] == "cpu"


# ---------------------------------------------------------------------------
# create_run — no overwrite guarantee
# ---------------------------------------------------------------------------

def test_no_overwrite_raises_on_collision(tmp_path):
    cfg = _cfg(tmp_path)
    create_run(cfg, base_dir=tmp_path / "runs", now=_FIXED_TS)
    with pytest.raises(FileExistsError):
        create_run(cfg, base_dir=tmp_path / "runs", now=_FIXED_TS)


def test_two_runs_different_timestamps_coexist(tmp_path):
    cfg = _cfg(tmp_path)
    ts1 = datetime(2026, 1, 1, 0, 0, 0)
    ts2 = datetime(2026, 1, 1, 0, 0, 1)
    ctx1 = create_run(cfg, base_dir=tmp_path / "runs", now=ts1)
    ctx2 = create_run(cfg, base_dir=tmp_path / "runs", now=ts2)
    assert ctx1.run_dir != ctx2.run_dir
    assert ctx1.run_dir.is_dir()
    assert ctx2.run_dir.is_dir()


# ---------------------------------------------------------------------------
# RunContext type
# ---------------------------------------------------------------------------

def test_run_context_is_dataclass(tmp_path):
    cfg = _cfg(tmp_path)
    ctx = create_run(cfg, base_dir=tmp_path / "runs", now=_FIXED_TS)
    assert isinstance(ctx, RunContext)
    assert isinstance(ctx.run_dir, Path)
