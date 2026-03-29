"""Tests for manifest generation (step 11)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from upscale_image import __version__
from upscale_image.config import AppConfig
from upscale_image.config.schema import ModelConfig, RuntimeConfig
from upscale_image.io.task import ImageTask
from upscale_image.pipeline.batch import BatchResult, ItemResult
from upscale_image.pipeline.manifest import write_manifest
from upscale_image.pipeline.run import RunContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(tmp_path: Path) -> RunContext:
    run_dir = tmp_path / "run_20260101_120000_mock_4x"
    run_dir.mkdir(parents=True)
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir()
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir()
    return RunContext(
        run_id="run_20260101_120000_mock_4x",
        run_dir=run_dir,
        outputs_dir=outputs_dir,
        metrics_dir=metrics_dir,
        logs_path=run_dir / "logs.txt",
        manifest_path=run_dir / "manifest.json",
        effective_config_path=run_dir / "effective_config.yaml",
    )


def _make_config(
    model_name: str = "mock",
    scale: int = 4,
    device: str = "cpu",
    precision: str = "fp32",
) -> AppConfig:
    return AppConfig(
        input_dir="/tmp/in",
        output_dir="/tmp/out",
        model=ModelConfig(name=model_name, scale=scale),
        runtime=RuntimeConfig(device=device, precision=precision),
    )


def _make_task(name: str = "img.png") -> ImageTask:
    return ImageTask(
        input_path=f"/tmp/in/{name}",
        output_path=f"/tmp/out/{name}",
        filename=name,
    )


def _make_batch(results: list[ItemResult], skipped_count: int = 0) -> BatchResult:
    from upscale_image.io import SkippedFile

    skipped = [
        SkippedFile(path=f"/tmp/in/skip{i}.txt", reason="unsupported")
        for i in range(skipped_count)
    ]
    batch = BatchResult(results=results, skipped=skipped, total_elapsed_s=2.5)
    return batch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWriteManifest:
    def test_creates_manifest_json(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        batch = _make_batch([])
        write_manifest(ctx, cfg, batch)
        assert ctx.manifest_path.exists()

    def test_manifest_is_valid_json(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        batch = _make_batch([])
        write_manifest(ctx, cfg, batch)
        data = json.loads(ctx.manifest_path.read_text())
        assert isinstance(data, dict)

    def test_run_id_matches(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        batch = _make_batch([])
        manifest = write_manifest(ctx, cfg, batch)
        assert manifest["run_id"] == ctx.run_id

    def test_model_fields(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config(model_name="realesrgan-x4", scale=4, device="cpu", precision="fp32")
        batch = _make_batch([])
        manifest = write_manifest(ctx, cfg, batch)
        m = manifest["model"]
        assert m["name"] == "realesrgan-x4"
        assert m["scale"] == 4
        assert m["device"] == "cpu"
        assert m["precision"] == "fp32"

    def test_runtime_fields(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        batch = _make_batch([])
        manifest = write_manifest(ctx, cfg, batch)
        rt = manifest["runtime"]
        assert rt["code_version"] == __version__
        assert "." in rt["python_version"]  # e.g. "3.11.5"

    def test_timing_total_elapsed(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        task = _make_task()
        result = ItemResult(
            task=task, status="done", elapsed=1.0, inference_time_ms=500.0,
            input_width=100, input_height=100, output_width=400, output_height=400,
        )
        batch = _make_batch([result])
        manifest = write_manifest(ctx, cfg, batch)
        assert manifest["timing"]["total_elapsed_s"] == pytest.approx(2.5, abs=0.01)

    def test_timing_inference_stats(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        tasks = [_make_task(f"img{i}.png") for i in range(3)]
        results = [
            ItemResult(task=tasks[0], status="done", elapsed=1.0, inference_time_ms=100.0),
            ItemResult(task=tasks[1], status="done", elapsed=1.0, inference_time_ms=200.0),
            ItemResult(task=tasks[2], status="done", elapsed=1.0, inference_time_ms=300.0),
        ]
        batch = _make_batch(results)
        manifest = write_manifest(ctx, cfg, batch)
        t = manifest["timing"]
        assert t["avg_inference_ms"] == pytest.approx(200.0, abs=0.01)
        assert t["min_inference_ms"] == pytest.approx(100.0, abs=0.01)
        assert t["max_inference_ms"] == pytest.approx(300.0, abs=0.01)

    def test_timing_inference_none_when_all_failed(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        task = _make_task()
        result = ItemResult(task=task, status="failed", elapsed=0.1, error="oops")
        batch = _make_batch([result])
        manifest = write_manifest(ctx, cfg, batch)
        t = manifest["timing"]
        assert t["avg_inference_ms"] is None
        assert t["min_inference_ms"] is None
        assert t["max_inference_ms"] is None

    def test_status_counts_all_done(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        tasks = [_make_task(f"img{i}.png") for i in range(3)]
        results = [
            ItemResult(task=t, status="done", elapsed=1.0, inference_time_ms=100.0)
            for t in tasks
        ]
        batch = _make_batch(results, skipped_count=2)
        manifest = write_manifest(ctx, cfg, batch)
        s = manifest["status"]
        assert s["total"] == 3
        assert s["done"] == 3
        assert s["failed"] == 0
        assert s["skipped"] == 2
        assert s["success_rate"] == pytest.approx(1.0)

    def test_status_counts_with_failures(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        tasks = [_make_task(f"img{i}.png") for i in range(4)]
        results = [
            ItemResult(task=tasks[0], status="done", elapsed=1.0, inference_time_ms=100.0),
            ItemResult(task=tasks[1], status="done", elapsed=1.0, inference_time_ms=100.0),
            ItemResult(task=tasks[2], status="done", elapsed=1.0, inference_time_ms=100.0),
            ItemResult(task=tasks[3], status="failed", elapsed=0.1, error="err"),
        ]
        batch = _make_batch(results)
        manifest = write_manifest(ctx, cfg, batch)
        s = manifest["status"]
        assert s["total"] == 4
        assert s["done"] == 3
        assert s["failed"] == 1
        assert s["success_rate"] == pytest.approx(0.75)

    def test_status_empty_run(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        batch = _make_batch([])
        manifest = write_manifest(ctx, cfg, batch)
        s = manifest["status"]
        assert s["total"] == 0
        assert s["done"] == 0
        assert s["failed"] == 0
        assert s["success_rate"] == pytest.approx(1.0)

    def test_artifacts_keys_present(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        batch = _make_batch([])
        manifest = write_manifest(ctx, cfg, batch)
        a = manifest["artifacts"]
        assert "effective_config" in a
        assert "outputs_dir" in a
        assert "metrics_dir" in a
        assert "log" in a

    def test_artifacts_relative_paths(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        batch = _make_batch([])
        manifest = write_manifest(ctx, cfg, batch)
        a = manifest["artifacts"]
        # All values should be plain filenames, not absolute paths
        for v in a.values():
            assert "/" not in v, f"Expected relative name, got: {v!r}"

    def test_required_top_level_keys(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        batch = _make_batch([])
        manifest = write_manifest(ctx, cfg, batch)
        for key in ("run_id", "model", "runtime", "timing", "status", "artifacts"):
            assert key in manifest, f"Missing key: {key!r}"

    def test_persisted_json_matches_returned_dict(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config()
        batch = _make_batch([])
        returned = write_manifest(ctx, cfg, batch)
        persisted = json.loads(ctx.manifest_path.read_text())
        assert returned == persisted

    def test_manifest_utf8_encoding(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        cfg = _make_config(model_name="modelo-ação")
        batch = _make_batch([])
        write_manifest(ctx, cfg, batch)
        raw = ctx.manifest_path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
        assert data["model"]["name"] == "modelo-ação"
