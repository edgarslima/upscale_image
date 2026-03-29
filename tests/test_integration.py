"""Integration tests (step 18) — full pipeline flows.

Each test exercises multiple subsystems together to protect against
structural regression. Unit tests for individual components live in
their own files; these tests focus on cross-module contracts and
end-to-end scenarios not covered by the unit test suite.

Gaps addressed:
  - CLI happy-path with real images (unit tests only tested empty dirs)
  - Exit code 2 when some items fail (ADR 0005 exit-code contract)
  - manifest.json produced by pipeline → loadable by compare module
  - Two sequential real pipeline runs → compare_runs produces deltas
  - effective_config.yaml round-trip (written by pipeline, parseable by YAML)
  - ADR 0004: pipeline resolves models via registry, not direct import
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from upscale_image.cli.main import app
from upscale_image.config import resolve_config
from upscale_image.models import MockSuperResolutionModel, available_models, resolve_model
from upscale_image.models.registry import ModelRegistry
from upscale_image.pipeline import BatchResult, create_run, run_batch, setup_run_logger, write_manifest
from upscale_image.reports.compare import compare_runs, load_run_snapshot

_RUNNER = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_image(path: Path, h: int = 16, w: int = 16) -> None:
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _run_pipeline(tmp_path: Path, n_images: int = 2, scale: int = 4,
                  ts_second: int = 0) -> tuple:
    """Run a complete pipeline with mock model; return (ctx, cfg, batch)."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    for i in range(n_images):
        _write_image(input_dir / f"img{i}.png")

    cfg = resolve_config(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "output"),
        model="mock",
        scale=scale,
    )
    ts = datetime(2026, 3, 28, 12, 0, ts_second)
    ctx = create_run(cfg, base_dir=tmp_path / "runs", now=ts)
    logger = setup_run_logger(ctx)

    model = resolve_model(cfg)
    model.load()
    batch = run_batch(cfg, ctx, model, logger)
    model.unload()
    logger.close()

    write_manifest(ctx, cfg, batch)
    return ctx, cfg, batch


# ---------------------------------------------------------------------------
# Full pipeline flow
# ---------------------------------------------------------------------------

class TestFullPipelineFlow:
    """End-to-end: config → discovery → run → batch → manifest."""

    def test_complete_run_creates_outputs(self, tmp_path, monkeypatch):
        """A successful run must produce one .png output per input image."""
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "a.png")
        _write_image(input_dir / "b.jpg")

        result = _RUNNER.invoke(app, [
            "upscale", str(input_dir),
            "--output", str(tmp_path / "out"),
            "--model", "mock",
        ])

        assert result.exit_code == 0
        # Outputs live inside runs/<run_id>/outputs/, not directly under --output
        outputs = list((tmp_path / "runs").rglob("outputs/*.png"))
        assert len(outputs) == 2

    def test_complete_run_creates_manifest(self, tmp_path, monkeypatch):
        """manifest.json must exist in the run directory after a successful run."""
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "img.png")

        _RUNNER.invoke(app, [
            "upscale", str(input_dir),
            "--output", str(tmp_path / "out"),
            "--model", "mock",
        ])

        manifests = list((tmp_path / "runs").rglob("manifest.json"))
        assert len(manifests) == 1

    def test_complete_run_creates_log(self, tmp_path, monkeypatch):
        """logs.txt must be written alongside the run artefacts."""
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "img.png")

        _RUNNER.invoke(app, [
            "upscale", str(input_dir),
            "--output", str(tmp_path / "out"),
            "--model", "mock",
        ])

        logs = list((tmp_path / "runs").rglob("logs.txt"))
        assert len(logs) == 1
        assert logs[0].stat().st_size > 0

    def test_complete_run_creates_effective_config(self, tmp_path, monkeypatch):
        """effective_config.yaml must be written in the run directory."""
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "img.png")

        _RUNNER.invoke(app, [
            "upscale", str(input_dir),
            "--output", str(tmp_path / "out"),
            "--model", "mock",
        ])

        configs = list((tmp_path / "runs").rglob("effective_config.yaml"))
        assert len(configs) == 1

    def test_exit_code_0_all_succeed(self, tmp_path, monkeypatch):
        """Exit code must be 0 when every image processes successfully."""
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "img.png")

        result = _RUNNER.invoke(app, [
            "upscale", str(input_dir),
            "--output", str(tmp_path / "out"),
            "--model", "mock",
        ])
        assert result.exit_code == 0

    def test_exit_code_2_when_some_items_fail(self, tmp_path, monkeypatch):
        """Exit code must be 2 (partial failure) when some items fail (ADR 0005)."""
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "good.png")

        # Inject a failing model into the registry temporarily
        from upscale_image.models.registry import _default_registry
        original = _default_registry._factories.get("mock")

        class FailingMock(MockSuperResolutionModel):
            def upscale(self, image, config):
                raise RuntimeError("forced failure")

        _default_registry._factories["mock"] = lambda scale: FailingMock(scale=scale)
        try:
            result = _RUNNER.invoke(app, [
                "upscale", str(input_dir),
                "--output", str(tmp_path / "out"),
                "--model", "mock",
            ])
        finally:
            _default_registry._factories["mock"] = original

        assert result.exit_code == 2

    def test_output_images_have_correct_scale(self, tmp_path, monkeypatch):
        """Output images must be exactly scale × input dimensions."""
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        img = np.zeros((8, 12, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "img.png"), img)

        _RUNNER.invoke(app, [
            "upscale", str(input_dir),
            "--output", str(tmp_path / "out"),
            "--model", "mock",
            "--scale", "4",
        ])

        outputs = list((tmp_path / "runs").rglob("outputs/img.png"))
        assert len(outputs) == 1
        out_img = cv2.imread(str(outputs[0]))
        assert out_img.shape == (32, 48, 3)


# ---------------------------------------------------------------------------
# Cross-module contracts
# ---------------------------------------------------------------------------

class TestCrossModuleContracts:
    """Data contracts between pipeline, manifest, and comparison modules."""

    def test_manifest_is_valid_json(self, tmp_path):
        """manifest.json written by write_manifest() must be valid JSON."""
        ctx, cfg, batch = _run_pipeline(tmp_path)
        content = ctx.manifest_path.read_text(encoding="utf-8")
        data = json.loads(content)   # raises if invalid
        assert isinstance(data, dict)

    def test_manifest_loadable_by_compare_module(self, tmp_path):
        """A manifest produced by the pipeline must be loadable by load_run_snapshot."""
        ctx, cfg, batch = _run_pipeline(tmp_path)
        snapshot = load_run_snapshot(ctx.run_dir)
        assert snapshot.run_id == ctx.run_id

    def test_snapshot_model_name_matches_config(self, tmp_path):
        """Snapshot loaded from manifest must reflect the model used in config."""
        ctx, cfg, batch = _run_pipeline(tmp_path)
        snapshot = load_run_snapshot(ctx.run_dir)
        assert snapshot.model_name == cfg.model.name

    def test_snapshot_status_counts_match_batch(self, tmp_path):
        """Snapshot status counts must match what run_batch returned."""
        ctx, cfg, batch = _run_pipeline(tmp_path, n_images=3)
        snapshot = load_run_snapshot(ctx.run_dir)
        assert snapshot.total_images == batch.total
        assert snapshot.done == batch.done
        assert snapshot.failed == batch.failed

    def test_effective_config_is_valid_yaml(self, tmp_path):
        """effective_config.yaml written by create_run must be parseable by PyYAML."""
        ctx, cfg, batch = _run_pipeline(tmp_path)
        content = ctx.effective_config_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)

    def test_effective_config_contains_model_section(self, tmp_path):
        """effective_config.yaml must contain a 'model' key with 'name' and 'scale'."""
        ctx, cfg, batch = _run_pipeline(tmp_path)
        content = ctx.effective_config_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert "model" in parsed
        assert "name" in parsed["model"]
        assert "scale" in parsed["model"]


# ---------------------------------------------------------------------------
# ADR 0004 — registry contract
# ---------------------------------------------------------------------------

class TestRegistryContract:
    """ADR 0004: pipeline resolves models through the registry, never directly."""

    def test_all_required_models_registered(self):
        """The three core models must be available in the default registry."""
        models = available_models()
        assert "mock" in models
        assert "bicubic" in models
        assert "realesrgan-x4" in models

    def test_custom_model_resolves_through_registry(self, tmp_path):
        """A model registered at runtime must be resolvable by the pipeline."""
        registry = ModelRegistry()
        registry.register("custom-mock", lambda scale: MockSuperResolutionModel(scale=scale))

        cfg = resolve_config(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            model="custom-mock",
            scale=2,
        )
        model = registry.resolve(cfg)
        assert model.name == "mock"
        assert model.scale == 2

    def test_resolve_model_returns_unloaded_instance(self, tmp_path):
        """resolve_model() must return an unloaded runner — caller controls lifecycle."""
        cfg = resolve_config(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            model="bicubic",
        )
        model = resolve_model(cfg)
        assert model.is_loaded is False

    def test_each_resolve_call_returns_independent_instance(self, tmp_path):
        """Two resolve_model() calls must return distinct objects (no shared state)."""
        cfg = resolve_config(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            model="mock",
        )
        m1 = resolve_model(cfg)
        m2 = resolve_model(cfg)
        assert m1 is not m2


# ---------------------------------------------------------------------------
# Two-run comparison flow
# ---------------------------------------------------------------------------

class TestComparisonFlow:
    """Full flow: two real pipeline runs → compare_runs → deltas."""

    def _make_two_runs(self, tmp_path):
        """Create two pipeline runs in isolated subdirs; return their run dirs."""
        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"
        dir_a.mkdir()
        dir_b.mkdir()
        ctx_a, _, _ = _run_pipeline(dir_a, n_images=2, scale=2, ts_second=0)
        ctx_b, _, _ = _run_pipeline(dir_b, n_images=2, scale=4, ts_second=1)
        return ctx_a.run_dir, ctx_b.run_dir

    def test_compare_runs_returns_two_snapshots(self, tmp_path):
        dir_a, dir_b = self._make_two_runs(tmp_path)
        result = compare_runs([str(dir_a), str(dir_b)])
        assert len(result.snapshots) == 2

    def test_compare_runs_returns_one_delta(self, tmp_path):
        dir_a, dir_b = self._make_two_runs(tmp_path)
        result = compare_runs([str(dir_a), str(dir_b)])
        assert len(result.deltas) == 1

    def test_comparison_snapshots_have_correct_scales(self, tmp_path):
        """Snapshots must reflect the scale used in each respective run."""
        dir_a, dir_b = self._make_two_runs(tmp_path)
        result = compare_runs([str(dir_a), str(dir_b)])
        assert result.snapshots[0].model_scale == 2
        assert result.snapshots[1].model_scale == 4

    def test_comparison_to_dict_is_json_serialisable(self, tmp_path):
        """comparison_to_dict() must produce a structure serialisable to JSON."""
        from upscale_image.reports.compare import comparison_to_dict
        dir_a, dir_b = self._make_two_runs(tmp_path)
        result = compare_runs([str(dir_a), str(dir_b)])
        data = comparison_to_dict(result)
        serialised = json.dumps(data)   # raises if not serialisable
        assert len(serialised) > 0

    def test_comparison_dict_has_runs_and_deltas_keys(self, tmp_path):
        """comparison_to_dict() must produce a dict with 'runs' and 'deltas' keys."""
        from upscale_image.reports.compare import comparison_to_dict
        dir_a, dir_b = self._make_two_runs(tmp_path)
        result = compare_runs([str(dir_a), str(dir_b)])
        data = comparison_to_dict(result)
        assert "runs" in data
        assert "deltas" in data

    def test_same_run_twice_produces_zero_success_rate_delta(self, tmp_path):
        """Comparing a run against itself must produce a zero success_rate delta."""
        dir_a = tmp_path / "run_a"
        dir_a.mkdir()
        ctx, _, _ = _run_pipeline(dir_a, n_images=1, scale=4, ts_second=0)
        run_dir = str(ctx.run_dir)

        # compare the run against itself (same data → delta = 0 for success_rate)
        result = compare_runs([run_dir, run_dir])
        delta = result.deltas[0]
        assert delta.delta_success_rate == 0.0
