"""Tests for the image optimization layer (Step 19).

Covers:
- OptimizeConfig defaults and overrides
- run_optimization happy path (webp + jpeg generated, sizes recorded)
- per_image.csv and summary.json written correctly
- source PNG files are never modified
- item-level failure is recoverable
- manifest patching
- CLI optimize subcommand (smoke test via CliRunner)
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from upscale_image.cli.main import app
from upscale_image.optimize import (
    OptimizeConfig,
    OptimizeSummary,
    run_optimization,
    default_optimize_config,
)
from upscale_image.pipeline.manifest import patch_manifest_with_optimization


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_png(path: Path, width: int = 64, height: int = 64) -> Path:
    """Write a small solid-colour PNG to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(path, format="PNG")
    return path


def _make_run_dir(tmp_path: Path, n_images: int = 2) -> Path:
    """Create a minimal run directory with synthetic outputs and a manifest."""
    run_dir = tmp_path / "run_test"
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True)

    for i in range(n_images):
        _make_png(outputs_dir / f"image_{i:02d}.png")

    # Minimal manifest (mirrors real schema)
    manifest = {
        "run_id": "run_test",
        "model": {"name": "mock", "scale": 2, "device": "cpu", "precision": "fp32"},
        "runtime": {"code_version": "0.1.0", "python_version": "3.10.0"},
        "timing": {"total_elapsed_s": 1.0},
        "status": {"total": n_images, "done": n_images, "failed": 0},
        "artifacts": {"outputs_dir": "outputs"},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return run_dir


# ---------------------------------------------------------------------------
# OptimizeConfig
# ---------------------------------------------------------------------------

class TestOptimizeConfig:
    def test_defaults(self):
        cfg = default_optimize_config()
        assert cfg.formats == ["webp", "jpeg"]
        assert 0 < cfg.webp_quality <= 100
        assert 0 < cfg.jpeg_quality <= 100

    def test_custom(self):
        cfg = OptimizeConfig(formats=["webp"], webp_quality=60, jpeg_quality=70)
        assert cfg.formats == ["webp"]
        assert cfg.webp_quality == 60


# ---------------------------------------------------------------------------
# run_optimization — happy path
# ---------------------------------------------------------------------------

class TestRunOptimizationHappyPath:
    def test_creates_optimized_dir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        run_optimization(run_dir)
        assert (run_dir / "optimized").is_dir()

    def test_creates_webp_subdir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        run_optimization(run_dir)
        assert (run_dir / "optimized" / "webp").is_dir()

    def test_creates_jpeg_subdir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        run_optimization(run_dir)
        assert (run_dir / "optimized" / "jpeg").is_dir()

    def test_webp_files_generated(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=2)
        run_optimization(run_dir)
        webp_files = list((run_dir / "optimized" / "webp").glob("*.webp"))
        assert len(webp_files) == 2

    def test_jpeg_files_generated(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=2)
        run_optimization(run_dir)
        jpg_files = list((run_dir / "optimized" / "jpeg").glob("*.jpg"))
        assert len(jpg_files) == 2

    def test_source_pngs_unchanged(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        outputs_dir = run_dir / "outputs"
        original_bytes = {p.name: p.read_bytes() for p in outputs_dir.glob("*.png")}
        run_optimization(run_dir)
        for name, data in original_bytes.items():
            assert (outputs_dir / name).read_bytes() == data, f"{name} was modified"

    def test_returns_summary(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        summary = run_optimization(run_dir)
        assert isinstance(summary, OptimizeSummary)

    def test_summary_eligible_count(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=3)
        summary = run_optimization(run_dir)
        assert summary.eligible == 3

    def test_summary_optimized_count(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=2)
        summary = run_optimization(run_dir, OptimizeConfig(formats=["webp", "jpeg"]))
        # 2 images × 2 formats = 4 ok results
        assert summary.optimized == 4

    def test_summary_failed_zero(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        summary = run_optimization(run_dir)
        assert summary.failed == 0

    def test_summary_bytes_recorded(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        summary = run_optimization(run_dir)
        assert summary.source_bytes_total > 0
        assert summary.optimized_bytes_total > 0

    def test_saving_ratio_is_float(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        summary = run_optimization(run_dir)
        # saving_ratio can be negative for tiny images where JPEG overhead exceeds PNG size
        assert isinstance(summary.saving_ratio_total, float)
        assert summary.saving_ratio_total <= 1.0

    def test_deterministic_on_rerun(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        s1 = run_optimization(run_dir)
        s2 = run_optimization(run_dir)
        assert s1.optimized_bytes_total == s2.optimized_bytes_total

    def test_webp_only_format(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        cfg = OptimizeConfig(formats=["webp"])
        summary = run_optimization(run_dir, cfg)
        assert not (run_dir / "optimized" / "jpeg").exists() or not list(
            (run_dir / "optimized" / "jpeg").glob("*")
        )
        assert summary.optimized == 1


# ---------------------------------------------------------------------------
# per_image.csv
# ---------------------------------------------------------------------------

class TestPerImageCsv:
    def test_csv_created(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        run_optimization(run_dir)
        assert (run_dir / "optimized" / "per_image.csv").is_file()

    def test_csv_has_header(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        run_optimization(run_dir)
        csv_path = run_dir / "optimized" / "per_image.csv"
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert "filename" in reader.fieldnames
            assert "status" in reader.fieldnames
            assert "saving_ratio" in reader.fieldnames

    def test_csv_row_count(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=2)
        run_optimization(run_dir, OptimizeConfig(formats=["webp", "jpeg"]))
        csv_path = run_dir / "optimized" / "per_image.csv"
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4  # 2 images × 2 formats

    def test_csv_status_ok(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        run_optimization(run_dir)
        csv_path = run_dir / "optimized" / "per_image.csv"
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert all(r["status"] == "ok" for r in rows)


# ---------------------------------------------------------------------------
# summary.json
# ---------------------------------------------------------------------------

class TestSummaryJson:
    def test_summary_json_created(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        run_optimization(run_dir)
        assert (run_dir / "optimized" / "summary.json").is_file()

    def test_summary_json_keys(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        run_optimization(run_dir)
        data = json.loads((run_dir / "optimized" / "summary.json").read_text())
        for key in ("eligible", "optimized", "failed", "source_bytes_total",
                    "optimized_bytes_total", "bytes_saved_total", "saving_ratio_total", "config"):
            assert key in data, f"Missing key: {key}"

    def test_summary_json_config_reflects_settings(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        cfg = OptimizeConfig(webp_quality=60, jpeg_quality=70)
        run_optimization(run_dir, cfg)
        data = json.loads((run_dir / "optimized" / "summary.json").read_text())
        assert data["config"]["webp_quality"] == 60
        assert data["config"]["jpeg_quality"] == 70


# ---------------------------------------------------------------------------
# Error handling (recoverable item failures)
# ---------------------------------------------------------------------------

class TestItemFailureRecoverable:
    def test_corrupt_png_does_not_abort(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        # Replace the valid PNG with corrupt bytes
        corrupt = run_dir / "outputs" / "corrupt.png"
        corrupt.write_bytes(b"not-a-png")
        summary = run_optimization(run_dir)
        # At least the originally valid image succeeded (2 ok) and corrupt failed (2 error)
        assert summary.failed > 0
        assert summary.optimized > 0

    def test_failed_items_in_csv(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=0)
        # Only a corrupt file
        corrupt = run_dir / "outputs" / "bad.png"
        corrupt.write_bytes(b"garbage")
        summary = run_optimization(run_dir)
        csv_path = run_dir / "optimized" / "per_image.csv"
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        error_rows = [r for r in rows if r["status"] == "error"]
        assert len(error_rows) > 0


# ---------------------------------------------------------------------------
# Error conditions (structural)
# ---------------------------------------------------------------------------

class TestStructuralErrors:
    def test_missing_run_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Run directory not found"):
            run_optimization(tmp_path / "nonexistent")

    def test_missing_manifest_raises(self, tmp_path):
        run_dir = tmp_path / "run_no_manifest"
        (run_dir / "outputs").mkdir(parents=True)
        _make_png(run_dir / "outputs" / "a.png")
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            run_optimization(run_dir)

    def test_empty_outputs_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=0)
        with pytest.raises(ValueError, match="No PNG files"):
            run_optimization(run_dir)


# ---------------------------------------------------------------------------
# Manifest patching
# ---------------------------------------------------------------------------

class TestManifestPatch:
    def test_patch_adds_optimization_key(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        summary = run_optimization(run_dir)
        patch_manifest_with_optimization(run_dir, summary)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "optimization" in manifest

    def test_patch_preserves_existing_fields(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        summary = run_optimization(run_dir)
        patch_manifest_with_optimization(run_dir, summary)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["run_id"] == "run_test"
        assert "model" in manifest
        assert "status" in manifest

    def test_patch_optimization_fields(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        summary = run_optimization(run_dir)
        patch_manifest_with_optimization(run_dir, summary)
        opt = json.loads((run_dir / "manifest.json").read_text())["optimization"]
        assert "eligible" in opt
        assert "optimized" in opt
        assert "bytes_saved_total" in opt
        assert opt["artifacts"]["optimized_dir"] == "optimized"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLIOptimize:
    def test_optimize_command_success(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        runner = CliRunner()
        result = runner.invoke(app, ["optimize", str(run_dir)])
        assert result.exit_code == 0, result.output

    def test_optimize_command_creates_artifacts(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        runner = CliRunner()
        runner.invoke(app, ["optimize", str(run_dir)])
        assert (run_dir / "optimized" / "summary.json").is_file()
        assert (run_dir / "optimized" / "per_image.csv").is_file()

    def test_optimize_command_patches_manifest(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, n_images=1)
        runner = CliRunner()
        runner.invoke(app, ["optimize", str(run_dir)])
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "optimization" in manifest

    def test_optimize_nonexistent_run_exits_1(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["optimize", str(tmp_path / "ghost")])
        assert result.exit_code == 1
