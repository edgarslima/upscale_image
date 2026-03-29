"""Tests for chained optimization inside the upscale command (Step 20).

Covers:
- Default behaviour unchanged when --optimize is not passed
- optimized/ created when --optimize is passed
- per_image.csv and summary.json written
- manifest patched with optimization key
- outputs/*.png remain intact
- --opt-format, --opt-webp-quality, --opt-jpeg-quality propagated correctly
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from upscale_image.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(path: Path, width: int = 32, height: int = 32) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((height, width, 3), 100, dtype=np.uint8)
    Image.fromarray(arr).save(path, format="PNG")
    return path


def _run_upscale(input_dir: str, *extra_args: str):
    """Run upscale with mock model. CWD must be set to tmp_path by caller."""
    return runner.invoke(
        app,
        [
            "upscale", input_dir,
            "--output", "runs",
            "--model", "mock",
            "--scale", "2",
            "--device", "cpu",
            *extra_args,
        ],
    )


def _get_run_dir(tmp_path: Path) -> Path:
    """Return the single run directory created under tmp_path/runs/."""
    run_dirs = list((tmp_path / "runs").iterdir())
    assert run_dirs, "No run directory was created"
    return run_dirs[0]


# ---------------------------------------------------------------------------
# Without --optimize: behaviour unchanged
# ---------------------------------------------------------------------------

class TestNoOptimize:
    def test_exits_0(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        img_dir = tmp_path / "input"
        _make_png(img_dir / "a.png")
        result = _run_upscale(str(img_dir))
        assert result.exit_code == 0, result.output

    def test_no_optimized_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        img_dir = tmp_path / "input"
        _make_png(img_dir / "a.png")
        _run_upscale(str(img_dir))
        assert not (_get_run_dir(tmp_path) / "optimized").exists()

    def test_manifest_has_no_optimization_key(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        img_dir = tmp_path / "input"
        _make_png(img_dir / "a.png")
        _run_upscale(str(img_dir))
        manifest = json.loads((_get_run_dir(tmp_path) / "manifest.json").read_text())
        assert "optimization" not in manifest


# ---------------------------------------------------------------------------
# With --optimize: chained flow
# ---------------------------------------------------------------------------

class TestWithOptimize:
    def _setup(self, tmp_path, monkeypatch, *extra):
        monkeypatch.chdir(tmp_path)
        img_dir = tmp_path / "input"
        _make_png(img_dir / "img.png")
        result = _run_upscale(str(img_dir), "--optimize", *extra)
        run_dir = _get_run_dir(tmp_path)
        return result, run_dir

    def test_exits_0(self, tmp_path, monkeypatch):
        result, _ = self._setup(tmp_path, monkeypatch)
        assert result.exit_code == 0, result.output

    def test_optimized_dir_created(self, tmp_path, monkeypatch):
        _, run_dir = self._setup(tmp_path, monkeypatch)
        assert (run_dir / "optimized").is_dir()

    def test_webp_generated(self, tmp_path, monkeypatch):
        _, run_dir = self._setup(tmp_path, monkeypatch)
        assert list((run_dir / "optimized" / "webp").glob("*.webp"))

    def test_jpeg_generated(self, tmp_path, monkeypatch):
        _, run_dir = self._setup(tmp_path, monkeypatch)
        assert list((run_dir / "optimized" / "jpeg").glob("*.jpg"))

    def test_per_image_csv_created(self, tmp_path, monkeypatch):
        _, run_dir = self._setup(tmp_path, monkeypatch)
        assert (run_dir / "optimized" / "per_image.csv").is_file()

    def test_summary_json_created(self, tmp_path, monkeypatch):
        _, run_dir = self._setup(tmp_path, monkeypatch)
        assert (run_dir / "optimized" / "summary.json").is_file()

    def test_manifest_has_optimization_key(self, tmp_path, monkeypatch):
        _, run_dir = self._setup(tmp_path, monkeypatch)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "optimization" in manifest

    def test_manifest_inference_fields_intact(self, tmp_path, monkeypatch):
        _, run_dir = self._setup(tmp_path, monkeypatch)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "model" in manifest
        assert "timing" in manifest
        assert "status" in manifest

    def test_outputs_png_unchanged(self, tmp_path, monkeypatch):
        _, run_dir = self._setup(tmp_path, monkeypatch)
        pngs = list((run_dir / "outputs").glob("*.png"))
        assert pngs
        for p in pngs:
            assert p.stat().st_size > 0
            img = Image.open(p)
            assert img.size[0] > 0

    def test_opt_webp_quality_propagated(self, tmp_path, monkeypatch):
        _, run_dir = self._setup(tmp_path, monkeypatch, "--opt-webp-quality", "50")
        data = json.loads((run_dir / "optimized" / "summary.json").read_text())
        assert data["config"]["webp_quality"] == 50

    def test_opt_jpeg_quality_propagated(self, tmp_path, monkeypatch):
        _, run_dir = self._setup(tmp_path, monkeypatch, "--opt-jpeg-quality", "60")
        data = json.loads((run_dir / "optimized" / "summary.json").read_text())
        assert data["config"]["jpeg_quality"] == 60

    def test_opt_format_webp_only(self, tmp_path, monkeypatch):
        _, run_dir = self._setup(tmp_path, monkeypatch, "--opt-format", "webp")
        assert (run_dir / "optimized" / "webp").is_dir()
        jpeg_dir = run_dir / "optimized" / "jpeg"
        assert not jpeg_dir.exists() or not list(jpeg_dir.glob("*"))

    def test_optimization_result_in_console_output(self, tmp_path, monkeypatch):
        result, _ = self._setup(tmp_path, monkeypatch)
        assert "ptimiz" in result.output

    def test_logs_txt_mentions_optimization(self, tmp_path, monkeypatch):
        _, run_dir = self._setup(tmp_path, monkeypatch)
        logs = (run_dir / "logs.txt").read_text(encoding="utf-8")
        assert "ptimiz" in logs
