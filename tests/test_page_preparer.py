"""Tests for pdf/page_preparer.py (Step 23).

Covers:
- prepare_pages_for_composition happy path
- compose_ready_pages/ directory is created inside run_dir/pdf/
- Output files are JPEG, not PNG (canonical outputs untouched)
- Progressive compression: lower quality used when budget is tight
- within_budget True when budget is achievable
- within_budget False when budget cannot be met (all quality steps exhausted)
- Correct ratio and budget_bytes in result
- FileNotFoundError when outputs_dir is missing
- ValueError when no page PNGs found
- patch_manifest_with_compose_ready writes correct keys
- CLI upscale --pdf-file creates compose_ready_pages/
- CLI upscale --pdf-file compose_ready_pages/ contains JPEGs, not PNGs
- CLI upscale --pdf-file with over-budget produces status output
- canonical outputs/*.png remain intact after preparation
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from upscale_image.cli.main import app
from upscale_image.pdf import ComposeReadyResult, PagePrepConfig, prepare_pages_for_composition
from upscale_image.pipeline.manifest import patch_manifest_with_compose_ready

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page_png(path: Path, width: int = 64, height: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((height, width, 3), 120, dtype=np.uint8)
    Image.fromarray(arr).save(path, format="PNG")
    return path


def _make_page_pngs(outputs_dir: Path, n: int = 3, width: int = 64, height: int = 64) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        _make_page_png(outputs_dir / f"page-{i:04d}.png", width=width, height=height)


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": "run_test", "model": {}, "status": {}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def _make_pdf(path: Path, n_pages: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=100, height=150)
        page.draw_rect(fitz.Rect(5, 5, 30, 30), color=(0.3, 0.6, 0.3), fill=(0.3, 0.6, 0.3))
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Unit tests: prepare_pages_for_composition
# ---------------------------------------------------------------------------

class TestPreparePages:
    def test_returns_compose_ready_result(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=2)
        result = prepare_pages_for_composition(outputs, run_dir, source_pdf_size_bytes=10_000)
        assert isinstance(result, ComposeReadyResult)

    def test_compose_ready_dir_created(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=2)
        result = prepare_pages_for_composition(outputs, run_dir, source_pdf_size_bytes=10_000)
        assert result.pages_dir.is_dir()
        assert result.pages_dir == run_dir / "pdf" / "compose_ready_pages"

    def test_output_files_are_jpeg(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=3)
        result = prepare_pages_for_composition(outputs, run_dir, source_pdf_size_bytes=10_000)
        files = list(result.pages_dir.iterdir())
        assert all(f.suffix.lower() == ".jpg" for f in files)

    def test_canonical_pngs_untouched(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=2)
        before = {f.name for f in outputs.iterdir()}
        prepare_pages_for_composition(outputs, run_dir, source_pdf_size_bytes=10_000)
        after = {f.name for f in outputs.iterdir()}
        assert before == after

    def test_pages_count_matches_input(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=4)
        result = prepare_pages_for_composition(outputs, run_dir, source_pdf_size_bytes=10_000)
        assert result.pages_count == 4

    def test_jpeg_count_matches_input(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=3)
        result = prepare_pages_for_composition(outputs, run_dir, source_pdf_size_bytes=10_000)
        jpegs = list(result.pages_dir.glob("*.jpg"))
        assert len(jpegs) == 3

    def test_within_budget_true_for_generous_budget(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=2, width=32, height=32)
        # Very generous budget: 100 MB
        result = prepare_pages_for_composition(
            outputs, run_dir, source_pdf_size_bytes=50_000_000
        )
        assert result.within_budget is True

    def test_within_budget_false_for_impossible_budget(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=3, width=64, height=64)
        # Budget of 1 byte — impossible
        cfg = PagePrepConfig(budget_ratio=1.0, quality_steps=[85, 70, 55, 40])
        result = prepare_pages_for_composition(
            outputs, run_dir, source_pdf_size_bytes=1, config=cfg
        )
        assert result.within_budget is False

    def test_budget_bytes_matches_formula(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=1)
        source_size = 20_000
        cfg = PagePrepConfig(budget_ratio=2.0)
        result = prepare_pages_for_composition(outputs, run_dir, source_size, cfg)
        assert result.budget_bytes == 40_000

    def test_source_pdf_bytes_recorded(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=1)
        result = prepare_pages_for_composition(outputs, run_dir, source_pdf_size_bytes=99_999)
        assert result.source_pdf_bytes == 99_999

    def test_ratio_is_estimated_over_source(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=2, width=32, height=32)
        source_size = 50_000
        result = prepare_pages_for_composition(outputs, run_dir, source_size)
        expected_ratio = result.estimated_bytes / source_size
        assert abs(result.ratio - round(expected_ratio, 4)) < 1e-6

    def test_lower_quality_used_when_budget_tight(self, tmp_path):
        """Progressive compression: tighter budget forces lower quality."""
        run_dir_a = _make_run_dir(tmp_path / "run_a")
        run_dir_b = _make_run_dir(tmp_path / "run_b")
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=2, width=128, height=128)

        # Generous budget — should stay at first (highest) quality
        cfg_generous = PagePrepConfig(budget_ratio=1000.0, quality_steps=[85, 55])
        result_generous = prepare_pages_for_composition(outputs, run_dir_a, 1, cfg_generous)

        # Tight budget — may force lower quality
        cfg_tight = PagePrepConfig(budget_ratio=0.0001, quality_steps=[85, 55])
        result_tight = prepare_pages_for_composition(outputs, run_dir_b, 1_000_000, cfg_tight)

        assert result_generous.preset_quality >= result_tight.preset_quality

    def test_missing_outputs_dir_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            prepare_pages_for_composition(tmp_path / "ghost", run_dir, 10_000)

    def test_no_page_pngs_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        empty = tmp_path / "outputs"
        empty.mkdir()
        (empty / "not_a_page.txt").write_text("ignored")
        with pytest.raises(ValueError, match="No page PNG files"):
            prepare_pages_for_composition(empty, run_dir, 10_000)

    def test_non_page_files_skipped(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=2)
        (outputs / "some_other_image.png").write_bytes(b"")  # not matching page-NNNN pattern
        result = prepare_pages_for_composition(outputs, run_dir, 10_000)
        assert result.pages_count == 2

    def test_default_config_used_when_none(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=1)
        result = prepare_pages_for_composition(outputs, run_dir, 10_000, config=None)
        assert result.preset_quality in [85, 70, 55, 40]


# ---------------------------------------------------------------------------
# Manifest patching
# ---------------------------------------------------------------------------

class TestManifestComposeReady:
    def test_patch_adds_pdf_compose_ready_key(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=1)
        result = prepare_pages_for_composition(outputs, run_dir, 10_000)
        patch_manifest_with_compose_ready(run_dir, result)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "pdf_compose_ready" in manifest

    def test_patch_preserves_other_keys(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=1)
        result = prepare_pages_for_composition(outputs, run_dir, 10_000)
        patch_manifest_with_compose_ready(run_dir, result)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["run_id"] == "run_test"

    def test_patch_fields_present(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=2)
        result = prepare_pages_for_composition(outputs, run_dir, 20_000)
        patch_manifest_with_compose_ready(run_dir, result)
        cr = json.loads((run_dir / "manifest.json").read_text())["pdf_compose_ready"]
        assert "source_pdf_bytes" in cr
        assert "budget_bytes" in cr
        assert "estimated_bytes" in cr
        assert "ratio" in cr
        assert "within_budget" in cr
        assert "preset_quality" in cr
        assert "pages_count" in cr
        assert "artifacts" in cr

    def test_patch_artifacts_has_dir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=1)
        result = prepare_pages_for_composition(outputs, run_dir, 10_000)
        patch_manifest_with_compose_ready(run_dir, result)
        cr = json.loads((run_dir / "manifest.json").read_text())["pdf_compose_ready"]
        assert "compose_ready_pages_dir" in cr["artifacts"]

    def test_patch_within_budget_value(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        outputs = tmp_path / "outputs"
        _make_page_pngs(outputs, n=1, width=16, height=16)
        result = prepare_pages_for_composition(outputs, run_dir, 50_000_000)
        patch_manifest_with_compose_ready(run_dir, result)
        cr = json.loads((run_dir / "manifest.json").read_text())["pdf_compose_ready"]
        assert cr["within_budget"] is True


# ---------------------------------------------------------------------------
# CLI integration: upscale --pdf-file
# ---------------------------------------------------------------------------

class TestCliComposeReady:
    def _invoke(self, tmp_path, monkeypatch, pdf: Path, *extra):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, [
            "upscale",
            "--pdf-file", str(pdf),
            "--output", "runs",
            "--model", "mock", "--scale", "2", "--device", "cpu",
            *extra,
        ])
        run_dir = next((tmp_path / "runs").iterdir()) if (tmp_path / "runs").exists() else None
        return result, run_dir

    def test_compose_ready_pages_dir_created(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        assert (run_dir / "pdf" / "compose_ready_pages").is_dir()

    def test_compose_ready_pages_are_jpeg(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        jpegs = list((run_dir / "pdf" / "compose_ready_pages").glob("*.jpg"))
        assert len(jpegs) == 2

    def test_canonical_outputs_are_png(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        pngs = list((run_dir / "outputs").glob("*.png"))
        assert len(pngs) == 2

    def test_canonical_outputs_unchanged(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        # Only .png files in outputs/ — no .jpg
        files_in_outputs = list((run_dir / "outputs").iterdir())
        assert all(f.suffix == ".png" for f in files_in_outputs)

    def test_manifest_has_pdf_compose_ready(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "pdf_compose_ready" in manifest

    def test_manifest_compose_ready_within_budget_field(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "within_budget" in manifest["pdf_compose_ready"]

    def test_rebuilt_pdf_still_created(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        rebuilt = list((run_dir / "pdf" / "rebuilt").glob("*.pdf"))
        assert rebuilt

    def test_exits_0_with_default_budget(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        result, _ = self._invoke(tmp_path, monkeypatch, pdf)
        assert result.exit_code == 0, result.output

    def test_custom_budget_ratio_accepted(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        result, run_dir = self._invoke(tmp_path, monkeypatch, pdf, "--pdf-budget-ratio", "3.0")
        assert result.exit_code == 0, result.output
        manifest = json.loads((run_dir / "manifest.json").read_text())
        # budget_bytes should reflect 3.0×
        source = manifest["pdf_compose_ready"]["source_pdf_bytes"]
        budget = manifest["pdf_compose_ready"]["budget_bytes"]
        assert budget == int(source * 3.0)

    def test_over_budget_warning_in_output(self, tmp_path, monkeypatch):
        """An impossible budget (ratio=0.000001) should produce a visible warning."""
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        result, _ = self._invoke(tmp_path, monkeypatch, pdf, "--pdf-budget-ratio", "0.000001")
        # CLI should still exit 0 (non-fatal) but print a budget warning
        assert "budget" in result.output.lower()

    def test_over_budget_within_budget_false_in_manifest(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf, "--pdf-budget-ratio", "0.000001")
        if run_dir is None:
            pytest.skip("run_dir not created")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["pdf_compose_ready"]["within_budget"] is False
