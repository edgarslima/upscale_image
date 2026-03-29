"""Tests for PDF page extraction layer (Step 21).

Covers:
- extract_pdf_pages happy path (pages generated, names deterministic)
- Page ordering and naming convention (page-0001, page-0002, ...)
- Source PDF copied into run_dir/pdf/source/
- Structural errors: missing file, invalid PDF, zero-page PDF
- PdfExtractionConfig defaults
- patch_manifest_with_pdf_source
- CLI pdf subcommand smoke tests
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import fitz  # pymupdf
import pytest
from typer.testing import CliRunner

from upscale_image.cli.main import app
from upscale_image.pdf import (
    PdfExtractionConfig,
    PdfExtractionResult,
    default_pdf_extraction_config,
    extract_pdf_pages,
)
from upscale_image.pipeline.manifest import patch_manifest_with_pdf_source

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pdf(path: Path, n_pages: int = 3, width: int = 200, height: int = 300) -> Path:
    """Create a minimal valid PDF with *n_pages* pages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=width, height=height)
        page.draw_rect(fitz.Rect(10, 10, 50, 50), color=(0.5, 0.5, 0.5), fill=(0.5, 0.5, 0.5))
    doc.save(str(path))
    doc.close()
    return path


def _make_run_dir(tmp_path: Path) -> Path:
    """Create a minimal run directory with a placeholder manifest."""
    run_dir = tmp_path / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": "run_test",
        "model": {"name": "mock", "scale": 2, "device": "cpu", "precision": "fp32"},
        "status": {"total": 0, "done": 0, "failed": 0},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# PdfExtractionConfig
# ---------------------------------------------------------------------------

class TestPdfExtractionConfig:
    def test_defaults(self):
        cfg = default_pdf_extraction_config()
        assert cfg.dpi > 0
        assert cfg.colorspace in ("rgb", "gray")

    def test_custom_dpi(self):
        cfg = PdfExtractionConfig(dpi=300)
        assert cfg.dpi == 300


# ---------------------------------------------------------------------------
# extract_pdf_pages — happy path
# ---------------------------------------------------------------------------

class TestExtractPdfPagesHappyPath:
    def test_returns_result(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        run_dir = _make_run_dir(tmp_path)
        result = extract_pdf_pages(pdf, run_dir)
        assert isinstance(result, PdfExtractionResult)

    def test_extracted_count_matches_pages(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=3)
        run_dir = _make_run_dir(tmp_path)
        result = extract_pdf_pages(pdf, run_dir)
        assert result.total_pages == 3
        assert result.extracted == 3
        assert result.failed == 0

    def test_extracted_dir_created(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        run_dir = _make_run_dir(tmp_path)
        result = extract_pdf_pages(pdf, run_dir)
        assert result.extracted_dir.is_dir()

    def test_source_copy_in_run_dir(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        run_dir = _make_run_dir(tmp_path)
        result = extract_pdf_pages(pdf, run_dir)
        assert result.source_copy.is_file()
        assert result.source_copy.name == "doc.pdf"

    def test_png_files_created(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        run_dir = _make_run_dir(tmp_path)
        result = extract_pdf_pages(pdf, run_dir)
        pngs = list(result.extracted_dir.glob("*.png"))
        assert len(pngs) == 2

    def test_page_naming_convention(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=3)
        run_dir = _make_run_dir(tmp_path)
        result = extract_pdf_pages(pdf, run_dir)
        names = sorted(p.name for p in result.extracted_dir.glob("*.png"))
        assert names == ["page-0001.png", "page-0002.png", "page-0003.png"]

    def test_page_ordering_deterministic(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=5)
        run_dir = _make_run_dir(tmp_path)
        r1 = extract_pdf_pages(pdf, run_dir)
        # Re-run cleans up and re-extracts into the same dir (exist_ok=True)
        r2 = extract_pdf_pages(pdf, run_dir)
        names1 = sorted(p.name for p in r1.extracted_dir.glob("*.png"))
        names2 = sorted(p.name for p in r2.extracted_dir.glob("*.png"))
        assert names1 == names2

    def test_single_page_pdf(self, tmp_path):
        pdf = _make_pdf(tmp_path / "single.pdf", n_pages=1)
        run_dir = _make_run_dir(tmp_path)
        result = extract_pdf_pages(pdf, run_dir)
        assert result.total_pages == 1
        assert result.extracted == 1
        pngs = list(result.extracted_dir.glob("*.png"))
        assert pngs[0].name == "page-0001.png"

    def test_page_info_has_dimensions(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        run_dir = _make_run_dir(tmp_path)
        result = extract_pdf_pages(pdf, run_dir)
        page = result.pages[0]
        assert page.width_px > 0
        assert page.height_px > 0
        assert page.status == "ok"

    def test_custom_dpi_affects_dimensions(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        run_dir1 = _make_run_dir(tmp_path / "r1")
        run_dir2 = _make_run_dir(tmp_path / "r2")
        r1 = extract_pdf_pages(pdf, run_dir1, PdfExtractionConfig(dpi=72))
        r2 = extract_pdf_pages(pdf, run_dir2, PdfExtractionConfig(dpi=144))
        assert r2.pages[0].width_px > r1.pages[0].width_px

    def test_structure_under_run_dir(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        run_dir = _make_run_dir(tmp_path)
        result = extract_pdf_pages(pdf, run_dir)
        assert (run_dir / "pdf" / "source").is_dir()
        assert (run_dir / "pdf" / "extracted_pages").is_dir()


# ---------------------------------------------------------------------------
# Structural errors
# ---------------------------------------------------------------------------

class TestStructuralErrors:
    def test_missing_pdf_raises_file_not_found(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            extract_pdf_pages(tmp_path / "ghost.pdf", run_dir)

    def test_invalid_pdf_raises_value_error(self, tmp_path):
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a pdf at all")
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(ValueError, match="Cannot open PDF"):
            extract_pdf_pages(bad_pdf, run_dir)

    def test_truncated_pdf_raises_value_error(self, tmp_path):
        """A PDF truncated after the header should raise ValueError."""
        truncated = tmp_path / "truncated.pdf"
        # Valid PDF header but no content — pymupdf raises on open
        truncated.write_bytes(b"%PDF-1.4\n%%EOF\n")
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(ValueError, match="Cannot open PDF|zero pages"):
            extract_pdf_pages(truncated, run_dir)


# ---------------------------------------------------------------------------
# Manifest patching
# ---------------------------------------------------------------------------

class TestManifestPdf:
    def test_patch_adds_pdf_source_key(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        run_dir = _make_run_dir(tmp_path)
        result = extract_pdf_pages(pdf, run_dir)
        patch_manifest_with_pdf_source(run_dir, result)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "pdf_source" in manifest

    def test_patch_preserves_existing_fields(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        run_dir = _make_run_dir(tmp_path)
        result = extract_pdf_pages(pdf, run_dir)
        patch_manifest_with_pdf_source(run_dir, result)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["run_id"] == "run_test"
        assert "model" in manifest

    def test_patch_pdf_source_fields(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        run_dir = _make_run_dir(tmp_path)
        result = extract_pdf_pages(pdf, run_dir)
        patch_manifest_with_pdf_source(run_dir, result)
        src = json.loads((run_dir / "manifest.json").read_text())["pdf_source"]
        assert src["input_mode"] == "pdf"
        assert src["total_pages"] == 2
        assert src["extracted"] == 2
        assert "source_copy" in src["artifacts"]
        assert "extracted_pages_dir" in src["artifacts"]


# ---------------------------------------------------------------------------
# CLI pdf subcommand
# ---------------------------------------------------------------------------

class TestCLIPdf:
    def test_pdf_command_exits_0(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        result = runner.invoke(app, [
            "pdf", str(pdf),
            "--output", "runs",
            "--model", "mock",
            "--scale", "2",
            "--device", "cpu",
        ])
        assert result.exit_code == 0, result.output

    def test_pdf_command_creates_outputs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        runner.invoke(app, [
            "pdf", str(pdf),
            "--output", "runs",
            "--model", "mock",
            "--scale", "2",
            "--device", "cpu",
        ])
        run_dirs = list((tmp_path / "runs").iterdir())
        assert run_dirs
        pngs = list((run_dirs[0] / "outputs").glob("*.png"))
        assert len(pngs) == 2

    def test_pdf_command_creates_extracted_pages(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        runner.invoke(app, [
            "pdf", str(pdf),
            "--output", "runs",
            "--model", "mock",
            "--scale", "2",
            "--device", "cpu",
        ])
        run_dir = next((tmp_path / "runs").iterdir())
        pages = list((run_dir / "pdf" / "extracted_pages").glob("*.png"))
        assert len(pages) == 2

    def test_pdf_command_patches_manifest(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        runner.invoke(app, [
            "pdf", str(pdf),
            "--output", "runs",
            "--model", "mock",
            "--scale", "2",
            "--device", "cpu",
        ])
        run_dir = next((tmp_path / "runs").iterdir())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "pdf_source" in manifest

    def test_pdf_nonexistent_file_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, [
            "pdf", str(tmp_path / "ghost.pdf"),
            "--output", "runs",
            "--model", "mock",
            "--scale", "2",
        ])
        assert result.exit_code == 1

    def test_pdf_with_optimize_flag(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        runner.invoke(app, [
            "pdf", str(pdf),
            "--output", "runs",
            "--model", "mock",
            "--scale", "2",
            "--device", "cpu",
            "--optimize",
        ])
        run_dir = next((tmp_path / "runs").iterdir())
        assert (run_dir / "optimized").is_dir()

    def test_image_pipeline_unaffected(self, tmp_path, monkeypatch):
        """Existing upscale command must still work without PDF."""
        import numpy as np
        from PIL import Image
        monkeypatch.chdir(tmp_path)
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        arr = np.full((32, 32, 3), 100, dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / "a.png")
        result = runner.invoke(app, [
            "upscale", str(img_dir),
            "--output", "runs",
            "--model", "mock",
            "--scale", "2",
            "--device", "cpu",
        ])
        assert result.exit_code == 0, result.output
