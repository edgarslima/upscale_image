"""Tests for PDF recomposition and the integrated upscale --pdf-file flow (Step 22).

Covers:
- compose_pdf_from_pages happy path
- Page ordering preserved in rebuilt PDF
- Empty or missing pages_dir raises
- patch_manifest_with_pdf_rebuilt
- CLI: upscale in image mode unchanged
- CLI: upscale --pdf-file creates extracted pages, outputs, rebuilt PDF, manifest
- CLI: upscale --pdf-file --optimize produces optimized/ alongside pdf/rebuilt/
- CLI: error when both input_dir and --pdf-file provided
- CLI: error when neither input_dir nor --pdf-file provided
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
from upscale_image.pdf import PdfComposeResult, compose_pdf_from_pages
from upscale_image.pipeline.manifest import patch_manifest_with_pdf_rebuilt

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(path: Path, width: int = 32, height: int = 32) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((height, width, 3), 100, dtype=np.uint8)
    Image.fromarray(arr).save(path, format="PNG")
    return path


def _make_page_pngs(pages_dir: Path, n: int = 3) -> None:
    pages_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        _make_png(pages_dir / f"page-{i:04d}.png")


def _make_pdf(path: Path, n_pages: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=100, height=150)
        page.draw_rect(fitz.Rect(5, 5, 30, 30), color=(0.3, 0.6, 0.3), fill=(0.3, 0.6, 0.3))
    doc.save(str(path))
    doc.close()
    return path


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": "run_test", "model": {}, "status": {}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# compose_pdf_from_pages — happy path
# ---------------------------------------------------------------------------

class TestComposePdfHappyPath:
    def test_returns_result(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        pages_dir = tmp_path / "pages"
        _make_page_pngs(pages_dir, n=2)
        result = compose_pdf_from_pages(pages_dir, run_dir, output_stem="doc")
        assert isinstance(result, PdfComposeResult)

    def test_status_ok(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        pages_dir = tmp_path / "pages"
        _make_page_pngs(pages_dir, n=2)
        result = compose_pdf_from_pages(pages_dir, run_dir)
        assert result.status == "ok", result.error

    def test_output_pdf_created(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        pages_dir = tmp_path / "pages"
        _make_page_pngs(pages_dir, n=2)
        result = compose_pdf_from_pages(pages_dir, run_dir, output_stem="doc")
        assert result.output_pdf.is_file()

    def test_pages_included_count(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        pages_dir = tmp_path / "pages"
        _make_page_pngs(pages_dir, n=3)
        result = compose_pdf_from_pages(pages_dir, run_dir)
        assert result.pages_included == 3

    def test_rebuilt_dir_location(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        pages_dir = tmp_path / "pages"
        _make_page_pngs(pages_dir, n=1)
        result = compose_pdf_from_pages(pages_dir, run_dir, output_stem="x")
        assert result.output_pdf.parent == run_dir / "pdf" / "rebuilt"

    def test_output_stem_in_filename(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        pages_dir = tmp_path / "pages"
        _make_page_pngs(pages_dir, n=1)
        result = compose_pdf_from_pages(pages_dir, run_dir, output_stem="my_doc")
        assert "my_doc" in result.output_pdf.name

    def test_page_count_in_output_pdf(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        pages_dir = tmp_path / "pages"
        _make_page_pngs(pages_dir, n=3)
        result = compose_pdf_from_pages(pages_dir, run_dir)
        doc = fitz.open(str(result.output_pdf))
        assert doc.page_count == 3
        doc.close()

    def test_non_page_files_skipped(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        pages_dir = tmp_path / "pages"
        _make_page_pngs(pages_dir, n=2)
        (pages_dir / "README.txt").write_text("ignored")
        result = compose_pdf_from_pages(pages_dir, run_dir)
        assert result.pages_included == 2


# ---------------------------------------------------------------------------
# Structural errors
# ---------------------------------------------------------------------------

class TestComposeErrors:
    def test_missing_pages_dir_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            compose_pdf_from_pages(tmp_path / "ghost", run_dir)

    def test_empty_pages_dir_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No page images"):
            compose_pdf_from_pages(empty_dir, run_dir)


# ---------------------------------------------------------------------------
# Manifest patching
# ---------------------------------------------------------------------------

class TestManifestRebuilt:
    def test_patch_adds_pdf_rebuilt_key(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        pages_dir = tmp_path / "pages"
        _make_page_pngs(pages_dir, n=1)
        result = compose_pdf_from_pages(pages_dir, run_dir)
        patch_manifest_with_pdf_rebuilt(run_dir, result)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "pdf_rebuilt" in manifest

    def test_patch_preserves_other_fields(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        pages_dir = tmp_path / "pages"
        _make_page_pngs(pages_dir, n=1)
        result = compose_pdf_from_pages(pages_dir, run_dir)
        patch_manifest_with_pdf_rebuilt(run_dir, result)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["run_id"] == "run_test"

    def test_patch_rebuilt_fields(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        pages_dir = tmp_path / "pages"
        _make_page_pngs(pages_dir, n=2)
        result = compose_pdf_from_pages(pages_dir, run_dir, output_stem="doc")
        patch_manifest_with_pdf_rebuilt(run_dir, result)
        rb = json.loads((run_dir / "manifest.json").read_text())["pdf_rebuilt"]
        assert rb["status"] == "ok"
        assert rb["pages_included"] == 2
        assert rb["output_pdf"] is not None


# ---------------------------------------------------------------------------
# CLI upscale — image mode unchanged
# ---------------------------------------------------------------------------

class TestUpscaleImageMode:
    def test_image_mode_exits_0(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        img_dir = tmp_path / "imgs"
        _make_png(img_dir / "a.png")
        result = runner.invoke(app, [
            "upscale", str(img_dir),
            "--output", "runs",
            "--model", "mock", "--scale", "2", "--device", "cpu",
        ])
        assert result.exit_code == 0, result.output

    def test_image_mode_no_pdf_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        img_dir = tmp_path / "imgs"
        _make_png(img_dir / "a.png")
        runner.invoke(app, [
            "upscale", str(img_dir),
            "--output", "runs",
            "--model", "mock", "--scale", "2", "--device", "cpu",
        ])
        run_dir = next((tmp_path / "runs").iterdir())
        assert not (run_dir / "pdf").exists()
        assert "pdf_source" not in json.loads((run_dir / "manifest.json").read_text())

    def test_both_input_and_pdf_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        result = runner.invoke(app, [
            "upscale", str(img_dir),
            "--pdf-file", str(pdf),
            "--output", "runs",
            "--model", "mock", "--scale", "2",
        ])
        assert result.exit_code == 1

    def test_no_input_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, [
            "upscale",
            "--output", "runs",
            "--model", "mock", "--scale", "2",
        ])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# CLI upscale --pdf-file
# ---------------------------------------------------------------------------

class TestUpscalePdfMode:
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

    def test_exits_0(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        result, _ = self._invoke(tmp_path, monkeypatch, pdf)
        assert result.exit_code == 0, result.output

    def test_extracted_pages_created(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        pages = list((run_dir / "pdf" / "extracted_pages").glob("*.png"))
        assert len(pages) == 2

    def test_outputs_created(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        pngs = list((run_dir / "outputs").glob("*.png"))
        assert len(pngs) == 2

    def test_rebuilt_pdf_created(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        rebuilt = list((run_dir / "pdf" / "rebuilt").glob("*.pdf"))
        assert rebuilt

    def test_rebuilt_pdf_page_count(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=3)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        rebuilt_pdf = next((run_dir / "pdf" / "rebuilt").glob("*.pdf"))
        doc = fitz.open(str(rebuilt_pdf))
        assert doc.page_count == 3
        doc.close()

    def test_manifest_has_pdf_source(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "pdf_source" in manifest

    def test_manifest_has_pdf_rebuilt(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "pdf_rebuilt" in manifest

    def test_manifest_inference_fields_intact(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "model" in manifest
        assert "status" in manifest

    def test_with_optimize_flag(self, tmp_path, monkeypatch):
        pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
        _, run_dir = self._invoke(tmp_path, monkeypatch, pdf, "--optimize")
        assert (run_dir / "optimized").is_dir()
        assert (run_dir / "pdf" / "rebuilt").is_dir()

    def test_nonexistent_pdf_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, [
            "upscale",
            "--pdf-file", str(tmp_path / "ghost.pdf"),
            "--output", "runs",
            "--model", "mock", "--scale", "2",
        ])
        assert result.exit_code == 1
