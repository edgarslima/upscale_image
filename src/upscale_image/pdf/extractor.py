"""PDF page extraction using PyMuPDF (fitz).

Pages are rendered to PNG images with deterministic naming (``page-NNNN.png``)
and persisted inside the run directory for full auditability (ADR 0002).

The extracted pages are designed to be consumed by the existing image
discovery and batch pipeline without any modification to the core pipeline.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class PdfExtractionConfig:
    """Configuration for PDF page extraction."""

    dpi: int = 150          # Render resolution; 150 dpi is a good balance for SR input
    colorspace: str = "rgb"  # "rgb" or "gray"


@dataclass
class PdfPageInfo:
    """Metadata for a single extracted page."""

    page_number: int        # 1-based
    extracted_path: Path    # Absolute path to the PNG file
    width_px: int
    height_px: int
    status: str             # "ok" | "error"
    error: str | None


@dataclass
class PdfExtractionResult:
    """Result of a full PDF extraction run."""

    source_pdf: Path
    total_pages: int
    extracted: int
    failed: int
    extracted_dir: Path     # run_dir/pdf/extracted_pages/
    source_copy: Path       # run_dir/pdf/source/<filename>
    pages: list[PdfPageInfo]


def default_pdf_extraction_config() -> PdfExtractionConfig:
    return PdfExtractionConfig()


def extract_pdf_pages(
    pdf_path: str | Path,
    run_dir: str | Path,
    config: PdfExtractionConfig | None = None,
) -> PdfExtractionResult:
    """Extract all pages of a PDF as numbered PNG images inside *run_dir*.

    Args:
        pdf_path: Path to the source PDF file.
        run_dir:  Path to the run directory (must already exist).
        config:   Extraction settings; defaults to :func:`default_pdf_extraction_config`.

    Returns:
        :class:`PdfExtractionResult` with per-page metadata.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        ValueError:        If the file is not a readable PDF or has zero pages.
    """
    import fitz  # pymupdf — imported lazily so the rest of the app works without it

    if config is None:
        config = default_pdf_extraction_config()

    pdf_path = Path(pdf_path)
    run_dir = Path(run_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Validate early
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF: {pdf_path} — {exc}") from exc

    total_pages = doc.page_count
    if total_pages == 0:
        doc.close()
        raise ValueError(f"PDF has zero pages: {pdf_path}")

    # Create directory structure
    pdf_dir = run_dir / "pdf"
    source_dir = pdf_dir / "source"
    extracted_dir = pdf_dir / "extracted_pages"
    source_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Copy original PDF into the run for auditability (ADR 0002)
    source_copy = source_dir / pdf_path.name
    shutil.copy2(str(pdf_path), str(source_copy))

    log.info(
        "PDF extraction started — file=%s  pages=%d  dpi=%d",
        pdf_path.name, total_pages, config.dpi,
    )

    colorspace = fitz.csRGB if config.colorspace == "rgb" else fitz.csGRAY
    pages: list[PdfPageInfo] = []

    for page_index in range(total_pages):
        page_number = page_index + 1
        out_name = f"page-{page_number:04d}.png"
        out_path = extracted_dir / out_name

        try:
            page = doc[page_index]
            matrix = fitz.Matrix(config.dpi / 72, config.dpi / 72)
            pix = page.get_pixmap(matrix=matrix, colorspace=colorspace, alpha=False)
            pix.save(str(out_path))

            pages.append(PdfPageInfo(
                page_number=page_number,
                extracted_path=out_path,
                width_px=pix.width,
                height_px=pix.height,
                status="ok",
                error=None,
            ))
            log.debug("  extracted page %d → %s (%dx%d)", page_number, out_name, pix.width, pix.height)

        except Exception as exc:  # noqa: BLE001
            log.warning("  failed page %d: %s", page_number, exc)
            pages.append(PdfPageInfo(
                page_number=page_number,
                extracted_path=out_path,
                width_px=0,
                height_px=0,
                status="error",
                error=str(exc),
            ))

    doc.close()

    extracted_count = sum(1 for p in pages if p.status == "ok")
    failed_count = sum(1 for p in pages if p.status == "error")

    log.info(
        "PDF extraction done — extracted=%d  failed=%d",
        extracted_count, failed_count,
    )

    return PdfExtractionResult(
        source_pdf=pdf_path,
        total_pages=total_pages,
        extracted=extracted_count,
        failed=failed_count,
        extracted_dir=extracted_dir,
        source_copy=source_copy,
        pages=pages,
    )
