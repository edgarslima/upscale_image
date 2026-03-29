"""PDF recomposition: rebuild a PDF from processed page images.

Reads canonical PNG pages from ``outputs/`` and assembles them into a new
PDF preserving the original page order.  The rebuilt PDF is persisted in
``pdf/rebuilt/`` inside the run directory, never overwriting any canonical
artefact (ADR 0010).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Matches names like page-0001.png, page-0001.jpg, page-0023.png, …
_PAGE_NAME_RE = re.compile(r"^page-(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


@dataclass
class PdfComposeResult:
    """Result of PDF recomposition."""

    source_pages_dir: Path   # directory where page images were read from
    output_pdf: Path         # path to the rebuilt PDF
    pages_included: int
    status: str              # "ok" | "error"
    error: str | None


def compose_pdf_from_pages(
    pages_dir: str | Path,
    run_dir: str | Path,
    output_stem: str = "output",
) -> PdfComposeResult:
    """Assemble a PDF from PNG pages in *pages_dir*.

    Pages are included in deterministic ascending order derived from their
    ``page-NNNN.png`` filenames.  Non-matching files are silently skipped.

    Args:
        pages_dir:   Directory containing ``page-NNNN.png`` files.
        run_dir:     Run directory; the PDF is written to ``pdf/rebuilt/``.
        output_stem: Base name (without extension) for the output PDF.

    Returns:
        :class:`PdfComposeResult` describing the outcome.

    Raises:
        FileNotFoundError: If *pages_dir* does not exist.
        ValueError:        If no eligible page images are found.
    """
    import fitz  # pymupdf

    pages_dir = Path(pages_dir)
    run_dir = Path(run_dir)

    if not pages_dir.exists():
        raise FileNotFoundError(f"Pages directory not found: {pages_dir}")

    # Collect and sort page files by page number
    page_files: list[tuple[int, Path]] = []
    for p in pages_dir.iterdir():
        m = _PAGE_NAME_RE.match(p.name)
        if m:
            page_files.append((int(m.group(1)), p))

    page_files.sort(key=lambda t: t[0])

    if not page_files:
        raise ValueError(f"No page images found in {pages_dir}")

    rebuilt_dir = run_dir / "pdf" / "rebuilt"
    rebuilt_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = rebuilt_dir / f"{output_stem}.upscaled.pdf"

    log.info(
        "PDF recomposition started — pages=%d  output=%s",
        len(page_files), output_pdf.name,
    )

    try:
        doc = fitz.open()
        for _page_num, img_path in page_files:
            img_doc = fitz.open(str(img_path))
            pdfbytes = img_doc.convert_to_pdf()
            img_doc.close()
            img_pdf = fitz.open("pdf", pdfbytes)
            doc.insert_pdf(img_pdf)
            img_pdf.close()

        doc.save(str(output_pdf))
        doc.close()

        log.info("PDF recomposition done — %s", output_pdf.name)

        return PdfComposeResult(
            source_pages_dir=pages_dir,
            output_pdf=output_pdf,
            pages_included=len(page_files),
            status="ok",
            error=None,
        )

    except Exception as exc:  # noqa: BLE001
        log.warning("PDF recomposition failed: %s", exc)
        return PdfComposeResult(
            source_pages_dir=pages_dir,
            output_pdf=output_pdf,
            pages_included=0,
            status="error",
            error=str(exc),
        )
