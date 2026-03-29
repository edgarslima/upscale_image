"""PDF ingestion and recomposition layer."""

from upscale_image.pdf.composer import PdfComposeResult, compose_pdf_from_pages
from upscale_image.pdf.extractor import (
    PdfExtractionConfig,
    PdfExtractionResult,
    PdfPageInfo,
    default_pdf_extraction_config,
    extract_pdf_pages,
)
from upscale_image.pdf.page_preparer import (
    ComposeReadyResult,
    PagePrepConfig,
    prepare_pages_for_composition,
)

__all__ = [
    "PdfExtractionConfig",
    "PdfExtractionResult",
    "PdfPageInfo",
    "PdfComposeResult",
    "default_pdf_extraction_config",
    "extract_pdf_pages",
    "compose_pdf_from_pages",
    "ComposeReadyResult",
    "PagePrepConfig",
    "prepare_pages_for_composition",
]
