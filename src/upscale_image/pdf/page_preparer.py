"""Pre-composition page preparation: compress PNG outputs to JPEG for PDF recomposition.

This module sits between the canonical outputs/ directory and pdf/rebuilt/.
It generates pdf/compose_ready_pages/ with pages optimized for PDF composition,
applying progressive JPEG compression to stay within the size budget.

The canonical outputs/*.png files are never modified (ADR 0010).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


_PAGE_PATTERN = re.compile(r"^page-(\d+)\.png$", re.IGNORECASE)


@dataclass
class PagePrepConfig:
    """Configuration for pre-composition page preparation.

    Args:
        budget_ratio:  Maximum allowed ratio of final PDF size to source PDF size.
                       Default 2.0 means the rebuilt PDF must be ≤ 2× the original.
        quality_steps: JPEG quality levels tried in descending order.
                       The first level that keeps estimated size within budget is used.
    """

    budget_ratio: float = 2.0
    quality_steps: list[int] = field(default_factory=lambda: [85, 70, 55, 40])


@dataclass
class ComposeReadyResult:
    """Result of the pre-composition preparation step.

    Attributes:
        pages_dir:       Path to pdf/compose_ready_pages/ directory.
        source_pdf_bytes: Size of the original PDF in bytes.
        budget_bytes:    Maximum allowed size for the rebuilt PDF.
        estimated_bytes: Sum of sizes of all prepared page files (proxy for PDF size).
        ratio:           estimated_bytes / source_pdf_bytes.
        within_budget:   True if estimated_bytes ≤ budget_bytes.
        preset_quality:  JPEG quality level that was ultimately applied.
        pages_count:     Number of pages prepared.
    """

    pages_dir: Path
    source_pdf_bytes: int
    budget_bytes: int
    estimated_bytes: int
    ratio: float
    within_budget: bool
    preset_quality: int
    pages_count: int


def prepare_pages_for_composition(
    outputs_dir: Path,
    run_dir: Path,
    source_pdf_size_bytes: int,
    config: PagePrepConfig | None = None,
) -> ComposeReadyResult:
    """Convert canonical PNG outputs to compressed JPEGs ready for PDF composition.

    Pages are written to ``run_dir/pdf/compose_ready_pages/``.  The function
    tries each quality level in *config.quality_steps* until the estimated
    total size fits within the budget, or until all levels are exhausted.

    The canonical ``outputs/*.png`` files are never modified.

    Args:
        outputs_dir:           Directory containing the canonical PNG outputs.
        run_dir:               Root of the current run (e.g. ``runs/<run_id>/``).
        source_pdf_size_bytes: File size of the original PDF (determines budget).
        config:                Preparation configuration; defaults to
                               :class:`PagePrepConfig` with its default values.

    Returns:
        :class:`ComposeReadyResult` describing the outcome.

    Raises:
        FileNotFoundError: If *outputs_dir* does not exist.
        ValueError:        If no page PNG files are found in *outputs_dir*.
    """
    if config is None:
        config = PagePrepConfig()

    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs_dir not found: {outputs_dir}")

    # Collect page PNG files sorted by page number
    page_files = sorted(
        (f for f in outputs_dir.iterdir() if _PAGE_PATTERN.match(f.name)),
        key=lambda f: int(_PAGE_PATTERN.match(f.name).group(1)),  # type: ignore[union-attr]
    )

    if not page_files:
        raise ValueError(f"No page PNG files found in {outputs_dir}")

    compose_dir = run_dir / "pdf" / "compose_ready_pages"
    compose_dir.mkdir(parents=True, exist_ok=True)

    budget_bytes = int(source_pdf_size_bytes * config.budget_ratio)

    final_quality = config.quality_steps[-1]
    estimated_bytes = 0
    within_budget = False

    for quality in config.quality_steps:
        final_quality = quality
        _write_pages(page_files, compose_dir, quality)
        estimated_bytes = _sum_dir_bytes(compose_dir)
        if estimated_bytes <= budget_bytes:
            within_budget = True
            break

    ratio = estimated_bytes / source_pdf_size_bytes if source_pdf_size_bytes > 0 else float("inf")

    return ComposeReadyResult(
        pages_dir=compose_dir,
        source_pdf_bytes=source_pdf_size_bytes,
        budget_bytes=budget_bytes,
        estimated_bytes=estimated_bytes,
        ratio=round(ratio, 4),
        within_budget=within_budget,
        preset_quality=final_quality,
        pages_count=len(page_files),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_pages(page_files: list[Path], dest_dir: Path, quality: int) -> None:
    """Convert *page_files* to JPEG at *quality* inside *dest_dir*."""
    for src in page_files:
        stem = src.stem  # e.g. "page-0001"
        dest = dest_dir / f"{stem}.jpg"
        img = Image.open(src).convert("RGB")
        img.save(dest, format="JPEG", quality=quality, optimize=True)


def _sum_dir_bytes(directory: Path) -> int:
    """Return the total size in bytes of all files in *directory*."""
    return sum(f.stat().st_size for f in directory.iterdir() if f.is_file())
