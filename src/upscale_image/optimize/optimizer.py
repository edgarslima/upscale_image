"""Optimization of canonical run outputs into compressed distribution formats.

This module reads ``runs/<run_id>/outputs/*.png`` and generates
compressed derivatives in ``runs/<run_id>/optimized/``.

Key constraints (ADR 0010):
- ``outputs/*.png`` are **never** modified.
- Derived files land exclusively in ``optimized/<format>/``.
- Per-item failures are recoverable; they do not invalidate the run.
- Execution is deterministic for the same run and configuration.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

_SUPPORTED_FORMATS = ("webp", "jpeg")


@dataclass
class OptimizeConfig:
    """Effective configuration for the optimization step."""

    formats: list[str] = field(default_factory=lambda: ["webp", "jpeg"])
    webp_quality: int = 80
    jpeg_quality: int = 85


@dataclass
class ImageOptResult:
    """Optimization result for a single source file / target format pair."""

    filename: str
    source_format: str
    target_format: str
    source_bytes: int
    optimized_bytes: int
    bytes_saved: int
    saving_ratio: float  # 0.0 – 1.0
    status: str          # "ok" | "error"
    error: str | None


@dataclass
class OptimizeSummary:
    """Aggregated result for the full optimization run."""

    eligible: int
    optimized: int
    failed: int
    source_bytes_total: int
    optimized_bytes_total: int
    bytes_saved_total: int
    saving_ratio_total: float
    config: dict
    results: list[ImageOptResult]


def default_optimize_config() -> OptimizeConfig:
    return OptimizeConfig()


def run_optimization(
    run_dir: str | Path,
    config: OptimizeConfig | None = None,
) -> OptimizeSummary:
    """Generate compressed derivatives from a completed run's canonical outputs.

    Args:
        run_dir: Path to an existing ``runs/<run_id>/`` directory.
        config:  Optimization settings; defaults to :func:`default_optimize_config`.

    Returns:
        :class:`OptimizeSummary` with per-item results and aggregate metrics.

    Raises:
        FileNotFoundError: If *run_dir* or ``manifest.json`` does not exist.
        ValueError: If ``outputs/`` contains no eligible PNG files.
    """
    if config is None:
        config = default_optimize_config()

    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    outputs_dir = run_dir / "outputs"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")
    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs/ not found in {run_dir}")

    pngs = sorted(outputs_dir.glob("*.png"), key=lambda p: p.name.lower())
    if not pngs:
        raise ValueError(f"No PNG files found in {outputs_dir}")

    optimized_dir = run_dir / "optimized"
    optimized_dir.mkdir(exist_ok=True)

    for fmt in config.formats:
        (optimized_dir / fmt).mkdir(exist_ok=True)

    formats_str = ", ".join(config.formats)
    log.info(
        "Optimization started — run=%s  files=%d  formats=%s",
        run_dir.name, len(pngs), formats_str,
    )

    results: list[ImageOptResult] = []

    for png_path in pngs:
        source_bytes = png_path.stat().st_size

        for fmt in config.formats:
            result = _optimize_one(png_path, source_bytes, fmt, optimized_dir, config)
            results.append(result)
            if result.status == "ok":
                log.info(
                    "  ok %s → %s  saved=%d bytes (%.1f%%)",
                    png_path.name,
                    fmt,
                    result.bytes_saved,
                    result.saving_ratio * 100,
                )
            else:
                log.warning("  failed %s → %s: %s", png_path.name, fmt, result.error)

    summary = _build_summary(results, config)

    _write_per_image_csv(optimized_dir / "per_image.csv", results)
    _write_summary_json(optimized_dir / "summary.json", summary)

    log.info(
        "Optimization done — optimized=%d  failed=%d  saved=%d bytes (%.1f%%)",
        summary.optimized,
        summary.failed,
        summary.bytes_saved_total,
        summary.saving_ratio_total * 100,
    )

    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _optimize_one(
    png_path: Path,
    source_bytes: int,
    fmt: str,
    optimized_dir: Path,
    config: OptimizeConfig,
) -> ImageOptResult:
    """Convert *png_path* to *fmt* and return a result record."""
    ext = "jpg" if fmt == "jpeg" else fmt
    out_path = optimized_dir / fmt / (png_path.stem + f".{ext}")

    try:
        img = Image.open(png_path).convert("RGB")
        save_kwargs: dict = {}
        if fmt == "webp":
            save_kwargs["quality"] = config.webp_quality
            save_kwargs["method"] = 6
        elif fmt == "jpeg":
            save_kwargs["quality"] = config.jpeg_quality
            save_kwargs["optimize"] = True

        img.save(out_path, format=fmt.upper(), **save_kwargs)

        optimized_bytes = out_path.stat().st_size
        bytes_saved = source_bytes - optimized_bytes
        saving_ratio = bytes_saved / source_bytes if source_bytes > 0 else 0.0

        return ImageOptResult(
            filename=png_path.name,
            source_format="png",
            target_format=fmt,
            source_bytes=source_bytes,
            optimized_bytes=optimized_bytes,
            bytes_saved=bytes_saved,
            saving_ratio=round(saving_ratio, 6),
            status="ok",
            error=None,
        )

    except Exception as exc:  # noqa: BLE001
        return ImageOptResult(
            filename=png_path.name,
            source_format="png",
            target_format=fmt,
            source_bytes=source_bytes,
            optimized_bytes=0,
            bytes_saved=0,
            saving_ratio=0.0,
            status="error",
            error=str(exc),
        )


def _build_summary(results: list[ImageOptResult], config: OptimizeConfig) -> OptimizeSummary:
    ok_results = [r for r in results if r.status == "ok"]
    failed_results = [r for r in results if r.status == "error"]

    source_bytes_total = sum(r.source_bytes for r in ok_results)
    optimized_bytes_total = sum(r.optimized_bytes for r in ok_results)
    bytes_saved_total = sum(r.bytes_saved for r in ok_results)
    saving_ratio_total = (
        bytes_saved_total / source_bytes_total if source_bytes_total > 0 else 0.0
    )

    # eligible = distinct source files (not format pairs)
    eligible = len({r.filename for r in results})

    return OptimizeSummary(
        eligible=eligible,
        optimized=len(ok_results),
        failed=len(failed_results),
        source_bytes_total=source_bytes_total,
        optimized_bytes_total=optimized_bytes_total,
        bytes_saved_total=bytes_saved_total,
        saving_ratio_total=round(saving_ratio_total, 6),
        config=asdict(config),
        results=results,
    )


def _write_per_image_csv(path: Path, results: list[ImageOptResult]) -> None:
    fieldnames = [
        "filename", "source_format", "target_format",
        "source_bytes", "optimized_bytes", "bytes_saved",
        "saving_ratio", "status", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def _write_summary_json(path: Path, summary: OptimizeSummary) -> None:
    data = {
        "eligible": summary.eligible,
        "optimized": summary.optimized,
        "failed": summary.failed,
        "source_bytes_total": summary.source_bytes_total,
        "optimized_bytes_total": summary.optimized_bytes_total,
        "bytes_saved_total": summary.bytes_saved_total,
        "saving_ratio_total": summary.saving_ratio_total,
        "config": summary.config,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
