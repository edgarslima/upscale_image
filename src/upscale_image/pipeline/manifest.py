"""Manifest generation: write manifest.json for each run.

The manifest is the primary audit artefact. It must allow a reader to
understand what happened in a run without consulting the terminal.

Schema (all top-level keys are stable):
  run_id    — unique identifier of this execution
  model     — logical name, scale, device, precision
  runtime   — code version, Python version
  timing    — wall-clock and inference statistics
  status    — counts (total / done / failed / skipped) and success rate
  artifacts — relative paths to co-located artefacts
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

from upscale_image import __version__
from upscale_image.config import AppConfig
from upscale_image.pipeline.batch import BatchResult
from upscale_image.pipeline.run import RunContext


def write_manifest(
    ctx: RunContext,
    config: AppConfig,
    batch: BatchResult,
) -> dict:
    """Build and persist ``manifest.json`` inside *ctx.run_dir*.

    Args:
        ctx:    Run context (paths and run_id).
        config: Effective configuration used in this run.
        batch:  Results produced by :func:`run_batch`.

    Returns:
        The manifest dict that was written (useful for tests).
    """
    stats = batch.stats()

    manifest = {
        "run_id": ctx.run_id,
        "model": {
            "name": config.model.name,
            "scale": config.model.scale,
            "device": config.runtime.device,
            "precision": config.runtime.precision,
        },
        "runtime": {
            "code_version": __version__,
            "python_version": platform.python_version(),
        },
        "timing": {
            "total_elapsed_s": round(stats.total_elapsed_s, 3),
            "avg_inference_ms": (
                round(stats.avg_inference_ms, 3)
                if stats.avg_inference_ms is not None
                else None
            ),
            "min_inference_ms": (
                round(stats.min_inference_ms, 3)
                if stats.min_inference_ms is not None
                else None
            ),
            "max_inference_ms": (
                round(stats.max_inference_ms, 3)
                if stats.max_inference_ms is not None
                else None
            ),
        },
        "status": {
            "total": stats.total,
            "done": stats.done,
            "failed": stats.failed,
            "skipped": len(batch.skipped),
            "success_rate": round(stats.success_rate, 4),
        },
        "artifacts": {
            "effective_config": ctx.effective_config_path.name,
            "outputs_dir": ctx.outputs_dir.name,
            "metrics_dir": ctx.metrics_dir.name,
            "log": ctx.logs_path.name,
        },
    }

    ctx.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return manifest


def patch_manifest_with_optimization(run_dir: "Path", summary: object) -> None:
    """Append optimization metadata to an existing ``manifest.json``.

    Only the ``optimization`` key is added/overwritten; all historical
    inference fields remain unchanged (ADR 0010).

    Args:
        run_dir: Path to the run directory containing ``manifest.json``.
        summary: Result returned by :func:`upscale_image.optimize.run_optimization`.
    """
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["optimization"] = {
        "eligible": summary.eligible,
        "optimized": summary.optimized,
        "failed": summary.failed,
        "source_bytes_total": summary.source_bytes_total,
        "optimized_bytes_total": summary.optimized_bytes_total,
        "bytes_saved_total": summary.bytes_saved_total,
        "saving_ratio_total": summary.saving_ratio_total,
        "artifacts": {
            "optimized_dir": "optimized",
            "per_image_csv": "optimized/per_image.csv",
            "summary_json": "optimized/summary.json",
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def patch_manifest_with_pdf_source(run_dir: "Path", result: object) -> None:
    """Append PDF-origin metadata to an existing ``manifest.json``.

    Only the ``pdf_source`` key is added; all inference fields remain intact.

    Args:
        run_dir: Path to the run directory containing ``manifest.json``.
        result:  :class:`upscale_image.pdf.PdfExtractionResult` instance.
    """
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["pdf_source"] = {
        "input_mode": "pdf",
        "source_file": result.source_pdf.name,
        "total_pages": result.total_pages,
        "extracted": result.extracted,
        "failed": result.failed,
        "artifacts": {
            "source_copy": str(result.source_copy.relative_to(run_dir)),
            "extracted_pages_dir": str(result.extracted_dir.relative_to(run_dir)),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def patch_manifest_with_compose_ready(run_dir: "Path", result: object) -> None:
    """Append pre-composition preparation metadata to an existing ``manifest.json``.

    Adds or updates the ``pdf_compose_ready`` key without touching any other field.

    Args:
        run_dir: Path to the run directory containing ``manifest.json``.
        result:  :class:`upscale_image.pdf.ComposeReadyResult` instance.
    """
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["pdf_compose_ready"] = {
        "source_pdf_bytes": result.source_pdf_bytes,
        "budget_bytes": result.budget_bytes,
        "estimated_bytes": result.estimated_bytes,
        "ratio": result.ratio,
        "within_budget": result.within_budget,
        "preset_quality": result.preset_quality,
        "pages_count": result.pages_count,
        "artifacts": {
            "compose_ready_pages_dir": str(result.pages_dir.relative_to(run_dir)),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def patch_manifest_with_pdf_rebuilt(run_dir: "Path", result: object) -> None:
    """Append PDF recomposition metadata to an existing ``manifest.json``.

    Adds or updates the ``pdf_rebuilt`` key without touching any other field.

    Args:
        run_dir: Path to the run directory containing ``manifest.json``.
        result:  :class:`upscale_image.pdf.PdfComposeResult` instance.
    """
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["pdf_rebuilt"] = {
        "status": result.status,
        "pages_included": result.pages_included,
        "output_pdf": str(result.output_pdf.relative_to(run_dir))
        if result.status == "ok"
        else None,
        "error": result.error,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
