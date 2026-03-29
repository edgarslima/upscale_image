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
