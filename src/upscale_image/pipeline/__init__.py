"""Batch inference pipeline, run management and manifest generation."""

from upscale_image.pipeline.batch import BatchResult, ItemResult, RunStats, run_batch
from upscale_image.pipeline.logger import RunLogger, setup_run_logger
from upscale_image.pipeline.manifest import write_manifest
from upscale_image.pipeline.run import RunContext, create_run, generate_run_id

__all__ = [
    "BatchResult",
    "ItemResult",
    "RunStats",
    "RunContext",
    "RunLogger",
    "create_run",
    "generate_run_id",
    "run_batch",
    "setup_run_logger",
    "write_manifest",
]
