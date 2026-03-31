"""Batch inference pipeline: loop over ImageTasks, call runner, persist outputs.

Rules (ADR 0005 / ADR 0006):
- Tasks are processed in the stable order produced by discover_images().
- Per-item exceptions are caught and logged; the run continues.
- Each item produces a structured ItemResult regardless of success or failure.
- Benchmarking, metrics and manifest generation are NOT mixed into this loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from upscale_image.config import AppConfig
from upscale_image.io import DiscoveryResult, SkippedFile, discover_images
from upscale_image.io.task import ImageTask
from upscale_image.models.base import SuperResolutionModel
from upscale_image.pipeline.logger import RunLogger
from upscale_image.pipeline.run import RunContext


# ---------------------------------------------------------------------------
# Per-item result
# ---------------------------------------------------------------------------

@dataclass
class ItemResult:
    """Structured outcome of processing a single image task.

    All fields are present for both success and failure so downstream
    consumers (manifest, benchmark) can treat the list uniformly.
    Dimension and timing fields are None when the step that produces them
    did not complete successfully.
    """

    task: ImageTask
    status: str                     # "done" | "failed"
    elapsed: float                  # total wall-clock seconds for this item
    inference_time_ms: float | None = None  # model.upscale() only, in ms
    input_width: int | None = None
    input_height: int | None = None
    output_width: int | None = None
    output_height: int | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Run-level aggregation
# ---------------------------------------------------------------------------

@dataclass
class RunStats:
    """Aggregated statistics computed from all ItemResults of a run."""

    total: int
    done: int
    failed: int
    total_elapsed_s: float
    success_rate: float             # done / total, or 1.0 for empty runs
    avg_inference_ms: float | None  # None when no item completed inference
    min_inference_ms: float | None
    max_inference_ms: float | None


@dataclass
class BatchResult:
    """Complete result of a batch run."""

    results: list[ItemResult] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)
    total_elapsed_s: float = 0.0

    # ------------------------------------------------------------------
    # Convenience counts
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def done(self) -> int:
        return sum(1 for r in self.results if r.status == "done")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def stats(self) -> RunStats:
        """Compute and return aggregated run statistics."""
        times = [
            r.inference_time_ms
            for r in self.results
            if r.status == "done" and r.inference_time_ms is not None
        ]
        return RunStats(
            total=self.total,
            done=self.done,
            failed=self.failed,
            total_elapsed_s=self.total_elapsed_s,
            success_rate=self.done / self.total if self.total > 0 else 1.0,
            avg_inference_ms=sum(times) / len(times) if times else None,
            min_inference_ms=min(times) if times else None,
            max_inference_ms=max(times) if times else None,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_batch(
    config: AppConfig,
    ctx: RunContext,
    model: SuperResolutionModel,
    logger: RunLogger,
    async_io: bool = False,
    prefetch_size: int = 4,
    write_workers: int = 2,
    batch_size: int = 1,
) -> BatchResult:
    """Run the full batch inference loop.

    Discovers input images, processes each one with *model*, and writes
    outputs to ``ctx.outputs_dir``.  Per-item failures are isolated: the
    loop continues and the failed task is recorded with ``status="failed"``.

    When *async_io* is True, I/O and inference are overlapped using a
    producer-consumer pipeline (ADR 0013). The serial path (async_io=False)
    is unchanged.

    Returns:
        BatchResult with per-item outcomes, skipped-file records and
        aggregated statistics.
    """
    discovery: DiscoveryResult = discover_images(
        config.input_dir, str(ctx.outputs_dir)
    )

    logger.log_run_start(
        ctx, config,
        task_count=len(discovery.tasks),
        skipped_count=len(discovery.skipped),
    )
    logger.log_skipped_files(discovery.skipped)

    run_start = time.monotonic()

    if config.runtime.multi_gpu:
        try:
            import torch
            n_gpus = torch.cuda.device_count()
        except ImportError:
            n_gpus = 0

        if n_gpus > 1:
            from upscale_image.pipeline.multi_gpu import run_batch_multi_gpu
            gpu_ids = config.runtime.gpu_ids or None
            results = run_batch_multi_gpu(
                config, discovery.tasks, lambda: model, gpu_ids=gpu_ids
            )
        else:
            logger.warning(
                "multi_gpu requested but fewer than 2 CUDA GPUs detected "
                f"({n_gpus} available) — falling back to single-GPU mode."
            )
            results = []
            for task in discovery.tasks:
                logger.log_item_start(task)
                item = _process_task(task, model, config, logger)
                results.append(item)
    elif async_io:
        from upscale_image.pipeline.async_worker import run_batch_async
        results = run_batch_async(
            config, discovery.tasks, model, logger,
            prefetch_size=prefetch_size,
            write_workers=write_workers,
            batch_size=batch_size,
        )
    else:
        results = []
        for task in discovery.tasks:
            logger.log_item_start(task)
            item = _process_task(task, model, config, logger)
            results.append(item)

    total_elapsed = time.monotonic() - run_start
    batch = BatchResult(
        results=results,
        skipped=discovery.skipped,
        total_elapsed_s=total_elapsed,
    )

    logger.log_run_summary(
        total=batch.total,
        done=batch.done,
        failed=batch.failed,
        elapsed=total_elapsed,
    )
    return batch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_task(
    task: ImageTask,
    model: SuperResolutionModel,
    config: AppConfig,
    logger: RunLogger,
) -> ItemResult:
    """Execute one image task, measuring time and dimensions at each stage.

    Never raises — all exceptions are caught and encoded in ItemResult.
    """
    t0 = time.monotonic()

    input_w: int | None = None
    input_h: int | None = None
    output_w: int | None = None
    output_h: int | None = None
    inference_time_ms: float | None = None

    try:
        # --- Read ---
        image = cv2.imread(task.input_path)
        if image is None:
            raise RuntimeError(
                f"cv2.imread returned None for {task.input_path!r}. "
                "The file may be corrupted or in an unsupported format."
            )
        input_h, input_w = image.shape[:2]

        # --- Inference (timed separately) ---
        t_infer = time.monotonic()
        output = model.upscale(image, config)
        inference_time_ms = (time.monotonic() - t_infer) * 1000.0

        output_h, output_w = output.shape[:2]

        # --- Write ---
        _save_output(output, task.output_path)

        elapsed = time.monotonic() - t0
        task.status = "done"
        logger.log_item_done(task, elapsed)

        return ItemResult(
            task=task,
            status="done",
            elapsed=elapsed,
            inference_time_ms=inference_time_ms,
            input_width=input_w,
            input_height=input_h,
            output_width=output_w,
            output_height=output_h,
        )

    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        task.status = "failed"
        logger.log_item_error(task, exc)

        return ItemResult(
            task=task,
            status="failed",
            elapsed=elapsed,
            inference_time_ms=inference_time_ms,
            input_width=input_w,
            input_height=input_h,
            output_width=output_w,
            output_height=output_h,
            error=str(exc),
        )


def group_tasks_by_size(
    tasks: list[ImageTask],
    batch_size: int,
    size_tolerance: float = 0.2,
) -> list[list[ImageTask]]:
    """Group tasks into batches of similar image dimensions to minimise padding.

    Strategy:
    1. Read each image's dimensions from its header (no pixel decode).
    2. Sort tasks by area (largest first) so the biggest images anchor each group.
    3. Accumulate tasks into a group while the area stays within *size_tolerance*
       (±20% by default) of the group's anchor; start a new group when the
       count reaches *batch_size* or the size diverges too far.

    Args:
        tasks:          Ordered list of ImageTask.
        batch_size:     Maximum tasks per group (≥ 1).
        size_tolerance: Fractional area tolerance within a group (0.0–1.0).

    Returns:
        List of groups (each group is a list of ImageTask). Groups preserve
        internal similarity but the overall order may differ from *tasks*.
    """
    if batch_size <= 1:
        return [[t] for t in tasks]

    from PIL import Image as _PILImage

    # Read (w, h) for each task without decoding pixels
    sizes: list[tuple[ImageTask, int]] = []
    for task in tasks:
        try:
            with _PILImage.open(task.input_path) as img:
                w, h = img.size
            area = w * h
        except Exception:
            area = 0  # unreadable images land at the end; sorted separately
        sizes.append((task, area))

    sizes.sort(key=lambda x: x[1], reverse=True)

    groups: list[list[ImageTask]] = []
    current_group: list[ImageTask] = []
    anchor_area: int = 0

    for task, area in sizes:
        if not current_group:
            current_group.append(task)
            anchor_area = area
        elif (
            len(current_group) < batch_size
            and anchor_area > 0
            and abs(area - anchor_area) / anchor_area <= size_tolerance
        ):
            current_group.append(task)
        else:
            groups.append(current_group)
            current_group = [task]
            anchor_area = area

    if current_group:
        groups.append(current_group)

    return groups


def estimate_safe_batch_size(
    sample_image: np.ndarray,
    model: SuperResolutionModel,
    config: AppConfig,
    safety_factor: float = 0.7,
) -> int:
    """Estimate the maximum safe batch_size based on available VRAM.

    Returns 1 when CUDA is not available (CPU doesn't benefit from batching).

    Args:
        sample_image:  Representative image (shape used for estimation).
        model:         Loaded SuperResolutionModel (provides scale).
        config:        Resolved AppConfig (provides model scale).
        safety_factor: Fraction of available VRAM to use (default 0.7).

    Returns:
        Estimated safe batch size (always ≥ 1).
    """
    try:
        import torch
    except ImportError:
        return 1

    if not torch.cuda.is_available():
        return 1

    total_vram = torch.cuda.get_device_properties(0).total_memory
    free_vram = total_vram - torch.cuda.memory_allocated(0)
    usable = int(free_vram * safety_factor)

    h, w, c = sample_image.shape
    scale = config.model.scale
    # RRDBNet x4 uses roughly 8× the input tensor size in VRAM during forward
    bytes_per_image = h * w * c * 4 * 8 * (scale ** 2)
    if bytes_per_image == 0:
        return 1

    return max(1, usable // bytes_per_image)


def _save_output(image: "np.ndarray", output_path: str) -> None:
    """Write *image* to *output_path*, creating parent directories if needed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(output_path, image)
    if not success:
        raise RuntimeError(f"cv2.imwrite failed for {output_path!r}.")
