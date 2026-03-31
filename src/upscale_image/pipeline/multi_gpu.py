"""Multi-GPU worker pool for batch inference.

Architecture (ADR 0013 extension):
- One process per GPU, each loading its own model instance.
- Tasks distributed via a shared multiprocessing.Queue (round-robin).
- Results collected from a result queue; order reconstructed to match input.
- mp.get_context("spawn") is mandatory — "fork" is unsafe with CUDA.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import TYPE_CHECKING, Callable

import cv2

from upscale_image.pipeline.batch import ItemResult, _save_output

if TYPE_CHECKING:
    from upscale_image.config import AppConfig
    from upscale_image.io.task import ImageTask
    from upscale_image.models.base import SuperResolutionModel


def _gpu_worker(
    gpu_id: int,
    model_factory: Callable[[], "SuperResolutionModel"],
    task_queue: "mp.Queue[ImageTask | None]",
    result_queue: "mp.Queue[tuple[str, ItemResult]]",
    config: "AppConfig",
) -> None:
    """Worker process: owns one GPU, processes tasks until it receives None.

    Sets CUDA_VISIBLE_DEVICES so the model is isolated to a single GPU.
    Each result is placed as a (input_path, ItemResult) tuple into
    result_queue for ordered reconstruction in the main process.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model: SuperResolutionModel = model_factory()
    model.load()

    try:
        while True:
            task = task_queue.get()
            if task is None:  # poison pill
                break

            t0 = time.monotonic()
            input_w: int | None = None
            input_h: int | None = None
            output_w: int | None = None
            output_h: int | None = None
            inference_time_ms: float | None = None

            try:
                image = cv2.imread(task.input_path)
                if image is None:
                    raise RuntimeError(
                        f"cv2.imread returned None for {task.input_path!r}."
                    )
                input_h, input_w = image.shape[:2]

                t_infer = time.monotonic()
                output = model.upscale(image, config)
                inference_time_ms = (time.monotonic() - t_infer) * 1000.0

                output_h, output_w = output.shape[:2]
                _save_output(output, task.output_path)

                elapsed = time.monotonic() - t0
                task.status = "done"
                result = ItemResult(
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
                result = ItemResult(
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

            result_queue.put((task.input_path, result))
    finally:
        model.unload()


def run_batch_multi_gpu(
    config: "AppConfig",
    tasks: list["ImageTask"],
    model_factory: Callable[[], "SuperResolutionModel"],
    gpu_ids: list[int] | None = None,
) -> list[ItemResult]:
    """Distribute *tasks* across multiple GPUs using a spawn-based worker pool.

    Args:
        config:        Resolved AppConfig passed to each worker.
        tasks:         Ordered list of ImageTask to process.
        model_factory: Callable that returns a fresh (unloaded) model instance.
                       Called once per worker process.
        gpu_ids:       GPU indices to use. None = auto-detect via
                       torch.cuda.device_count().

    Returns:
        List of ItemResult in the same order as *tasks*.
    """
    import torch

    if gpu_ids is None:
        n_gpus = torch.cuda.device_count()
        gpu_ids = list(range(n_gpus))

    if not gpu_ids:
        gpu_ids = [0]

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()

    # Enqueue all tasks
    for task in tasks:
        task_queue.put(task)

    # Enqueue one poison pill per worker
    for _ in gpu_ids:
        task_queue.put(None)

    # Start one process per GPU
    processes = []
    for gid in gpu_ids:
        p = ctx.Process(
            target=_gpu_worker,
            args=(gid, model_factory, task_queue, result_queue, config),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # Collect all results
    results_map: dict[str, ItemResult] = {}
    for _ in tasks:
        input_path, result = result_queue.get()
        results_map[input_path] = result

    # Wait for all workers to finish
    for p in processes:
        p.join()

    # Reconstruct original order
    return [results_map[task.input_path] for task in tasks]
