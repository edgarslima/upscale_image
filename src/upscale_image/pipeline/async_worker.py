"""Asynchronous producer-consumer pipeline for batch inference.

Architecture (ADR 0013):
  Thread-Read  →  read_queue  →  Thread-Infer (main)  →  write_queue  →  Pool-Write

- Thread de Leitura: percorre tasks em ordem, lê cada imagem e enfileira _ReadResult.
- Thread de Inferência (chamador): consome read_queue, chama model.upscale() ou
  model.upscale_batch() (quando batch_size > 1), enfileira _InferResult.
- Pool de Escrita (write_workers threads): consome write_queue, chama cv2.imwrite(),
  registra resultado.

Garantias:
- Resultados são retornados na mesma ordem de `tasks` (ADR 0005 — determinismo de ordem).
- Falhas em qualquer estágio geram ItemResult(status="failed") sem derrubar a run.
- Cada write worker recebe exatamente um sentinel _STOP.
- Nenhuma thread fica travada após a conclusão do batch.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from upscale_image.pipeline.batch import ItemResult

if TYPE_CHECKING:
    import numpy as np

    from upscale_image.config import AppConfig
    from upscale_image.io.task import ImageTask
    from upscale_image.models.base import SuperResolutionModel
    from upscale_image.pipeline.logger import RunLogger


# ---------------------------------------------------------------------------
# Sentinel — signals end-of-stream to write worker threads
# ---------------------------------------------------------------------------

_STOP = object()


# ---------------------------------------------------------------------------
# Internal pipeline dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _ReadResult:
    """Result of the read stage for one task."""

    task: "ImageTask"
    idx: int                        # index in original tasks list (preserves order)
    image: "np.ndarray | None"
    error: str | None
    t_read_start: float             # monotonic timestamp at read start


@dataclass
class _InferResult:
    """Result of the inference stage for one task."""

    task: "ImageTask"
    idx: int
    output: "np.ndarray | None"
    inference_time_ms: float | None
    input_w: int | None
    input_h: int | None
    error: str | None
    t_read_start: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_batch_async(
    config: "AppConfig",
    tasks: list["ImageTask"],
    model: "SuperResolutionModel",
    logger: "RunLogger",
    prefetch_size: int = 4,
    write_workers: int = 2,
    batch_size: int = 1,
) -> list[ItemResult]:
    """Run the batch with overlapping disk I/O and GPU inference.

    Returns a list of ItemResult **in the same order as** `tasks`.
    Individual failures in any stage are converted to ItemResult(status="failed")
    without aborting the run (ADR 0005).

    When *batch_size* > 1, the inference thread accumulates that many images
    and calls ``model.upscale_batch()`` for a single GPU forward pass per group
    (ADR 0012). A whole-batch failure marks every item in the group as failed.

    Args:
        config:        Resolved AppConfig (device, precision, tile_size, …).
        tasks:         Ordered list of ImageTask from discover_images().
        model:         Loaded SuperResolutionModel (already called .load()).
        logger:        RunLogger bound to the current run.
        prefetch_size: Maximum images buffered between read and infer stages.
        write_workers: Number of parallel write threads.
        batch_size:    Images per GPU forward pass (1 = per-image, ADR 0012).

    Returns:
        list[ItemResult] with len == len(tasks), in the same order.
    """
    if not tasks:
        return []

    effective_batch = max(1, batch_size)
    read_queue: queue.Queue[_ReadResult | object] = queue.Queue(maxsize=prefetch_size)
    write_queue: queue.Queue[_InferResult | object] = queue.Queue(maxsize=prefetch_size * 2)

    # Pre-allocate result slots indexed by position
    results: dict[int, ItemResult] = {}

    # ------------------------------------------------------------------
    # Thread 1 — Reader
    # ------------------------------------------------------------------

    def _reader() -> None:
        for idx, task in enumerate(tasks):
            t0 = time.monotonic()
            logger.log_item_start(task)
            try:
                image = cv2.imread(task.input_path)
                if image is None:
                    raise RuntimeError(
                        f"cv2.imread returned None for {task.input_path!r}. "
                        "The file may be corrupted or in an unsupported format."
                    )
                read_queue.put(_ReadResult(task=task, idx=idx, image=image,
                                           error=None, t_read_start=t0))
            except Exception as exc:  # noqa: BLE001
                read_queue.put(_ReadResult(task=task, idx=idx, image=None,
                                           error=str(exc), t_read_start=t0))
        read_queue.put(_STOP)

    reader_thread = threading.Thread(target=_reader, daemon=True, name="async-reader")
    reader_thread.start()

    # ------------------------------------------------------------------
    # Thread pool — Writers
    # ------------------------------------------------------------------

    def _writer() -> None:
        while True:
            item = write_queue.get()
            if item is _STOP:
                write_queue.task_done()
                break
            assert isinstance(item, _InferResult)
            infer_result: _InferResult = item
            task = infer_result.task
            idx = infer_result.idx
            elapsed = time.monotonic() - infer_result.t_read_start

            if infer_result.error is not None:
                # Failure from earlier stage — propagate as failed result
                task.status = "failed"
                logger.log_item_error(task, Exception(infer_result.error))
                results[idx] = ItemResult(
                    task=task,
                    status="failed",
                    elapsed=elapsed,
                    inference_time_ms=infer_result.inference_time_ms,
                    input_width=infer_result.input_w,
                    input_height=infer_result.input_h,
                    error=infer_result.error,
                )
            else:
                try:
                    assert infer_result.output is not None
                    out_h, out_w = infer_result.output.shape[:2]
                    _save_output(infer_result.output, task.output_path)
                    elapsed = time.monotonic() - infer_result.t_read_start
                    task.status = "done"
                    logger.log_item_done(task, elapsed)
                    results[idx] = ItemResult(
                        task=task,
                        status="done",
                        elapsed=elapsed,
                        inference_time_ms=infer_result.inference_time_ms,
                        input_width=infer_result.input_w,
                        input_height=infer_result.input_h,
                        output_width=out_w,
                        output_height=out_h,
                    )
                except Exception as exc:  # noqa: BLE001
                    elapsed = time.monotonic() - infer_result.t_read_start
                    task.status = "failed"
                    logger.log_item_error(task, exc)
                    results[idx] = ItemResult(
                        task=task,
                        status="failed",
                        elapsed=elapsed,
                        inference_time_ms=infer_result.inference_time_ms,
                        input_width=infer_result.input_w,
                        input_height=infer_result.input_h,
                        error=str(exc),
                    )

            write_queue.task_done()

    writer_threads = [
        threading.Thread(target=_writer, daemon=True, name=f"async-writer-{i}")
        for i in range(write_workers)
    ]
    for t in writer_threads:
        t.start()

    # ------------------------------------------------------------------
    # Main thread — Inference (single-threaded GPU, ADR 0013)
    # ------------------------------------------------------------------

    if effective_batch == 1:
        _run_inference_serial(read_queue, write_queue, model, config)
    else:
        _run_inference_batched(read_queue, write_queue, model, config, effective_batch)

    # Signal each write worker exactly once
    for _ in range(write_workers):
        write_queue.put(_STOP)

    # Wait for all writes to complete before returning
    for t in writer_threads:
        t.join()
    reader_thread.join()

    # Return results in original task order
    return [results[i] for i in range(len(tasks))]


# ---------------------------------------------------------------------------
# Inference helpers (called from main thread)
# ---------------------------------------------------------------------------

def _run_inference_serial(
    read_queue: "queue.Queue[_ReadResult | object]",
    write_queue: "queue.Queue[_InferResult | object]",
    model: "SuperResolutionModel",
    config: "AppConfig",
) -> None:
    """Single-image inference loop (batch_size == 1)."""
    while True:
        read_item = read_queue.get()
        if read_item is _STOP:
            read_queue.task_done()
            break

        assert isinstance(read_item, _ReadResult)
        rr: _ReadResult = read_item

        if rr.error is not None:
            write_queue.put(_InferResult(
                task=rr.task, idx=rr.idx, output=None,
                inference_time_ms=None,
                input_w=None, input_h=None,
                error=rr.error,
                t_read_start=rr.t_read_start,
            ))
            read_queue.task_done()
            continue

        image = rr.image
        assert image is not None
        input_h, input_w = image.shape[:2]

        try:
            t_infer = time.monotonic()
            output = model.upscale(image, config)
            inference_time_ms = (time.monotonic() - t_infer) * 1000.0
            write_queue.put(_InferResult(
                task=rr.task, idx=rr.idx, output=output,
                inference_time_ms=inference_time_ms,
                input_w=input_w, input_h=input_h,
                error=None,
                t_read_start=rr.t_read_start,
            ))
        except Exception as exc:  # noqa: BLE001
            write_queue.put(_InferResult(
                task=rr.task, idx=rr.idx, output=None,
                inference_time_ms=None,
                input_w=input_w, input_h=input_h,
                error=str(exc),
                t_read_start=rr.t_read_start,
            ))

        read_queue.task_done()


def _run_inference_batched(
    read_queue: "queue.Queue[_ReadResult | object]",
    write_queue: "queue.Queue[_InferResult | object]",
    model: "SuperResolutionModel",
    config: "AppConfig",
    batch_size: int,
) -> None:
    """Batch inference loop: accumulate `batch_size` reads, call upscale_batch()."""
    pending: list[_ReadResult] = []

    def _flush(batch: list[_ReadResult]) -> None:
        """Process one group: forward failed reads, call upscale_batch() on ok reads."""
        if not batch:
            return

        # Separate failed reads (read error) from successful reads
        ok: list[_ReadResult] = []
        for rr in batch:
            if rr.error is not None:
                write_queue.put(_InferResult(
                    task=rr.task, idx=rr.idx, output=None,
                    inference_time_ms=None,
                    input_w=None, input_h=None,
                    error=rr.error,
                    t_read_start=rr.t_read_start,
                ))
            else:
                ok.append(rr)

        if not ok:
            return

        images = [rr.image for rr in ok]
        sizes = [(rr.image.shape[1], rr.image.shape[0]) for rr in ok]  # (w, h)

        try:
            t_infer = time.monotonic()
            outputs = model.upscale_batch(images, config)
            per_item_ms = (time.monotonic() - t_infer) * 1000.0 / len(ok)
            for rr, out, (w, h) in zip(ok, outputs, sizes):
                write_queue.put(_InferResult(
                    task=rr.task, idx=rr.idx, output=out,
                    inference_time_ms=per_item_ms,
                    input_w=w, input_h=h,
                    error=None,
                    t_read_start=rr.t_read_start,
                ))
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            for rr, (w, h) in zip(ok, sizes):
                write_queue.put(_InferResult(
                    task=rr.task, idx=rr.idx, output=None,
                    inference_time_ms=None,
                    input_w=w, input_h=h,
                    error=err,
                    t_read_start=rr.t_read_start,
                ))

    while True:
        read_item = read_queue.get()
        if read_item is _STOP:
            read_queue.task_done()
            _flush(pending)
            pending = []
            break

        assert isinstance(read_item, _ReadResult)
        pending.append(read_item)
        read_queue.task_done()

        if len(pending) >= batch_size:
            _flush(pending)
            pending = []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_output(image: "np.ndarray", output_path: str) -> None:
    """Write *image* to *output_path*, creating parent directories if needed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(output_path, image)
    if not success:
        raise RuntimeError(f"cv2.imwrite failed for {output_path!r}.")
