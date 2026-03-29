"""Run-scoped logging: Rich console output + plain-text file persistence.

Each RunLogger is independent (no global state). The file handler writes
to ``<run_dir>/logs.txt``; the console handler uses Rich for structured,
human-readable output.

Minimum events required per ADR 0006:
  - run start (run_id, model, device, precision, input_dir)
  - discovered tasks and skipped files
  - per-item start, completion, and failure
  - run summary (total, done, failed, elapsed)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from upscale_image.config import AppConfig
    from upscale_image.io.task import ImageTask, SkippedFile
    from upscale_image.pipeline.run import RunContext

_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RunLogger:
    """Dual-output logger bound to a single run.

    All messages go to both the Rich console and ``logs.txt`` inside the run
    directory. Use the event helpers (``log_run_start``, ``log_item_done``,
    etc.) to emit structured events; use ``info/warning/error`` for ad-hoc
    messages.
    """

    def __init__(self, ctx: RunContext) -> None:
        self._console = Console(stderr=False)
        self._logger = self._build_logger(ctx.logs_path, ctx.run_id)

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    @staticmethod
    def _build_logger(logs_path: Path, run_id: str) -> logging.Logger:
        logger = logging.getLogger(f"upscale_image.run.{run_id}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.handlers:
            logger.handlers.clear()

        # Console: Rich-formatted, INFO and above
        console_handler = RichHandler(
            level=logging.INFO,
            show_path=False,
            rich_tracebacks=False,
            markup=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(console_handler)

        # File: plain text, DEBUG and above
        file_handler = logging.FileHandler(str(logs_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(file_handler)

        return logger

    # ------------------------------------------------------------------
    # Raw log primitives
    # ------------------------------------------------------------------

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    # ------------------------------------------------------------------
    # Structured events
    # ------------------------------------------------------------------

    def log_run_start(
        self,
        ctx: RunContext,
        cfg: AppConfig,
        task_count: int,
        skipped_count: int,
    ) -> None:
        self.info(f"Run started: [bold]{ctx.run_id}[/bold]")
        self.info(f"  model    : {cfg.model.name}  scale={cfg.model.scale}x")
        self.info(f"  device   : {cfg.runtime.device}  precision={cfg.runtime.precision}")
        self.info(f"  input    : {cfg.input_dir}")
        self.info(f"  output   : {cfg.output_dir}")
        self.info(f"  tasks    : {task_count} valid  |  {skipped_count} skipped")

    def log_skipped_files(self, skipped: list[SkippedFile]) -> None:
        for sf in skipped:
            self.warning(f"Skipped {sf.path!r}: {sf.reason}")

    def log_item_start(self, task: ImageTask) -> None:
        self.debug(f"Processing: {task.filename}")

    def log_item_done(self, task: ImageTask, elapsed: float) -> None:
        self.info(f"  [green]done[/green] {task.filename}  ({elapsed:.2f}s)")

    def log_item_error(self, task: ImageTask, exc: Exception) -> None:
        self.error(f"  [red]failed[/red] {task.filename}: {exc}")

    def log_run_summary(
        self,
        total: int,
        done: int,
        failed: int,
        elapsed: float,
    ) -> None:
        self.info(
            f"Run finished — total={total}  done=[green]{done}[/green]"
            f"  failed=[red]{failed}[/red]  elapsed={elapsed:.2f}s"
        )

    def close(self) -> None:
        """Flush and close all handlers. Call after the run completes."""
        for handler in self._logger.handlers[:]:
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)


def setup_run_logger(ctx: RunContext) -> RunLogger:
    """Factory: create and return a :class:`RunLogger` bound to *ctx*."""
    return RunLogger(ctx)
