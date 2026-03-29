"""Run management: ID generation, directory tree creation, config persistence.

Each execution is encapsulated in:
    runs/<run_id>/
        outputs/
        metrics/
        logs.txt           (placeholder path — populated by logging step)
        manifest.json      (placeholder path — populated by pipeline)
        effective_config.yaml

run_id format: run_<YYYYMMDD_HHMMSS>_<model>_<scale>x
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from upscale_image.config import AppConfig, save_effective_config

_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9]")


def _sanitize(name: str) -> str:
    """Replace non-alphanumeric characters with hyphens for safe directory names."""
    return _UNSAFE_CHARS.sub("-", name).strip("-")


def generate_run_id(model_name: str, scale: int, *, now: datetime | None = None) -> str:
    """Return a unique, human-readable run identifier.

    Format: ``run_<YYYYMMDD_HHMMSS>_<model>_<scale>x``

    Args:
        model_name: Logical model name (e.g. ``realesrgan-x4``).
        scale:      Upscale factor.
        now:        Timestamp override (useful in tests to produce deterministic IDs).
    """
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_model = _sanitize(model_name)
    return f"run_{ts}_{safe_model}_{scale}x"


@dataclass
class RunContext:
    """All paths and identifiers for a single execution.

    Created once by :func:`create_run` and passed through the pipeline;
    no module should recompute these paths independently.
    """

    run_id: str
    run_dir: Path
    outputs_dir: Path
    metrics_dir: Path
    logs_path: Path
    manifest_path: Path
    effective_config_path: Path


def create_run(
    config: AppConfig,
    *,
    base_dir: str | Path = "runs",
    now: datetime | None = None,
) -> RunContext:
    """Create the run directory tree and persist the effective configuration.

    Args:
        config:   Resolved :class:`AppConfig` for this execution.
        base_dir: Root directory for all runs (default ``runs/``).
        now:      Timestamp override for deterministic run IDs in tests.

    Returns:
        A :class:`RunContext` with all paths pre-built and ready for use.

    Raises:
        FileExistsError: If a run with the same ID already exists (clock collision).
    """
    run_id = generate_run_id(config.model.name, config.model.scale, now=now)
    run_dir = Path(base_dir) / run_id

    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            "Wait one second and retry, or check for duplicate invocations."
        )

    outputs_dir = run_dir / "outputs"
    metrics_dir = run_dir / "metrics"

    run_dir.mkdir(parents=True)
    outputs_dir.mkdir()
    metrics_dir.mkdir()

    ctx = RunContext(
        run_id=run_id,
        run_dir=run_dir,
        outputs_dir=outputs_dir,
        metrics_dir=metrics_dir,
        logs_path=run_dir / "logs.txt",
        manifest_path=run_dir / "manifest.json",
        effective_config_path=run_dir / "effective_config.yaml",
    )

    save_effective_config(config, str(ctx.effective_config_path))

    return ctx
