"""Run comparison: read artefacts from multiple runs and produce consolidated output.

Reads from each run directory:
  manifest.json              — model params, runtime, timing, status
  metrics/summary.json       — FR benchmark averages (optional)
  metrics/niqe_summary.json  — NR benchmark averages (optional)

Never recalculates metrics — reads what was persisted.

ADR 0009: comparisons must consider params, runtime, timing and quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunSnapshot:
    """All comparable data extracted from a single run's artefacts."""

    run_id: str
    run_dir: str

    # From manifest.json
    model_name: str
    model_scale: int
    device: str
    precision: str

    # timing
    total_elapsed_s: float
    avg_inference_ms: float | None

    # status
    total_images: int
    done: int
    failed: int
    skipped: int
    success_rate: float

    # optional metrics (None when benchmark was not run)
    avg_psnr: float | None = None
    avg_ssim: float | None = None
    avg_lpips: float | None = None
    avg_niqe: float | None = None


@dataclass
class RunDelta:
    """Numeric differences between two snapshots (b − a)."""

    run_id_a: str
    run_id_b: str
    delta_total_elapsed_s: float | None
    delta_avg_inference_ms: float | None
    delta_success_rate: float | None
    delta_avg_psnr: float | None
    delta_avg_ssim: float | None
    delta_avg_lpips: float | None
    delta_avg_niqe: float | None


@dataclass
class ComparisonResult:
    """Comparison of two or more runs."""

    snapshots: list[RunSnapshot] = field(default_factory=list)
    deltas: list[RunDelta] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run_snapshot(run_dir: str | Path) -> RunSnapshot:
    """Read a run's artefacts and return a :class:`RunSnapshot`.

    Args:
        run_dir: Path to the run directory (``runs/<run_id>/``).

    Raises:
        FileNotFoundError: If ``manifest.json`` is missing.
        ValueError:        If manifest is malformed.
    """
    run_path = Path(run_dir)
    manifest_path = run_path / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {run_dir!r}. "
            "Has this run completed successfully?"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid manifest.json in {run_dir!r}: {exc}") from exc

    # Optional metric files
    fr_summary = _load_json_optional(run_path / "metrics" / "summary.json")
    nr_summary = _load_json_optional(run_path / "metrics" / "niqe_summary.json")

    return RunSnapshot(
        run_id=manifest["run_id"],
        run_dir=str(run_path),
        model_name=manifest["model"]["name"],
        model_scale=manifest["model"]["scale"],
        device=manifest["model"]["device"],
        precision=manifest["model"]["precision"],
        total_elapsed_s=manifest["timing"]["total_elapsed_s"],
        avg_inference_ms=manifest["timing"].get("avg_inference_ms"),
        total_images=manifest["status"]["total"],
        done=manifest["status"]["done"],
        failed=manifest["status"]["failed"],
        skipped=manifest["status"]["skipped"],
        success_rate=manifest["status"]["success_rate"],
        avg_psnr=fr_summary.get("avg_psnr") if fr_summary else None,
        avg_ssim=fr_summary.get("avg_ssim") if fr_summary else None,
        avg_lpips=fr_summary.get("avg_lpips") if fr_summary else None,
        avg_niqe=nr_summary.get("avg_niqe") if nr_summary else None,
    )


def _load_json_optional(path: Path) -> dict | None:
    """Return parsed JSON or None if file is absent."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_runs(
    run_dirs: list[str | Path],
) -> ComparisonResult:
    """Load and compare runs from *run_dirs*.

    Deltas are computed sequentially: each run is compared against the
    preceding one (run[i] − run[i-1]), giving a chain of improvements
    or regressions.

    Args:
        run_dirs: List of run directory paths (at least one).

    Returns:
        :class:`ComparisonResult` with snapshots and sequential deltas.

    Raises:
        ValueError: If fewer than one run directory is provided.
    """
    if not run_dirs:
        raise ValueError("At least one run directory is required.")

    snapshots = [load_run_snapshot(d) for d in run_dirs]
    deltas = [
        _compute_delta(snapshots[i - 1], snapshots[i])
        for i in range(1, len(snapshots))
    ]
    return ComparisonResult(snapshots=snapshots, deltas=deltas)


def _delta(a: float | None, b: float | None) -> float | None:
    """Return b − a, or None if either value is missing."""
    if a is None or b is None:
        return None
    return round(b - a, 4)


def _compute_delta(a: RunSnapshot, b: RunSnapshot) -> RunDelta:
    return RunDelta(
        run_id_a=a.run_id,
        run_id_b=b.run_id,
        delta_total_elapsed_s=_delta(a.total_elapsed_s, b.total_elapsed_s),
        delta_avg_inference_ms=_delta(a.avg_inference_ms, b.avg_inference_ms),
        delta_success_rate=_delta(a.success_rate, b.success_rate),
        delta_avg_psnr=_delta(a.avg_psnr, b.avg_psnr),
        delta_avg_ssim=_delta(a.avg_ssim, b.avg_ssim),
        delta_avg_lpips=_delta(a.avg_lpips, b.avg_lpips),
        delta_avg_niqe=_delta(a.avg_niqe, b.avg_niqe),
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def comparison_to_dict(result: ComparisonResult) -> dict[str, Any]:
    """Serialise a :class:`ComparisonResult` to a plain dict (JSON-safe)."""
    return {
        "runs": [
            {
                "run_id": s.run_id,
                "run_dir": s.run_dir,
                "model": {
                    "name": s.model_name,
                    "scale": s.model_scale,
                    "device": s.device,
                    "precision": s.precision,
                },
                "timing": {
                    "total_elapsed_s": s.total_elapsed_s,
                    "avg_inference_ms": s.avg_inference_ms,
                },
                "status": {
                    "total": s.total_images,
                    "done": s.done,
                    "failed": s.failed,
                    "skipped": s.skipped,
                    "success_rate": s.success_rate,
                },
                "metrics": {
                    "avg_psnr": s.avg_psnr,
                    "avg_ssim": s.avg_ssim,
                    "avg_lpips": s.avg_lpips,
                    "avg_niqe": s.avg_niqe,
                },
            }
            for s in result.snapshots
        ],
        "deltas": [
            {
                "from": d.run_id_a,
                "to": d.run_id_b,
                "delta_total_elapsed_s": d.delta_total_elapsed_s,
                "delta_avg_inference_ms": d.delta_avg_inference_ms,
                "delta_success_rate": d.delta_success_rate,
                "delta_avg_psnr": d.delta_avg_psnr,
                "delta_avg_ssim": d.delta_avg_ssim,
                "delta_avg_lpips": d.delta_avg_lpips,
                "delta_avg_niqe": d.delta_avg_niqe,
            }
            for d in result.deltas
        ],
    }
