"""No-reference image quality benchmark: NIQE.

Evaluates output images without a ground truth reference.
Used when full-reference mode is not available (no HR dataset).

Outputs written to ``metrics_dir``:
  niqe_per_image.csv   — one row per output image
  niqe_summary.json    — aggregated NIQE statistics

ADR 0008: FR and NR modes must be clearly distinguished.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".png"})

_CSV_FIELDNAMES = ["filename", "niqe", "error"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NiqeResult:
    """NIQE score for a single output image."""

    filename: str
    niqe: float | None = None
    error: str | None = None    # None → success


@dataclass
class NrBenchmarkSummary:
    """Aggregated statistics for a no-reference benchmark run."""

    mode: str = "no_reference"
    total: int = 0
    computed: int = 0
    skipped: int = 0
    avg_niqe: float | None = None


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------

def _load_tensor(path: str) -> torch.Tensor | None:
    """Load image as float32 NCHW tensor in [0, 1]. Returns None on failure."""
    img = cv2.imread(path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0)  # HWC → NCHW


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_no_reference_benchmark(
    outputs_dir: str | Path,
    metrics_dir: str | Path,
    device: str = "cpu",
    *,
    _niqe_metric=None,  # injectable for tests
) -> NrBenchmarkSummary:
    """Compute NIQE for every .png file in *outputs_dir*.

    Args:
        outputs_dir: Directory with .png outputs from the pipeline.
        metrics_dir: Directory where ``niqe_per_image.csv`` and
                     ``niqe_summary.json`` will be written.
        device:      Compute device for NIQE (``"cpu"`` or ``"cuda"``).
        _niqe_metric: Pre-created pyiqa NIQE metric (injected in tests).

    Returns:
        Aggregated :class:`NrBenchmarkSummary`.
    """
    import pyiqa  # lazy import — heavy dependency

    out_path = Path(outputs_dir)
    met_path = Path(metrics_dir)
    met_path.mkdir(parents=True, exist_ok=True)

    if _niqe_metric is None:
        niqe_metric = pyiqa.create_metric("niqe", device=device)
    else:
        niqe_metric = _niqe_metric

    output_files = sorted(out_path.glob("*.png"), key=lambda p: p.name.lower())
    results: list[NiqeResult] = []

    for img_path in output_files:
        result = _compute_niqe(str(img_path), img_path.name, niqe_metric)
        results.append(result)

    _save_niqe_csv(results, met_path / "niqe_per_image.csv")
    summary = _build_nr_summary(results)
    _save_nr_summary_json(summary, met_path / "niqe_summary.json")

    return summary


def _compute_niqe(path: str, filename: str, niqe_metric) -> NiqeResult:
    """Compute NIQE for one image. Never raises."""
    tensor = _load_tensor(path)
    if tensor is None:
        return NiqeResult(filename=filename, error="load_error")

    try:
        with torch.no_grad():
            score = niqe_metric(tensor)
        return NiqeResult(filename=filename, niqe=float(score.item()))
    except Exception as exc:  # noqa: BLE001
        return NiqeResult(filename=filename, error=f"compute_error: {exc}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_niqe_csv(results: list[NiqeResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filename": r.filename,
                "niqe": "" if r.niqe is None else round(r.niqe, 4),
                "error": r.error or "",
            })


def _build_nr_summary(results: list[NiqeResult]) -> NrBenchmarkSummary:
    computed = [r for r in results if r.error is None and r.niqe is not None]
    skipped = [r for r in results if r.error is not None]
    niqe_vals = [r.niqe for r in computed]

    avg_niqe: float | None = None
    if niqe_vals:
        avg_niqe = round(sum(niqe_vals) / len(niqe_vals), 4)

    return NrBenchmarkSummary(
        mode="no_reference",
        total=len(results),
        computed=len(computed),
        skipped=len(skipped),
        avg_niqe=avg_niqe,
    )


def _save_nr_summary_json(summary: NrBenchmarkSummary, path: Path) -> None:
    data = {
        "mode": summary.mode,
        "total": summary.total,
        "computed": summary.computed,
        "skipped": summary.skipped,
        "avg_niqe": summary.avg_niqe,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
