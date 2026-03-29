"""Full-reference image quality benchmark: PSNR, SSIM, LPIPS.

Pairing rule: output and reference are matched by file stem (case-insensitive).
Outputs are always .png (produced by the pipeline). References can use any
supported image extension.

Outputs written to ``metrics_dir``:
  per_image.csv   — one row per pair (or unpaired output)
  summary.json    — aggregated statistics

ADR 0008: FR and NR modes must be clearly distinguished; results must be
persisted per-image AND as an aggregate summary.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as _compute_psnr
from skimage.metrics import structural_similarity as _compute_ssim

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
)

_CSV_FIELDNAMES = [
    "filename",
    "psnr",
    "ssim",
    "lpips",
    "error",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImagePair:
    """A matched (output, reference) pair ready for metric computation."""

    filename: str        # stem used for matching
    output_path: str
    reference_path: str


@dataclass
class UnpairedOutput:
    """An output file that could not be paired with a reference."""

    filename: str
    output_path: str
    reason: str          # "no_reference"


@dataclass
class PairResult:
    """Metric results for a single output/reference pair.

    Fields are ``None`` when the corresponding step did not complete
    (e.g. size mismatch prevents PSNR/SSIM computation).
    """

    filename: str
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    error: str | None = None    # None → success


@dataclass
class BenchmarkSummary:
    """Aggregated statistics for a full-reference benchmark run."""

    mode: str = "full_reference"
    total_pairs: int = 0
    computed: int = 0
    skipped: int = 0
    avg_psnr: float | None = None
    avg_ssim: float | None = None
    avg_lpips: float | None = None


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def pair_outputs_with_references(
    outputs_dir: str | Path,
    reference_dir: str | Path,
) -> tuple[list[ImagePair], list[UnpairedOutput]]:
    """Match files in *outputs_dir* to files in *reference_dir* by stem.

    Matching is case-insensitive on the stem.  References can use any
    supported extension; outputs are always ``.png``.

    Returns:
        (pairs, unpaired) — paired outputs ready for metrics, and outputs
        that had no matching reference file.
    """
    out_path = Path(outputs_dir)
    ref_path = Path(reference_dir)

    # Index reference files by lower-cased stem → path
    ref_index: dict[str, Path] = {}
    for p in ref_path.iterdir():
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS:
            ref_index[p.stem.lower()] = p

    pairs: list[ImagePair] = []
    unpaired: list[UnpairedOutput] = []

    for out_file in sorted(out_path.glob("*.png"), key=lambda p: p.name.lower()):
        stem_lower = out_file.stem.lower()
        if stem_lower in ref_index:
            pairs.append(
                ImagePair(
                    filename=out_file.name,
                    output_path=str(out_file),
                    reference_path=str(ref_index[stem_lower]),
                )
            )
        else:
            unpaired.append(
                UnpairedOutput(
                    filename=out_file.name,
                    output_path=str(out_file),
                    reason="no_reference",
                )
            )

    return pairs, unpaired


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def _load_rgb(path: str) -> np.ndarray | None:
    """Load image as RGB uint8 (H, W, 3). Returns None on failure."""
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    """Convert RGB uint8 HWC to float32 NCHW tensor in [0, 1]."""
    t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0)


def compute_psnr(img: np.ndarray, ref: np.ndarray) -> float:
    """Compute PSNR (dB) between two RGB uint8 images of the same size."""
    return float(_compute_psnr(ref, img, data_range=255))


def compute_ssim(img: np.ndarray, ref: np.ndarray) -> float:
    """Compute SSIM between two RGB uint8 images of the same size."""
    return float(_compute_ssim(ref, img, data_range=255, channel_axis=2))


def compute_lpips(
    img: np.ndarray,
    ref: np.ndarray,
    lpips_metric,
) -> float:
    """Compute LPIPS distance using a pre-created pyiqa metric object."""
    t_img = _to_tensor(img)
    t_ref = _to_tensor(ref)
    with torch.no_grad():
        score = lpips_metric(t_img, t_ref)
    return float(score.item())


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_full_reference_benchmark(
    outputs_dir: str | Path,
    reference_dir: str | Path,
    metrics_dir: str | Path,
    device: str = "cpu",
    *,
    _lpips_metric=None,  # injectable for tests
) -> BenchmarkSummary:
    """Compute PSNR, SSIM, LPIPS for all matched output/reference pairs.

    Args:
        outputs_dir:    Directory with .png outputs from the pipeline.
        reference_dir:  Directory with ground truth reference images.
        metrics_dir:    Directory where ``per_image.csv`` and
                        ``summary.json`` will be written.
        device:         Compute device for LPIPS (``"cpu"`` or ``"cuda"``).
        _lpips_metric:  Pre-created pyiqa LPIPS metric (injected in tests
                        to avoid network downloads).

    Returns:
        Aggregated :class:`BenchmarkSummary`.
    """
    import pyiqa  # lazy import — heavy dependency

    metrics_path = Path(metrics_dir)
    metrics_path.mkdir(parents=True, exist_ok=True)

    pairs, unpaired = pair_outputs_with_references(outputs_dir, reference_dir)

    # Create LPIPS metric once
    if _lpips_metric is None:
        lpips_metric = pyiqa.create_metric("lpips", device=device)
    else:
        lpips_metric = _lpips_metric

    results: list[PairResult] = []

    for pair in pairs:
        result = _compute_pair(pair, lpips_metric)
        results.append(result)

    # Unpaired outputs produce rows with error="no_reference"
    for up in unpaired:
        results.append(PairResult(filename=up.filename, error="no_reference"))

    # Sort by filename for deterministic output
    results.sort(key=lambda r: r.filename.lower())

    _save_per_image_csv(results, metrics_path / "per_image.csv")
    summary = _build_summary(results)
    _save_summary_json(summary, metrics_path / "summary.json")

    return summary


def _compute_pair(pair: ImagePair, lpips_metric) -> PairResult:
    """Compute all three metrics for one pair. Never raises."""
    out_img = _load_rgb(pair.output_path)
    ref_img = _load_rgb(pair.reference_path)

    if out_img is None or ref_img is None:
        return PairResult(filename=pair.filename, error="load_error")

    if out_img.shape != ref_img.shape:
        return PairResult(filename=pair.filename, error="size_mismatch")

    try:
        psnr = compute_psnr(out_img, ref_img)
        ssim = compute_ssim(out_img, ref_img)
        lpips = compute_lpips(out_img, ref_img, lpips_metric)
    except Exception as exc:  # noqa: BLE001
        return PairResult(filename=pair.filename, error=f"compute_error: {exc}")

    return PairResult(filename=pair.filename, psnr=psnr, ssim=ssim, lpips=lpips)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_per_image_csv(results: list[PairResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filename": r.filename,
                "psnr": "" if r.psnr is None else round(r.psnr, 4),
                "ssim": "" if r.ssim is None else round(r.ssim, 4),
                "lpips": "" if r.lpips is None else round(r.lpips, 4),
                "error": r.error or "",
            })


def _build_summary(results: list[PairResult]) -> BenchmarkSummary:
    computed = [r for r in results if r.error is None]
    skipped = [r for r in results if r.error is not None]

    psnr_vals = [r.psnr for r in computed if r.psnr is not None]
    ssim_vals = [r.ssim for r in computed if r.ssim is not None]
    lpips_vals = [r.lpips for r in computed if r.lpips is not None]

    def avg(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    return BenchmarkSummary(
        mode="full_reference",
        total_pairs=len(results),
        computed=len(computed),
        skipped=len(skipped),
        avg_psnr=avg(psnr_vals),
        avg_ssim=avg(ssim_vals),
        avg_lpips=avg(lpips_vals),
    )


def _save_summary_json(summary: BenchmarkSummary, path: Path) -> None:
    data = {
        "mode": summary.mode,
        "total_pairs": summary.total_pairs,
        "computed": summary.computed,
        "skipped": summary.skipped,
        "avg_psnr": summary.avg_psnr,
        "avg_ssim": summary.avg_ssim,
        "avg_lpips": summary.avg_lpips,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
