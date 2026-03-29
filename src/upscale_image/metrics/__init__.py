"""Quality metrics: PSNR, SSIM, LPIPS (full-reference) and NIQE (no-reference)."""

from upscale_image.metrics.full_reference import (
    BenchmarkSummary,
    ImagePair,
    PairResult,
    UnpairedOutput,
    pair_outputs_with_references,
    run_full_reference_benchmark,
)
from upscale_image.metrics.no_reference import (
    NiqeResult,
    NrBenchmarkSummary,
    run_no_reference_benchmark,
)

__all__ = [
    "BenchmarkSummary",
    "ImagePair",
    "NiqeResult",
    "NrBenchmarkSummary",
    "PairResult",
    "UnpairedOutput",
    "pair_outputs_with_references",
    "run_full_reference_benchmark",
    "run_no_reference_benchmark",
]
