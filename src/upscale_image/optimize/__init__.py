"""Optimization layer: generate compressed derivatives from canonical run outputs."""

from upscale_image.optimize.optimizer import (
    OptimizeConfig,
    OptimizeSummary,
    ImageOptResult,
    run_optimization,
    default_optimize_config,
)

__all__ = [
    "OptimizeConfig",
    "OptimizeSummary",
    "ImageOptResult",
    "run_optimization",
    "default_optimize_config",
]
