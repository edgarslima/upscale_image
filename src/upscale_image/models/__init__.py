"""Model abstraction layer: contract, registry and runners."""

from upscale_image.models.base import SuperResolutionModel
from upscale_image.models.bicubic import BicubicRunner
from upscale_image.models.mock import MockSuperResolutionModel
from upscale_image.models.registry import (
    ModelRegistry,
    available_models,
    register,
    resolve_model,
)

__all__ = [
    "BicubicRunner",
    "SuperResolutionModel",
    "MockSuperResolutionModel",
    "ModelRegistry",
    "register",
    "resolve_model",
    "available_models",
]
