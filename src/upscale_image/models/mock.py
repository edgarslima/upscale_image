"""Mock super-resolution runner for testing and contract validation.

Upscales by nearest-neighbour resize — no weights, no GPU required.
Behaviour is deterministic and fast, suitable for unit and integration tests.
"""

from __future__ import annotations

import cv2
import numpy as np

from upscale_image.config import AppConfig
from upscale_image.models.base import SuperResolutionModel


class MockSuperResolutionModel(SuperResolutionModel):
    """Trivial runner that resizes images with cv2 nearest-neighbour interpolation.

    Used only for pipeline wiring tests; never shipped as a real model.
    """

    def __init__(self, scale: int = 4) -> None:
        self._scale = scale
        self._loaded = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mock"

    @property
    def scale(self) -> int:
        return self._scale

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        self._loaded = True

    def upscale(self, image: np.ndarray, config: AppConfig) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() before upscale().")
        h, w = image.shape[:2]
        return cv2.resize(
            image,
            (w * self._scale, h * self._scale),
            interpolation=cv2.INTER_NEAREST,
        )

    def unload(self) -> None:
        self._loaded = False
