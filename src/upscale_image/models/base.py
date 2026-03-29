"""Abstract base contract for all super-resolution model runners.

Rules (ADR 0004):
- The pipeline calls only this interface; it never imports concrete runners.
- Every model must implement load(), upscale() and unload().
- Metadata (name, scale, is_loaded) is available for logging and manifests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from upscale_image.config import AppConfig


class SuperResolutionModel(ABC):
    """Base contract for super-resolution runners.

    Lifecycle::

        model.load()
        output = model.upscale(image, config)   # may be called N times
        model.unload()

    ``image`` and the return value are always ``np.ndarray`` in BGR uint8
    format (OpenCV convention).
    """

    # ------------------------------------------------------------------
    # Metadata — consumed by logger and manifest; never by the pipeline loop
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Logical model name (e.g. ``'realesrgan-x4'``)."""

    @property
    @abstractmethod
    def scale(self) -> int:
        """Upscale factor this runner was configured for."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True after :meth:`load` succeeds, False after :meth:`unload`."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load model weights and prepare for inference.

        Must be called before :meth:`upscale`.
        Raises :exc:`RuntimeError` if loading fails.
        """

    @abstractmethod
    def upscale(self, image: np.ndarray, config: AppConfig) -> np.ndarray:
        """Run super-resolution on a single image.

        Args:
            image:  Input BGR uint8 numpy array (H × W × 3).
            config: Resolved :class:`~upscale_image.config.AppConfig`
                    providing device, precision and tiling parameters.

        Returns:
            Upscaled BGR uint8 numpy array (scale·H × scale·W × 3).

        Raises:
            :exc:`RuntimeError`: If called before :meth:`load`.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free associated resources."""
