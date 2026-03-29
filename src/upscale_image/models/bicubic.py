"""Bicubic super-resolution runner.

Implements the standard bicubic upsampling baseline using
``torch.nn.functional.interpolate``.  No pretrained weights are required —
``load()`` is a no-op in terms of network parameters.

This runner is useful as a quality floor for comparison: if a deep learning
model does not beat bicubic, the model is not adding value.

Registered in the default registry as ``"bicubic"``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from upscale_image.config import AppConfig
from upscale_image.models.base import SuperResolutionModel


class BicubicRunner(SuperResolutionModel):
    """Bicubic upsampling via PyTorch — no weights, deterministic output.

    I/O contract (same as all runners):
      - Input:  BGR uint8 numpy array (H × W × 3)
      - Output: BGR uint8 numpy array (scale·H × scale·W × 3)
    """

    def __init__(self, scale: int = 4) -> None:
        self._scale = scale
        self._loaded = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "bicubic"

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
        """No weights to load; marks the runner as ready."""
        self._loaded = True

    def upscale(self, image: np.ndarray, config: AppConfig) -> np.ndarray:
        """Upscale *image* using bicubic interpolation.

        Args:
            image:  BGR uint8 numpy array (H × W × 3).
            config: Runtime config (device is respected; precision is
                    ignored — bicubic is always float32 internally).

        Returns:
            BGR uint8 numpy array at scale × original resolution.

        Raises:
            RuntimeError: If :meth:`load` was not called.
        """
        if not self._loaded:
            raise RuntimeError("BicubicRunner is not loaded. Call load() first.")

        device = torch.device(config.runtime.device)

        # BGR uint8 HWC → float32 NCHW in [0, 1]
        t = torch.from_numpy(image.astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)

        h, w = t.shape[2], t.shape[3]
        t_up = F.interpolate(
            t,
            size=(h * self._scale, w * self._scale),
            mode="bicubic",
            align_corners=False,
        )

        # Clamp to [0, 1], convert back to BGR uint8 HWC
        t_up = t_up.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0)
        return (t_up.cpu().numpy() * 255.0).round().astype(np.uint8)

    def unload(self) -> None:
        """Mark as unloaded (no resources to free)."""
        self._loaded = False
