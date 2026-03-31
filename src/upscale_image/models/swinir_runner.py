"""SwinIR runner for Real-SR (Large, x4).

Implements the SuperResolutionModel contract (ADR 0004) using the SwinIR
architecture. Requires the `swinir` package and `timm>=1.0.0`.

Key constraint: SwinIR requires input H and W to be multiples of window_size=8.
This runner pads the input tensor with reflect-padding, runs the forward pass,
then crops the output back to (orig_H * scale, orig_W * scale, 3).

Installation:
    pip install -r requirements/swinir.txt

Weights: place `weights/swinir-x4.pth` before calling load().
Official weights: https://github.com/JingyunLiang/SwinIR/releases
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from upscale_image.config import AppConfig
from upscale_image.models.base import SuperResolutionModel

_WINDOW_SIZE = 8


class SwinIRRunner(SuperResolutionModel):
    """Runner for SwinIR Real-SR (Large, x4).

    Args:
        scale:        Upscale factor (default 4).
        weights_path: Path to the ``.pth`` weight file.
    """

    def __init__(
        self,
        scale: int = 4,
        weights_path: str = "weights/swinir-x4.pth",
    ) -> None:
        self._scale = scale
        self._weights_path = Path(weights_path)
        self._net: nn.Module | None = None
        self._eager_net: nn.Module | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "swinir-x4"

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
        """Load SwinIR weights from disk.

        Raises:
            ImportError:      If the `swinir` package is not installed.
            FileNotFoundError: If the weights file does not exist.
            RuntimeError:     If state-dict loading fails.
        """
        try:
            from swinir import SwinIR as SwinIRNet  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "SwinIR não instalado. Instale com: "
                "pip install -r requirements/swinir.txt"
            )

        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"Pesos SwinIR não encontrados: {self._weights_path}\n"
                "Baixe de: https://github.com/JingyunLiang/SwinIR/releases"
            )

        net = SwinIRNet(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=_WINDOW_SIZE,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=240,
            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
            mlp_ratio=2,
            upsampler="nearest+conv",
            resi_connection="3conv",
        )

        loadnet = torch.load(
            str(self._weights_path), map_location="cpu", weights_only=True
        )

        if "params_ema" in loadnet:
            state_dict = loadnet["params_ema"]
        elif "params" in loadnet:
            state_dict = loadnet["params"]
        else:
            state_dict = loadnet

        try:
            net.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load SwinIR state dict from {self._weights_path}: {exc}"
            ) from exc

        net.eval()

        self._eager_net = net
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            net = torch.compile(net, mode="reduce-overhead")

        self._net = net
        self._loaded = True

    def upscale(self, image: np.ndarray, config: AppConfig) -> np.ndarray:
        """Run SwinIR super-resolution on a single image.

        Args:
            image:  BGR uint8 numpy array (H × W × 3).
            config: Resolved AppConfig providing device and precision.

        Returns:
            Upscaled BGR uint8 numpy array (scale·H × scale·W × 3).

        Raises:
            RuntimeError: If called before load(), or if device is unavailable.
        """
        if not self._loaded or self._net is None:
            raise RuntimeError("Model not loaded. Call load() before upscale().")

        device = self._resolve_device(config.runtime.device)
        net = self._net.to(device).float()

        use_amp = (config.runtime.precision == "fp16") and (device.type == "cuda")

        # BGR uint8 → RGB float32 tensor [0, 1] — (1, C, H, W)
        img_rgb = image[:, :, ::-1].copy()
        orig_h, orig_w = img_rgb.shape[:2]
        tensor = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

        # Pad to multiples of window_size=8.
        # reflect mode requires pad < input dim; fall back to replicate for small images.
        pad_h = ((_WINDOW_SIZE - orig_h % _WINDOW_SIZE) % _WINDOW_SIZE)
        pad_w = ((_WINDOW_SIZE - orig_w % _WINDOW_SIZE) % _WINDOW_SIZE)
        if pad_h > 0 or pad_w > 0:
            pad_mode = "reflect" if (pad_h < orig_h and pad_w < orig_w) else "replicate"
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=pad_mode)

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                try:
                    output = net(tensor)
                except Exception:
                    if self._eager_net is not None and net is not self._eager_net:
                        self._net = self._eager_net
                        self._eager_net = None
                        net = self._net.to(device).float()
                        output = net(tensor)
                    else:
                        raise

        # Crop output to (orig_h * scale, orig_w * scale)
        out_h = orig_h * self._scale
        out_w = orig_w * self._scale
        output = output[:, :, :out_h, :out_w]

        out_np = (
            output.squeeze(0)
            .float()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return out_np[:, :, ::-1].copy()  # RGB → BGR

    def unload(self) -> None:
        """Release model weights and free GPU memory."""
        self._net = None
        self._eager_net = None
        self._loaded = False
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda requested but CUDA is not available on this system."
            )
        return torch.device(device_str)
