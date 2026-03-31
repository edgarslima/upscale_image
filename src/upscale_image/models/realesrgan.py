"""Real-ESRGAN runner using RRDBNet architecture from basicsr.

Supports:
- CPU and CUDA inference
- fp32 and fp16 precision (fp16 only on CUDA)
- Standard Real-ESRGAN weight files (.pth) with 'params_ema', 'params'
  or raw state-dict layouts

Weights path convention: ``weights/<model_name>.pth``
Standard arch for realesrgan-x4: num_feat=64, num_block=23, num_grow_ch=32
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import upscale_image.models._compat  # noqa: F401 — apply torchvision shim first
from basicsr.archs.rrdbnet_arch import RRDBNet

from upscale_image.config import AppConfig
from upscale_image.models.base import SuperResolutionModel


class RealESRGANRunner(SuperResolutionModel):
    """Runner for Real-ESRGAN models based on the RRDBNet architecture.

    Args:
        scale:        Upscale factor (2, 4, or 8).
        weights_path: Path to the ``.pth`` weight file.
        num_feat:     Number of features in each RRDB block (default 64).
        num_block:    Number of RRDB blocks (default 23 — full model).
        num_grow_ch:  Growth channels inside each residual block (default 32).
    """

    def __init__(
        self,
        scale: int,
        weights_path: str,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        self._scale = scale
        self._weights_path = Path(weights_path)
        self._num_feat = num_feat
        self._num_block = num_block
        self._num_grow_ch = num_grow_ch
        self._net: nn.Module | None = None
        self._eager_net: nn.Module | None = None  # uncompiled fallback
        self._loaded = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "realesrgan-x4"

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
        """Load and validate weights from disk onto CPU.

        Raises:
            FileNotFoundError: If the weights file does not exist.
            RuntimeError:      If state-dict loading fails.
        """
        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {self._weights_path}. "
                "Download the model weights and place them in weights/."
            )

        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=self._num_feat,
            num_block=self._num_block,
            num_grow_ch=self._num_grow_ch,
            scale=self._scale,
        )

        loadnet = torch.load(str(self._weights_path), map_location="cpu", weights_only=True)

        # Handle common weight file layouts
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
                f"Failed to load state dict from {self._weights_path}: {exc}"
            ) from exc

        net.eval()

        self._eager_net = net  # always keep the uncompiled version for fallback
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            net = torch.compile(net, mode="reduce-overhead")

        self._net = net
        self._loaded = True

    def upscale(self, image: np.ndarray, config: AppConfig) -> np.ndarray:
        """Run Real-ESRGAN super-resolution on a single image.

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
        net = self._net.to(device)

        # Model always kept in FP32 in memory; autocast manages per-op precision.
        net = net.float()

        # BGR uint8 → RGB float32 tensor [0, 1] — always FP32 on input
        img_rgb = image[:, :, ::-1].copy()  # BGR → RGB
        tensor = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

        use_amp = (config.runtime.precision == "fp16") and (device.type == "cuda")

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                try:
                    if config.runtime.tile_size > 0:
                        output = self._upscale_tiled(tensor, net, config)
                    else:
                        output = net(tensor)
                except Exception:
                    # torch.compile may fail in environments missing build tools
                    # (e.g. Python.h not found). Fall back to eager mode for this
                    # and all future calls.
                    if self._eager_net is not None and net is not self._eager_net:
                        self._net = self._eager_net
                        self._eager_net = None
                        net = self._net.to(device).float()
                        if config.runtime.tile_size > 0:
                            output = self._upscale_tiled(tensor, net, config)
                        else:
                            output = net(tensor)
                    else:
                        raise

        # Output always back to FP32 before converting to uint8
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

    def upscale_batch(
        self,
        images: list[np.ndarray],
        config: AppConfig,
    ) -> list[np.ndarray]:
        """Run Real-ESRGAN on a batch of images in a single GPU forward pass.

        All images are padded to the largest dimensions in the batch, stacked
        into a single (N, C, H, W) tensor, processed together, then cropped
        back to each image's original upscaled size.

        When tiling is enabled (tile_size > 0), falls back to serial upscale()
        because tiled inference is not compatible with batch padding.

        Args:
            images: List of BGR uint8 numpy arrays, may differ in size.
            config: Resolved AppConfig.

        Returns:
            List of BGR uint8 numpy arrays; item i has shape
            (orig_h_i * scale, orig_w_i * scale, 3).
        """
        if not self._loaded or self._net is None:
            raise RuntimeError("Model not loaded. Call load() before upscale_batch().")

        # Tiling and batch padding are incompatible — fall back to serial
        if config.runtime.tile_size > 0:
            return [self.upscale(img, config) for img in images]

        if not images:
            return []

        device = self._resolve_device(config.runtime.device)
        net = self._net.to(device).float()
        use_amp = (config.runtime.precision == "fp16") and (device.type == "cuda")

        # BGR uint8 → RGB float32 tensors [0, 1]
        tensors = []
        original_sizes: list[tuple[int, int]] = []
        for img in images:
            h, w = img.shape[:2]
            original_sizes.append((h, w))
            rgb = img[:, :, ::-1].copy()
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
            tensors.append(t)

        # Pad each tensor to max(H) × max(W) across the batch
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        padded = []
        for t in tensors:
            pad_h = max_h - t.shape[1]
            pad_w = max_w - t.shape[2]
            import torch.nn.functional as F
            padded.append(F.pad(t, (0, pad_w, 0, pad_h), mode="reflect"))

        batch_tensor = torch.stack(padded, dim=0).to(device)  # (N, C, H, W)

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                batch_out = net(batch_tensor)  # (N, C, scale*H, scale*W)

        # Crop each output to the original upscaled size and convert to BGR uint8
        results: list[np.ndarray] = []
        for i, (orig_h, orig_w) in enumerate(original_sizes):
            out_h = orig_h * self._scale
            out_w = orig_w * self._scale
            out = batch_out[i, :, :out_h, :out_w]
            out_np = (
                out.float().clamp(0.0, 1.0).mul(255.0).round()
                   .byte().permute(1, 2, 0).cpu().numpy()
            )
            results.append(out_np[:, :, ::-1].copy())
        return results

    def unload(self) -> None:
        """Release model weights and free GPU memory."""
        self._net = None
        self._eager_net = None
        self._loaded = False
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upscale_tiled(
        self, tensor: torch.Tensor, net: nn.Module, config: "AppConfig"
    ) -> torch.Tensor:
        """Process the image in overlapping tiles with Hann-window blending.

        Each tile is weighted by a 2-D Hann window so tile boundaries blend
        smoothly instead of producing hard-cut artefacts.
        """
        tile_size = config.runtime.tile_size
        tile_pad  = config.runtime.tile_pad
        scale     = self._scale

        _, channel, height, width = tensor.shape
        # Weighted-sum accumulators
        output_sum = tensor.new_zeros(1, channel, height * scale, width * scale)
        weight_sum = tensor.new_zeros(1, 1,       height * scale, width * scale)

        tiles_x = math.ceil(width  / tile_size)
        tiles_y = math.ceil(height / tile_size)

        for iy in range(tiles_y):
            for ix in range(tiles_x):
                # Tile boundaries in the input
                in_x1 = ix * tile_size
                in_x2 = min(in_x1 + tile_size, width)
                in_y1 = iy * tile_size
                in_y2 = min(in_y1 + tile_size, height)

                # Padded tile boundaries (clamped to image edges)
                px1 = max(in_x1 - tile_pad, 0)
                px2 = min(in_x2 + tile_pad, width)
                py1 = max(in_y1 - tile_pad, 0)
                py2 = min(in_y2 + tile_pad, height)

                tile_in = tensor[:, :, py1:py2, px1:px2]
                with torch.inference_mode():
                    tile_out = net(tile_in)

                # Valid region (without padding) in output space
                cx1 = (in_x1 - px1) * scale
                cx2 = cx1 + (in_x2 - in_x1) * scale
                cy1 = (in_y1 - py1) * scale
                cy2 = cy1 + (in_y2 - in_y1) * scale
                valid = tile_out[:, :, cy1:cy2, cx1:cx2]

                h_out = cy2 - cy1
                w_out = cx2 - cx1
                mask = self._hann_window(h_out, w_out, device=tensor.device)

                out_y1 = in_y1 * scale
                out_y2 = in_y2 * scale
                out_x1 = in_x1 * scale
                out_x2 = in_x2 * scale

                output_sum[:, :, out_y1:out_y2, out_x1:out_x2] += valid * mask
                weight_sum[:,  :, out_y1:out_y2, out_x1:out_x2] += mask

        return output_sum / weight_sum.clamp(min=1e-6)

    @staticmethod
    def _hann_window(h: int, w: int, device: torch.device) -> torch.Tensor:
        """2-D Hann window: peaks at 1 in the centre, tapers to 0 at edges.

        Returns shape (1, 1, h, w) for broadcasting with (1, C, h, w) tensors.
        """
        hann_h = torch.hann_window(h, periodic=False, device=device)
        hann_w = torch.hann_window(w, periodic=False, device=device)
        # outer product → (H, W); unsqueeze for (1, 1, H, W)
        return (hann_h.unsqueeze(1) * hann_w.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve and validate the requested device.

        Raises:
            RuntimeError: If CUDA is requested but not available.
        """
        if device_str == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda requested but CUDA is not available on this system."
            )
        return torch.device(device_str)
