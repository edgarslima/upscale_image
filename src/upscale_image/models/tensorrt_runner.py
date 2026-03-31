"""TensorRT runner for Real-ESRGAN (and compatible architectures).

Loads a TensorRT-compiled engine (.ep) exported offline by
``scripts/export_tensorrt.py``. The engine is hardware-specific and must be
regenerated when the GPU or TensorRT version changes (ADR 0014).

Requirements:
    pip install -r requirements/performance.txt

Usage:
    # 1. Export engine once per hardware target:
    #    python scripts/export_tensorrt.py --weights weights/realesrgan-x4.pth \\
    #        --output weights/realesrgan-x4-trt-fp16.ep --precision fp16
    #
    # 2. Use via CLI:
    #    upscale-image upscale input/ -o out/ --model realesrgan-x4-trt-fp16 --device cuda

Note: ``torch_tensorrt`` and ``onnxruntime-gpu`` must NOT coexist in the same
environment due to CUDA runtime conflicts (ADR 0014).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from upscale_image.config import AppConfig
from upscale_image.models.base import SuperResolutionModel


class TensorRTRunner(SuperResolutionModel):
    """Runner that executes a TensorRT-compiled engine (.ep).

    Args:
        scale:       Upscale factor (must match the exported engine).
        engine_path: Path to the ``.ep`` TensorRT engine file.
        precision:   ``"fp16"`` or ``"fp32"`` — must match the exported engine.
    """

    def __init__(
        self,
        scale: int = 4,
        engine_path: str = "weights/realesrgan-x4-trt-fp16.ep",
        precision: str = "fp16",
    ) -> None:
        self._scale = scale
        self._engine_path = Path(engine_path)
        self._precision = precision
        self._net: nn.Module | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"realesrgan-x4-trt-{self._precision}"

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
        """Load the TensorRT engine from disk.

        Raises:
            ImportError:       If ``torch-tensorrt`` is not installed.
            FileNotFoundError: If the engine file does not exist.
            RuntimeError:      If the engine cannot be loaded.
        """
        try:
            import torch_tensorrt  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch-tensorrt não instalado. "
                "Instale com: pip install -r requirements/performance.txt"
            )

        if not self._engine_path.exists():
            raise FileNotFoundError(
                f"Engine TensorRT não encontrado: {self._engine_path}\n"
                "Gere o engine com: python scripts/export_tensorrt.py"
            )

        try:
            net = torch_tensorrt.load(str(self._engine_path))
            net = net.cuda().eval()
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao carregar engine TensorRT de {self._engine_path}: {exc}"
            ) from exc

        self._net = net
        self._loaded = True

    def upscale(self, image: np.ndarray, config: AppConfig) -> np.ndarray:
        """Run inference through the TensorRT engine.

        Args:
            image:  BGR uint8 numpy array (H × W × 3).
            config: Resolved AppConfig (device must be ``"cuda"``).

        Returns:
            Upscaled BGR uint8 numpy array (scale·H × scale·W × 3).

        Raises:
            RuntimeError: If called before :meth:`load`.
        """
        if not self._loaded or self._net is None:
            raise RuntimeError("Model not loaded. Call load() before upscale().")

        dtype = torch.float16 if self._precision == "fp16" else torch.float32

        img_rgb = image[:, :, ::-1].copy()
        tensor = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .to(dtype=dtype)
            .div(255.0)
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output = self._net(tensor)

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
        """Release the engine and free GPU memory."""
        self._net = None
        self._loaded = False
        torch.cuda.empty_cache()
