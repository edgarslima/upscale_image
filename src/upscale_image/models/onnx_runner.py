"""ONNX Runtime runner for Real-ESRGAN (and compatible architectures).

Provides cross-vendor inference (NVIDIA, AMD, Intel, CPU) via ``onnxruntime``.
The model must be exported first with ``scripts/export_onnx.py``.

Supported execution providers:
  - ``CUDAExecutionProvider``   — GPU (NVIDIA), registered as ``realesrgan-x4-onnx-cuda``
  - ``CPUExecutionProvider``    — CPU fallback, registered as ``realesrgan-x4-onnx-cpu``

Requirements (separate from torch-tensorrt — see ADR 0014):
    pip install onnxruntime-gpu==1.20.0

Usage:
    # 1. Export ONNX model once:
    #    python scripts/export_onnx.py --weights weights/realesrgan-x4.pth \\
    #        --output weights/realesrgan-x4.onnx
    #
    # 2. Use via CLI:
    #    upscale-image upscale input/ -o out/ --model realesrgan-x4-onnx-cuda
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from upscale_image.config import AppConfig
from upscale_image.models.base import SuperResolutionModel


class OnnxRunner(SuperResolutionModel):
    """Runner that executes inference via ONNX Runtime.

    Uses a pure NumPy pipeline — no ``torch`` dependency at inference time.

    Args:
        scale:    Upscale factor (must match the exported model).
        onnx_path: Path to the ``.onnx`` model file.
        provider: Primary ONNX execution provider.
                  ``"CUDAExecutionProvider"`` or ``"CPUExecutionProvider"``.
    """

    def __init__(
        self,
        scale: int = 4,
        onnx_path: str = "weights/realesrgan-x4.onnx",
        provider: str = "CUDAExecutionProvider",
    ) -> None:
        self._scale = scale
        self._onnx_path = Path(onnx_path)
        self._provider = provider
        self._session = None          # onnxruntime.InferenceSession
        self._input_name: str = ""
        self._loaded = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        suffix = "cuda" if "CUDA" in self._provider else "cpu"
        return f"realesrgan-x4-onnx-{suffix}"

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
        """Load the ONNX model and create an inference session.

        Raises:
            ImportError:       If ``onnxruntime`` is not installed.
            FileNotFoundError: If the ONNX model file does not exist.
            RuntimeError:      If the session cannot be created.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime não instalado. "
                "Instale com: pip install onnxruntime-gpu==1.20.0"
            )

        if not self._onnx_path.exists():
            raise FileNotFoundError(
                f"Modelo ONNX não encontrado: {self._onnx_path}\n"
                "Exporte com: python scripts/export_onnx.py"
            )

        try:
            session = ort.InferenceSession(
                str(self._onnx_path),
                providers=[self._provider, "CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao criar sessão ONNX para {self._onnx_path}: {exc}"
            ) from exc

        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._loaded = True

    def upscale(self, image: np.ndarray, config: AppConfig) -> np.ndarray:
        """Run ONNX inference on a single image.

        Pure NumPy pipeline: no ``torch`` dependency at inference time.

        Args:
            image:  BGR uint8 numpy array (H × W × 3).
            config: Resolved AppConfig (device used for EP selection at load time).

        Returns:
            Upscaled BGR uint8 numpy array (scale·H × scale·W × 3).

        Raises:
            RuntimeError: If called before :meth:`load`.
        """
        if not self._loaded or self._session is None:
            raise RuntimeError("Model not loaded. Call load() before upscale().")

        orig_h, orig_w = image.shape[:2]

        # BGR uint8 → RGB float32 (1, C, H, W) in [0, 1]
        img_rgb = image[:, :, ::-1].copy()
        inp = (
            img_rgb.transpose(2, 0, 1)          # (C, H, W)
            .astype(np.float32) / 255.0
        )[np.newaxis, ...]                        # (1, C, H, W)

        outputs = self._session.run(None, {self._input_name: inp})
        out = outputs[0]  # (1, C, scale*H, scale*W) float32

        out_h = orig_h * self._scale
        out_w = orig_w * self._scale

        # Crop to expected output size (handles dynamic-axis models)
        out = out[0, :, :out_h, :out_w]          # (C, H*scale, W*scale)

        out_np = (
            np.clip(out, 0.0, 1.0) * 255.0 + 0.5
        ).astype(np.uint8).transpose(1, 2, 0)    # (H*scale, W*scale, C)

        return out_np[:, :, ::-1].copy()          # RGB → BGR

    def unload(self) -> None:
        """Release the ONNX session."""
        self._session = None
        self._loaded = False
