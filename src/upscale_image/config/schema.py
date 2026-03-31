"""Canonical configuration schema for upscale_image.

All modules consume AppConfig exclusively; no module reads raw CLI flags
or YAML dicts directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    name: str = "realesrgan-x4"
    scale: int = 4


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    precision: Literal["fp32", "fp16"] = "fp32"
    tile_size: int = 0   # 0 means no tiling
    tile_pad: int = 32
    async_io: bool = False
    prefetch_size: int = 4
    write_workers: int = 2
    batch_size: int = 1   # images per GPU forward pass; 0 = auto-detect by VRAM
    multi_gpu: bool = False
    gpu_ids: list[int] = field(default_factory=list)  # [] = use all detected GPUs


@dataclass
class AppConfig:
    input_dir: str = ""
    output_dir: str = ""
    config_file: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
