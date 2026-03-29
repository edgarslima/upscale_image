"""Serialize AppConfig to a plain dict or YAML string for run persistence."""

from __future__ import annotations

import yaml

from upscale_image.config.schema import AppConfig


def config_to_dict(config: AppConfig) -> dict:
    return {
        "input_dir": config.input_dir,
        "output_dir": config.output_dir,
        "config_file": config.config_file,
        "model": {
            "name": config.model.name,
            "scale": config.model.scale,
        },
        "runtime": {
            "device": config.runtime.device,
            "precision": config.runtime.precision,
            "tile_size": config.runtime.tile_size,
            "tile_pad": config.runtime.tile_pad,
        },
    }


def config_to_yaml(config: AppConfig) -> str:
    return yaml.safe_dump(config_to_dict(config), sort_keys=False, allow_unicode=True)


def save_effective_config(config: AppConfig, path: str) -> None:
    """Write the effective configuration to *path* as YAML."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(config_to_yaml(config))
