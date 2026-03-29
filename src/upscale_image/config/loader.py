"""Configuration loader: reads YAML, merges with CLI overrides, validates.

Precedence: CLI > YAML > defaults.
Validation happens once here; downstream modules trust AppConfig unconditionally.
"""

from __future__ import annotations

import yaml

from upscale_image.config.schema import AppConfig, ModelConfig, RuntimeConfig

_VALID_DEVICES = {"cpu", "cuda"}
_VALID_PRECISIONS = {"fp32", "fp16"}
_SUPPORTED_SCALES = {2, 4, 8}


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _merge_yaml_into_config(config: AppConfig, yaml_data: dict) -> None:
    """Apply YAML values into config (lower priority than CLI overrides)."""
    model_data = yaml_data.get("model", {})
    if "name" in model_data:
        config.model.name = model_data["name"]
    if "scale" in model_data:
        config.model.scale = int(model_data["scale"])

    runtime_data = yaml_data.get("runtime", {})
    if "device" in runtime_data:
        config.runtime.device = runtime_data["device"]
    if "precision" in runtime_data:
        config.runtime.precision = runtime_data["precision"]
    if "tile_size" in runtime_data:
        config.runtime.tile_size = int(runtime_data["tile_size"])
    if "tile_pad" in runtime_data:
        config.runtime.tile_pad = int(runtime_data["tile_pad"])


def _apply_cli_overrides(
    config: AppConfig,
    *,
    input_dir: str | None,
    output_dir: str | None,
    model: str | None,
    scale: int | None,
    device: str | None,
) -> None:
    """Overwrite config fields with explicitly provided CLI values."""
    if input_dir is not None:
        config.input_dir = input_dir
    if output_dir is not None:
        config.output_dir = output_dir
    if model is not None:
        config.model.name = model
    if scale is not None:
        config.model.scale = scale
    if device is not None:
        config.runtime.device = device


def _validate(config: AppConfig) -> None:
    """Raise ValueError early if the configuration is inconsistent."""
    errors: list[str] = []

    if not config.input_dir:
        errors.append("input_dir is required.")
    if not config.output_dir:
        errors.append("output_dir is required.")
    if not config.model.name:
        errors.append("model.name is required.")
    if config.model.scale not in _SUPPORTED_SCALES:
        errors.append(
            f"model.scale={config.model.scale} is not supported. "
            f"Valid values: {sorted(_SUPPORTED_SCALES)}."
        )
    if config.runtime.device not in _VALID_DEVICES:
        errors.append(
            f"runtime.device='{config.runtime.device}' is invalid. "
            f"Valid values: {sorted(_VALID_DEVICES)}."
        )
    if config.runtime.precision not in _VALID_PRECISIONS:
        errors.append(
            f"runtime.precision='{config.runtime.precision}' is invalid. "
            f"Valid values: {sorted(_VALID_PRECISIONS)}."
        )
    if config.runtime.tile_size < 0:
        errors.append("runtime.tile_size must be >= 0 (0 disables tiling).")
    if config.runtime.tile_pad < 0:
        errors.append("runtime.tile_pad must be >= 0.")
    if config.runtime.precision == "fp16" and config.runtime.device == "cpu":
        errors.append("runtime.precision=fp16 is not supported on device=cpu.")

    if errors:
        raise ValueError("Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))


def resolve_config(
    *,
    input_dir: str | None = None,
    output_dir: str | None = None,
    model: str | None = None,
    scale: int | None = None,
    device: str | None = None,
    config_file: str | None = None,
) -> AppConfig:
    """Build the final AppConfig applying precedence: CLI > YAML > defaults.

    Args:
        input_dir:   Path to the directory with input images (CLI).
        output_dir:  Path to the output directory (CLI).
        model:       Model name override (CLI).
        scale:       Upscale factor override (CLI).
        device:      Compute device override (CLI).
        config_file: Optional path to a YAML configuration file.

    Returns:
        Validated AppConfig ready for pipeline consumption.

    Raises:
        ValueError: If the resolved configuration is invalid.
        FileNotFoundError: If config_file is specified but does not exist.
    """
    config = AppConfig(
        model=ModelConfig(),
        runtime=RuntimeConfig(),
    )

    if config_file:
        config.config_file = config_file
        yaml_data = _load_yaml(config_file)
        _merge_yaml_into_config(config, yaml_data)

    _apply_cli_overrides(
        config,
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        scale=scale,
        device=device,
    )

    _validate(config)
    return config
