"""Tests for the configuration layer (schema, loader, serializer)."""

from __future__ import annotations

import textwrap
import pytest

from upscale_image.config import (
    AppConfig,
    ModelConfig,
    RuntimeConfig,
    config_to_dict,
    config_to_yaml,
    resolve_config,
    save_effective_config,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_defaults_pure():
    cfg = resolve_config(input_dir="/in", output_dir="/out")
    assert cfg.model.name == "realesrgan-x4"
    assert cfg.model.scale == 4
    assert cfg.runtime.device == "cpu"
    assert cfg.runtime.precision == "fp32"
    assert cfg.runtime.tile_size == 0
    assert cfg.runtime.tile_pad == 32


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def test_yaml_valid(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(textwrap.dedent("""\
        model:
          name: realesrgan-x4plus
          scale: 4
        runtime:
          device: cpu
          precision: fp32
          tile_size: 512
          tile_pad: 32
    """))
    cfg = resolve_config(input_dir="/in", output_dir="/out", config_file=str(yaml_file))
    assert cfg.model.name == "realesrgan-x4plus"
    assert cfg.runtime.tile_size == 512
    assert cfg.runtime.tile_pad == 32


def test_yaml_missing_file():
    with pytest.raises(FileNotFoundError):
        resolve_config(input_dir="/in", output_dir="/out", config_file="/nonexistent.yaml")


def test_yaml_empty_file(tmp_path):
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("")
    # empty YAML should use defaults without crashing
    cfg = resolve_config(input_dir="/in", output_dir="/out", config_file=str(yaml_file))
    assert cfg.model.name == "realesrgan-x4"


# ---------------------------------------------------------------------------
# CLI > YAML precedence
# ---------------------------------------------------------------------------

def test_cli_overrides_yaml(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(textwrap.dedent("""\
        model:
          name: from-yaml
          scale: 4
        runtime:
          device: cpu
    """))
    cfg = resolve_config(
        input_dir="/in",
        output_dir="/out",
        model="from-cli",
        config_file=str(yaml_file),
    )
    assert cfg.model.name == "from-cli"


def test_cli_device_overrides_yaml(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("runtime:\n  device: cpu\n")
    cfg = resolve_config(
        input_dir="/in",
        output_dir="/out",
        device="cpu",
        config_file=str(yaml_file),
    )
    assert cfg.runtime.device == "cpu"


def test_yaml_wins_over_defaults(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("model:\n  scale: 2\n")
    cfg = resolve_config(input_dir="/in", output_dir="/out", config_file=str(yaml_file))
    assert cfg.model.scale == 2


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_missing_input_dir():
    with pytest.raises(ValueError, match="input_dir"):
        resolve_config(output_dir="/out")


def test_missing_output_dir():
    with pytest.raises(ValueError, match="output_dir"):
        resolve_config(input_dir="/in")


def test_invalid_device():
    with pytest.raises(ValueError, match="device"):
        resolve_config(input_dir="/in", output_dir="/out", device="tpu")


def test_invalid_scale():
    with pytest.raises(ValueError, match="scale"):
        resolve_config(input_dir="/in", output_dir="/out", scale=3)


def test_fp16_on_cpu_rejected(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("runtime:\n  precision: fp16\n  device: cpu\n")
    with pytest.raises(ValueError, match="fp16"):
        resolve_config(input_dir="/in", output_dir="/out", config_file=str(yaml_file))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_config_to_dict_structure():
    cfg = resolve_config(input_dir="/in", output_dir="/out")
    d = config_to_dict(cfg)
    assert d["input_dir"] == "/in"
    assert d["output_dir"] == "/out"
    assert d["model"]["name"] == "realesrgan-x4"
    assert d["model"]["scale"] == 4
    assert d["runtime"]["device"] == "cpu"
    assert d["runtime"]["precision"] == "fp32"


def test_config_to_yaml_is_parseable():
    import yaml
    cfg = resolve_config(input_dir="/in", output_dir="/out")
    text = config_to_yaml(cfg)
    parsed = yaml.safe_load(text)
    assert parsed["model"]["scale"] == 4


def test_save_effective_config(tmp_path):
    import yaml
    cfg = resolve_config(input_dir="/in", output_dir="/out")
    dest = tmp_path / "effective_config.yaml"
    save_effective_config(cfg, str(dest))
    assert dest.exists()
    parsed = yaml.safe_load(dest.read_text())
    assert parsed["model"]["name"] == "realesrgan-x4"
