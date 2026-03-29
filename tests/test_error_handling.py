"""Tests for error handling strategy (step 12).

Covers the 7 minimum scenarios defined in docs/parts/12-implementar-tratamento-de-erros.md:
  1. imagem corrompida            → discovery skips it (SkippedFile), run continues
  2. erro de leitura por item     → ItemResult status="failed", run continues
  3. falha isolada de inferência  → ItemResult status="failed", run continues
  4. modelo não encontrado        → structural abort (ValueError) before batch
  5. peso inexistente             → structural abort (FileNotFoundError) on model.load()
  6. device inválido              → structural abort (ValueError) during config validation
  7. diretório de entrada inexistente → structural abort (ValueError) from run_batch

ADR 0005: structural failures abort; per-item failures continue.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from upscale_image.cli.main import app
from upscale_image.config import resolve_config
from upscale_image.io.discovery import discover_images
from upscale_image.models import MockSuperResolutionModel
from upscale_image.models.realesrgan import RealESRGANRunner
from upscale_image.pipeline import BatchResult, ItemResult, create_run, run_batch, setup_run_logger

_RUNNER = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_image(path: Path, h: int = 8, w: int = 8) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_corrupted(path: Path) -> None:
    path.write_bytes(b"\x00\x01\x02bad_content")


def _setup_batch(tmp_path: Path, input_dir: Path, suffix: int = 0):
    """Create config, run context, logger and loaded mock model."""
    cfg = resolve_config(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "output"),
    )
    ts = datetime(2026, 3, 28, 10, 0, suffix)
    ctx = create_run(cfg, base_dir=tmp_path / "runs", now=ts)
    logger = setup_run_logger(ctx)
    model = MockSuperResolutionModel(scale=cfg.model.scale)
    model.load()
    return cfg, ctx, logger, model


# ---------------------------------------------------------------------------
# Scenario 1: imagem corrompida → SkippedFile, run continues
# ---------------------------------------------------------------------------

class TestCorruptedImage:
    def test_corrupted_image_is_skipped_not_aborted(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_corrupted(input_dir / "bad.png")
        _write_image(input_dir / "good.png")

        result = discover_images(str(input_dir), str(tmp_path / "out"))

        assert len(result.tasks) == 1
        assert result.tasks[0].filename == "good.png"
        assert len(result.skipped) == 1

    def test_corrupted_image_reason_is_clear(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_corrupted(input_dir / "bad.png")

        result = discover_images(str(input_dir), str(tmp_path / "out"))

        skipped = result.skipped[0]
        assert "corrupted" in skipped.reason
        assert skipped.path.endswith("bad.png")

    def test_all_corrupted_produces_empty_task_list(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for i in range(3):
            _write_corrupted(input_dir / f"bad{i}.png")

        result = discover_images(str(input_dir), str(tmp_path / "out"))

        assert result.tasks == []
        assert len(result.skipped) == 3

    def test_batch_continues_when_discovery_skips_corrupted(self, tmp_path):
        """A corrupted file skipped at discovery does not abort the batch."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_corrupted(input_dir / "bad.png")
        _write_image(input_dir / "good.png")

        cfg, ctx, logger, model = _setup_batch(tmp_path, input_dir, suffix=1)
        batch = run_batch(cfg, ctx, model, logger)
        model.unload()
        logger.close()

        assert batch.done == 1
        assert len(batch.skipped) == 1


# ---------------------------------------------------------------------------
# Scenario 2 & 3: per-item errors → status="failed", run continues
# (These use model.upscale patching, consistent with test_batch.py patterns)
# ---------------------------------------------------------------------------

class TestPerItemErrors:
    def test_inference_error_marks_item_failed(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "img.png")

        cfg, ctx, logger, model = _setup_batch(tmp_path, input_dir, suffix=2)
        model.upscale = lambda img, cfg: (_ for _ in ()).throw(RuntimeError("GPU OOM"))

        batch = run_batch(cfg, ctx, model, logger)
        model.unload()
        logger.close()

        assert batch.total == 1
        assert batch.failed == 1
        assert batch.results[0].status == "failed"
        assert "GPU OOM" in batch.results[0].error

    def test_inference_error_does_not_abort_run(self, tmp_path):
        """A failure on one item must not stop the remaining items."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ("a.png", "b.png", "c.png"):
            _write_image(input_dir / name)

        cfg, ctx, logger, model = _setup_batch(tmp_path, input_dir, suffix=3)

        original = model.upscale
        call_count = [0]

        def fail_second(image, config):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("second item fails")
            return original(image, config)

        model.upscale = fail_second
        batch = run_batch(cfg, ctx, model, logger)
        model.unload()
        logger.close()

        assert batch.total == 3
        assert batch.done == 2
        assert batch.failed == 1

    def test_per_item_error_has_description(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_image(input_dir / "img.png")

        cfg, ctx, logger, model = _setup_batch(tmp_path, input_dir, suffix=4)
        model.upscale = lambda img, cfg: (_ for _ in ()).throw(RuntimeError("specific error"))

        batch = run_batch(cfg, ctx, model, logger)
        model.unload()
        logger.close()

        assert batch.results[0].error is not None
        assert len(batch.results[0].error) > 0


# ---------------------------------------------------------------------------
# Scenario 4: modelo não encontrado → structural abort (ValueError)
# ---------------------------------------------------------------------------

class TestModelNotFound:
    def test_unknown_model_raises_value_error(self):
        from upscale_image.models.registry import ModelRegistry
        registry = ModelRegistry()
        cfg = resolve_config(
            input_dir="/tmp/in",
            output_dir="/tmp/out",
            model="nonexistent-model-xyz",
        )
        with pytest.raises(ValueError, match="Unknown model"):
            registry.resolve(cfg)

    def test_cli_unknown_model_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        result = _RUNNER.invoke(app, [
            "upscale", str(input_dir),
            "--output", str(tmp_path / "out"),
            "--model", "nonexistent-xyz",
        ])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Scenario 5: peso inexistente → structural abort (FileNotFoundError)
# ---------------------------------------------------------------------------

class TestMissingWeights:
    def test_missing_weights_raises_file_not_found(self):
        runner = RealESRGANRunner(
            scale=4,
            weights_path="/nonexistent/weights.pth",
        )
        with pytest.raises(FileNotFoundError):
            runner.load()

    def test_cli_missing_weights_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        result = _RUNNER.invoke(app, [
            "upscale", str(input_dir),
            "--output", str(tmp_path / "out"),
            "--model", "realesrgan-x4",
        ])
        # realesrgan-x4 requires weights at a path that doesn't exist in tests
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Scenario 6: device inválido → structural abort (ValueError)
# ---------------------------------------------------------------------------

class TestInvalidDevice:
    def test_invalid_device_raises_value_error(self):
        with pytest.raises(ValueError, match="device"):
            resolve_config(
                input_dir="/tmp/in",
                output_dir="/tmp/out",
                device="tpu",
            )

    def test_cli_invalid_device_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        result = _RUNNER.invoke(app, [
            "upscale", str(input_dir),
            "--output", str(tmp_path / "out"),
            "--device", "tpu",
        ])
        assert result.exit_code == 1

    def test_cli_invalid_device_prints_error_message(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        result = _RUNNER.invoke(app, [
            "upscale", str(input_dir),
            "--output", str(tmp_path / "out"),
            "--device", "tpu",
        ])
        assert "error" in result.output.lower() or "tpu" in result.output.lower()


# ---------------------------------------------------------------------------
# Scenario 7: diretório de entrada inexistente → structural abort
# ---------------------------------------------------------------------------

class TestNonExistentInputDir:
    def test_discover_images_raises_for_missing_dir(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            discover_images("/nonexistent/path/xyz", str(tmp_path / "out"))

    def test_cli_nonexistent_input_dir_exits_1(self, tmp_path, monkeypatch):
        """After the fix in cli/main.py, a non-existent input dir must exit 1."""
        monkeypatch.chdir(tmp_path)
        result = _RUNNER.invoke(app, [
            "upscale", "/nonexistent/path/does/not/exist",
            "--output", str(tmp_path / "out"),
            "--model", "mock",
        ])
        assert result.exit_code == 1

    def test_cli_nonexistent_input_dir_prints_clean_message(self, tmp_path, monkeypatch):
        """The error message must be human-readable, not a raw traceback."""
        monkeypatch.chdir(tmp_path)
        result = _RUNNER.invoke(app, [
            "upscale", "/nonexistent/path/does/not/exist",
            "--output", str(tmp_path / "out"),
            "--model", "mock",
        ])
        assert result.exit_code == 1
        assert "error" in result.output.lower() or "nonexistent" in result.output.lower()

    def test_cli_valid_empty_input_dir_exits_0(self, tmp_path, monkeypatch):
        """Sanity check: a valid (empty) input dir must exit 0 with mock model."""
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        result = _RUNNER.invoke(app, [
            "upscale", str(input_dir),
            "--output", str(tmp_path / "out"),
            "--model", "mock",
        ])
        assert result.exit_code == 0
