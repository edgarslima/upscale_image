"""Tests for input discovery: filtering, validation, ordering, task structure."""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from upscale_image.io import ImageTask, SkippedFile, discover_images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_valid_png(path: Path) -> None:
    """Write a real 4x4 RGB PNG that OpenCV can decode."""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_corrupted_file(path: Path) -> None:
    """Write a file with a valid extension but invalid image content."""
    path.write_bytes(b"\x00\x01\x02\x03\xff\xfe")


# ---------------------------------------------------------------------------
# Basic discovery
# ---------------------------------------------------------------------------

def test_empty_directory(tmp_path):
    result = discover_images(str(tmp_path), str(tmp_path / "out"))
    assert result.tasks == []
    assert result.skipped == []


def test_valid_images_discovered(tmp_path):
    _write_valid_png(tmp_path / "a.png")
    _write_valid_png(tmp_path / "b.png")
    result = discover_images(str(tmp_path), str(tmp_path / "out"))
    assert len(result.tasks) == 2
    assert result.skipped == []


def test_unsupported_extension_skipped(tmp_path):
    _write_valid_png(tmp_path / "img.png")
    (tmp_path / "doc.txt").write_text("hello")
    (tmp_path / "data.csv").write_text("a,b")
    result = discover_images(str(tmp_path), str(tmp_path / "out"))
    assert len(result.tasks) == 1
    assert len(result.skipped) == 2
    reasons = {s.reason for s in result.skipped}
    assert all("unsupported extension" in r for r in reasons)


def test_corrupted_file_skipped(tmp_path):
    _write_valid_png(tmp_path / "good.png")
    _write_corrupted_file(tmp_path / "bad.png")
    result = discover_images(str(tmp_path), str(tmp_path / "out"))
    assert len(result.tasks) == 1
    assert len(result.skipped) == 1
    assert "corrupted" in result.skipped[0].reason


def test_mixed_directory(tmp_path):
    _write_valid_png(tmp_path / "valid.jpg")
    _write_corrupted_file(tmp_path / "broken.png")
    (tmp_path / "readme.txt").write_text("ignore me")
    result = discover_images(str(tmp_path), str(tmp_path / "out"))
    assert len(result.tasks) == 1
    assert len(result.skipped) == 2


# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"])
def test_all_supported_extensions_accepted(tmp_path, ext):
    img_path = tmp_path / f"image{ext}"
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    result = discover_images(str(tmp_path), str(tmp_path / "out"))
    assert len(result.tasks) == 1, f"Expected 1 task for extension {ext}"


# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------

def test_order_is_stable(tmp_path):
    names = ["c.png", "a.png", "b.png"]
    for name in names:
        _write_valid_png(tmp_path / name)
    result = discover_images(str(tmp_path), str(tmp_path / "out"))
    filenames = [t.filename for t in result.tasks]
    assert filenames == ["a.png", "b.png", "c.png"]


def test_order_is_case_insensitive(tmp_path):
    _write_valid_png(tmp_path / "Z.png")
    _write_valid_png(tmp_path / "a.png")
    _write_valid_png(tmp_path / "M.png")
    result = discover_images(str(tmp_path), str(tmp_path / "out"))
    filenames = [t.filename for t in result.tasks]
    assert filenames == sorted(filenames, key=str.lower)


# ---------------------------------------------------------------------------
# ImageTask structure
# ---------------------------------------------------------------------------

def test_task_fields_populated(tmp_path):
    out_dir = tmp_path / "out"
    _write_valid_png(tmp_path / "photo.jpg")
    result = discover_images(str(tmp_path), str(out_dir))
    task = result.tasks[0]
    assert task.filename == "photo.jpg"
    assert task.input_path.endswith("photo.jpg")
    assert task.output_path.endswith("photo.png")  # always .png output
    assert str(out_dir) in task.output_path
    assert task.status == "pending"


def test_output_path_uses_png_extension(tmp_path):
    _write_valid_png(tmp_path / "img.bmp")
    result = discover_images(str(tmp_path), str(tmp_path / "out"))
    assert result.tasks[0].output_path.endswith(".png")


# ---------------------------------------------------------------------------
# Structural errors
# ---------------------------------------------------------------------------

def test_nonexistent_input_dir_raises():
    with pytest.raises(ValueError, match="does not exist"):
        discover_images("/nonexistent/path/xyz", "/out")


def test_input_dir_is_file_raises(tmp_path):
    f = tmp_path / "afile.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        discover_images(str(f), str(tmp_path / "out"))
