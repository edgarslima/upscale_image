"""Input discovery: scan directory, filter extensions, validate readability.

Rules (ADR 0005):
- Order is deterministic (sorted by filename, case-insensitive).
- Unsupported extensions are silently skipped (logged as SkippedFile).
- Corrupted/unreadable files are marked as SkippedFile, never raise globally.
- Missing or non-directory input_dir raises ValueError (structural error → abort).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2

from upscale_image.io.task import ImageTask, SkippedFile

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
)


@dataclass
class DiscoveryResult:
    tasks: list[ImageTask]
    skipped: list[SkippedFile]


def _is_readable(path: str) -> bool:
    """Return True if OpenCV can decode the file (validates content, not just extension)."""
    img = cv2.imread(path)
    return img is not None


def _build_output_path(input_path: Path, output_dir: Path) -> str:
    """Compute output path preserving filename, forcing .png extension."""
    return str(output_dir / (input_path.stem + ".png"))


def discover_images(input_dir: str, output_dir: str) -> DiscoveryResult:
    """Scan *input_dir* and build a stable list of ImageTasks ready for the pipeline.

    Args:
        input_dir:  Path to the directory containing input images.
        output_dir: Path to the directory where processed images will be saved.
                    The directory does not need to exist at this point.

    Returns:
        DiscoveryResult with a sorted list of valid ImageTasks and a list of
        SkippedFiles describing every excluded file.

    Raises:
        ValueError: If input_dir does not exist or is not a directory.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.exists():
        raise ValueError(f"input_dir does not exist: {input_dir!r}")
    if not in_path.is_dir():
        raise ValueError(f"input_dir is not a directory: {input_dir!r}")

    tasks: list[ImageTask] = []
    skipped: list[SkippedFile] = []

    # Collect all regular files, sorted deterministically by lowercased filename.
    candidates = sorted(
        (p for p in in_path.iterdir() if p.is_file()),
        key=lambda p: p.name.lower(),
    )

    for file_path in candidates:
        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            skipped.append(SkippedFile(path=str(file_path), reason=f"unsupported extension {ext!r}"))
            continue

        if not _is_readable(str(file_path)):
            skipped.append(SkippedFile(path=str(file_path), reason="file is unreadable or corrupted"))
            continue

        tasks.append(
            ImageTask(
                input_path=str(file_path),
                output_path=_build_output_path(file_path, out_path),
                filename=file_path.name,
            )
        )

    return DiscoveryResult(tasks=tasks, skipped=skipped)
