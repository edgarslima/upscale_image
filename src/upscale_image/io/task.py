"""Image task structure used throughout the pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImageTask:
    """Represents a single image to be processed.

    Constructed during input discovery; output_path is resolved here so
    downstream pipeline stages never need to compute it again.
    """

    input_path: str
    output_path: str
    filename: str
    status: str = "pending"  # pending | skipped | corrupted | done | failed


@dataclass
class SkippedFile:
    """A file that was found but excluded from the pipeline."""

    path: str
    reason: str
