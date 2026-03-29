"""Compatibility shim for torchvision / basicsr import mismatch.

basicsr 1.4.x references ``torchvision.transforms.functional_tensor`` which
was removed in torchvision >= 0.16. The functions were merged into
``torchvision.transforms.functional``.

Import this module before any basicsr import to patch sys.modules.
"""

from __future__ import annotations

import sys


def _apply_torchvision_shim() -> None:
    """Register functional_tensor alias in sys.modules if missing."""
    key = "torchvision.transforms.functional_tensor"
    if key not in sys.modules:
        import torchvision.transforms.functional as _f  # noqa: PLC0415
        sys.modules[key] = _f  # type: ignore[assignment]


_apply_torchvision_shim()
