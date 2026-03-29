"""Model registry: maps logical names to runner factories.

Rules (ADR 0004):
- The pipeline resolves runners by name; no if/else per model.
- Registration is explicit, never an import side-effect.
- An unknown name raises ValueError immediately (fail-early).

Usage::

    registry = ModelRegistry()
    registry.register("mock", lambda scale: MockSuperResolutionModel(scale))
    model = registry.resolve(config)   # returns an unloaded SuperResolutionModel
    model.load()
    ...
    model.unload()

A module-level default registry is provided via ``register()`` and
``resolve_model()`` for production use.
"""

from __future__ import annotations

from typing import Callable

from upscale_image.config import AppConfig
from upscale_image.models.base import SuperResolutionModel

# Factory: receives scale and returns an *unloaded* runner instance.
FactoryFn = Callable[[int], SuperResolutionModel]


class ModelRegistry:
    """Maps logical model names to runner factories.

    Each entry is a ``FactoryFn`` — a callable ``(scale: int) ->
    SuperResolutionModel`` that creates an unloaded runner instance.
    """

    def __init__(self) -> None:
        self._factories: dict[str, FactoryFn] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, factory: FactoryFn) -> None:
        """Register *factory* under *name*.

        Args:
            name:    Logical model name (must match the value used in config).
            factory: Callable ``(scale: int) -> SuperResolutionModel``.

        Raises:
            ValueError: If *name* is already registered.
        """
        if name in self._factories:
            raise ValueError(
                f"Model {name!r} is already registered. "
                "Use a different name or deregister first."
            )
        self._factories[name] = factory

    def deregister(self, name: str) -> None:
        """Remove *name* from the registry (useful in tests)."""
        self._factories.pop(name, None)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, config: AppConfig) -> SuperResolutionModel:
        """Instantiate and return the runner for ``config.model.name``.

        The returned instance is **not** loaded; the caller must invoke
        ``.load()`` before running inference.

        Args:
            config: Resolved :class:`~upscale_image.config.AppConfig`.

        Returns:
            An unloaded :class:`~upscale_image.models.base.SuperResolutionModel`.

        Raises:
            ValueError: If ``config.model.name`` is not registered.
        """
        name = config.model.name
        factory = self._factories.get(name)
        if factory is None:
            available = sorted(self._factories)
            raise ValueError(
                f"Unknown model {name!r}. "
                f"Registered models: {available or ['(none)']}"
            )
        return factory(config.model.scale)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def available(self) -> list[str]:
        """Return sorted list of registered model names."""
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self._factories


# ---------------------------------------------------------------------------
# Default registry — populated explicitly, never by import side-effects
# ---------------------------------------------------------------------------

_default_registry = ModelRegistry()


def register(name: str, factory: FactoryFn) -> None:
    """Register *factory* in the default registry."""
    _default_registry.register(name, factory)


def resolve_model(config: AppConfig) -> SuperResolutionModel:
    """Resolve a runner from the default registry using *config*."""
    return _default_registry.resolve(config)


def available_models() -> list[str]:
    """Return sorted names of models in the default registry."""
    return _default_registry.available()


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------
# Import here to avoid circular imports; registration is explicit, not
# triggered merely by importing models.base or models.mock elsewhere.

from upscale_image.models.bicubic import BicubicRunner  # noqa: E402
from upscale_image.models.mock import MockSuperResolutionModel  # noqa: E402
from upscale_image.models.realesrgan import RealESRGANRunner  # noqa: E402

_default_registry.register(
    "mock",
    lambda scale: MockSuperResolutionModel(scale=scale),
)

_default_registry.register(
    "bicubic",
    lambda scale: BicubicRunner(scale=scale),
)

_default_registry.register(
    "realesrgan-x4",
    lambda scale: RealESRGANRunner(
        scale=scale,
        weights_path=f"weights/realesrgan-x4.pth",
    ),
)
