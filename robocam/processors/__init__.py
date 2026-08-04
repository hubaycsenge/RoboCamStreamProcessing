"""Processor registry.

Adding a model means writing a Processor subclass and registering it here; the
server picks it up via ``processor.name`` in the YAML config.

Planned additions (see README):
    "yolo"   -> ultralytics detection, per-frame
    "mast3r" -> two-view matching / relative pose, on keyframe pairs
    "vggt"   -> multi-view geometry over a sliding keyframe window

A detector will most likely want to subclass or borrow from "fusion", which
already turns an image column into metres using the LDS-02 scan attached to the
frame.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .base import Frame, Processor

REGISTRY: Dict[str, Callable[..., Processor]] = {}


def register(name: str, factory: Callable[..., Processor]) -> None:
    if name in REGISTRY:
        raise ValueError(f"processor {name!r} is already registered")
    REGISTRY[name] = factory


def build(name: str, options: Optional[Dict[str, Any]] = None) -> Processor:
    """Instantiate a processor by registered name."""
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "(none)"
        raise KeyError(f"unknown processor {name!r}; registered: {known}")
    return REGISTRY[name](**(options or {}))


def available() -> list[str]:
    return sorted(REGISTRY)


# -- built-ins ---------------------------------------------------------------

from .fusion import FusionProcessor  # noqa: E402
from .noop import NoopProcessor  # noqa: E402
from .stats import StatsProcessor  # noqa: E402

register("stats", StatsProcessor)
register("noop", NoopProcessor)
register("fusion", FusionProcessor)

__all__ = [
    "Frame",
    "Processor",
    "REGISTRY",
    "register",
    "build",
    "available",
    "StatsProcessor",
    "NoopProcessor",
    "FusionProcessor",
]
