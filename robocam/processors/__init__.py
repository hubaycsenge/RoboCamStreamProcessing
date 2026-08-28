"""Processor registry.

Adding a model means writing a Processor subclass and registering it here; the
server picks it up via ``processor.name`` in the YAML config.

Registered models:
    "deep3r" -> CUT3R streaming reconstruction; a metric point cloud per frame

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

from .deep3r import Deep3RProcessor  # noqa: E402
from .fusion import FusionProcessor  # noqa: E402
from .noop import NoopProcessor  # noqa: E402
from .stats import StatsProcessor  # noqa: E402

register("stats", StatsProcessor)
register("noop", NoopProcessor)
register("fusion", FusionProcessor)
# Importing this one must stay cheap: torch and CUT3R load in setup(), on the
# worker thread, so a server running "stats" on a machine without CUDA is not
# stopped at startup by a processor it never selected.
register("deep3r", Deep3RProcessor)

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
    "Deep3RProcessor",
]
