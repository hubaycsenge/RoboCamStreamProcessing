"""The processor interface.

A processor receives decoded frames and returns a JSON-serialisable dict that is
sent back to the robot in ``result.data``.  This is the seam where YOLO, MASt3R
and VGGT will attach — the wire protocol does not change when they do.

Contract
--------
* ``setup`` runs once in the worker thread before any frame.  Load weights and
  do warm-up passes here, not in ``__init__``, so that startup cost is paid on
  the thread that owns the CUDA context.
* ``process`` must return a dict or None.  It must not mutate ``frame.image``
  in place unless it owns the copy — other processors may share it later.
* ``process`` raising is not fatal: the server reports an unsuccessful result
  for that frame and carries on.
* ``close`` runs once at shutdown.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Frame:
    """One decoded frame on its way to a processor."""

    seq: int
    session_id: str
    image: np.ndarray
    # The header exactly as the client sent it.
    header: Dict[str, Any] = field(default_factory=dict)
    # Server monotonic clock, ns, when the payload came off the socket.
    recv_ts_ns: int = 0
    # Milliseconds spent decoding, measured by the IO thread.
    decode_ms: float = 0.0
    # Size of the encoded payload on the wire.
    payload_bytes: int = 0

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def channels(self) -> int:
        return int(self.image.shape[2]) if self.image.ndim == 3 else 1


class Processor(abc.ABC):
    """Base class for anything that consumes frames."""

    #: Name used in logs and reported back in ``result.processor``.
    name: str = "processor"

    def __init__(self, **options: Any) -> None:
        self.options = options

    def setup(self) -> None:
        """Called once in the worker thread before the first frame."""

    @abc.abstractmethod
    def process(self, frame: Frame) -> Optional[Dict[str, Any]]:
        """Handle one frame and return JSON-serialisable data for the robot."""

    def close(self) -> None:
        """Called once at shutdown."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"
