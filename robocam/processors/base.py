"""The processor interface.

A processor receives decoded frames and returns a JSON-serialisable dict that is
sent back to the robot in ``result.data``.  This is the seam where YOLO, MASt3R
and VGGT will attach — the wire protocol does not change when they do.

Contract
--------
* ``configure`` runs once, on the server's thread, before ``setup``.  It hands
  over the robot's LiDAR geometry and IMU thresholds so that a processor which
  projects scans into the image does not need its own copy of numbers that
  describe the hardware.
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

from ..imu import ImuBatch
from ..lidar import Scan


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
    # The most recent LiDAR revolution, if one arrived recently enough to still
    # describe the same world (lidar.stale_after_ms).  None whenever the robot
    # has no scanner, the server has LiDAR disabled, or the scan went stale —
    # a processor must handle that case rather than assume ranges are there.
    # Its ``summary`` is already computed; see robocam/lidar.py.
    scan: Optional[Scan] = None
    # Age of that scan in milliseconds when it was attached, on the server's
    # clock.  Nonzero even for a fresh scan: at 5 Hz, ~100 ms is normal.
    scan_age_ms: float = 0.0
    # The most recent burst of inertial samples from the OpenCR, on the same
    # terms as ``scan``: None whenever the robot has no IMU, the server has it
    # disabled, or the burst went stale (imu.stale_after_ms).  Its ``summary``
    # is already computed; see robocam/imu.py.
    imu: Optional[ImuBatch] = None
    # Age of that burst in milliseconds when it was attached.  At ~100 Hz and a
    # burst per frame this is small; tens of milliseconds is normal.
    imu_age_ms: float = 0.0

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

    def configure(self, lidar_cfg: Any, imu_cfg: Any = None) -> None:
        """Called once with the server's sensor config before ``setup``.

        Ignore it unless the processor needs to relate scan bearings to image
        columns; the mounting yaw and the lens field of view are properties of
        the robot, so they live in the server config rather than in each
        processor's options.  ``imu_cfg`` defaults to None so that a processor
        written before the IMU existed still satisfies this signature.
        """

    def setup(self) -> None:
        """Called once in the worker thread before the first frame."""

    @abc.abstractmethod
    def process(self, frame: Frame) -> Optional[Dict[str, Any]]:
        """Handle one frame and return JSON-serialisable data for the robot."""

    def close(self) -> None:
        """Called once at shutdown."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"
