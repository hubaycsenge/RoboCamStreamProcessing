"""The processor interface.

A processor receives decoded frames and returns a JSON-serialisable dict that is
sent back to the robot in ``result.data``.  This is the seam where YOLO, MASt3R
and VGGT will attach — the wire protocol does not change when they do.

A frame now arrives with more than an image on it.  Whatever the robot's other
streams last sent — a LiDAR revolution, an inertial burst, the pose the frame was
taken at, the occupancy grid the robot is building, its exit candidates — is
attached to it, already parsed and already summarised.  That is what makes the
right-hand column of the system diagram implementable inside a processor: a
stage that wants to compare its reconstruction against the robot's map has both
in hand, in one object, on one thread, without asking for anything.

Announcements
-------------
Returning a dict answers the frame.  Some products are not answers to a frame:
the patch Compare wants merged into the robot's map, the correction it offers
SLAM, the target the decision stage found.  Those are announcements, and a
processor emits one with :meth:`Frame.announce`, which the server sends ahead of
the frame's own result.  The server stamps the sequence number — the counters are
per-session and per-stream and belong to it, not to a worker thread that may be
one of several.

Contract
--------
* ``configure`` runs once, on the server's thread, before ``setup``.  It hands
  over the robot's LiDAR geometry and IMU thresholds so that a processor which
  projects scans into the image does not need its own copy of numbers that
  describe the hardware, and the whole server config for the stages that need
  the camera mount or the height slice.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..imu import ImuBatch
from ..lidar import Scan
from ..occupancy import Grid
from ..odometry import Odom


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
    # The pose the robot believes it was at when this frame was taken, on the
    # same terms as the two above: None when the robot sends no odometry, the
    # server has it disabled, or the pose went stale (odom.stale_after_ms).
    # Without it a reconstruction cannot be placed in the robot's map frame at
    # all, so a stage that compares the two must check rather than assume.
    odom: Optional[Odom] = None
    # Age of that pose in milliseconds when it was attached.
    odom_age_ms: float = 0.0
    # The robot's latest occupancy grid, or None if it has not sent one.  Not
    # copied per frame: this is the session's grid and every frame in flight
    # shares it, so a processor must not write to ``grid.cells`` in place.
    grid: Optional[Grid] = None
    # Age of that grid in milliseconds.  Much larger than the others by design —
    # a map is uploaded every few seconds, not every frame.
    grid_age_ms: float = 0.0
    # The robot's latest exit candidates, already validated.  Empty when it has
    # sent none; see robocam/mission.py for the field meanings.
    exits: List[Dict[str, Any]] = field(default_factory=list)
    # Which row of the system diagram this session is on: "t1" explore, "t2"
    # seek.  A processor that behaves identically in both may ignore it; one
    # that runs a decision stage must not, since the robot is only listening for
    # a ``found`` in t2.
    phase: str = "t1"
    # The mission, when there is one: at minimum {"target": "..."}.
    mission: Dict[str, Any] = field(default_factory=dict)
    # Announcements this frame produced, as (header, payload) pairs.  Appended
    # to with announce(); drained by the server after process() returns.
    announcements: List[Tuple[Dict[str, Any], bytes]] = field(default_factory=list)

    def announce(self, header: Dict[str, Any], payload: bytes = b"") -> None:
        """Queue an unsolicited message to the robot.

        For the products that are not answers to this frame — ``map_update``,
        ``pose_hint``, ``found``.  The server sends them *before* this frame's
        result, so that a robot acting on the result already holds whatever the
        announcement carried; and it overwrites ``seq``, because sequence
        numbers are per-session state and a worker thread has no business
        allocating them.
        """
        self.announcements.append((header, payload))

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

    def configure(self, lidar_cfg: Any, imu_cfg: Any = None, config: Any = None) -> None:
        """Called once with the server's config before ``setup``.

        Ignore it unless the processor needs to relate scan bearings to image
        columns; the mounting yaw and the lens field of view are properties of
        the robot, so they live in the server config rather than in each
        processor's options.  ``imu_cfg`` and ``config`` default to None so that
        a processor written before the IMU, or before the map, still satisfies
        this signature — the two arguments were added in that order and each
        would otherwise have broken every processor written against the previous
        one.

        ``config`` is the whole :class:`robocam.config.Config`.  A stage that
        compares a reconstruction against the robot's map needs the camera mount
        and the height slice, which describe the robot rather than the model, and
        duplicating them into processor options would let the two disagree.
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
