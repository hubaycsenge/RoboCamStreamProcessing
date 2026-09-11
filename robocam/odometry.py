"""Odometry poses: decoding, and the small amount of geometry worth doing once.

The robot already knows where it is.  This module exists so the server can know
it too, in the same frame, because that is the precondition for every arrow in
the right-hand half of the system diagram: a point cloud is a cloud of numbers
until something says where the camera was when it was taken.

Two frames, and the difference matters
--------------------------------------
A pose in ``odom`` is dead reckoning — smooth, continuous, and wrong by however
much the wheels have slipped since the robot started.  A pose in ``map`` is that
same pose plus SLAM's accumulated correction: it jumps when a loop closes, and it
is the one the occupancy grid is drawn in.  They can differ by a metre after a
few minutes on a carpet.

So a cloud placed with an ``odom`` pose and compared against a ``map`` grid finds
differences everywhere and calls them obstacles.  :func:`decode_odom` therefore
keeps ``frame`` as a first-class field and the server refuses the comparison
unless it matches the grid's, rather than assuming the robot sent the right one.
Refusing is not pedantry — a silent mismatch produces a map that looks plausible
and is wrong, which is the worst of the available outcomes.

Planar, deliberately
--------------------
The robot drives on a floor.  Everything downstream of here — the grid, the
exits, the goal poses — is (x, y, yaw), so this module reduces a full 3D pose to
that as early as possible and keeps ``z`` only to report it.  The one place the
third dimension is not dropped is the cloud itself, where height is exactly what
the LiDAR could not see; see :mod:`robocam.compare`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

#: A pose whose translation exceeds this is rejected.  Not a map size limit — a
#: sanity limit.  10 km from the origin means a parse or a units error (metres
#: read as millimetres is the classic), and letting it through would place a
#: cloud so far from the grid that the comparison finds nothing and reports
#: healthily that there is nothing to report.
MAX_ABS_M = 10_000.0


class OdomError(Exception):
    """Raised when an odometry header does not parse."""


@dataclass
class Odom:
    """One decoded pose.

    ``yaw`` is always present and always in radians; when the sender supplied a
    quaternion it is derived from that rather than from the sender's own ``yaw``
    field, so there is exactly one Euler convention in this system and it is the
    one written down in :func:`yaw_from_quaternion`.
    """

    x: float
    y: float
    yaw: float
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    seq: int = -1
    frame: str = "map"
    child_frame: str = "base_link"
    quaternion: Optional[Tuple[float, float, float, float]] = None
    pose_cov_diag: Optional[Tuple[float, ...]] = None
    source: str = ""
    # Server monotonic clock, ns, when the pose came off the socket.  Frame
    # association is measured against this and never against a client clock.
    recv_ts_ns: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)
    #: ``T_child_optical``: where the camera was relative to ``child_frame`` for
    #: the one frame this pose was attached to, when the robot said.  It takes
    #: the place of the configured :class:`robocam.compare.CameraMount`, and
    #: exists because the Mecanumbot's camera is on a neck that moves.  None on
    #: every stream pose; see :func:`decode_frame_pose`.
    camera: Optional[np.ndarray] = None
    camera_source: str = ""

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def matrix2d(self) -> np.ndarray:
        """The 3x3 planar transform taking points from ``child_frame`` to ``frame``.

        This is what puts a cloud expressed in the camera's world into the
        robot's map, and it is a matrix rather than three numbers because the
        alternative is writing the same two sines and cosines out at every call
        site, which is where the sign errors live.
        """
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s, self.x],
                         [s, c, self.y],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


def yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    """Yaw about +z, radians, from a ROS-ordered quaternion.

    The standard ZYX extraction.  Written once and used everywhere so that the
    server cannot disagree with itself about which way is positive; the robot's
    own convention is REP-103 (x forward, y left, z up, yaw counter-clockwise)
    and this matches it.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a: float) -> float:
    """Fold an angle into [-pi, pi], which is what atan2 returns.

    Every angular difference in this package goes through here.  A yaw error of
    ``2*pi - 0.01`` is an error of 0.01 radians, and code that averages the
    unwrapped version produces a correction pointing the wrong way.
    """
    return math.atan2(math.sin(a), math.cos(a))


def decode_odom(header: Dict[str, Any], recv_ts_ns: int = 0) -> Odom:
    """Turn one ``odom`` header into an :class:`Odom`.

    There is no payload to parse, so this is all validation — and validation is
    most of the value.  A pose is three numbers that look fine whatever they
    contain; the failures worth catching (NaN from an uninitialised filter, a
    frame name that does not match the map, a translation in the wrong units)
    all look like ordinary floats.
    """
    try:
        x = float(header.get("x", 0.0))
        y = float(header.get("y", 0.0))
        z = float(header.get("z", 0.0))
    except (TypeError, ValueError) as exc:
        raise OdomError(f"pose is not numeric: {exc}") from exc

    for name, value in (("x", x), ("y", y), ("z", z)):
        if not math.isfinite(value):
            raise OdomError(f"{name} is {value}")
        if abs(value) > MAX_ABS_M:
            raise OdomError(
                f"{name} = {value} m is beyond the {MAX_ABS_M:.0f} m sanity limit; "
                "millimetres sent as metres look exactly like this"
            )

    quaternion = None
    if all(k in header for k in ("qw", "qx", "qy", "qz")):
        try:
            quat = (float(header["qw"]), float(header["qx"]),
                    float(header["qy"]), float(header["qz"]))
        except (TypeError, ValueError) as exc:
            raise OdomError(f"quaternion is not numeric: {exc}") from exc
        norm = math.sqrt(sum(v * v for v in quat))
        if not math.isfinite(norm) or norm < 1e-6:
            raise OdomError(f"quaternion has norm {norm}, which is not a rotation")
        # Normalise rather than reject a slightly-off quaternion: a filter's
        # output drifts from unit norm by parts per million and that is not an
        # error, while a norm of zero (an uninitialised message) is.
        quaternion = tuple(v / norm for v in quat)
        yaw = yaw_from_quaternion(*quaternion)
    else:
        try:
            yaw = float(header.get("yaw", 0.0))
        except (TypeError, ValueError) as exc:
            raise OdomError(f"yaw is not numeric: {exc}") from exc
        if not math.isfinite(yaw):
            raise OdomError(f"yaw is {yaw}")
        yaw = wrap_angle(yaw)

    def _vel(key: str) -> float:
        try:
            v = float(header.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        # A non-finite velocity is not worth refusing the whole pose over: the
        # pose is what the comparison needs, and the twist is only reported.
        return v if math.isfinite(v) else 0.0

    cov = header.get("pose_cov_diag")
    cov_tuple = None
    if isinstance(cov, (list, tuple)) and cov:
        try:
            cov_tuple = tuple(float(v) for v in cov)
        except (TypeError, ValueError):
            cov_tuple = None

    frame = str(header.get("frame", "map") or "map")
    return Odom(
        x=x, y=y, z=z, yaw=yaw,
        vx=_vel("vx"), vy=_vel("vy"), vyaw=_vel("vyaw"),
        seq=int(header.get("seq", -1)),
        frame=frame,
        child_frame=str(header.get("child_frame", "base_link") or "base_link"),
        quaternion=quaternion,
        pose_cov_diag=cov_tuple,
        source=str(header.get("source", "")),
        recv_ts_ns=recv_ts_ns,
    )


#: A camera further than this from the base frame's origin is not on the robot.
#: The Mecanumbot's is about 25 cm from ``base_link``; two metres leaves room for
#: any mast and still catches millimetres sent as metres.
MAX_CAMERA_OFFSET_M = 2.0


def _matrix_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def decode_camera(block: Dict[str, Any], base_frame: str) -> np.ndarray:
    """Turn a pose's ``camera`` block into the 4x4 ``T_base_optical``.

    This is the matrix :meth:`robocam.compare.CameraMount.matrix` would return,
    supplied by the robot for one frame instead of by the config for all of
    them.  It is held to the same convention: the child frame is **optical** (x
    right, y down, z forward), and no flip is applied here, because the robot
    has already applied it.

    ``frame`` must name the base the pose itself names.  An extrinsic measured
    from ``base_footprint`` composed onto a ``base_link`` pose is a centimetre
    out on this robot and arbitrarily out on another, and nothing downstream
    could tell, so a mismatch is refused rather than assumed away.
    """
    if not isinstance(block, dict):
        raise OdomError(f"camera is {type(block).__name__}, expected an object")
    frame = str(block.get("frame", "") or "")
    if frame != base_frame:
        raise OdomError(
            f"camera pose is relative to {frame!r} but the pose is of {base_frame!r}; "
            "the two cannot be composed")
    try:
        values = [float(block[k]) for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    except KeyError as exc:
        raise OdomError(f"camera pose is missing {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise OdomError(f"camera pose is not numeric: {exc}") from exc
    if not all(math.isfinite(v) for v in values):
        raise OdomError(f"camera pose is not finite: {values}")
    x, y, z, qw, qx, qy, qz = values
    if math.sqrt(x * x + y * y + z * z) > MAX_CAMERA_OFFSET_M:
        raise OdomError(
            f"camera is {math.sqrt(x * x + y * y + z * z):.1f} m from {base_frame!r}, "
            f"beyond the {MAX_CAMERA_OFFSET_M:.0f} m a camera on the robot can be")
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-6:
        raise OdomError(f"camera quaternion has norm {norm}, which is not a rotation")
    out = np.eye(4)
    out[:3, :3] = _matrix_from_quaternion(qw / norm, qx / norm, qy / norm, qz / norm)
    out[:3, 3] = (x, y, z)
    return out


def decode_frame_pose(pose: Any, seq: int = -1, recv_ts_ns: int = 0) -> Tuple[Odom, float]:
    """Decode the pose a robot attached to one frame, and how old it was.

    The odom stream is paired with frames by arrival on this server, so the
    pose a frame gets from it is whichever arrived last -- up to one pose period
    away from the instant the image was taken, and blind to anything that moved
    the camera but not the base.  A robot that knows better attaches the pose to
    the frame itself.  It goes through :func:`decode_odom`, so it meets every
    check a stream pose does, plus the ``camera`` block when there is one.

    The age is the robot's own figure for how far the newest input it used is
    from the image's timestamp, measured on its clock and reported, never
    compared with this server's.
    """
    if not isinstance(pose, dict):
        raise OdomError(f"frame pose is {type(pose).__name__}, expected an object")
    header = dict(pose)
    header.setdefault("seq", seq)
    odom = decode_odom(header, recv_ts_ns=recv_ts_ns)
    try:
        age_ms = abs(float(pose.get("age_ms", 0.0) or 0.0))
    except (TypeError, ValueError) as exc:
        raise OdomError(f"age_ms is not numeric: {exc}") from exc
    if not math.isfinite(age_ms):
        raise OdomError(f"age_ms is {age_ms}")
    if pose.get("camera") is not None:
        odom.camera = decode_camera(pose["camera"], base_frame=odom.child_frame)
        odom.camera_source = str(pose["camera"].get("source", "") or "robot")
    return odom, age_ms


def analyse(odom: Odom, previous: Optional[Odom] = None,
            still_speed_ms: float = 0.02, still_yaw_rate_dps: float = 2.0) -> Dict[str, Any]:
    """The handful of numbers worth returning with every pose.

    ``moved_m`` and ``turned_deg`` come from the previous pose rather than from
    the reported twist, because they are what the reconstruction actually cares
    about: CUT3R's state is stepped per frame, and a robot that has not moved
    between two frames gives it no new parallax whatever the wheels report.

    Both are also the cheapest available check that the odometry is alive at all.
    A source that has frozen reports a perfectly plausible pose forever, and the
    only thing that distinguishes it from a stationary robot is that a stationary
    robot's IMU is stationary too.
    """
    summary: Dict[str, Any] = {
        "x": round(odom.x, 4),
        "y": round(odom.y, 4),
        "yaw_deg": round(math.degrees(odom.yaw), 2),
        "frame": odom.frame,
        "speed_ms": round(odom.speed(), 4),
        "yaw_rate_dps": round(math.degrees(odom.vyaw), 2),
    }
    if previous is not None:
        moved = math.hypot(odom.x - previous.x, odom.y - previous.y)
        turned = wrap_angle(odom.yaw - previous.yaw)
        dt_s = (odom.recv_ts_ns - previous.recv_ts_ns) / 1e9
        summary["moved_m"] = round(moved, 4)
        summary["turned_deg"] = round(math.degrees(turned), 2)
        if dt_s > 0:
            summary["dt_ms"] = round(dt_s * 1000.0, 1)
            summary["measured_speed_ms"] = round(moved / dt_s, 4)
    summary["still"] = (
        odom.speed() < still_speed_ms
        and abs(math.degrees(odom.vyaw)) < still_yaw_rate_dps
    )
    return summary


def compose(base: Odom, dx: float, dy: float, dyaw: float) -> Tuple[float, float, float]:
    """Apply a correction expressed in ``base``'s own frame, returning (x, y, yaw).

    This is what a robot does with a ``pose_hint`` if it decides to accept one.
    It lives here rather than on the robot so that both ends compute it the same
    way: the correction is a rigid transform *of the map frame*, so the
    translation is added in map coordinates and only the heading composes.
    """
    return (base.x + dx, base.y + dy, wrap_angle(base.yaw + dyaw))
