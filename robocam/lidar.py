"""LiDAR scans: decoding, and the geometry worth computing on every one.

The robot carries an LDS-02 — a planar 360° scanner, roughly 5 Hz, 0.12–12 m,
360 points per revolution.  It measures one horizontal slice at its own mounting
height, so it says nothing about what is above or below that plane; a table top
at 70 cm is invisible to it and a table leg is not.  Everything here is written
with that limitation in mind: the scan is treated as a source of *ranges along
bearings*, never as a description of the scene.

Two things happen to every scan.

**Analysis** (:func:`analyse`) reduces 360 ranges to the handful of numbers a
robot steers on — nearest obstacle, per-sector clearance, the widest free
direction.  It costs about 60 µs, so the server runs it inline on the IO thread
and answers the scan immediately rather than queueing it behind a model.

**Projection** (:func:`project_columns`, :func:`range_for_box`) maps bearings
onto image columns, which is the whole reason for having both sensors on one
link.  A YOLO box gives you *what* and *where in the image*; the scan turns the
second half of that into metres.  The mapping is a pinhole assumption plus the
mounting yaw and nothing else — the two sensors are not calibrated to each other
here, so treat the ranges it returns as "the obstacle in roughly that direction",
good to a few degrees, not as a depth measurement of that object.

Angle convention throughout: bearings are radians, counter-clockwise positive,
zero along the *camera's* optical axis after ``mount_yaw`` is applied.  So a
negative bearing is to the left of the image centre in a normal (non-mirrored)
camera, and ``bearing_deg`` in the output can be compared directly against an
image column without further sign juggling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import wire

# A revolution of an LDS-02 is 360 points.  The cap is a guard against a corrupt
# count field turning into a large allocation, not a real device limit.
MAX_POINTS = 8192

TWO_PI = 2.0 * math.pi


class ScanError(Exception):
    """Raised when a scan header or payload does not parse."""


@dataclass
class Scan:
    """One decoded revolution.

    ``ranges`` is metres with NaN for every no-return, which is the honest
    representation: an LDS-02 reports 0 both for "nothing within range" and for
    "the return was too weak", and those must not average in as a distance of
    zero.  Every consumer here masks on ``valid`` rather than trusting arithmetic
    to do the right thing with the gaps.
    """

    ranges: np.ndarray            # float32 metres, NaN where there was no return
    bearings: np.ndarray          # float32 radians, CCW, zero = sensor forward
    seq: int = -1
    angle_min: float = 0.0
    angle_increment: float = 0.0
    range_min: float = 0.0
    range_max: float = 0.0
    scan_time: float = 0.0
    intensities: Optional[np.ndarray] = None
    source: str = ""
    # Server monotonic clock, ns, when the scan came off the socket.  This is
    # what frame/scan association is measured against — never a client clock.
    recv_ts_ns: int = 0
    # Filled in by the server: the result of analyse(), computed once on the IO
    # thread so that processors and the scan reply share one answer.
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.ranges.shape[0])

    @property
    def valid(self) -> np.ndarray:
        """Boolean mask of points that carry a real measurement."""
        return np.isfinite(self.ranges)

    def bearings_from(self, yaw_rad: float = 0.0) -> np.ndarray:
        """Bearings re-referenced to an axis ``yaw_rad`` off the sensor's zero."""
        return wrap_pi(self.bearings - yaw_rad)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Scan seq={self.seq} points={self.count} valid={int(self.valid.sum())}>"


def wrap_pi(angles):
    """Wrap radians into (-π, π].  Accepts a scalar or an array."""
    return (np.asarray(angles) + math.pi) % TWO_PI - math.pi


# ---------------------------------------------------------------------------
# Wire <-> Scan
# ---------------------------------------------------------------------------


def decode_scan(header: Dict[str, Any], payload: bytes, recv_ts_ns: int = 0) -> Scan:
    """Turn a ``scan`` message into a :class:`Scan`.

    Raises :class:`ScanError` on anything malformed; the caller turns that into
    an unsuccessful scan result rather than dropping the session, for the same
    reason a corrupt JPEG does not: one bad revolution is not a broken robot.
    """
    encoding = str(header.get("encoding", wire.SCAN_ENC_U16_MM))
    try:
        count = int(header.get("count", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ScanError(f"count is not an integer: {header.get('count')!r}") from exc

    if count <= 0:
        raise ScanError(f"scan has no points (count={count})")
    if count > MAX_POINTS:
        raise ScanError(f"count {count} exceeds the {MAX_POINTS} point limit")

    want_intensities = bool(header.get("intensities", False))

    if encoding == wire.SCAN_ENC_U16_MM:
        need = count * 2
        if len(payload) < need:
            raise ScanError(f"payload is {len(payload)} bytes, expected at least {need} for {count} u16 points")
        raw = np.frombuffer(payload, dtype="<u2", count=count).astype(np.float32) / 1000.0
        ranges = np.where(raw > 0.0, raw, np.nan).astype(np.float32)
    elif encoding == wire.SCAN_ENC_F32_M:
        need = count * 4
        if len(payload) < need:
            raise ScanError(f"payload is {len(payload)} bytes, expected at least {need} for {count} f32 points")
        raw = np.frombuffer(payload, dtype="<f4", count=count).astype(np.float32)
        # ROS uses inf for "nothing out there" and nan for "no reading"; both
        # arrive here as gaps, which is what they are.
        ranges = np.where(np.isfinite(raw) & (raw > 0.0), raw, np.nan).astype(np.float32)
    else:
        raise ScanError(
            f"unsupported scan encoding {encoding!r}; "
            f"server supports {', '.join(wire.SUPPORTED_SCAN_ENCODINGS)}"
        )

    intensities = None
    if want_intensities:
        offset = need
        if len(payload) >= offset + count:
            intensities = np.frombuffer(payload, dtype=np.uint8, count=count, offset=offset).copy()

    range_min = _as_float(header.get("range_min"), 0.0)
    range_max = _as_float(header.get("range_max"), 0.0)
    # Trusting a return from outside the device's rated window is how a spurious
    # 4 cm echo becomes an emergency stop.  Discard rather than clamp: a clamped
    # value would be indistinguishable from a real reading at the limit.
    if range_min > 0.0:
        ranges = np.where(ranges >= range_min, ranges, np.nan)
    if range_max > 0.0:
        ranges = np.where(ranges <= range_max, ranges, np.nan)

    angle_min = _as_float(header.get("angle_min"), 0.0)
    angle_increment = _as_float(header.get("angle_increment"), 0.0)
    if angle_increment == 0.0:
        # The common case — a full even revolution — so clients do not have to
        # carry a float that is always the same thing.
        angle_increment = TWO_PI / count

    bearings = wrap_pi(angle_min + angle_increment * np.arange(count, dtype=np.float32)).astype(np.float32)

    return Scan(
        ranges=ranges,
        bearings=bearings,
        seq=int(header.get("seq", -1) or -1),
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        scan_time=_as_float(header.get("scan_time"), 0.0),
        intensities=intensities,
        source=str(header.get("source", "")),
        recv_ts_ns=recv_ts_ns,
    )


def encode_scan_payload(ranges_m, intensities=None) -> bytes:
    """Pack metres into the default ``u16mm`` payload.

    The deployable client has its own copy of this — it must not import the
    server package — so this one exists for tests and for anything server-side
    that wants to replay a scan.
    """
    arr = np.asarray(ranges_m, dtype=np.float32)
    mm = np.where(np.isfinite(arr) & (arr > 0.0), arr * 1000.0, 0.0)
    # 65.535 m is beyond any device we would put on this robot, but clipping
    # keeps a bad reading from wrapping around into a very short one.
    packed = np.clip(np.rint(mm), 0, 65535).astype("<u2").tobytes()
    if intensities is None:
        return packed
    return packed + np.asarray(intensities, dtype=np.uint8).tobytes()


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyse(
    scan: Scan,
    *,
    sectors: int = 12,
    obstacle_m: float = 0.5,
    clear_m: float = 1.0,
    front_deg: float = 60.0,
    min_free_deg: float = 15.0,
    mount_yaw_deg: float = 0.0,
    hz: float = 0.0,
) -> Dict[str, Any]:
    """Reduce a revolution to what a robot can act on.

    Everything is reported in the camera-referenced frame (see the module
    docstring), in degrees, because that is what a human reads in a log line and
    what pairs with an image column.
    """
    yaw = math.radians(mount_yaw_deg)
    bearings = scan.bearings_from(yaw)
    ranges = scan.ranges
    valid = np.isfinite(ranges)
    n_valid = int(valid.sum())

    out: Dict[str, Any] = {
        "points": scan.count,
        "valid": n_valid,
        # Low coverage is the LiDAR equivalent of a lens cap: a scanner facing a
        # window or spinning below its rated speed returns mostly nothing.
        "coverage": round(n_valid / scan.count, 3) if scan.count else 0.0,
        "sector_width_deg": round(360.0 / sectors, 2) if sectors > 0 else 0.0,
        "hz": round(hz, 2),
    }
    if scan.source:
        out["source"] = scan.source

    if n_valid == 0:
        # Not an error — a scanner in the middle of a large empty room does this
        # legitimately — but nothing below would mean anything, so say so plainly.
        out.update({
            "blind": True,
            "nearest_m": None,
            "nearest_deg": None,
            "front_min_m": None,
            "obstacle": False,
            "sector_min_m": [None] * max(sectors, 0),
            "free_deg": None,
            "free_width_deg": 0.0,
        })
        return out
    out["blind"] = False

    vb = bearings[valid]
    vr = ranges[valid]

    nearest = int(np.argmin(vr))
    out["nearest_m"] = round(float(vr[nearest]), 3)
    out["nearest_deg"] = round(math.degrees(float(vb[nearest])), 1)

    half_front = math.radians(front_deg) / 2.0
    front = np.abs(vb) <= half_front
    if front.any():
        front_min = float(np.min(vr[front]))
        out["front_min_m"] = round(front_min, 3)
        out["front_valid"] = int(front.sum())
        # The number the robot stops on.  Deliberately the minimum over the arc
        # rather than a percentile: one real return at 20 cm is a collision, and
        # a filter that smooths it away is a filter that drives into a chair leg.
        out["obstacle"] = bool(front_min < obstacle_m)
    else:
        out["front_min_m"] = None
        out["front_valid"] = 0
        out["obstacle"] = False

    sector_min = _sector_minima(vb, vr, sectors)
    out["sector_min_m"] = [None if math.isnan(v) else round(float(v), 3) for v in sector_min]

    free_deg, free_width = _widest_free(ranges, bearings, clear_m, min_free_deg)
    out["free_deg"] = None if free_deg is None else round(free_deg, 1)
    out["free_width_deg"] = round(free_width, 1)
    return out


def _sector_minima(bearings: np.ndarray, ranges: np.ndarray, sectors: int) -> np.ndarray:
    """Minimum range per angular sector, NaN where a sector saw nothing.

    Sector 0 starts at -180° and they run counter-clockwise, so sector index
    ``sectors // 2`` is the one centred on straight ahead.
    """
    if sectors <= 0:
        return np.empty(0, dtype=np.float32)
    width = TWO_PI / sectors
    idx = np.floor((bearings + math.pi) / width).astype(np.int64)
    np.clip(idx, 0, sectors - 1, out=idx)
    # np.minimum.at is the reduction that respects duplicate indices; starting
    # from +inf and converting after keeps empty sectors distinguishable from
    # sectors whose nearest return happens to be far away.
    out = np.full(sectors, np.inf, dtype=np.float32)
    np.minimum.at(out, idx, ranges)
    return np.where(np.isinf(out), np.nan, out)


def _widest_free(
    ranges: np.ndarray,
    bearings: np.ndarray,
    clear_m: float,
    min_free_deg: float,
) -> Tuple[Optional[float], float]:
    """Centre bearing (degrees) and width of the widest gap the robot could take.

    Measured at the scan's own resolution rather than per sector: a doorway is
    often narrower than a 30° sector, and quantising to sectors would report the
    only way out of a room as blocked.  A run must be at least ``min_free_deg``
    wide to count, which is both what keeps a single spurious long reading
    between two walls from looking like an exit, and a crude stand-in for the
    fact that the robot has a width.

    A no-return counts as clear: on a 12 m scanner, nothing coming back from a
    direction means nothing is close in it.  Runs wrap around the end of the
    array — the widest gap is often the one straddling ±180°, and a scan is a
    circle whatever the array indexing suggests.
    """
    n = int(ranges.shape[0])
    if n == 0:
        return None, 0.0

    # Contiguity in index is contiguity in angle: the points of a revolution are
    # reported in angular order, whichever way the device spins.
    clear = ~np.isfinite(ranges) | (ranges >= clear_m)
    if not clear.any():
        return None, 0.0

    step_deg = 360.0 / n
    if clear.all():
        return 0.0, 360.0

    best_start, best_len = 0, 0
    start, length = None, 0
    # Two laps so a run crossing the array boundary is seen whole.
    for i in range(2 * n):
        if clear[i % n]:
            if start is None:
                start, length = i, 0
            length += 1
            if length > best_len:
                best_len, best_start = length, start
        else:
            start, length = None, 0
        if best_len >= n:
            break

    best_len = min(best_len, n)
    width_deg = best_len * step_deg
    if width_deg < min_free_deg:
        return None, width_deg

    centre = int(round(best_start + (best_len - 1) / 2.0)) % n
    return float(math.degrees(float(bearings[centre]))), float(width_deg)


# ---------------------------------------------------------------------------
# Projection into the camera
# ---------------------------------------------------------------------------


def bearing_to_column(bearing_rad, width: int, hfov_deg: float) -> np.ndarray:
    """Image column a bearing falls on, under a pinhole assumption.

    ``x = cx − fx·tan(bearing)`` with ``fx = (width/2) / tan(hfov/2)``.

    The minus sign is the one that matters and the easiest to get backwards: a
    positive bearing is counter-clockwise, which is to the robot's *left*, and
    the left of the scene is the *low* columns of the image.  Get it wrong and
    every range still looks reasonable while being attached to the object on the
    opposite side of the frame.

    The tangent matters too — a linear degrees-to-pixels mapping is off by
    several percent at the edges of a 70° lens, which is exactly where an
    obstacle is interesting.  Bearings outside the field of view come back as
    NaN rather than as a clamped or wrapped-around column.
    """
    b = np.asarray(bearing_rad, dtype=np.float32)
    half = math.radians(hfov_deg) / 2.0
    fx = (width / 2.0) / math.tan(half)
    col = (width / 2.0) - fx * np.tan(np.clip(b, -half, half))
    return np.where(np.abs(b) <= half, col, np.nan).astype(np.float32)


def project_columns(
    scan: Scan,
    width: int,
    hfov_deg: float = 70.0,
    bins: int = 32,
    mount_yaw_deg: float = 0.0,
) -> np.ndarray:
    """Nearest range per horizontal slice of the image, NaN where nothing hit.

    Binned rather than per-pixel on purpose: 360 points across a 70° field is
    roughly 70 usable returns, so a 1280-long array would be 95% interpolation
    presented as measurement.  32 bins keeps one number per ~2° which is close to
    the honest resolution of the pairing.
    """
    if bins <= 0 or width <= 0:
        return np.empty(0, dtype=np.float32)

    bearings = scan.bearings_from(math.radians(mount_yaw_deg))
    valid = np.isfinite(scan.ranges)
    cols = bearing_to_column(bearings[valid], width, hfov_deg)
    ranges = scan.ranges[valid]

    in_view = np.isfinite(cols)
    cols, ranges = cols[in_view], ranges[in_view]

    out = np.full(bins, np.inf, dtype=np.float32)
    if cols.size:
        idx = np.clip((cols / width * bins).astype(np.int64), 0, bins - 1)
        np.minimum.at(out, idx, ranges)
    return np.where(np.isinf(out), np.nan, out)


def range_for_box(
    scan: Scan,
    x0: float,
    x1: float,
    width: int,
    hfov_deg: float = 70.0,
    mount_yaw_deg: float = 0.0,
    reducer: str = "median",
) -> Optional[float]:
    """Range to whatever the scan sees across an image box's columns.

    This is the primitive a detector plugs into: hand it a box's left and right
    edges and get metres back.  ``median`` is the default because a box usually
    spans some background either side of the object, and the minimum would then
    report the wall behind it whenever the object is narrow; pass ``min`` when
    the question is "how close is the nearest thing in that direction", which is
    the collision question rather than the object question.

    Returns None when no beam fell inside the box — a tall thin object at the
    edge of a 360-point scan genuinely produces this, and a made-up number would
    be worse than an admitted gap.
    """
    if width <= 0 or x1 < x0:
        return None

    bearings = scan.bearings_from(math.radians(mount_yaw_deg))
    valid = np.isfinite(scan.ranges)
    cols = bearing_to_column(bearings[valid], width, hfov_deg)
    ranges = scan.ranges[valid]

    hit = np.isfinite(cols) & (cols >= x0) & (cols <= x1)
    if not hit.any():
        return None

    picked = ranges[hit]
    value = float(np.min(picked)) if reducer == "min" else float(np.median(picked))
    return round(value, 3)


def to_xy(scan: Scan, mount_yaw_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Valid returns as (x, y) metres — x forward along the camera axis, y left."""
    bearings = scan.bearings_from(math.radians(mount_yaw_deg))
    valid = np.isfinite(scan.ranges)
    r = scan.ranges[valid]
    b = bearings[valid]
    return (r * np.cos(b)).astype(np.float32), (r * np.sin(b)).astype(np.float32)
