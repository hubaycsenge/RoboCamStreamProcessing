"""Inertial bursts: decoding, and the attitude worth computing on every one.

The robot's IMU lives on its OpenCR board — the same board that drives the
wheels — and reaches this server one of two ways: through the ``opencr_state``
topic the robot's own ROS 2 node publishes, or straight off the board's serial
link when no ROS is running.  Either way it arrives here as ~100 Hz of angular
rate, specific force, magnetic field and a board-computed orientation
quaternion, delivered in bursts so that the message rate stays at the camera's
while the sample rate stays at the sensor's.

What an IMU can and cannot tell you is worth being blunt about, because the
temptation to treat it as a pose source is strong and the failure is silent:

* **Attitude is honest.** Roll and pitch are observable — gravity is a permanent
  reference — so "the robot is tipping" or "this ramp is 8°" is real information.
* **Yaw is not.** Nothing in a gyro fixes an absolute heading, so ``yaw_deg`` is
  an arbitrary origin that drifts, degrees per minute on a board like this.  Use
  ``yaw_rate_dps`` for control and yaw only for short differences.  The
  magnetometer would fix it in principle and does not in practice: it sits
  centimetres from four motors whose field swamps the Earth's.
* **Position is not, at all.**  Double-integrating this accelerometer gives
  metres of error in seconds.  Wheel odometry and the LiDAR are the position
  sensors on this robot; the IMU tells you how it is *oriented* and whether it is
  *moving*, which is exactly what neither of those does well.

Axes follow the ROS body convention the OpenCR firmware uses: **x forward, y
left, z up**, so a level robot at rest reads ``az ≈ +9.81`` and nothing else.
That is the assumption most likely to be wrong on a rebuilt robot, and
:func:`analyse` is written so it shows up rather than hides: a board mounted on
its side reports a permanent 90° tilt from the first burst.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import wire

# A burst is normally a handful of samples.  The cap is a guard against a corrupt
# count field turning into a large allocation, not a real device limit; at 100 Hz
# it is 40 seconds of samples, far more than any burst should ever hold.
MAX_SAMPLES = 4096

#: Channels that must be present as a set for the derived vector to exist.
GYRO_FIELDS = ("wx", "wy", "wz")
ACCEL_FIELDS = ("ax", "ay", "az")
MAG_FIELDS = ("mx", "my", "mz")
QUAT_FIELDS = ("qw", "qx", "qy", "qz")

#: Standard gravity, for the ``g`` -> ``m/s2`` conversion and the plausibility
#: check.  Local gravity differs from this in the fourth digit, which is far
#: below anything a MEMS accelerometer in a robot can resolve.
G_MS2 = 9.80665

DEG = 180.0 / math.pi


class ImuError(Exception):
    """Raised when an IMU header or payload does not parse."""


@dataclass
class ImuBatch:
    """One burst of samples, in SI units whatever the client declared.

    ``values`` is ``(n, k)`` with the columns named by ``fields``; the decoder
    has already converted to rad/s and m/s² so that nothing downstream has to
    carry the unit question.  Non-finite entries are left alone rather than
    filtered: an orientation filter that has not converged emits NaN quaternions
    for its first samples, and that is information — every consumer here masks
    on finiteness instead of letting a NaN propagate into a mean.
    """

    values: np.ndarray                   # (n, k) float32, SI units
    fields: Tuple[str, ...]
    # Microseconds after the burst's own first sample.  Client clock, so only
    # differences within one burst mean anything; never compared across machines.
    offsets_us: np.ndarray               # (n,) int64
    seq: int = -1
    # What the source says it produces, against which the measured rate is judged.
    declared_rate_hz: float = 0.0
    # Samples the client's buffer discarded before this burst was sent.
    dropped: int = 0
    source: str = ""
    # Server monotonic clock, ns, when the burst came off the socket.  This is
    # what frame/IMU association is measured against — never a client clock.
    recv_ts_ns: int = 0
    # Filled in by the server: the result of analyse(), computed once on the IO
    # thread so processors and the IMU reply share one answer.
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.values.shape[0])

    @property
    def span_s(self) -> float:
        """Seconds from the first sample to the last.  Zero for a single sample."""
        if self.count < 2:
            return 0.0
        return float(self.offsets_us[-1] - self.offsets_us[0]) / 1e6

    @property
    def rate_hz(self) -> float:
        """Sample rate measured from the offsets, 0 when a burst is too short.

        The gap between samples, not the gap between bursts: this is the rate the
        sensor actually delivered, which is the number that disagrees with
        ``declared_rate_hz`` when something upstream is dropping samples.
        """
        span = self.span_s
        return (self.count - 1) / span if span > 0 else 0.0

    def column(self, name: str) -> Optional[np.ndarray]:
        """One named channel, or None when this client does not send it."""
        try:
            index = self.fields.index(name)
        except ValueError:
            return None
        return self.values[:, index]

    def block(self, names: Sequence[str]) -> Optional[np.ndarray]:
        """``(n, len(names))`` of the named channels, or None if any is missing.

        All-or-nothing on purpose: two thirds of an angular rate vector is not a
        rate you can do anything with, and a caller that got a partial array
        would have to re-check every column anyway.
        """
        columns = [self.column(name) for name in names]
        if any(c is None for c in columns):
            return None
        return np.stack(columns, axis=1)

    @property
    def gyro(self) -> Optional[np.ndarray]:
        """Angular rate, rad/s, ``(n, 3)`` as x/y/z."""
        return self.block(GYRO_FIELDS)

    @property
    def accel(self) -> Optional[np.ndarray]:
        """Specific force, m/s², ``(n, 3)``.  Includes gravity — at rest it *is* gravity."""
        return self.block(ACCEL_FIELDS)

    @property
    def mag(self) -> Optional[np.ndarray]:
        return self.block(MAG_FIELDS)

    @property
    def quat(self) -> Optional[np.ndarray]:
        """Board-computed orientation, ``(n, 4)`` as w/x/y/z."""
        return self.block(QUAT_FIELDS)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ImuBatch seq={self.seq} samples={self.count} fields={len(self.fields)}>"


# ---------------------------------------------------------------------------
# Wire <-> ImuBatch
# ---------------------------------------------------------------------------


def decode_imu(header: Dict[str, Any], payload: bytes, recv_ts_ns: int = 0) -> ImuBatch:
    """Turn an ``imu`` message into an :class:`ImuBatch`.

    Raises :class:`ImuError` on anything malformed; the caller turns that into an
    unsuccessful IMU result rather than dropping the session, for the same reason
    a corrupt JPEG does not.
    """
    encoding = str(header.get("encoding", wire.IMU_ENC_F32))
    if encoding != wire.IMU_ENC_F32:
        raise ImuError(
            f"unsupported imu encoding {encoding!r}; "
            f"server supports {', '.join(wire.SUPPORTED_IMU_ENCODINGS)}"
        )

    try:
        count = int(header.get("count", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ImuError(f"count is not an integer: {header.get('count')!r}") from exc
    if count <= 0:
        raise ImuError(f"burst has no samples (count={count})")
    if count > MAX_SAMPLES:
        raise ImuError(f"count {count} exceeds the {MAX_SAMPLES} sample limit")

    raw_fields = header.get("fields") or list(wire.IMU_FIELDS)
    if not isinstance(raw_fields, (list, tuple)) or not raw_fields:
        raise ImuError(f"fields must be a non-empty list, got {raw_fields!r}")
    fields = tuple(str(name) for name in raw_fields)
    if len(set(fields)) != len(fields):
        raise ImuError(f"fields contains duplicates: {fields}")

    width = len(fields)
    offsets_bytes = count * 4
    values_bytes = count * width * 4
    need = offsets_bytes + values_bytes
    if len(payload) < need:
        raise ImuError(
            f"payload is {len(payload)} bytes, expected at least {need} for "
            f"{count} samples of {width} channels"
        )

    offsets = np.frombuffer(payload, dtype="<i4", count=count).astype(np.int64)
    values = np.frombuffer(
        payload, dtype="<f4", count=count * width, offset=offsets_bytes
    ).reshape(count, width).astype(np.float32)  # a copy: the units are rescaled below

    _to_si(values, fields,
           gyro_units=str(header.get("gyro_units", "rad/s")),
           accel_units=str(header.get("accel_units", "m/s2")))

    return ImuBatch(
        values=values,
        fields=fields,
        offsets_us=offsets,
        seq=int(header.get("seq", -1) or -1),
        declared_rate_hz=_as_float(header.get("rate_hz"), 0.0),
        dropped=max(0, int(_as_float(header.get("dropped"), 0.0))),
        source=str(header.get("source", "")),
        recv_ts_ns=recv_ts_ns,
    )


def _to_si(values: np.ndarray, fields: Tuple[str, ...], gyro_units: str, accel_units: str) -> None:
    """Rescale in place so everything downstream is rad/s and m/s².

    An unknown unit is an error rather than an assumption: a gyro treated as
    rad/s when it is deg/s under-reports every turn by 57x, and every number
    derived from it still looks entirely reasonable.
    """
    if gyro_units not in wire.IMU_GYRO_UNITS:
        raise ImuError(f"unknown gyro_units {gyro_units!r}; expected one of "
                       f"{', '.join(wire.IMU_GYRO_UNITS)}")
    if accel_units not in wire.IMU_ACCEL_UNITS:
        raise ImuError(f"unknown accel_units {accel_units!r}; expected one of "
                       f"{', '.join(wire.IMU_ACCEL_UNITS)}")

    if gyro_units == "deg/s":
        for name in GYRO_FIELDS:
            if name in fields:
                values[:, fields.index(name)] *= np.float32(math.pi / 180.0)
    if accel_units == "g":
        for name in ACCEL_FIELDS:
            if name in fields:
                values[:, fields.index(name)] *= np.float32(G_MS2)


def encode_imu_payload(values, offsets_us=None) -> bytes:
    """Pack samples into an ``f32`` payload.

    The deployable client has its own copy of this — it must not import the
    server package — so this one exists for tests and for anything server-side
    that wants to replay a burst.  ``offsets_us`` defaults to all-zero, which is
    the honest encoding of "these samples came with no timing of their own".
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    count = arr.shape[0]
    if offsets_us is None:
        offsets = np.zeros(count, dtype="<i4")
    else:
        offsets = np.asarray(offsets_us, dtype="<i4")
    return offsets.tobytes() + np.ascontiguousarray(arr, dtype="<f4").tobytes()


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


# ---------------------------------------------------------------------------
# Attitude
# ---------------------------------------------------------------------------


def quat_to_euler_deg(q) -> Tuple[float, float, float]:
    """``(roll, pitch, yaw)`` degrees from a ``(w, x, y, z)`` quaternion.

    Intrinsic Z-Y-X, the aerospace and ROS convention: yaw about up, then pitch
    about left, then roll about forward.  Pitch is clamped before the arcsine so
    a quaternion that is a rounding error away from vertical produces ±90°
    instead of NaN.
    """
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll * DEG, pitch * DEG, yaw * DEG


def tilt_from_quat_deg(q) -> float:
    """Angle between the robot's up axis and the world vertical, degrees.

    One number for "how far off level", independent of which way it leans, which
    is what a tip-over check actually wants.  It is the third diagonal element of
    the rotation matrix — the cosine between body +z and world +z — so it needs
    no Euler angles and has no gimbal problem.
    """
    w, x, y, z = (float(v) for v in q)
    cos_tilt = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_tilt))))


def _quat_is_usable(q) -> bool:
    """A quaternion the board has actually computed, rather than a placeholder.

    An unconverged filter emits NaN, and a firmware that never fills the field
    leaves all zeros; both would otherwise come through as a confident attitude
    of exactly level, which is the worst possible failure mode for this number.
    """
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape != (4,) or not np.isfinite(arr).all():
        return False
    norm = float(np.linalg.norm(arr))
    # Generous band: some firmwares emit a slightly unnormalised quaternion, and
    # rejecting those would throw away a perfectly good attitude.
    return 0.5 < norm < 1.5


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyse(
    batch: ImuBatch,
    *,
    still_gyro_dps: float = 2.0,
    still_accel_ms2: float = 0.35,
    tilt_warn_deg: float = 15.0,
    shock_ms2: float = 25.0,
    gravity_tolerance_ms2: float = 2.0,
) -> Dict[str, Any]:
    """Reduce a burst to what a robot can act on.

    Everything is reported in degrees and m/s² because that is what a human reads
    in a log line, and every value is rounded and JSON-safe — NaN is not valid
    JSON and a bare ``NaN`` token is what strict parsers on the robot choke on.
    """
    out: Dict[str, Any] = {
        "samples": batch.count,
        "span_ms": round(batch.span_s * 1000.0, 2),
        "rate_hz": round(batch.rate_hz, 1),
        "declared_rate_hz": round(batch.declared_rate_hz, 1),
        # Samples the client's own buffer lost before this burst went out.  A
        # nonzero value here means the client is the bottleneck, which is a
        # different fix from a rate that is low because the sensor is slow.
        "dropped": batch.dropped,
        "fields": list(batch.fields),
    }
    if batch.source:
        out["source"] = batch.source

    _analyse_rotation(batch, out, still_gyro_dps)
    _analyse_force(batch, out, still_accel_ms2, shock_ms2, gravity_tolerance_ms2)
    _analyse_attitude(batch, out, tilt_warn_deg)
    _analyse_mag(batch, out)

    # "Still" needs both halves and means neither is happening.  Note what it
    # does not mean: a robot rolling at constant speed across a smooth floor is
    # still by this test, because constant velocity is genuinely invisible to an
    # accelerometer.  It is "not turning and not changing speed", not "stopped".
    turning = out.get("turning")
    shaking = out.get("shaking")
    out["still"] = bool(turning is False and shaking is False)
    return out


def _analyse_rotation(batch: ImuBatch, out: Dict[str, Any], still_gyro_dps: float) -> None:
    gyro = batch.gyro
    rows = _finite_rows(gyro)
    if rows is None:
        out.update({"gyro": False, "yaw_rate_dps": None, "roll_rate_dps": None,
                    "pitch_rate_dps": None, "gyro_max_dps": None, "turning": None})
        return

    out["gyro"] = True
    dps = rows * DEG
    # Mean over the burst, because a single 100 Hz sample of a MEMS gyro is
    # mostly noise and the burst spans tens of milliseconds — far shorter than
    # any manoeuvre this robot makes, so averaging costs no responsiveness.
    mean = dps.mean(axis=0)
    out["roll_rate_dps"] = round(float(mean[0]), 2)
    out["pitch_rate_dps"] = round(float(mean[1]), 2)
    # The one a differential-drive robot steers on.
    out["yaw_rate_dps"] = round(float(mean[2]), 2)
    # Max over samples, not of the mean: a jolt inside the burst is exactly what
    # the average is designed to hide, and here it is the thing worth seeing.
    out["gyro_max_dps"] = round(float(np.abs(dps).max()), 2)
    out["turning"] = bool(np.linalg.norm(mean) > still_gyro_dps)


def _analyse_force(batch: ImuBatch, out: Dict[str, Any], still_accel_ms2: float,
                   shock_ms2: float, gravity_tolerance_ms2: float) -> None:
    accel = batch.accel
    rows = _finite_rows(accel)
    if rows is None:
        out.update({"accel": False, "accel_mag_mean": None, "accel_mag_max": None,
                    "vibration_ms2": None, "shock": None, "shaking": None,
                    "gravity_ok": None})
        return

    out["accel"] = True
    magnitude = np.linalg.norm(rows, axis=1)
    mean_mag = float(magnitude.mean())
    out["accel_mag_mean"] = round(mean_mag, 3)
    out["accel_mag_max"] = round(float(magnitude.max()), 3)
    # Spread of the magnitude rather than of the components: it is invariant to
    # how the board is mounted, so the same threshold means the same thing on a
    # robot whose IMU was rotated 90° during a rebuild.
    out["vibration_ms2"] = round(float(magnitude.std()), 3)
    out["shaking"] = bool(magnitude.std() > still_accel_ms2)
    out["shock"] = bool(magnitude.max() > shock_ms2)
    # At rest the accelerometer should read exactly gravity.  A mean far from it
    # over a burst this short is a units mistake or a dead axis long before it is
    # a real manoeuvre — a robot cannot sustain 2 m/s² of extra specific force
    # for a tenth of a second without something dramatic happening.
    out["gravity_ok"] = bool(abs(mean_mag - G_MS2) <= gravity_tolerance_ms2)


def _analyse_attitude(batch: ImuBatch, out: Dict[str, Any], tilt_warn_deg: float) -> None:
    """Roll/pitch/yaw and tilt, from the quaternion if there is one.

    The quaternion is preferred because it stays right while the robot
    accelerates, which is precisely when the accelerometer's idea of "down" is
    contaminated by the robot's own motion.  The accelerometer fallback exists
    for a client that sends no orientation at all, and it is only trusted when
    the specific force looks like gravity and nothing else.
    """
    quat = batch.quat
    if quat is not None and quat.shape[0]:
        # The last sample: attitude is a state, and the freshest estimate of a
        # state beats an average that includes where the robot used to be.
        last = quat[-1]
        if _quat_is_usable(last):
            roll, pitch, yaw = quat_to_euler_deg(last)
            out["attitude_from"] = "quaternion"
            out["roll_deg"] = round(roll, 2)
            out["pitch_deg"] = round(pitch, 2)
            # Documented as drifting and origin-less; see the module docstring.
            out["yaw_deg"] = round(yaw, 2)
            out["yaw_absolute"] = False
            out["tilt_deg"] = round(tilt_from_quat_deg(last), 2)
            out["tilted"] = bool(out["tilt_deg"] > tilt_warn_deg)
            return

    accel = batch.accel
    rows = _finite_rows(accel)
    if rows is not None and out.get("gravity_ok"):
        ax, ay, az = (float(v) for v in rows.mean(axis=0))
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        out["attitude_from"] = "accelerometer"
        out["roll_deg"] = round(math.degrees(math.atan2(ay, az)), 2)
        out["pitch_deg"] = round(math.degrees(math.atan2(-ax, math.hypot(ay, az))), 2)
        # An accelerometer has no idea which way is north, and saying so is
        # better than a zero that reads as "pointing forward".
        out["yaw_deg"] = None
        out["yaw_absolute"] = False
        out["tilt_deg"] = round(math.degrees(math.acos(max(-1.0, min(1.0, az / norm)))), 2)
        out["tilted"] = bool(out["tilt_deg"] > tilt_warn_deg)
        return

    out.update({"attitude_from": None, "roll_deg": None, "pitch_deg": None,
                "yaw_deg": None, "yaw_absolute": False, "tilt_deg": None,
                "tilted": None})


def _analyse_mag(batch: ImuBatch, out: Dict[str, Any]) -> None:
    """Magnetic field magnitude only — deliberately no heading.

    A heading from this magnetometer would be a number that looks like a compass
    and is not one: it sits centimetres from four motors, and their field moves
    with the wheel currents.  The magnitude is still worth reporting because it
    is how you *see* that happening — it swings wildly under load, and a value
    that never changes at all means the sensor is not being read.
    """
    rows = _finite_rows(batch.mag)
    if rows is None:
        out["mag"] = False
        out["mag_norm"] = None
        return
    out["mag"] = True
    out["mag_norm"] = round(float(np.linalg.norm(rows, axis=1).mean()), 3)


def _finite_rows(block: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Rows of a ``(n, k)`` block where every channel is finite, or None."""
    if block is None or block.size == 0:
        return None
    keep = np.isfinite(block).all(axis=1)
    if not keep.any():
        return None
    return block[keep].astype(np.float64)
