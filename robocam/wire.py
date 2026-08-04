"""Wire protocol for the RoboCam video link.

Transport is ZeroMQ: the robot opens a DEALER socket, the server binds a ROUTER.
Every application message is a two-frame multipart message::

    [ header_json, payload ]

``header`` is a small UTF-8 JSON object, ``payload`` is opaque bytes (the encoded
image for ``frame`` messages, the packed ranges for ``scan`` messages, empty for
everything else).  ROUTER prepends the peer identity on receive and strips it on
send, so the server actually handles ``[identity, header, payload]``.  DEALER
does not insert the empty delimiter frame that REQ/REP would, so the framing is
exactly as written above.

The robot sends three streams down this one socket: camera frames at ~30 Hz,
LiDAR scans at ~5 Hz and IMU bursts at the frame rate (carrying ~100 Hz of
inertial samples).  They are separate message types rather than one combined
message because the sensors are not synchronised and never will be — pairing
them at the source would mean holding a frame back to wait for a scan, which
costs latency to buy an alignment the server can do better itself.

A JSON header costs roughly 200 bytes per frame.  That is noise next to a JPEG
payload and it makes the stream inspectable with tcpdump, which is worth a lot
when the other end is a robot you cannot easily attach a debugger to.

Clocks
------
The Orin and the server do not share a clock, so no timestamp is ever compared
across machines.  The client stamps ``t_capture_ns`` and ``t_send_ns`` from its
own monotonic clock; the server echoes them back untouched so the client can
compute round-trip time against its own clock.  The server separately reports
``server_ms``, measured entirely on the server's monotonic clock.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Tuple

PROTOCOL_VERSION = 1

# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

MSG_HELLO = "hello"      # client -> server, opens a session
MSG_WELCOME = "welcome"  # server -> client, session accepted
MSG_FRAME = "frame"      # client -> server, one encoded image
MSG_RESULT = "result"    # server -> client, one result per frame
MSG_SCAN = "scan"        # client -> server, one LiDAR revolution
MSG_SCAN_RESULT = "scan_result"  # server -> client, one result per scan
MSG_IMU = "imu"          # client -> server, a burst of inertial samples
MSG_IMU_RESULT = "imu_result"    # server -> client, one result per burst
MSG_PING = "ping"        # client -> server, liveness probe
MSG_PONG = "pong"        # server -> client
MSG_BYE = "bye"          # either direction, graceful close
MSG_ERROR = "error"      # server -> client, request-level failure

# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------

CODEC_JPEG = "jpeg"
CODEC_RAW_BGR = "raw_bgr"  # uncompressed, header must carry width/height
CODEC_H264 = "h264"        # optional, needs PyAV on the server

SUPPORTED_CODECS = (CODEC_JPEG, CODEC_RAW_BGR, CODEC_H264)

# ---------------------------------------------------------------------------
# Scan payload encodings
# ---------------------------------------------------------------------------
#
# An LDS-02 revolution is 360 points.  As uint16 millimetres that is 720 bytes,
# roughly 4% of a 720p JPEG, so there is no reason to compress it — and the
# device's own resolution is millimetres, which makes the conversion exact.
# ``f32m`` exists because that is what a ROS ``LaserScan`` already holds, and
# forcing every client to quantise would be a trap for the one that has a
# longer-range scanner later.

SCAN_ENC_U16_MM = "u16mm"  # count * uint16 LE, millimetres, 0 = no return
SCAN_ENC_F32_M = "f32m"    # count * float32 LE, metres, 0/inf/nan = no return

SUPPORTED_SCAN_ENCODINGS = (SCAN_ENC_U16_MM, SCAN_ENC_F32_M)

# ---------------------------------------------------------------------------
# IMU payload encoding
# ---------------------------------------------------------------------------
#
# The OpenCR runs its IMU at ~100 Hz, three times the camera rate, so unlike a
# scan an inertial sample is not something to send one message at a time — the
# JSON header would cost more than the data and the message rate would be the
# only thing on this link that scales with the sensor rather than with the
# frame.  A burst carries every sample taken since the previous one, so the full
# rate survives while the message rate stays at the camera's.
#
# Layout of an ``f32`` payload, for ``count`` samples over ``len(fields)``
# channels::
#
#     count * int32   microseconds of each sample after ``t_capture_ns``
#     count * k * f32 the channels of each sample, in ``fields`` order
#
# Offsets are relative and 32-bit because a burst spans tens of milliseconds;
# carrying an absolute int64 per sample would double the payload to describe a
# clock the server is forbidden from comparing against anyway (see *Clocks*).
#
# ``fields`` is named rather than positional so a client without one of the
# sensors sends a shorter row instead of padding: a ``sensor_msgs/Imu`` source
# has no magnetometer, and inventing zeros for it would read downstream as a
# magnetometer that measures zero.

IMU_ENC_F32 = "f32"

SUPPORTED_IMU_ENCODINGS = (IMU_ENC_F32,)

#: Every channel the protocol knows how to interpret: angular rate, specific
#: force, magnetic field and the orientation the sensor's own filter produced.
#: A client sends the subset it has; anything outside this tuple is carried
#: through to the summary untouched but nothing is derived from it.
IMU_FIELDS = ("wx", "wy", "wz", "ax", "ay", "az",
              "mx", "my", "mz", "qw", "qx", "qy", "qz")

#: Units a client may declare.  Naming them beats assuming: a gyro read as rad/s
#: when it is deg/s is wrong by 57x and still entirely plausible-looking.
IMU_GYRO_UNITS = ("rad/s", "deg/s")
IMU_ACCEL_UNITS = ("m/s2", "g")

# Why a result can be unsuccessful.  These end up in ``result.reason``.
REASON_OK = "ok"
REASON_DROPPED = "dropped"        # queue was full, frame never reached a worker
REASON_DECODE_FAILED = "decode_failed"
REASON_PROCESSOR_FAILED = "processor_failed"
REASON_UNSUPPORTED_CODEC = "unsupported_codec"
REASON_BAD_SCAN = "bad_scan"           # scan header/payload did not parse
REASON_LIDAR_DISABLED = "lidar_disabled"  # server is configured to ignore scans
REASON_BAD_IMU = "bad_imu"             # imu header/payload did not parse
REASON_IMU_DISABLED = "imu_disabled"   # server is configured to ignore imu bursts


class ProtocolError(Exception):
    """Raised when a peer sends something that does not parse as a message."""


def monotonic_ns() -> int:
    """Monotonic nanoseconds.  Never compare this across machines."""
    return time.monotonic_ns()


def encode(header: Dict[str, Any], payload: bytes = b"") -> Tuple[bytes, bytes]:
    """Serialise a message into the two ZeroMQ frames that go on the wire."""
    return json.dumps(header, separators=(",", ":")).encode("utf-8"), payload


def decode(frames) -> Tuple[Dict[str, Any], bytes]:
    """Parse the two application frames of a received multipart message.

    The caller is responsible for having already stripped the ROUTER identity.
    """
    if len(frames) == 1:
        header_bytes, payload = frames[0], b""
    elif len(frames) == 2:
        header_bytes, payload = frames[0], frames[1]
    else:
        raise ProtocolError(f"expected 1 or 2 frames, got {len(frames)}")

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"header is not valid JSON: {exc}") from exc

    if not isinstance(header, dict):
        raise ProtocolError("header must be a JSON object")
    if "type" not in header:
        raise ProtocolError("header has no 'type' field")

    return header, payload


# ---------------------------------------------------------------------------
# Constructors.  These exist so the field names live in exactly one place.
# ---------------------------------------------------------------------------


def hello(
    client_id: str,
    codec: str = CODEC_JPEG,
    width: int = 0,
    height: int = 0,
    fps: float = 0.0,
    camera: str = "",
    lidar: Dict[str, Any] | None = None,
    imu: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    header = {
        "type": MSG_HELLO,
        "protocol": PROTOCOL_VERSION,
        "client_id": client_id,
        "codec": codec,
        "width": width,
        "height": height,
        "fps": fps,
        "camera": camera,
        # Empty when the robot has no scanner or no OpenCR attached; the server
        # logs which sensors a session actually brought rather than guessing
        # from traffic.
        "lidar": lidar or {},
        "imu": imu or {},
        "t_send_ns": monotonic_ns(),
    }
    if extra:
        header["extra"] = extra
    return header


def welcome(
    session_id: str,
    processor: str,
    accepted: bool = True,
    message: str = "",
    server_info: Dict[str, Any] | None = None,
    echo_t_send_ns: int | None = None,
) -> Dict[str, Any]:
    header = {
        "type": MSG_WELCOME,
        "protocol": PROTOCOL_VERSION,
        "session_id": session_id,
        "processor": processor,
        "accepted": accepted,
        "message": message,
        "server": server_info or {},
    }
    if echo_t_send_ns is not None:
        header["t_send_ns"] = echo_t_send_ns
    return header


def frame(
    seq: int,
    codec: str,
    width: int,
    height: int,
    t_capture_ns: int,
    channels: int = 3,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Header for one image.

    ``width``/``height`` are what the client believes it sent.  The server
    reports the dimensions it actually decoded, which is how you catch a camera
    that silently renegotiated its format.
    """
    header = {
        "type": MSG_FRAME,
        "seq": seq,
        "codec": codec,
        "width": width,
        "height": height,
        "channels": channels,
        "t_capture_ns": t_capture_ns,
        "t_send_ns": monotonic_ns(),
    }
    if extra:
        header["extra"] = extra
    return header


def result(
    seq: int,
    ok: bool,
    reason: str = REASON_OK,
    *,
    width: int = 0,
    height: int = 0,
    channels: int = 0,
    dtype: str = "",
    nbytes: int = 0,
    payload_bytes: int = 0,
    codec: str = "",
    decode_ms: float = 0.0,
    process_ms: float = 0.0,
    server_ms: float = 0.0,
    queue_ms: float = 0.0,
    processor: str = "",
    data: Dict[str, Any] | None = None,
    t_capture_ns: int | None = None,
    t_send_ns: int | None = None,
    scan_seq: int | None = None,
    scan_age_ms: float | None = None,
    imu_seq: int | None = None,
    imu_age_ms: float | None = None,
) -> Dict[str, Any]:
    """Header for one result.

    ``t_capture_ns`` and ``t_send_ns`` are echoed straight back from the frame
    header so the client can compute latency against its own clock.

    ``scan_seq``/``scan_age_ms`` appear only when a LiDAR scan was attached to
    this frame, so ``scan_seq`` missing from a result is the honest signal that
    the processor saw no ranges — as opposed to a scan of all no-returns, which
    is a different failure and reads as ``scan_seq`` present with zero coverage.
    ``imu_seq``/``imu_age_ms`` say the same thing about the inertial burst.
    """
    header = {
        "type": MSG_RESULT,
        "seq": seq,
        "ok": ok,
        "reason": reason,
        # What the server actually decoded, not what the client claimed.
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": dtype,
        "nbytes": nbytes,
        "payload_bytes": payload_bytes,
        "codec": codec,
        # Server-side timings, all from the server's monotonic clock.
        "decode_ms": round(decode_ms, 3),
        "process_ms": round(process_ms, 3),
        "queue_ms": round(queue_ms, 3),
        "server_ms": round(server_ms, 3),
        "processor": processor,
        # Per-processor payload.  Empty today, YOLO boxes tomorrow.
        "data": data or {},
    }
    if t_capture_ns is not None:
        header["t_capture_ns"] = t_capture_ns
    if t_send_ns is not None:
        header["t_send_ns"] = t_send_ns
    # Only present when a LiDAR scan was fresh enough to attach to this frame.
    if scan_seq is not None:
        header["scan_seq"] = scan_seq
    if scan_age_ms is not None:
        header["scan_age_ms"] = round(scan_age_ms, 2)
    # Likewise for the inertial burst.
    if imu_seq is not None:
        header["imu_seq"] = imu_seq
    if imu_age_ms is not None:
        header["imu_age_ms"] = round(imu_age_ms, 2)
    return header


def scan(
    seq: int,
    count: int,
    t_capture_ns: int,
    *,
    encoding: str = SCAN_ENC_U16_MM,
    angle_min: float = 0.0,
    angle_increment: float = 0.0,
    range_min: float = 0.0,
    range_max: float = 0.0,
    scan_time: float = 0.0,
    intensities: bool = False,
    source: str = "",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Header for one LiDAR revolution.

    Angles follow the ``sensor_msgs/LaserScan`` convention: the bearing of point
    ``i`` is ``angle_min + i * angle_increment`` radians, counter-clockwise, with
    zero along the sensor's forward axis.  ``angle_increment`` may be negative
    for a device that reports clockwise; the server normalises.  Defaulting it to
    0 means "a full even revolution", which the server expands to ``2π / count``.

    ``range_min``/``range_max`` are the device's own limits (0.12 m and 12 m on
    an LDS-02).  Returns outside them are treated as no-returns rather than
    trusted, which is what stops a 0.05 m spurious echo from reading as an
    imminent collision.
    """
    header = {
        "type": MSG_SCAN,
        "seq": seq,
        "encoding": encoding,
        "count": count,
        "angle_min": angle_min,
        "angle_increment": angle_increment,
        "range_min": range_min,
        "range_max": range_max,
        "scan_time": scan_time,
        "intensities": bool(intensities),
        "source": source,
        "t_capture_ns": t_capture_ns,
        "t_send_ns": monotonic_ns(),
    }
    if extra:
        header["extra"] = extra
    return header


def scan_result(
    seq: int,
    ok: bool,
    reason: str = REASON_OK,
    *,
    points: int = 0,
    payload_bytes: int = 0,
    encoding: str = "",
    parse_ms: float = 0.0,
    server_ms: float = 0.0,
    data: Dict[str, Any] | None = None,
    t_capture_ns: int | None = None,
    t_send_ns: int | None = None,
) -> Dict[str, Any]:
    """Header for one scan result.

    Scans are answered from the IO thread rather than the worker pool: the whole
    analysis is a handful of numpy reductions over 360 floats, and routing it
    through the queue would delay obstacle information behind a model that may be
    taking 30 ms per frame.  Anything expensive belongs in a processor, which
    sees the same scan attached to the next frame.
    """
    header = {
        "type": MSG_SCAN_RESULT,
        "seq": seq,
        "ok": ok,
        "reason": reason,
        "points": points,
        "payload_bytes": payload_bytes,
        "encoding": encoding,
        "parse_ms": round(parse_ms, 3),
        "server_ms": round(server_ms, 3),
        "data": data or {},
    }
    if t_capture_ns is not None:
        header["t_capture_ns"] = t_capture_ns
    if t_send_ns is not None:
        header["t_send_ns"] = t_send_ns
    return header


def imu(
    seq: int,
    count: int,
    t_capture_ns: int,
    *,
    fields=IMU_FIELDS,
    encoding: str = IMU_ENC_F32,
    gyro_units: str = "rad/s",
    accel_units: str = "m/s2",
    rate_hz: float = 0.0,
    dropped: int = 0,
    source: str = "",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Header for one burst of inertial samples.

    ``t_capture_ns`` belongs to the *first* sample in the burst; the rest are
    placed by the microsecond offsets in the payload.  So a burst carries its
    own internal timing exactly, and only its position on the client's clock is
    unusable server-side — which is the same deal every other message here gets.

    ``rate_hz`` is what the source believes it produces, not what arrived.  The
    server reports the rate it measures from the offsets, and the two disagreeing
    is the signal that samples are being lost somewhere between the sensor and
    the socket.  ``dropped`` counts samples the client's own buffer had to
    discard since the last burst, which is the other half of that story: a gap
    the client knows about is worth stating rather than leaving to be inferred
    from a rate that came out low.
    """
    header = {
        "type": MSG_IMU,
        "seq": seq,
        "encoding": encoding,
        "count": count,
        "fields": list(fields),
        "gyro_units": gyro_units,
        "accel_units": accel_units,
        "rate_hz": rate_hz,
        "dropped": dropped,
        "source": source,
        "t_capture_ns": t_capture_ns,
        "t_send_ns": monotonic_ns(),
    }
    if extra:
        header["extra"] = extra
    return header


def imu_result(
    seq: int,
    ok: bool,
    reason: str = REASON_OK,
    *,
    samples: int = 0,
    payload_bytes: int = 0,
    encoding: str = "",
    parse_ms: float = 0.0,
    server_ms: float = 0.0,
    data: Dict[str, Any] | None = None,
    t_capture_ns: int | None = None,
    t_send_ns: int | None = None,
) -> Dict[str, Any]:
    """Header for one IMU burst result.

    Answered from the IO thread, for the same reason scans are: reducing a
    hundred samples to attitude, rates and a still/moving decision is a few
    numpy passes, and inertial data is the most perishable thing on the link —
    routing it through the frame queue would deliver "the robot is tipping" a
    model's worth of latency after it started to.
    """
    header = {
        "type": MSG_IMU_RESULT,
        "seq": seq,
        "ok": ok,
        "reason": reason,
        "samples": samples,
        "payload_bytes": payload_bytes,
        "encoding": encoding,
        "parse_ms": round(parse_ms, 3),
        "server_ms": round(server_ms, 3),
        "data": data or {},
    }
    if t_capture_ns is not None:
        header["t_capture_ns"] = t_capture_ns
    if t_send_ns is not None:
        header["t_send_ns"] = t_send_ns
    return header


def ping(nonce: int) -> Dict[str, Any]:
    return {"type": MSG_PING, "nonce": nonce, "t_send_ns": monotonic_ns()}


def pong(nonce: int, echo_t_send_ns: int | None = None) -> Dict[str, Any]:
    header = {"type": MSG_PONG, "nonce": nonce}
    if echo_t_send_ns is not None:
        header["t_send_ns"] = echo_t_send_ns
    return header


def bye(reason: str = "") -> Dict[str, Any]:
    return {"type": MSG_BYE, "reason": reason}


def error(message: str, seq: int | None = None) -> Dict[str, Any]:
    header = {"type": MSG_ERROR, "message": message}
    if seq is not None:
        header["seq"] = seq
    return header
