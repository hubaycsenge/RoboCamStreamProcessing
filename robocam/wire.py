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

The robot sends five streams down this one socket: camera frames at ~30 Hz,
LiDAR scans at ~5 Hz, IMU bursts at the frame rate (carrying ~100 Hz of inertial
samples), odometry poses at the frame rate, and its SLAM occupancy grid every
few seconds.  They are separate message types rather than one combined message
because the sensors are not synchronised and never will be — pairing them at the
source would mean holding a frame back to wait for a scan, which costs latency to
buy an alignment the server can do better itself.

Replies and announcements
-------------------------
Every robot message gets exactly one reply, because the client sizes each
stream's in-flight window from outstanding replies and a silently dropped
message would stall that stream alone.

Three server messages are *not* replies: ``map_update``, ``pose_hint`` and
``found``.  They are the server's own products — the "Compare" and
"LLM/decision" boxes of the system diagram — and they appear when those finish,
not when the robot asks.  The robot must therefore treat the socket as
bidirectional rather than as request/response; DEALER/ROUTER already is.

The two phases
--------------
The diagram splits the session in two, and so does this protocol:

``t1``  explore and map.  Frames and scans go up, the reconstruction comes back
        as a point cloud, ``compare`` diffs that cloud against the robot's own
        occupancy grid, and what it finds returns as ``map_update`` (obstacles
        the LiDAR's single plane cannot see) and ``pose_hint`` (an *offer* of a
        correction to SLAM, never a command).  The robot's frontier candidates
        go up as ``exits`` and come back ranked.

``t2``  seek.  The robot navigates its own map; the server watches the same
        frames for the mission target and announces ``found`` when a decision
        stage concludes the target is in view, with a pose for the robot's
        behaviour tree to drive to.

The phase is declared in ``hello`` and changed with a ``phase`` message from
either end.  Nothing about the transport changes with it — what changes is which
side is producing decisions, so it is worth being explicit rather than inferring
it from traffic.

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

#: Bumped to 2 when odometry, the occupancy grid, the exit candidates and the
#: server's three announcements were added.  Everything in v2 is additive, so a
#: v1 client talks to a v2 server perfectly well — it simply never sends a map
#: and never hears a ``map_update``.  The server therefore accepts any version in
#: [MIN_PROTOCOL_VERSION, PROTOCOL_VERSION] rather than demanding equality; what
#: it must still refuse is a version it has never heard of, because that one may
#: mean something different by a field name it recognises.
PROTOCOL_VERSION = 2
MIN_PROTOCOL_VERSION = 1

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
MSG_ODOM = "odom"        # client -> server, one pose in the robot's own frame
MSG_ODOM_RESULT = "odom_result"  # server -> client, one result per pose
MSG_MAP = "map"          # client -> server, the robot's 2D occupancy grid
MSG_MAP_RESULT = "map_result"    # server -> client, one result per grid
MSG_EXITS = "exits"      # client -> server, the robot's exit/frontier candidates
MSG_EXITS_RESULT = "exits_result"  # server -> client, the same candidates ranked

# The three server products.  Unlike everything above these are announcements,
# not replies: they are emitted when Compare or the decision stage finishes, and
# the robot never asks for one.
MSG_MAP_UPDATE = "map_update"    # server -> client, cells Compare wants merged
MSG_POSE_HINT = "pose_hint"      # server -> client, a correction *offered* to SLAM
MSG_FOUND = "found"              # server -> client, T2: the target and where it is

MSG_PHASE = "phase"      # either direction, move between t1 and t2
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

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
#
# The two rows of the system diagram.  They are named rather than inferred from
# traffic because the same messages flow in both: what differs is what the
# server does with them, and a robot that thinks it is exploring while the
# server is already seeking is a failure that would otherwise look like silence.

PHASE_EXPLORE = "t1"   # build the map; Compare and the exit ranking are live
PHASE_SEEK = "t2"      # navigate the map; the decision stage is live

SUPPORTED_PHASES = (PHASE_EXPLORE, PHASE_SEEK)

# ---------------------------------------------------------------------------
# Occupancy grid payload encodings
# ---------------------------------------------------------------------------
#
# A ``nav_msgs/OccupancyGrid`` is int8 per cell: -1 unknown, 0..100 the
# probability of occupancy in percent.  This protocol keeps that representation
# exactly, because anything else would need a lossy conversion at both ends of a
# link whose whole point is that the two ends agree about the map.
#
# Size is the reason for the second encoding.  A 20 m room at 5 cm is 400x400 =
# 160 kB, which is a fine thing to send once but not every second; the same grid
# is mostly -1 and mostly runs of equal values, so zlib takes it to a few kB.
# ``i8`` stays for the small patches, where the compression header would be a
# noticeable fraction of the payload.

MAP_ENC_I8 = "i8"        # width * height int8, row-major, row 0 at the origin
MAP_ENC_I8_ZLIB = "i8z"  # the same bytes through zlib.compress

SUPPORTED_MAP_ENCODINGS = (MAP_ENC_I8, MAP_ENC_I8_ZLIB)

#: Cell value meaning "never observed".  ROS's own sentinel; kept as a named
#: constant because -1 appearing bare in a threshold comparison is exactly the
#: kind of thing that quietly turns unknown space into free space.
MAP_UNKNOWN = -1

# How a ``map_update`` patch is to be merged into the robot's grid.
#
# ``max`` is the default and the honest one: the server is looking at a monocular
# reconstruction, so what it can contribute is *obstacles the LiDAR's single
# horizontal plane could not see* — a table top, a shelf edge, a low sill.  It is
# in no position to clear a cell the robot's own scanner marked occupied, and a
# merge that let it do so would turn a reconstruction artefact into a collision.
# ``replace`` exists for the case where the server is authoritative about a
# region (a stage that re-derives it wholesale), and must be asked for.
MAP_MERGE_MAX = "max"
MAP_MERGE_REPLACE = "replace"

SUPPORTED_MAP_MERGES = (MAP_MERGE_MAX, MAP_MERGE_REPLACE)

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
REASON_BAD_ODOM = "bad_odom"           # odom header did not parse
REASON_ODOM_DISABLED = "odom_disabled" # server is configured to ignore odometry
REASON_BAD_MAP = "bad_map"             # map header/payload did not parse
REASON_MAP_DISABLED = "map_disabled"   # server is configured to ignore grids
REASON_MAP_TOO_LARGE = "map_too_large"  # more cells than map.max_cells allows
REASON_BAD_EXITS = "bad_exits"         # exits header did not parse
REASON_BAD_PHASE = "bad_phase"         # unknown phase name


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
    odom: Dict[str, Any] | None = None,
    map_info: Dict[str, Any] | None = None,
    phase: str = PHASE_EXPLORE,
    mission: Dict[str, Any] | None = None,
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
        # Likewise for the two streams that come from the robot's own software
        # rather than from a device: the odometry source and the SLAM node
        # publishing the grid.  A robot running neither says so by leaving these
        # empty, and the server then knows the reconstruction has nothing to be
        # compared against, rather than waiting for a map that is never coming.
        "odom": odom or {},
        "map": map_info or {},
        # Which row of the system diagram this session starts on, and what it is
        # looking for if it starts on the second.
        "phase": phase,
        "mission": mission or {},
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
    odom_seq: int | None = None,
    odom_age_ms: float | None = None,
    map_seq: int | None = None,
    map_id: str | None = None,
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
    # And for the pose the frame was taken at.  Its absence is what tells the
    # robot that anything the server derived about *where* something is lives in
    # the reconstruction's own frame and not in the map — the difference between
    # a cloud that can be compared against the grid and one that cannot.
    if odom_seq is not None:
        header["odom_seq"] = odom_seq
    if odom_age_ms is not None:
        header["odom_age_ms"] = round(odom_age_ms, 2)
    # Which of the robot's uploaded grids this result was computed against.
    if map_seq is not None:
        header["map_seq"] = map_seq
    if map_id is not None:
        header["map_id"] = map_id
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


def odom(
    seq: int,
    t_capture_ns: int,
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
    quaternion: Tuple[float, float, float, float] | None = None,
    vx: float = 0.0,
    vy: float = 0.0,
    vyaw: float = 0.0,
    frame: str = "map",
    child_frame: str = "base_link",
    pose_cov_diag=None,
    source: str = "",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Header for one pose.  No payload — one pose is smaller than its header.

    ``frame`` is the single most important field here and the easiest to get
    wrong.  A pose in ``odom`` and a pose in ``map`` differ by exactly the
    accumulated SLAM correction, which is the quantity nobody notices is missing
    until a reconstruction is fused a metre away from where it belongs.  The
    server refuses to compare a cloud against a grid unless this names the same
    frame the grid declared, rather than assuming the robot meant the right one.

    ``yaw`` is carried alongside the optional quaternion because everything on
    the 2D side of this system — the grid, the exits, the goal poses — is planar,
    and re-deriving a yaw from a quaternion at four different call sites is four
    chances to pick a different Euler convention.  When both are present the
    quaternion is authoritative and ``yaw`` must agree with it.
    """
    header = {
        "type": MSG_ODOM,
        "seq": seq,
        "frame": frame,
        "child_frame": child_frame,
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "yaw": float(yaw),
        "vx": float(vx),
        "vy": float(vy),
        "vyaw": float(vyaw),
        "source": source,
        "t_capture_ns": t_capture_ns,
        "t_send_ns": monotonic_ns(),
    }
    if quaternion is not None:
        qw, qx, qy, qz = (float(v) for v in quaternion)
        header.update({"qw": qw, "qx": qx, "qy": qy, "qz": qz})
    if pose_cov_diag is not None:
        # Six numbers, x y z roll pitch yaw, straight off a ROS covariance
        # diagonal.  Only the planar three are used, but dropping the other
        # three at the source would make this un-round-trippable.
        header["pose_cov_diag"] = [float(v) for v in pose_cov_diag]
    if extra:
        header["extra"] = extra
    return header


def odom_result(
    seq: int,
    ok: bool,
    reason: str = REASON_OK,
    *,
    server_ms: float = 0.0,
    data: Dict[str, Any] | None = None,
    t_capture_ns: int | None = None,
    t_send_ns: int | None = None,
) -> Dict[str, Any]:
    """Header for one odometry reply.

    Deliberately thin.  There is nothing to analyse in a single pose — the
    interesting quantities (drift, whether the robot is where it thinks) come
    from comparing it with something else, which is what ``pose_hint`` is for.
    This exists because the client counts replies, and because ``data`` carries
    back the one thing the robot cannot know on its own: whether the server was
    able to place this pose in the frame of the grid it uploaded.
    """
    header = {
        "type": MSG_ODOM_RESULT,
        "seq": seq,
        "ok": ok,
        "reason": reason,
        "server_ms": round(server_ms, 3),
        "data": data or {},
    }
    if t_capture_ns is not None:
        header["t_capture_ns"] = t_capture_ns
    if t_send_ns is not None:
        header["t_send_ns"] = t_send_ns
    return header


def occupancy_map(
    seq: int,
    width: int,
    height: int,
    resolution: float,
    t_capture_ns: int,
    *,
    encoding: str = MAP_ENC_I8_ZLIB,
    origin=(0.0, 0.0, 0.0),
    frame: str = "map",
    map_id: str = "",
    occupied_min: int = 65,
    free_max: int = 25,
    full: bool = True,
    x0: int = 0,
    y0: int = 0,
    source: str = "",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Header for one occupancy grid, or one patch of one.

    Geometry follows ``nav_msgs/OccupancyGrid`` exactly: ``origin`` is the pose
    of cell (0, 0)'s corner as ``(x, y, yaw)`` in ``frame``, cells are square
    with edge ``resolution`` metres, and the payload is row-major with row 0 at
    the origin.  Matching ROS here is not deference — it is that the robot has a
    grid in this layout already, and any re-origining on the way out is a sign
    error waiting to flip a map north-south.

    ``occupied_min`` and ``free_max`` travel with the grid rather than living in
    the server's config because they are the robot's *own* costmap tuning.  A
    server that hardcoded 65 would silently disagree with a robot configured at
    50 about which cells are obstacles, and the disagreement would show up only
    as a comparison that finds differences everywhere.

    ``map_id`` changes whenever SLAM restarts its map.  It is a string rather
    than a counter because the robot has one and the server does not; the server
    only ever tests it for equality, which is what makes a stale ``map_update``
    for a map that no longer exists discardable instead of destructive.

    ``full`` distinguishes a whole grid from an incremental patch at
    ``(x0, y0)``.  A robot that maps a corridor for ten minutes should send the
    grid once and patches after that; the server holds the last full grid and
    applies patches onto it.
    """
    header = {
        "type": MSG_MAP,
        "seq": seq,
        "encoding": encoding,
        "width": int(width),
        "height": int(height),
        "resolution": float(resolution),
        "origin": [float(v) for v in origin],
        "frame": frame,
        "map_id": map_id,
        "occupied_min": int(occupied_min),
        "free_max": int(free_max),
        "unknown": MAP_UNKNOWN,
        "full": bool(full),
        "x0": int(x0),
        "y0": int(y0),
        "source": source,
        "t_capture_ns": t_capture_ns,
        "t_send_ns": monotonic_ns(),
    }
    if extra:
        header["extra"] = extra
    return header


def map_result(
    seq: int,
    ok: bool,
    reason: str = REASON_OK,
    *,
    cells: int = 0,
    payload_bytes: int = 0,
    encoding: str = "",
    parse_ms: float = 0.0,
    server_ms: float = 0.0,
    data: Dict[str, Any] | None = None,
    t_capture_ns: int | None = None,
    t_send_ns: int | None = None,
) -> Dict[str, Any]:
    """Header for one occupancy-grid reply.

    ``data`` reports what the server made of the grid — cell counts by class,
    and whether it now has everything Compare needs (a grid, a pose in the same
    frame, and a cloud).  That last flag is worth its bytes: "the server is not
    comparing anything" and "the server is comparing and finding nothing" look
    identical from the robot otherwise.
    """
    header = {
        "type": MSG_MAP_RESULT,
        "seq": seq,
        "ok": ok,
        "reason": reason,
        "cells": int(cells),
        "payload_bytes": int(payload_bytes),
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


def map_update(
    seq: int,
    width: int,
    height: int,
    resolution: float,
    *,
    encoding: str = MAP_ENC_I8_ZLIB,
    x0: int = 0,
    y0: int = 0,
    origin=(0.0, 0.0, 0.0),
    frame: str = "map",
    map_id: str = "",
    merge: str = MAP_MERGE_MAX,
    cells_changed: int = 0,
    source: str = "compare",
    from_frame_seq: int | None = None,
    cloud_map_id: int | None = None,
    server_ms: float = 0.0,
    data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Header for the patch Compare wants merged into the robot's grid.

    This is the solid arrow from Compare back to the robot's 2D occupancy grid.
    It is a patch, not a map: the server has no business restating the parts of
    the robot's map it has nothing to say about, and a full grid every time would
    also make every merge a chance to overwrite something newer.

    ``merge`` says how, and defaults to ``max`` — see :data:`MAP_MERGE_MAX` for
    why the server may add obstacles but not clear them.

    ``map_id`` is the robot's, echoed back.  A patch that arrives after SLAM has
    restarted its map names a map that no longer exists and must be dropped;
    without this field it would instead be applied to the new map at coordinates
    that mean something entirely different.
    """
    header = {
        "type": MSG_MAP_UPDATE,
        "seq": seq,
        "encoding": encoding,
        "x0": int(x0),
        "y0": int(y0),
        "width": int(width),
        "height": int(height),
        "resolution": float(resolution),
        "origin": [float(v) for v in origin],
        "frame": frame,
        "map_id": map_id,
        "merge": merge,
        "unknown": MAP_UNKNOWN,
        "cells_changed": int(cells_changed),
        "source": source,
        "server_ms": round(server_ms, 3),
        "data": data or {},
        "t_send_ns": monotonic_ns(),
    }
    if from_frame_seq is not None:
        header["from_frame_seq"] = int(from_frame_seq)
    if cloud_map_id is not None:
        header["cloud_map_id"] = int(cloud_map_id)
    return header


def pose_hint(
    seq: int,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    dyaw: float = 0.0,
    confidence: float = 0.0,
    inliers: int = 0,
    method: str = "",
    frame: str = "map",
    map_id: str = "",
    from_frame_seq: int | None = None,
    data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Header for the correction Compare *offers* to SLAM.

    The dotted arrow in the diagram is dotted on purpose and so is this message.
    It carries a planar correction to apply to the robot's pose — the transform
    that would bring the reconstruction into agreement with the grid — and
    nothing about it obliges SLAM to take it.  SLAM has constraints the server
    cannot see (its own graph, its loop closures, the scan history) and is the
    only thing entitled to decide what the robot's pose is.

    ``advisory`` is therefore always true and is stated in the message rather
    than left as a convention, so that a future consumer reading only the header
    cannot mistake this for a command.  ``confidence`` and ``inliers`` are what
    it should be weighed by; a hint with few inliers is the reconstruction
    agreeing with a corner of the room and nothing else.
    """
    header = {
        "type": MSG_POSE_HINT,
        "seq": seq,
        "frame": frame,
        "map_id": map_id,
        "dx": round(float(dx), 4),
        "dy": round(float(dy), 4),
        "dyaw": round(float(dyaw), 5),
        "confidence": round(float(confidence), 3),
        "inliers": int(inliers),
        "method": method,
        "advisory": True,
        "data": data or {},
        "t_send_ns": monotonic_ns(),
    }
    if from_frame_seq is not None:
        header["from_frame_seq"] = int(from_frame_seq)
    return header


def exits(
    seq: int,
    candidates,
    t_capture_ns: int,
    *,
    frame: str = "map",
    map_id: str = "",
    source: str = "frontier",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Header for the robot's exit candidates.  No payload; there are a handful.

    Each candidate is a dict — see :func:`robocam.mission.normalise_exit` for the
    fields and what they mean.  They go up rather than being found on the server
    because the robot is the one that knows which of them it has already driven
    to and which turned out to be a wall; the server ranks, the robot remembers.
    """
    header = {
        "type": MSG_EXITS,
        "seq": seq,
        "frame": frame,
        "map_id": map_id,
        "candidates": list(candidates),
        "source": source,
        "t_capture_ns": t_capture_ns,
        "t_send_ns": monotonic_ns(),
    }
    if extra:
        header["extra"] = extra
    return header


def exits_result(
    seq: int,
    ok: bool,
    reason: str = REASON_OK,
    *,
    ranked=None,
    chosen: str = "",
    server_ms: float = 0.0,
    data: Dict[str, Any] | None = None,
    t_capture_ns: int | None = None,
    t_send_ns: int | None = None,
) -> Dict[str, Any]:
    """Header for the ranked exits coming back.

    ``chosen`` is the server's single recommendation and ``ranked`` is the whole
    ordering with a reason per entry.  Both, rather than just the winner, because
    the robot may reject the recommendation for a reason the server cannot see —
    the corridor is blocked, the battery is low — and it needs a second choice
    without another round trip.
    """
    header = {
        "type": MSG_EXITS_RESULT,
        "seq": seq,
        "ok": ok,
        "reason": reason,
        "ranked": list(ranked or []),
        "chosen": chosen,
        "server_ms": round(server_ms, 3),
        "data": data or {},
    }
    if t_capture_ns is not None:
        header["t_capture_ns"] = t_capture_ns
    if t_send_ns is not None:
        header["t_send_ns"] = t_send_ns
    return header


def found(
    seq: int,
    target: str,
    *,
    found: bool = True,
    confidence: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
    approach=None,
    frame: str = "map",
    map_id: str = "",
    phase: str = PHASE_SEEK,
    decider: str = "",
    rationale: str = "",
    basis: str = "live",
    age_s: float = 0.0,
    reachable: bool | None = None,
    reach: Dict[str, Any] | None = None,
    evidence: Dict[str, Any] | None = None,
    data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Header for the T2 announcement: the target, and where to go for it.

    The bottom-right box of the diagram, arriving at the robot's Seek behaviour
    tree.  Two poses rather than one: ``x, y, z, yaw`` is where the target *is*,
    and ``approach`` is where the robot should stand to act on it.  They differ
    by the robot's own reach and by the fact that driving to the centroid of a
    thing means driving into it, and a behaviour tree handed only the first has
    to invent the second from a standoff distance it has no way to know.

    ``found: false`` is a real and useful message — the decision stage having
    looked and concluded the target is not here is what lets the robot stop
    waiting and pick another exit, rather than seeking until it times out.

    ``basis`` says how the coordinate was arrived at, and the robot's two search
    branches are exactly its two values:

    ``live``    the decision stage is looking at the target in this frame.  Drive
                to it; it is there now.
    ``memory``  the target was seen earlier — usually in T1, before it was named
                — and this is where it was, ``age_s`` seconds ago.  Drive there,
                keep watching, and expect to have to search if it has moved.

    ``reachable`` is a three-valued answer and the third value carries weight.
    True and False are the gripper envelope; ``null`` means the height could not
    be measured for this coordinate, and the robot's correct response is to go
    and look rather than to assume either.  ``reach.verdict`` says which case it
    is in one word — see :mod:`robocam.seek` — and ``reach.reason`` says it in a
    sentence, for the log.

    ``rationale`` is free text from whatever made the decision.  It is for the
    log and for the human reading it afterwards; nothing on the robot should
    parse it.
    """
    header = {
        "type": MSG_FOUND,
        "seq": seq,
        "phase": phase,
        "target": target,
        "found": bool(found),
        "confidence": round(float(confidence), 3),
        "frame": frame,
        "map_id": map_id,
        "x": round(float(x), 4),
        "y": round(float(y), 4),
        "z": round(float(z), 4),
        "yaw": round(float(yaw), 5),
        "basis": str(basis),
        "age_s": round(float(age_s), 2),
        # Explicitly three-valued: None survives JSON as null, and a robot that
        # tests `if header["reachable"]` treats it as "no" while one that tests
        # `is False` treats it as "go and look".  The docstring says which is
        # meant; collapsing it to a bool here would remove the choice.
        "reachable": None if reachable is None else bool(reachable),
        "reach": reach or {},
        "decider": decider,
        "rationale": rationale,
        "evidence": evidence or {},
        "data": data or {},
        "t_send_ns": monotonic_ns(),
    }
    if approach is not None:
        ax, ay, ayaw = (float(v) for v in approach)
        header["approach"] = {"x": round(ax, 4), "y": round(ay, 4), "yaw": round(ayaw, 5)}
    return header


def phase(
    name: str,
    *,
    reason: str = "",
    mission: Dict[str, Any] | None = None,
    accepted: bool | None = None,
    seq: int | None = None,
) -> Dict[str, Any]:
    """Header for a phase change, in either direction.

    Either end may send it: the robot when it decides the map is good enough,
    the server when the decision stage concludes there is nothing left to
    explore.  The receiver answers with the same message and ``accepted`` set,
    which is what keeps the two ends from spending a minute in different phases
    after a lost message.
    """
    header = {"type": MSG_PHASE, "phase": name, "reason": reason,
              "t_send_ns": monotonic_ns()}
    if mission is not None:
        header["mission"] = mission
    if accepted is not None:
        header["accepted"] = bool(accepted)
    if seq is not None:
        header["seq"] = int(seq)
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
