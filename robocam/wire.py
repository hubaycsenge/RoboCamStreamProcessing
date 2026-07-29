"""Wire protocol for the RoboCam video link.

Transport is ZeroMQ: the robot opens a DEALER socket, the server binds a ROUTER.
Every application message is a two-frame multipart message::

    [ header_json, payload ]

``header`` is a small UTF-8 JSON object, ``payload`` is opaque bytes (the encoded
image for ``frame`` messages, empty for everything else).  ROUTER prepends the
peer identity on receive and strips it on send, so the server actually handles
``[identity, header, payload]``.  DEALER does not insert the empty delimiter
frame that REQ/REP would, so the framing is exactly as written above.

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

# Why a result can be unsuccessful.  These end up in ``result.reason``.
REASON_OK = "ok"
REASON_DROPPED = "dropped"        # queue was full, frame never reached a worker
REASON_DECODE_FAILED = "decode_failed"
REASON_PROCESSOR_FAILED = "processor_failed"
REASON_UNSUPPORTED_CODEC = "unsupported_codec"


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
) -> Dict[str, Any]:
    """Header for one result.

    ``t_capture_ns`` and ``t_send_ns`` are echoed straight back from the frame
    header so the client can compute latency against its own clock.
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
