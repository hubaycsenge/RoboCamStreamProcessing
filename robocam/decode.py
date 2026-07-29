"""Turning wire payloads back into numpy images.

JPEG and raw are stateless, so any frame can be decoded on its own and a dropped
frame costs nothing.  H.264 is stateful and therefore needs one decoder instance
per session, which is why decoding lives behind a per-session object rather than
a module-level function.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

from . import wire

log = logging.getLogger(__name__)


class DecodeError(Exception):
    pass


class SessionDecoder:
    """Decodes the payloads of a single session.

    One instance per connected robot: the H.264 path carries state across
    frames, and mixing two streams through one decoder produces garbage.
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self._h264 = None  # lazily created PyAV codec context

    # -- public -----------------------------------------------------------

    def decode(self, header: Dict[str, Any], payload: bytes) -> np.ndarray:
        """Decode one payload to an HxWx3 BGR uint8 array.

        Raises DecodeError on anything malformed; the caller turns that into an
        unsuccessful result rather than dropping the connection.
        """
        codec = header.get("codec", wire.CODEC_JPEG)
        if not payload:
            raise DecodeError("empty payload")

        if codec == wire.CODEC_JPEG:
            return self._decode_jpeg(payload)
        if codec == wire.CODEC_RAW_BGR:
            return self._decode_raw(header, payload)
        if codec == wire.CODEC_H264:
            return self._decode_h264(payload)
        raise DecodeError(f"unsupported codec {codec!r}")

    def close(self) -> None:
        if self._h264 is not None:
            try:
                self._h264.close()
            except Exception:  # pragma: no cover - teardown best effort
                pass
            self._h264 = None

    # -- codecs -----------------------------------------------------------

    @staticmethod
    def _decode_jpeg(payload: bytes) -> np.ndarray:
        buf = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise DecodeError("cv2.imdecode returned None (truncated or non-JPEG payload)")
        return img

    @staticmethod
    def _decode_raw(header: Dict[str, Any], payload: bytes) -> np.ndarray:
        try:
            w = int(header["width"])
            h = int(header["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DecodeError("raw_bgr requires integer width/height in the header") from exc
        c = int(header.get("channels", 3))
        if w <= 0 or h <= 0 or c not in (1, 3, 4):
            raise DecodeError(f"implausible raw geometry {w}x{h}x{c}")

        expected = w * h * c
        if len(payload) != expected:
            raise DecodeError(f"raw payload is {len(payload)} bytes, expected {expected} for {w}x{h}x{c}")

        img = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, c)
        if c == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif c == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # frombuffer gives a read-only view of the zmq buffer; processors will
        # want to write to it, so hand back an owned copy.
        return np.ascontiguousarray(img)

    def _decode_h264(self, payload: bytes) -> np.ndarray:
        ctx = self._ensure_h264()
        try:
            import av  # noqa: F401  (import checked in _ensure_h264)

            packets = ctx.parse(payload)
            frames = []
            for packet in packets:
                frames.extend(ctx.decode(packet))
        except Exception as exc:
            raise DecodeError(f"h264 decode failed: {exc}") from exc

        if not frames:
            # Normal at stream start: the decoder needs a keyframe before it can
            # emit anything.  Treated as a decode failure for this frame only.
            raise DecodeError("h264 decoder produced no frame yet (waiting for keyframe)")
        return frames[-1].to_ndarray(format="bgr24")

    def _ensure_h264(self):
        if self._h264 is None:
            try:
                import av
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise DecodeError(
                    "h264 support needs PyAV: uv pip install av"
                ) from exc
            self._h264 = av.CodecContext.create("h264", "r")
        return self._h264


def encode_jpeg(img: np.ndarray, quality: int = 85) -> Optional[bytes]:
    """Encode a BGR array as JPEG.  Used for snapshots and by the test client."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return buf.tobytes()
