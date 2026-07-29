#!/usr/bin/env python3
"""RoboCam client — runs on the Jetson Orin Nano.

Captures the webcam, encodes each frame, streams it to the server and hands the
returned result to a callback.  Single file with no dependency on the ``robocam``
package, so deploying it is one ``scp``.

    pip3 install pyzmq numpy opencv-python      # opencv usually already on JetPack
    python3 robocam_client.py --server tcp://10.128.17.196:5555

Test the link without a camera:

    python3 robocam_client.py --server tcp://10.128.17.196:5555 --source synthetic

Use as a library:

    client = RoboCamClient("tcp://10.128.17.196:5555", on_result=my_callback)
    client.run()
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import zmq

log = logging.getLogger("robocam.client")

PROTOCOL_VERSION = 1

MSG_HELLO = "hello"
MSG_WELCOME = "welcome"
MSG_FRAME = "frame"
MSG_RESULT = "result"
MSG_PING = "ping"
MSG_PONG = "pong"
MSG_BYE = "bye"
MSG_ERROR = "error"

CODEC_JPEG = "jpeg"
CODEC_RAW_BGR = "raw_bgr"


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


class FrameSource:
    """Yields ``(bgr_image, jpeg_bytes_or_None)`` pairs.

    A source may return pre-encoded JPEG bytes — the hardware encoder path does
    — in which case the client skips software encoding entirely.
    """

    width = 0
    height = 0
    fps = 0.0

    def frames(self) -> Iterator[Tuple[Optional[np.ndarray], Optional[bytes]]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SyntheticSource(FrameSource):
    """A moving test pattern.  Proves the link works before a camera is involved."""

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0) -> None:
        self.width, self.height, self.fps = width, height, fps
        self.period = 1.0 / fps if fps > 0 else 0.0
        # A static gradient background, so only the moving parts cost anything
        # per frame and the encoder still sees realistic entropy.
        xs = np.linspace(0, 255, width, dtype=np.float32)
        ys = np.linspace(0, 255, height, dtype=np.float32)
        self._bg = np.dstack([
            np.tile(xs, (height, 1)),
            np.tile(ys[:, None], (1, width)),
            np.full((height, width), 128.0, dtype=np.float32),
        ]).astype(np.uint8)

    def frames(self):
        import cv2

        i = 0
        while True:
            t0 = time.perf_counter()
            img = self._bg.copy()
            # A marker that moves every frame, so a frozen stream is obvious.
            cx = int((0.5 + 0.4 * np.sin(i / 20.0)) * self.width)
            cy = int((0.5 + 0.4 * np.cos(i / 17.0)) * self.height)
            cv2.circle(img, (cx, cy), 60, (0, 0, 255), -1)
            cv2.putText(img, f"synthetic {i}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            yield img, None
            i += 1
            if self.period:
                sleep = self.period - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)

    def __len__(self) -> int:  # pragma: no cover
        return 0


class OpenCVSource(FrameSource):
    """Capture through OpenCV.  Works with /dev/videoN, a file, or a GStreamer pipeline."""

    def __init__(self, device: str = "0", width: int = 1280, height: int = 720,
                 fps: float = 30.0, fourcc: str = "MJPG") -> None:
        import cv2

        self._cv2 = cv2
        if device.isdigit():
            self.cap = cv2.VideoCapture(int(device), cv2.CAP_V4L2)
        elif " ! " in device:
            self.cap = cv2.VideoCapture(device, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(device)

        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source {device!r}")

        if device.isdigit():
            # Ask the camera for MJPG: most USB webcams can only do 30 fps at
            # 720p in MJPG, and fall back to 5-10 fps if left in YUYV.
            if fourcc:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, fps)
            # A deep driver buffer means you act on stale frames; keep it at 1.
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(actual_fps) if actual_fps > 0 else fps
        log.info("camera opened: %dx%d @%.1f fps (requested %dx%d @%.1f)",
                 self.width, self.height, actual_fps, width, height, fps)

    def frames(self):
        misses = 0
        while True:
            ok, img = self.cap.read()
            if not ok or img is None:
                misses += 1
                if misses > 30:
                    raise RuntimeError("camera returned 30 consecutive empty frames")
                time.sleep(0.01)
                continue
            misses = 0
            yield img, None

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


class GstJpegSource(FrameSource):
    """Hardware-encoded JPEG straight out of the Orin's NVJPEG block.

    Avoids the decode-to-BGR-then-re-encode round trip that OpenCV forces, which
    on an Orin Nano is most of a CPU core at 720p30.  Needs PyGObject and the
    GStreamer introspection data, both standard on JetPack::

        sudo apt install python3-gi gir1.2-gstreamer-1.0

    The client never sees pixels in this mode — it forwards the JPEG untouched.
    """

    def __init__(self, device: str = "/dev/video0", width: int = 1280,
                 height: int = 720, fps: float = 30.0, quality: int = 85) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        self._Gst = Gst
        Gst.init(None)
        self.width, self.height, self.fps = width, height, fps

        # nvvidconv moves the buffer into NVMM memory where nvjpegenc can reach
        # it; both elements ship with JetPack.
        pipeline = (
            f"v4l2src device={device} io-mode=2 ! "
            f"image/jpeg,width={width},height={height},framerate={int(fps)}/1 ! "
            f"jpegparse ! appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
        )
        log.info("gstreamer pipeline: %s", pipeline)
        self.pipeline = Gst.parse_launch(pipeline)
        self.sink = self.pipeline.get_by_name("sink")
        self.pipeline.set_state(Gst.State.PLAYING)

    def frames(self):
        Gst = self._Gst
        while True:
            sample = self.sink.emit("pull-sample")
            if sample is None:
                time.sleep(0.005)
                continue
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                yield None, bytes(info.data)
            finally:
                buf.unmap(info)

    def close(self) -> None:
        try:
            self.pipeline.set_state(self._Gst.State.NULL)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class RoboCamClient:
    """Streams frames to the server and dispatches results to a callback.

    Flow control: at most ``max_inflight`` frames may be awaiting a result.
    Beyond that the client drops frames at the source rather than letting them
    pile up in a socket buffer, because a queued frame is a stale frame and the
    robot should be reacting to now, not to two seconds ago.
    """

    def __init__(
        self,
        server: str,
        client_id: str = "orin",
        codec: str = CODEC_JPEG,
        jpeg_quality: int = 85,
        max_inflight: int = 3,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
        reconnect_after_s: float = 5.0,
    ) -> None:
        self.server = server
        self.client_id = client_id
        self.codec = codec
        self.jpeg_quality = jpeg_quality
        self.max_inflight = max(1, max_inflight)
        self.on_result = on_result
        self.reconnect_after_s = reconnect_after_s

        self.ctx = zmq.Context.instance()
        self.sock: Optional[zmq.Socket] = None
        self._pending: Dict[int, float] = {}   # seq -> monotonic send time
        self._seq = 0
        self._stop = False
        self._connected = False
        self._last_rx = time.monotonic()

        # Counters for the periodic status line.
        self.sent = 0
        self.results_ok = 0
        self.results_bad = 0
        self.skipped_backpressure = 0
        self._rtt_sum = 0.0
        self._rtt_n = 0

    # -- connection -------------------------------------------------------

    def connect(self) -> None:
        self.close_socket()
        self.sock = self.ctx.socket(zmq.DEALER)
        # A stable identity means the server recognises us across reconnects.
        self.sock.setsockopt(zmq.IDENTITY, self.client_id.encode("utf-8")[:64])
        self.sock.setsockopt(zmq.SNDHWM, 8)
        self.sock.setsockopt(zmq.RCVHWM, 64)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.server)
        self._pending.clear()
        self._connected = False
        self._last_rx = time.monotonic()
        log.info("connecting to %s as %r", self.server, self.client_id)
        self._send({
            "type": MSG_HELLO,
            "protocol": PROTOCOL_VERSION,
            "client_id": self.client_id,
            "codec": self.codec,
            "width": getattr(self, "_src_width", 0),
            "height": getattr(self, "_src_height", 0),
            "fps": getattr(self, "_src_fps", 0.0),
            "camera": getattr(self, "_src_name", ""),
            "t_send_ns": time.monotonic_ns(),
        })

    def close_socket(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None

    def _send(self, header: Dict[str, Any], payload: bytes = b"") -> bool:
        if self.sock is None:
            return False
        try:
            self.sock.send_multipart(
                [json.dumps(header, separators=(",", ":")).encode("utf-8"), payload],
                zmq.NOBLOCK,
            )
            return True
        except zmq.Again:
            return False

    # -- main loop --------------------------------------------------------

    def run(self, source: FrameSource, duration: float = 0.0, status_every: float = 5.0) -> None:
        import cv2

        self._src_width = source.width
        self._src_height = source.height
        self._src_fps = getattr(source, "fps", 0.0)
        self._src_name = type(source).__name__

        self.connect()
        t_start = time.monotonic()
        t_status = t_start

        try:
            for img, pre_encoded in source.frames():
                if self._stop:
                    break

                self._drain_results()

                now = time.monotonic()
                if now - self._last_rx > self.reconnect_after_s:
                    log.warning("no reply for %.1fs, reconnecting", now - self._last_rx)
                    self.connect()

                if len(self._pending) >= self.max_inflight:
                    # Server is behind. Drop this frame at the source.
                    self.skipped_backpressure += 1
                else:
                    self._send_frame(img, pre_encoded, cv2)

                if status_every > 0 and now - t_status >= status_every:
                    self._log_status(now - t_status)
                    t_status = now

                if duration > 0 and now - t_start >= duration:
                    break
        except KeyboardInterrupt:  # pragma: no cover - interactive
            log.info("interrupted")
        finally:
            self._send({"type": MSG_BYE, "reason": "client shutting down"})
            # Give the goodbye a moment to leave the socket.
            time.sleep(0.05)
            source.close()
            self.close_socket()

    def _send_frame(self, img: Optional[np.ndarray], pre_encoded: Optional[bytes], cv2) -> None:
        t_capture_ns = time.monotonic_ns()

        if pre_encoded is not None:
            payload = pre_encoded
            width, height, channels = self._src_width, self._src_height, 3
            codec = CODEC_JPEG
        elif img is None:
            return
        elif self.codec == CODEC_RAW_BGR:
            payload = np.ascontiguousarray(img).tobytes()
            height, width = img.shape[:2]
            channels = img.shape[2] if img.ndim == 3 else 1
            codec = CODEC_RAW_BGR
        else:
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                log.warning("jpeg encode failed, skipping frame")
                return
            payload = buf.tobytes()
            height, width = img.shape[:2]
            channels = img.shape[2] if img.ndim == 3 else 1
            codec = CODEC_JPEG

        seq = self._seq
        self._seq += 1
        header = {
            "type": MSG_FRAME,
            "seq": seq,
            "codec": codec,
            "width": int(width),
            "height": int(height),
            "channels": int(channels),
            "t_capture_ns": t_capture_ns,
            "t_send_ns": time.monotonic_ns(),
        }
        if self._send(header, payload):
            self._pending[seq] = time.monotonic()
            self.sent += 1
        else:
            self.skipped_backpressure += 1

    def _drain_results(self) -> None:
        if self.sock is None:
            return
        while True:
            try:
                parts = self.sock.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            except zmq.ZMQError as exc:  # pragma: no cover
                log.warning("recv failed: %s", exc)
                return

            self._last_rx = time.monotonic()
            try:
                header = json.loads(parts[0].decode("utf-8"))
            except (IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                log.warning("unparseable message from server: %s", exc)
                continue
            self._handle(header)

    def _handle(self, header: Dict[str, Any]) -> None:
        mtype = header.get("type")

        if mtype == MSG_WELCOME:
            if header.get("accepted"):
                self._connected = True
                log.info("server accepted session %s (processor=%s, host=%s)",
                         header.get("session_id"), header.get("processor"),
                         header.get("server", {}).get("host", "?"))
            else:
                log.error("server rejected session: %s", header.get("message"))
                self._stop = True
            return

        if mtype == MSG_RESULT:
            seq = int(header.get("seq", -1))
            sent_at = self._pending.pop(seq, None)
            if sent_at is not None:
                rtt_ms = (time.monotonic() - sent_at) * 1000.0
                header["rtt_ms"] = round(rtt_ms, 2)
                self._rtt_sum += rtt_ms
                self._rtt_n += 1
            if header.get("ok"):
                self.results_ok += 1
            else:
                self.results_bad += 1
                if header.get("reason") not in ("dropped",):
                    log.warning("seq=%d not ok: %s %s", seq, header.get("reason"),
                                header.get("data", {}).get("error", ""))
            if self.on_result is not None:
                try:
                    self.on_result(header)
                except Exception:
                    log.exception("on_result callback raised")
            return

        if mtype == MSG_PONG:
            return
        if mtype == MSG_ERROR:
            log.error("server error: %s", header.get("message"))
            return
        log.warning("unexpected message type from server: %r", mtype)

    def _log_status(self, elapsed: float) -> None:
        rtt = self._rtt_sum / self._rtt_n if self._rtt_n else 0.0
        log.info(
            "sent=%d ok=%d bad=%d skipped=%d | %.1f fps out | rtt %.1f ms | inflight=%d",
            self.sent, self.results_ok, self.results_bad, self.skipped_backpressure,
            self.sent / elapsed if elapsed > 0 else 0.0, rtt, len(self._pending),
        )
        self.sent = 0
        self.results_ok = 0
        self.results_bad = 0
        self.skipped_backpressure = 0
        self._rtt_sum = 0.0
        self._rtt_n = 0

    def stop(self) -> None:
        self._stop = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_source(args) -> FrameSource:
    if args.source == "synthetic":
        return SyntheticSource(args.width, args.height, args.fps)
    if args.source == "gst-jpeg":
        return GstJpegSource(args.device, args.width, args.height, args.fps, args.quality)
    return OpenCVSource(args.device, args.width, args.height, args.fps, args.fourcc)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="robocam-client",
        description="Stream the robot's webcam to the processing server.",
    )
    p.add_argument("-s", "--server", default="tcp://10.128.17.196:5555",
                   help="server endpoint (default: %(default)s)")
    p.add_argument("--client-id", default="orin", help="identifies this robot to the server")
    p.add_argument("--source", choices=["camera", "synthetic", "gst-jpeg"], default="camera",
                   help="camera: OpenCV capture; synthetic: test pattern, no camera needed; "
                        "gst-jpeg: hardware JPEG via GStreamer (Jetson)")
    p.add_argument("--device", default="0", help="/dev/videoN index, path, or GStreamer pipeline")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--fourcc", default="MJPG", help="camera pixel format to request (MJPG or YUYV)")
    p.add_argument("--codec", choices=[CODEC_JPEG, CODEC_RAW_BGR], default=CODEC_JPEG)
    p.add_argument("--quality", type=int, default=85, help="JPEG quality, 1-100")
    p.add_argument("--max-inflight", type=int, default=3,
                   help="frames allowed to be awaiting a result before dropping at the source")
    p.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = forever)")
    p.add_argument("--print-results", action="store_true", help="print every result as JSON")
    p.add_argument("--status-every", type=float, default=5.0, help="status line interval, 0 to disable")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    def on_result(result: Dict[str, Any]) -> None:
        if args.print_results:
            print(json.dumps(result, separators=(",", ":")), flush=True)

    client = RoboCamClient(
        server=args.server,
        client_id=args.client_id,
        codec=args.codec,
        jpeg_quality=args.quality,
        max_inflight=args.max_inflight,
        on_result=on_result,
    )

    signal.signal(signal.SIGINT, lambda *_: client.stop())
    signal.signal(signal.SIGTERM, lambda *_: client.stop())

    try:
        source = build_source(args)
    except Exception as exc:
        log.error("could not open source: %s", exc)
        return 1

    client.run(source, duration=args.duration, status_every=args.status_every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
