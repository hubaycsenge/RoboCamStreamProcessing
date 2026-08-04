#!/usr/bin/env python3
"""RoboCam client — runs on the Jetson Orin Nano.

Captures the webcam, encodes each frame, streams it to the server and hands the
returned result to a callback.  If the robot's LDS-02 LiDAR is attached, its
scans go down the same socket alongside the frames.  Single file with no
dependency on the ``robocam`` package, so deploying it is one ``scp``.

    pip3 install pyzmq numpy opencv-python      # opencv usually already on JetPack
    python3 robocam_client.py --server tcp://10.128.17.196:5555

Test the link without a camera:

    python3 robocam_client.py --server tcp://10.128.17.196:5555 --source synthetic

Camera plus LiDAR, the normal robot configuration:

    python3 robocam_client.py --server tcp://10.128.17.196:5555 --lidar auto

Use as a library:

    client = RoboCamClient("tcp://10.128.17.196:5555", on_result=my_callback)
    client.run(OpenCVSource("0"), scan_source=build_scan_source_auto())

The camera and the LiDAR are read independently — the scanner runs on its own
thread with a latest-wins slot — because they run at different rates (~30 Hz and
~5 Hz) and neither should ever wait for the other.  The server pairs them by
arrival time; see the module docstring of robocam/lidar.py for what that pairing
does and does not justify.
"""

from __future__ import annotations

import argparse
import errno
import glob
import json
import logging
import math
import os
import signal
import stat
import struct
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import zmq

log = logging.getLogger("robocam.client")

PROTOCOL_VERSION = 1

MSG_HELLO = "hello"
MSG_WELCOME = "welcome"
MSG_FRAME = "frame"
MSG_RESULT = "result"
MSG_SCAN = "scan"
MSG_SCAN_RESULT = "scan_result"
MSG_PING = "ping"
MSG_PONG = "pong"
MSG_BYE = "bye"
MSG_ERROR = "error"

CODEC_JPEG = "jpeg"
CODEC_RAW_BGR = "raw_bgr"

SCAN_ENC_U16_MM = "u16mm"

TWO_PI = 2.0 * math.pi

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


class NullSource(FrameSource):
    """No camera at all, paced so the main loop keeps servicing the LiDAR.

    For running the robot's scanner on its own — while the camera is unplugged,
    or to measure what the LiDAR path costs by itself.
    """

    def __init__(self, hz: float = 50.0) -> None:
        self.width = self.height = 0
        self.fps = 0.0
        self.period = 1.0 / hz if hz > 0 else 0.02

    def frames(self):
        while True:
            time.sleep(self.period)
            yield None, None


# ---------------------------------------------------------------------------
# LiDAR sources
# ---------------------------------------------------------------------------
#
# The robot's LDS-02 is a planar 360° scanner: ~5 Hz, 0.12–12 m, 360 points a
# revolution, reported in millimetres.  Two ways in, and which one you have
# depends on how the Orin is set up rather than on anything here:
#
#   ros2    — the LDS-02 is already published as sensor_msgs/LaserScan on /scan
#             by ld08_driver.  Use this if ROS is running; it is the one that
#             keeps working when someone re-mounts the scanner and updates the
#             URDF instead of telling you.
#   serial  — talk to the device directly over its USB serial link (115200
#             baud, the LD08/LD19 packet format).  Use this if there is no ROS
#             on the robot, or to take the scanner away from ROS entirely.
#
# ``--lidar auto`` tries them in that order.  It deliberately does *not* fall
# back to the synthetic scanner: a robot that silently reports a fictional room
# is far worse than one that reports no LiDAR at all.
#
# Silence from a scanner has several causes that look identical from the outside
# — nothing plugged in, nothing spinning, wrong baud, wrong device, someone else
# holding the port.  They are distinguishable by *how far the data gets*, so both
# sources count the stages they reach and report the furthest one instead of
# simply producing nothing.  That is what :class:`_ScanHealth` is for.


#: Where the scanner appears in ``/dev``.  The udev rule on the robot matches the
#: scanner's CP2102 bridge (10c4:ea60) and gives it a stable name; which name you
#: get depends on which revision of that rule is installed, and both have been in
#: use on this fleet.  ``tb3_lidar`` is what the robot currently has.
#:
#: Raw ``ttyUSBn`` numbers are deliberately absent: they are not stable across
#: boots or replugs — the scanner is on ttyUSB1 today — and on this robot ttyUSB0
#: is as likely to be the OpenCR board as the scanner.  Opening a motor
#: controller and reading it as ranges is worse than failing, so a raw device is
#: only ever used when ``--lidar-port`` names one explicitly.
LIDAR_PORT_ALIASES = ("/dev/tb3_lidar", "/dev/ld08_lidar")
DEFAULT_LIDAR_PORT = LIDAR_PORT_ALIASES[0]

#: The LDS-02's line rate.  The LD08 inside it shares the LD06/LD19 *packet
#: format* but not their baud — those run at 230400 and the LD08 at 115200 — and
#: the two are easy to conflate because every other detail matches.  Reading at
#: 230400 samples every bit twice: bytes arrive steadily, at roughly double the
#: true rate, and not one of them frames.  See :meth:`SerialLdsSource._explain_no_frames`.
LDS02_BAUD = 115200


class LidarUnavailable(RuntimeError):
    """No scanner could be opened.  The message says which stage failed and why."""


def _describe_port(path: str) -> str:
    """What ``path`` actually is — symlink target, device type, permissions."""
    if not os.path.exists(path):
        return f"{path}: no such device node"

    bits = []
    target = os.path.realpath(path)
    bits.append(f"{path} -> {target}" if target != path else path)
    try:
        st = os.stat(path)
        bits.append("mode %04o" % (st.st_mode & 0o7777))
        if not stat.S_ISCHR(st.st_mode):
            bits.append("NOT a character device")
    except OSError as exc:
        bits.append(f"stat failed: {exc}")
    if not os.access(path, os.R_OK | os.W_OK):
        bits.append("not read/writable by this user")
    return ", ".join(bits)


def _serial_devices_present() -> List[str]:
    """Serial devices that exist right now, for when the wanted one does not."""
    found: List[str] = []
    for pattern in ("/dev/tb3_lidar", "/dev/ld08_lidar", "/dev/opencr",
                    "/dev/arduino_nano", "/dev/ttyUSB*", "/dev/ttyACM*"):
        for path in sorted(glob.glob(pattern)):
            target = os.path.realpath(path)
            entry = f"{path} -> {os.path.basename(target)}" if target != path else path
            if entry not in found:
                found.append(entry)
    return found


def _explain_open_failure(path: str, exc: Exception) -> str:
    """Turn a pyserial exception into something that names the actual fix."""
    hints = []
    code = getattr(exc, "errno", None)
    if not os.path.exists(path):
        hints.append(
            "the udev rule that creates this name may not be installed — see "
            "mecanumbot_description/udev/99-turtlebot3-cdc.rules, then "
            "`sudo udevadm control --reload-rules && sudo udevadm trigger`"
        )
    elif code == errno.EACCES:
        hints.append("permission denied — `sudo usermod -aG dialout $USER`, "
                     "then log out and back in")
    elif code in (errno.EBUSY, errno.EAGAIN):
        hints.append("the port is already open — ld08_driver is most likely "
                     "holding it; stop the ROS driver, or use --lidar ros2 "
                     "and share it through the topic instead")

    detail = "%s (%s)" % (exc, _describe_port(path))
    return "%s. %s" % (detail, " ".join(hints)) if hints else detail


def _devices_present_line() -> str:
    present = _serial_devices_present()
    return "Serial devices present: %s." % (", ".join(present) if present else "none")


class _ScanHealth:
    """Periodic progress reports while a scanner is being read.

    Two jobs.  Until the first revolution arrives it says which stage the data
    reached, because "no scans" alone does not distinguish a dead scanner from a
    misconfigured one.  After that it reports rate and coverage, so a scanner
    that is technically alive but returning almost nothing is visible rather
    than merely quiet.

    ``diagnose`` is supplied by the caller and returns the stage description; it
    is only consulted while nothing has arrived yet.
    """

    def __init__(self, what: str, diagnose: Callable[[float], str], every: float = 5.0) -> None:
        self.what = what
        self.every = float(every)
        self._diagnose = diagnose
        self.t0 = time.monotonic()
        self._t_last = self.t0
        self._t_last_rev = self.t0
        self.revolutions = 0
        self.last_filled = 0
        self.last_points = 0

    def revolution(self, filled: int, points: int) -> None:
        """One complete revolution came out."""
        self.revolutions += 1
        self.last_filled, self.last_points = filled, points
        if self.revolutions == 1:
            log.info("lidar: %s — first revolution after %.1fs, %d/%d points (%.0f%% coverage)",
                     self.what, time.monotonic() - self.t0, filled, points,
                     100.0 * filled / max(1, points))
        self._t_last_rev = time.monotonic()

    def tick(self) -> None:
        """Call often; emits at most one line per ``every`` seconds."""
        if self.every <= 0:
            return
        now = time.monotonic()
        if now - self._t_last < self.every:
            return
        self._t_last = now

        if self.revolutions == 0:
            log.warning("lidar: %s — no scan yet after %.1fs. %s",
                        self.what, now - self.t0, self._diagnose(now - self.t0))
            return

        stale = now - self._t_last_rev
        if stale > self.every:
            log.warning("lidar: %s — %d revolutions, then nothing for %.0fs. "
                        "The scanner stopped producing; check power and that the "
                        "rotor is still turning.", self.what, self.revolutions, stale)
            return
        log.info("lidar: %s — %d revolutions, %.1f/s, %d/%d points (%.0f%% coverage)",
                 self.what, self.revolutions, self.revolutions / max(1e-9, now - self.t0),
                 self.last_filled, self.last_points,
                 100.0 * self.last_filled / max(1, self.last_points))


class ScanReading:
    """One revolution, in the form the wire wants it.

    ``ranges_mm`` is uint16 millimetres with 0 for "no return", which is the
    LDS-02's own convention and needs no conversion in either direction.
    """

    __slots__ = ("ranges_mm", "angle_min", "angle_increment", "range_min",
                 "range_max", "scan_time", "intensities", "t_capture_ns")

    def __init__(
        self,
        ranges_mm: np.ndarray,
        angle_min: float = 0.0,
        angle_increment: float = 0.0,
        range_min: float = 0.0,
        range_max: float = 0.0,
        scan_time: float = 0.0,
        intensities: Optional[np.ndarray] = None,
        t_capture_ns: int = 0,
    ) -> None:
        self.ranges_mm = ranges_mm
        self.angle_min = angle_min
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max
        self.scan_time = scan_time
        self.intensities = intensities
        self.t_capture_ns = t_capture_ns or time.monotonic_ns()

    def __len__(self) -> int:
        return int(self.ranges_mm.shape[0])


class ScanSource:
    """Yields :class:`ScanReading` objects, one per revolution."""

    #: Reported to the server in ``hello.lidar``, and used in log lines.
    model = "lidar"

    def info(self) -> Dict[str, Any]:
        return {"model": self.model}

    def scans(self) -> Iterator[ScanReading]:
        raise NotImplementedError

    def request_stop(self) -> None:
        """Ask :meth:`scans` to return at its next opportunity.

        Separate from :meth:`close` because the reader is a *thread* sitting in
        a blocking read: closing the device out from under it raises inside
        somebody else's library, which reads as a crash in the log even though
        the process was only shutting down.  Asking first and closing after the
        thread has left is the difference between a clean exit and a traceback.
        """

    def close(self) -> None:
        pass


class SyntheticScanSource(ScanSource):
    """A fake room: rectangular walls and one obstacle circling the robot.

    Never selected automatically — pass ``--lidar synthetic`` on purpose.  Its
    job is to prove the scan path end to end (client, wire, server analysis,
    overlay) without needing the robot powered up, and the shapes are chosen so
    that a wrong sign or a wrong mounting yaw is visible in the bird's-eye plot
    rather than hidden in plausible-looking numbers.
    """

    model = "synthetic"

    def __init__(self, points: int = 360, hz: float = 5.0,
                 room: Tuple[float, float, float, float] = (-2.0, 4.0, -3.0, 3.0)) -> None:
        self.points = int(points)
        self.hz = float(hz)
        self.room = room
        self.range_min, self.range_max = 0.12, 12.0
        self._stopping = False

    def info(self) -> Dict[str, Any]:
        return {"model": self.model, "points": self.points, "hz": self.hz,
                "note": "SYNTHETIC — not a real sensor"}

    def request_stop(self) -> None:
        self._stopping = True

    def scans(self) -> Iterator[ScanReading]:
        period = 1.0 / self.hz if self.hz > 0 else 0.2
        bearings = np.linspace(0.0, TWO_PI, self.points, endpoint=False, dtype=np.float64)
        dx, dy = np.cos(bearings), np.sin(bearings)
        xmin, xmax, ymin, ymax = self.room
        i = 0

        while not self._stopping:
            t0 = time.perf_counter()
            # Distance to the walls of an axis-aligned box, from inside it.
            t = np.full(self.points, np.inf)
            for delta, direction in ((xmin, dx), (xmax, dx), (ymin, dy), (ymax, dy)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    hit = delta / np.where(np.abs(direction) < 1e-9, np.nan, direction)
                t = np.minimum(t, np.where(hit > 0, hit, np.inf))

            # One obstacle orbiting the robot, so successive scans differ and a
            # frozen feed is as obvious as a frozen camera.
            angle = i * 0.05
            ox, oy, radius = 1.2 * math.cos(angle), 1.2 * math.sin(angle), 0.30
            b = -2.0 * (dx * ox + dy * oy)
            c = ox * ox + oy * oy - radius * radius
            disc = b * b - 4.0 * c
            with np.errstate(invalid="ignore"):
                root = (-b - np.sqrt(np.where(disc > 0, disc, np.nan))) / 2.0
            t = np.minimum(t, np.where(np.isfinite(root) & (root > 0), root, np.inf))

            ranges = np.where(np.isfinite(t) & (t <= self.range_max), t, 0.0)
            yield ScanReading(
                ranges_mm=_pack_mm(ranges),
                angle_min=0.0,
                angle_increment=TWO_PI / self.points,
                range_min=self.range_min,
                range_max=self.range_max,
                scan_time=period,
            )
            i += 1
            sleep = period - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)


class Ros2ScanSource(ScanSource):
    """sensor_msgs/LaserScan off a ROS 2 topic, normally ``/scan``.

    Uses ``rclpy`` with a best-effort, depth-1 subscription: the sensor QoS
    profile most LiDAR drivers publish with, and the one that matches what this
    client wants anyway — the newest revolution, never a backlog of old ones.
    """

    model = "lds02-ros2"

    def __init__(self, topic: str = "/scan", timeout_s: float = 5.0,
                 require_publisher: bool = False, discovery_s: float = 2.0,
                 health_every: Optional[float] = None) -> None:
        import rclpy
        from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import LaserScan

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("robocam_lidar_client")
        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._slot: Optional[Any] = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stopping = False
        self._topic = topic
        self._timeout_s = timeout_s
        self._sub = self._node.create_subscription(LaserScan, topic, self._on_msg, qos)
        log.info("lidar: subscribed to %s", topic)

        # Subscribing always succeeds, even to a topic nobody publishes — so
        # without this check ``auto`` would settle on ROS 2 whenever rclpy
        # imports and then sit silent forever, never trying the serial port.
        # Waiting for a publisher is what makes "ROS 2 is not the answer here"
        # an answer rather than a hang.
        publishers = self._await_publisher(discovery_s)
        if publishers:
            log.info("lidar: %d publisher(s) on %s", publishers, topic)
        elif require_publisher:
            self.close()
            raise LidarUnavailable(
                f"no publisher on {topic} after {discovery_s:.0f}s — ld08_driver is not "
                f"running, or ROS_DOMAIN_ID differs from this process's. "
                f"Check with `ros2 topic info {topic}`"
            )
        else:
            log.warning("lidar: nothing publishes %s yet — waiting. If the driver "
                        "should be up, check `ros2 topic info %s`", topic, topic)

        self._health = _ScanHealth(
            f"ros2 {topic}", self._diagnose,
            every=timeout_s if health_every is None else health_every,
        )

    def _await_publisher(self, deadline_s: float) -> int:
        """Spin briefly so discovery can complete, and report what turned up."""
        end = time.monotonic() + max(0.0, deadline_s)
        while True:
            count = self._node.count_publishers(self._topic)
            if count or time.monotonic() >= end:
                return count
            self._rclpy.spin_once(self._node, timeout_sec=0.05)

    def _diagnose(self, waited_s: float) -> str:
        count = self._node.count_publishers(self._topic)
        if count:
            return (f"{count} publisher(s) are advertising {self._topic} but none has sent "
                    f"a message. The driver is up and the scanner is not producing — check "
                    f"that the rotor is spinning and that the driver is not logging port errors.")
        return (f"nothing is publishing {self._topic}. Start ld08_driver, or check it is not "
                f"failing to open its port (`ros2 topic info {self._topic}`, then the "
                f"driver's own log).")

    def info(self) -> Dict[str, Any]:
        return {"model": self.model, "topic": self._topic, "transport": "ros2"}

    def request_stop(self) -> None:
        self._stopping = True

    def _on_msg(self, msg) -> None:
        with self._lock:
            self._slot = msg
        self._new.set()

    def scans(self) -> Iterator[ScanReading]:
        while not self._stopping:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
            self._health.tick()
            if not self._new.is_set():
                continue
            self._new.clear()
            with self._lock:
                msg = self._slot
            if msg is None:
                continue

            ranges = np.asarray(msg.ranges, dtype=np.float32)
            packed = _pack_mm(ranges)
            self._health.revolution(int((packed > 0).sum()), int(packed.shape[0]))
            # ROS uses inf for "nothing within range" and nan for "no reading";
            # both are no-returns, which _pack_mm encodes as zero.
            yield ScanReading(
                ranges_mm=packed,
                angle_min=float(msg.angle_min),
                angle_increment=float(msg.angle_increment),
                range_min=float(msg.range_min),
                range_max=float(msg.range_max),
                scan_time=float(getattr(msg, "scan_time", 0.0) or 0.0),
            )

    def close(self) -> None:
        try:
            self._node.destroy_node()
        except Exception:
            pass


class SerialLdsSource(ScanSource):
    """The LDS-02 over its USB serial link, no ROS involved.

    The device streams 47-byte packets at 115200 baud, each carrying 12 points
    with a start and end angle to interpolate between (the LD08/LD19 format that
    the LDS-02 uses).  Points are accumulated into a fixed 360-bin table and a
    revolution is emitted when the angle wraps past zero.

    Three details worth knowing before you debug this against hardware:

    * **Baud.** :data:`LDS02_BAUD`, 115200 — *not* the 230400 that LD06 and LD19
      use, though all three speak the same packets.  At the wrong rate the port
      opens, bytes flow steadily, and nothing ever frames; ``--lidar-baud`` is
      the knob and :meth:`_explain_no_frames` works out which way to turn it.
    * **Spin direction.** The device numbers its angles in the direction it
      turns, which is clockwise seen from above, while ``LaserScan`` and this
      protocol are counter-clockwise-positive.  That is expressed as a negative
      ``angle_increment`` rather than by reversing the array, so the raw bins
      stay in device order and only one number carries the convention.  If the
      bird's-eye plot in the snapshot comes out mirrored, this is the flag:
      ``--lidar-spin ccw``.
    * **CRC.** Packets carry a CRC-8 (polynomial 0x4D).  The default ``auto``
      mode checks it, and if nearly everything fails in the first second it
      concludes the checksum convention differs on this unit and carries on
      without it rather than reporting a dead scanner.  ``--lidar-crc check``
      makes a mismatch fatal to the packet; ``ignore`` skips it entirely.
    """

    model = "lds02-serial"

    HEADER = 0x54
    VER_LEN = 0x2C
    POINTS_PER_PACKET = 12
    PACKET_BYTES = 47

    #: What one revolution costs on the wire: 360 points, 12 to a 47-byte packet,
    #: at the scanner's nominal 5 Hz.  Only used to judge an observed byte rate,
    #: so the nominal figure is close enough.
    EXPECTED_BYTES_PER_S = PACKET_BYTES * (360 / POINTS_PER_PACKET) * 5.0

    def __init__(self, port: str = DEFAULT_LIDAR_PORT, baud: int = LDS02_BAUD, points: int = 360,
                 spin: str = "cw", crc: str = "auto", timeout: float = 1.0,
                 health_every: float = 5.0) -> None:
        import serial  # pyserial

        self.baud = int(baud)
        self._stopping = False
        self.points = int(points)
        self.spin = spin
        self.crc_mode = crc
        self.range_min, self.range_max = 0.12, 12.0
        self._buf = bytearray()
        self._crc_ok = 0
        self._crc_bad = 0
        self._crc_checking = crc != "ignore"
        self._speed_dps = 0.0
        self._bytes = 0
        self._packet_count = 0

        self.port, self._ser = self._open(serial, port, timeout)
        log.info("lidar: opened %s at %d baud (%s spin, crc=%s) — %s",
                 self.port, self.baud, spin, crc, _describe_port(self.port))
        self._health = _ScanHealth(self.port, self._diagnose, every=health_every)

    def _open(self, serial, port: str, timeout: float):
        """Open ``port``, falling back to the scanner's other udev name.

        The fallback covers only :data:`LIDAR_PORT_ALIASES`, and only when the
        caller did not name a port itself.  Those names are created by one udev
        rule for one device, so trying the sibling cannot open something that is
        not the scanner.  A port the operator asked for by name is never
        second-guessed — silently opening a different device is how a motor
        controller ends up being read as ranges.
        """
        candidates = [port]
        if port == DEFAULT_LIDAR_PORT:
            candidates += [alias for alias in LIDAR_PORT_ALIASES if alias != port]

        failures = []
        for index, candidate in enumerate(candidates):
            try:
                handle = serial.Serial(candidate, self.baud, timeout=timeout)
            except Exception as exc:
                failures.append(_explain_open_failure(candidate, exc))
                remaining = candidates[index + 1:]
                if remaining:
                    log.info("lidar: %s did not open (%s), trying %s",
                             candidate, exc, remaining[0])
                continue
            if candidate != port:
                log.warning("lidar: %s was not available, using %s instead", port, candidate)
            return candidate, handle

        # Only the first candidate's failure is spelled out: the aliases fail the
        # same way for the same reason, and repeating the udev advice per name
        # buries it.
        message = failures[0]
        if len(candidates) > 1:
            message += " (also tried %s)" % ", ".join(candidates[1:])
        raise LidarUnavailable("%s %s" % (message, _devices_present_line()))

    def _diagnose(self, waited_s: float) -> str:
        """Which stage the bytes reached — the thing that separates the causes."""
        if self._bytes == 0:
            return (f"not one byte has arrived. The port opened, so the device node is real, "
                    f"but nothing is transmitting: check the scanner's power lead and that the "
                    f"rotor is actually spinning. A {self.port} that points at some other "
                    f"device looks exactly like this too.")
        if self._packet_count == 0:
            return self._explain_no_frames(waited_s)
        return (f"{self._bytes} bytes and {self._packet_count} packets parsed, but no complete "
                f"revolution yet. Normal for the first second while the rotor comes up to "
                f"speed; if it persists, fewer than a quarter of the {self.points} bins are "
                f"filling, so most beams are returning nothing.")

    def _explain_no_frames(self, waited_s: float) -> str:
        """Bytes arrive and none of them frame — which way is the baud wrong?

        The observed byte rate answers that, because a UART reading faster than
        its transmitter samples every bit more than once and inflates the byte
        count by the ratio of the two rates, while one reading too slowly drops
        most of them.  So the rate relative to
        :data:`EXPECTED_BYTES_PER_S` says which direction to move, and naming a
        rate to try beats naming the symptom.
        """
        rate = self._bytes / waited_s if waited_s > 0 else 0.0
        ratio = rate / self.EXPECTED_BYTES_PER_S
        other = 230400 if self.baud == LDS02_BAUD else LDS02_BAUD
        head = (f"{self._bytes} bytes arrived ({rate:.0f} B/s) but not one valid LD08 packet "
                f"was framed, reading at {self.baud} baud.")

        if ratio >= 1.5:
            return (f"{head} An LDS-02 sends about {self.EXPECTED_BYTES_PER_S:.0f} B/s, so this "
                    f"is ~{ratio:.1f}x too much — the signature of reading faster than the "
                    f"device transmits. Try `--lidar-baud {other}`. (The LDS-02 is "
                    f"{LDS02_BAUD}; LD06 and LD19 are 230400 and are easy to confuse with it.)")
        if ratio <= 0.6:
            return (f"{head} An LDS-02 sends about {self.EXPECTED_BYTES_PER_S:.0f} B/s, so this "
                    f"is only ~{ratio:.1f}x that — either the rotor is not up to speed, or "
                    f"this is reading slower than the device transmits. Try "
                    f"`--lidar-baud {other}`.")
        return (f"{head} The rate is about right for an LDS-02, so the baud is probably correct "
                f"and the bytes are not LD08 packets: that is what some other device on this "
                f"port looks like. Check that {self.port} really is the scanner "
                f"(`ls -l /dev/serial/by-id/` — the LDS-02 is the CP2102).")

    def info(self) -> Dict[str, Any]:
        return {"model": self.model, "port": self.port, "points": self.points,
                "spin": self.spin, "transport": "serial"}

    def request_stop(self) -> None:
        self._stopping = True

    def scans(self) -> Iterator[ScanReading]:
        ranges = np.zeros(self.points, dtype=np.uint16)
        intensities = np.zeros(self.points, dtype=np.uint8)
        filled = 0
        last_start_cdeg = None

        while True:
            if self._stopping:
                return
            chunk = self._ser.read(512)
            # Ticked before the empty-chunk check: a port that reads nothing at
            # all is precisely the case the health report exists to describe,
            # and the read timeout guarantees we get here at least once a second.
            self._bytes += len(chunk) if chunk else 0
            self._health.tick()
            if not chunk:
                continue
            self._buf.extend(chunk)

            for start_cdeg, points in self._packets():
                # A start angle lower than the previous one means the scanner
                # has come back round past zero: that is one revolution.
                if last_start_cdeg is not None and start_cdeg < last_start_cdeg:
                    if filled >= self.points // 4:
                        self._health.revolution(filled, self.points)
                        yield self._reading(ranges, intensities)
                        ranges = np.zeros(self.points, dtype=np.uint16)
                        intensities = np.zeros(self.points, dtype=np.uint8)
                        filled = 0
                    else:
                        # Too sparse to be a revolution — a partial scan from
                        # mid-stream startup, or the device spinning up.
                        log.debug("lidar: discarding sparse revolution (%d/%d points)",
                                  filled, self.points)
                        ranges[:] = 0
                        intensities[:] = 0
                        filled = 0
                last_start_cdeg = start_cdeg

                for cdeg, distance_mm, intensity in points:
                    if distance_mm == 0:
                        continue
                    index = int(round(cdeg / 100.0 * self.points / 360.0)) % self.points
                    if ranges[index] == 0:
                        filled += 1
                    # Nearest wins where two beams land in one bin: for
                    # obstacle avoidance the closer return is the one that
                    # matters, and averaging across a depth discontinuity
                    # invents a surface that is not there.
                    if ranges[index] == 0 or distance_mm < ranges[index]:
                        ranges[index] = distance_mm
                        intensities[index] = intensity

    def _reading(self, ranges: np.ndarray, intensities: np.ndarray) -> ScanReading:
        # Clockwise device, counter-clockwise protocol: carried entirely by the
        # sign of angle_increment.
        step = TWO_PI / self.points
        return ScanReading(
            ranges_mm=ranges.copy(),
            angle_min=0.0,
            angle_increment=-step if self.spin == "cw" else step,
            range_min=self.range_min,
            range_max=self.range_max,
            scan_time=(360.0 / self._speed_dps) if self._speed_dps > 0 else 0.0,
            intensities=intensities.copy(),
        )

    def _packets(self):
        """Yield ``(start_centidegrees, [(centideg, mm, intensity), ...])``."""
        buf = self._buf
        while True:
            start = buf.find(bytes([self.HEADER, self.VER_LEN]))
            if start < 0:
                # Keep one byte in case a header straddles two reads.
                del buf[:max(0, len(buf) - 1)]
                return
            if start:
                del buf[:start]
            if len(buf) < self.PACKET_BYTES:
                return

            packet = bytes(buf[:self.PACKET_BYTES])
            if self._crc_checking and not self._check_crc(packet):
                # 0x54 0x2C can occur inside a distance field, so a failed CRC
                # is as likely to mean "synced to the wrong offset" as "corrupt
                # packet".  Advance one byte and look again rather than
                # swallowing 47 bytes of what may be a real packet.
                del buf[:1]
                continue
            del buf[:self.PACKET_BYTES]
            self._packet_count += 1

            speed, start_cdeg = struct.unpack_from("<HH", packet, 2)
            end_cdeg, = struct.unpack_from("<H", packet, 42)
            self._speed_dps = float(speed)

            span = (end_cdeg - start_cdeg) % 36000
            step = span / (self.POINTS_PER_PACKET - 1) if self.POINTS_PER_PACKET > 1 else 0.0

            points = []
            for i in range(self.POINTS_PER_PACKET):
                distance, intensity = struct.unpack_from("<HB", packet, 6 + i * 3)
                points.append(((start_cdeg + step * i) % 36000, distance, intensity))
            yield start_cdeg, points

    def _check_crc(self, packet: bytes) -> bool:
        if _crc8(packet[:-1]) == packet[-1]:
            self._crc_ok += 1
            return True
        self._crc_bad += 1
        if (self.crc_mode == "auto" and self._crc_bad >= 50
                and self._crc_bad > 4 * self._crc_ok):
            # Data is clearly arriving — packets are being framed — but the
            # checksum does not agree.  Refusing every packet would present a
            # working scanner as a dead one, so say what happened and use it.
            log.warning("lidar: %d of %d packets failed CRC; disabling the check. "
                        "Ranges are still being framed, but a corrupt packet will "
                        "now get through — pass --lidar-crc check to be strict.",
                        self._crc_bad, self._crc_bad + self._crc_ok)
            self._crc_checking = False
        return False

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:
            pass


def _crc8_table() -> List[int]:
    """CRC-8 table for polynomial 0x4D, as used by the LD08/LD19 packet format."""
    table = []
    for i in range(256):
        value = i
        for _ in range(8):
            value = ((value << 1) ^ 0x4D) & 0xFF if value & 0x80 else (value << 1) & 0xFF
        table.append(value)
    return table


_CRC8_TABLE = _crc8_table()


def _crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = _CRC8_TABLE[(crc ^ byte) & 0xFF]
    return crc


def _pack_mm(ranges_m) -> np.ndarray:
    """Metres (float, possibly inf/nan) to uint16 millimetres, 0 = no return."""
    arr = np.asarray(ranges_m, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(arr) & (arr > 0.0)
        mm = np.where(valid, arr * 1000.0, 0.0)
    return np.clip(np.rint(mm), 0, 65535).astype("<u2")


class LidarFeed:
    """Runs a scan source on its own thread and keeps only the newest scan.

    The main loop is paced by the camera, so it cannot also sit in a blocking
    serial read.  One slot, latest wins: if the loop is busy for 400 ms, the
    scanner has produced two revolutions and only the second one is worth
    anything.  ``dropped`` counts the rest, which is how you tell the difference
    between a slow link and a slow scanner.
    """

    def __init__(self, source: ScanSource) -> None:
        self.source = source
        self._slot: Optional[ScanReading] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.received = 0
        self.dropped = 0
        self.failed = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="lidar", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for reading in self.source.scans():
                if self._stop.is_set():
                    break
                with self._lock:
                    if self._slot is not None:
                        self.dropped += 1
                    self._slot = reading
                    self.received += 1
        except Exception:
            if self._stop.is_set():
                # Shutting down.  The source was asked to stop and did not get
                # there before its device was closed; whatever it raised on the
                # way out is noise, and an ERROR with a traceback here makes a
                # normal Ctrl-C look like a crash.
                log.debug("lidar feed raised during shutdown", exc_info=True)
                return
            # The camera stream must survive a scanner that was unplugged.
            self.failed = True
            log.exception("lidar feed stopped")

    def take(self) -> Optional[ScanReading]:
        with self._lock:
            reading, self._slot = self._slot, None
        return reading

    def stop(self, timeout: float = 2.0) -> None:
        """Wind the reader down, then close the device — in that order.

        Closing first is what produced a traceback on every Ctrl-C: the thread
        is parked in a blocking read, and pyserial closing underneath it leaves
        the read holding a file descriptor that is now ``None``.  So ask the
        source to return, give it ``timeout`` to do so, and only then close.
        A source that ignores the request still gets closed — a shutdown that
        hangs is worse than a noisy one — but ``_run`` now knows to expect the
        fallout.
        """
        self._stop.set()
        try:
            self.source.request_stop()
        except Exception:
            pass
        if self._thread is not None:
            # The serial read timeout bounds how long this can take; the ROS
            # spin is 50 ms.  Either way it is well inside the default.
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning("lidar reader did not stop within %.1fs; closing anyway", timeout)
            self._thread = None
        try:
            self.source.close()
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
        max_inflight_scans: int = 2,
        on_scan_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.server = server
        self.client_id = client_id
        self.codec = codec
        self.jpeg_quality = jpeg_quality
        self.max_inflight = max(1, max_inflight)
        self.on_result = on_result
        self.reconnect_after_s = reconnect_after_s
        # Scans get their own window.  Sharing one with the camera would let a
        # backlog of frames stop the obstacle data, which is exactly backwards:
        # if the link is congested, ranges are the part worth keeping.
        self.max_inflight_scans = max(1, max_inflight_scans)
        self.on_scan_result = on_scan_result

        self.ctx = zmq.Context.instance()
        self.sock: Optional[zmq.Socket] = None
        self._pending: Dict[int, float] = {}   # seq -> monotonic send time
        self._pending_scans: Dict[int, float] = {}
        self._seq = 0
        self._scan_seq = 0
        self._stop = False
        self._connected = False
        self._last_rx = time.monotonic()
        self._lidar_info: Dict[str, Any] = {}
        # Set from the welcome: a server with lidar.enabled false answers every
        # scan with an error, so there is no point sending them.
        self._server_wants_scans = True

        # Counters for the periodic status line.
        self.sent = 0
        self.results_ok = 0
        self.results_bad = 0
        self.skipped_backpressure = 0
        self.scans_sent = 0
        self.scan_results_ok = 0
        self.scan_results_bad = 0
        self.scans_skipped = 0
        self._rtt_sum = 0.0
        self._rtt_n = 0
        self._last_scan_summary: Dict[str, Any] = {}

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
        self._pending_scans.clear()
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
            "lidar": self._lidar_info,
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

    def run(self, source: FrameSource, duration: float = 0.0, status_every: float = 5.0,
            scan_source: Optional[ScanSource] = None) -> None:
        import cv2

        self._src_width = source.width
        self._src_height = source.height
        self._src_fps = getattr(source, "fps", 0.0)
        self._src_name = type(source).__name__

        feed: Optional[LidarFeed] = None
        if scan_source is not None:
            self._lidar_info = scan_source.info()
            feed = LidarFeed(scan_source)
            feed.start()
            log.info("lidar: %s", ", ".join(f"{k}={v}" for k, v in self._lidar_info.items()))

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

                # Before the frame: a scan is smaller, rarer and more perishable,
                # and the frame that follows it on the server will be paired with
                # it rather than with the one before.
                if feed is not None:
                    self._pump_lidar(feed)

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
            if feed is not None:
                feed.stop()
            source.close()
            self.close_socket()

    def _pump_lidar(self, feed: LidarFeed) -> None:
        """Send the newest revolution, if there is one and the server wants it."""
        reading = feed.take()
        if reading is None or not self._server_wants_scans:
            return
        if len(self._pending_scans) >= self.max_inflight_scans:
            self.scans_skipped += 1
            return
        self._send_scan(reading)

    def _send_scan(self, reading: ScanReading) -> None:
        payload = reading.ranges_mm.astype("<u2").tobytes()
        if reading.intensities is not None:
            payload += np.asarray(reading.intensities, dtype=np.uint8).tobytes()

        seq = self._scan_seq
        self._scan_seq += 1
        header = {
            "type": MSG_SCAN,
            "seq": seq,
            "encoding": SCAN_ENC_U16_MM,
            "count": int(len(reading)),
            "angle_min": float(reading.angle_min),
            "angle_increment": float(reading.angle_increment),
            "range_min": float(reading.range_min),
            "range_max": float(reading.range_max),
            "scan_time": float(reading.scan_time),
            "intensities": reading.intensities is not None,
            "source": self._lidar_info.get("model", ""),
            "t_capture_ns": int(reading.t_capture_ns),
            "t_send_ns": time.monotonic_ns(),
        }
        if self._send(header, payload):
            self._pending_scans[seq] = time.monotonic()
            self.scans_sent += 1
        else:
            self.scans_skipped += 1

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
                server_lidar = header.get("server", {}).get("lidar", {})
                # An older server has no opinion on scans; assume it wants them
                # and let it answer with errors if not.
                self._server_wants_scans = bool(server_lidar.get("enabled", True))
                if self._lidar_info and not self._server_wants_scans:
                    log.warning("server has lidar disabled — not sending scans "
                                "(start it without --no-lidar to use the LDS-02)")
                elif self._lidar_info and server_lidar:
                    log.info("server lidar geometry: yaw %+.0f deg, hfov %.0f deg, "
                             "scans older than %.0f ms not fused",
                             server_lidar.get("mount_yaw_deg", 0.0),
                             server_lidar.get("camera_hfov_deg", 0.0),
                             server_lidar.get("stale_after_ms", 0.0))
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

        if mtype == MSG_SCAN_RESULT:
            seq = int(header.get("seq", -1))
            sent_at = self._pending_scans.pop(seq, None)
            if sent_at is not None:
                header["rtt_ms"] = round((time.monotonic() - sent_at) * 1000.0, 2)
            if header.get("ok"):
                self.scan_results_ok += 1
                self._last_scan_summary = header.get("data", {}) or {}
            else:
                self.scan_results_bad += 1
                reason = header.get("reason")
                if reason == "lidar_disabled":
                    # Stop rather than keep paying to be refused; the operator
                    # has told the server not to look at scans.
                    self._server_wants_scans = False
                    log.warning("server refused a scan: %s", header.get("data", {}).get("error", ""))
                else:
                    log.warning("scan seq=%d not ok: %s %s", seq, reason,
                                header.get("data", {}).get("error", ""))
            if self.on_scan_result is not None:
                try:
                    self.on_scan_result(header)
                except Exception:
                    log.exception("on_scan_result callback raised")
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
            "sent=%d ok=%d bad=%d skipped=%d | %.1f fps out | rtt %.1f ms | inflight=%d%s",
            self.sent, self.results_ok, self.results_bad, self.skipped_backpressure,
            self.sent / elapsed if elapsed > 0 else 0.0, rtt, len(self._pending),
            self._lidar_status(elapsed),
        )
        self.sent = 0
        self.results_ok = 0
        self.results_bad = 0
        self.skipped_backpressure = 0
        self.scans_sent = 0
        self.scan_results_ok = 0
        self.scan_results_bad = 0
        self.scans_skipped = 0
        self._rtt_sum = 0.0
        self._rtt_n = 0

    def _lidar_status(self, elapsed: float) -> str:
        """The LiDAR half of the status line, empty when there is no scanner."""
        if not self._lidar_info:
            return ""
        rate = self.scans_sent / elapsed if elapsed > 0 else 0.0
        summary = self._last_scan_summary
        detail = ""
        if summary:
            nearest = summary.get("nearest_m")
            # The number that says the scanner is really measuring the room, not
            # just producing traffic.
            detail = (" nearest %.2f m @%+.0f°" % (nearest, summary.get("nearest_deg", 0.0))
                      if nearest is not None else " no returns")
            if summary.get("obstacle"):
                detail += " OBSTACLE"
        return (" | lidar %.1f Hz ok=%d bad=%d skipped=%d%s"
                % (rate, self.scan_results_ok, self.scan_results_bad, self.scans_skipped, detail))

    def stop(self) -> None:
        self._stop = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_source(args) -> FrameSource:
    if args.source == "synthetic":
        return SyntheticSource(args.width, args.height, args.fps)
    if args.source == "none":
        return NullSource()
    if args.source == "gst-jpeg":
        return GstJpegSource(args.device, args.width, args.height, args.fps, args.quality)
    return OpenCVSource(args.device, args.width, args.height, args.fps, args.fourcc)


def build_scan_source(args) -> Optional[ScanSource]:
    """Open the LiDAR, or return None if there is not one to open.

    ``auto`` tries ROS 2 then serial and gives up.  It never substitutes the
    synthetic scanner: a robot acting on invented ranges is worse than a robot
    that knows it is blind.
    """
    mode = args.lidar
    health_every = getattr(args, "lidar_health_every", 5.0)
    if mode == "off":
        return None
    if mode == "synthetic":
        return SyntheticScanSource(points=args.lidar_points, hz=args.lidar_hz)
    if mode == "ros2":
        return Ros2ScanSource(args.lidar_topic, health_every=health_every)
    if mode == "serial":
        return SerialLdsSource(args.lidar_port, args.lidar_baud, args.lidar_points,
                               spin=args.lidar_spin, crc=args.lidar_crc,
                               health_every=health_every)

    errors = []
    log.info("lidar: auto — trying ROS 2 topic %s", args.lidar_topic)
    try:
        # require_publisher, because a subscription to a topic nobody publishes
        # succeeds happily and would end the search here with a source that can
        # never produce anything.
        return Ros2ScanSource(args.lidar_topic, require_publisher=True,
                              health_every=health_every)
    except Exception as exc:
        errors.append("ros2: %s" % exc)
        log.info("lidar: ROS 2 unavailable — %s", exc)

    log.info("lidar: auto — trying serial %s", args.lidar_port)
    try:
        return SerialLdsSource(args.lidar_port, args.lidar_baud, args.lidar_points,
                               spin=args.lidar_spin, crc=args.lidar_crc,
                               health_every=health_every)
    except Exception as exc:
        errors.append("serial: %s" % exc)
        log.info("lidar: serial unavailable — %s", exc)

    log.warning("no lidar found, streaming camera only. Tried:\n  %s", "\n  ".join(errors))
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="robocam-client",
        description="Stream the robot's webcam to the processing server.",
    )
    p.add_argument("-s", "--server", default="tcp://10.128.17.196:5555",
                   help="server endpoint (default: %(default)s)")
    p.add_argument("--client-id", default="orin", help="identifies this robot to the server")
    p.add_argument("--source", choices=["camera", "synthetic", "gst-jpeg", "none"], default="camera",
                   help="camera: OpenCV capture; synthetic: test pattern, no camera needed; "
                        "gst-jpeg: hardware JPEG via GStreamer (Jetson); none: lidar only")
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

    lidar = p.add_argument_group(
        "lidar (LDS-02)",
        "auto tries ROS 2 then the serial device and gives up if neither is there; "
        "it never silently substitutes the synthetic scanner",
    )
    lidar.add_argument("--lidar", choices=["off", "auto", "ros2", "serial", "synthetic"],
                       default="off", help="how to read the scanner (default: %(default)s)")
    lidar.add_argument("--lidar-topic", default="/scan", help="ROS 2 LaserScan topic")
    lidar.add_argument("--lidar-port", default=DEFAULT_LIDAR_PORT,
                       help="serial device (default: %(default)s, the udev symlink for the "
                            "scanner; falls back to /dev/ld08_lidar. Name a raw /dev/ttyUSBn "
                            "only if you are sure which device it is)")
    lidar.add_argument("--lidar-health-every", type=float, default=5.0,
                       help="seconds between lidar progress reports while reading, 0 to "
                            "disable (default: %(default)s)")
    lidar.add_argument("--lidar-baud", type=int, default=LDS02_BAUD,
                       help="serial baud rate (default: %(default)s, the LDS-02/LD08 rate. "
                            "LD06 and LD19 use 230400 with the same packet format, so if "
                            "bytes arrive but nothing frames, this is the first thing to try)")
    lidar.add_argument("--lidar-points", type=int, default=360, help="points per revolution")
    lidar.add_argument("--lidar-spin", choices=["cw", "ccw"], default="cw",
                       help="direction the device numbers its angles in; flip this if the "
                            "bird's-eye plot in the server's snapshot is mirrored")
    lidar.add_argument("--lidar-crc", choices=["auto", "check", "ignore"], default="auto",
                       help="serial packet checksum handling")
    lidar.add_argument("--lidar-hz", type=float, default=5.0, help="synthetic scanner rate")
    lidar.add_argument("--max-inflight-scans", type=int, default=2,
                       help="scans allowed to be awaiting a result before dropping")
    lidar.add_argument("--print-scans", action="store_true",
                       help="print every scan result as JSON")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    def on_result(result: Dict[str, Any]) -> None:
        if args.print_results:
            print(json.dumps(result, separators=(",", ":")), flush=True)

    def on_scan_result(result: Dict[str, Any]) -> None:
        if args.print_scans:
            print(json.dumps(result, separators=(",", ":")), flush=True)

    client = RoboCamClient(
        server=args.server,
        client_id=args.client_id,
        codec=args.codec,
        jpeg_quality=args.quality,
        max_inflight=args.max_inflight,
        on_result=on_result,
        max_inflight_scans=args.max_inflight_scans,
        on_scan_result=on_scan_result,
    )

    signal.signal(signal.SIGINT, lambda *_: client.stop())
    signal.signal(signal.SIGTERM, lambda *_: client.stop())

    try:
        source = build_source(args)
    except Exception as exc:
        log.error("could not open source: %s", exc)
        return 1

    try:
        scan_source = build_scan_source(args)
    except Exception as exc:
        # An explicitly requested scanner that will not open is a failure, not
        # something to shrug off: the operator asked for ranges.
        log.error("could not open lidar (%s): %s", args.lidar, exc)
        source.close()
        return 1

    client.run(source, duration=args.duration, status_every=args.status_every,
               scan_source=scan_source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
