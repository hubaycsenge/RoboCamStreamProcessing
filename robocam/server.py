"""The stream server.

One thread owns the ZeroMQ socket and does nothing slow: receive, decode,
enqueue, and send back whatever the workers have finished.  Everything
expensive happens in the worker pool (see pipeline.py).

ZeroMQ sockets are not thread-safe, so results travel from the workers to the
IO thread through a plain ``queue.Queue`` which the IO loop drains after every
poll.  That keeps all socket calls on one thread without any locking.

The non-camera streams are the exception to "the IO thread does nothing slow",
and only because the work is genuinely tiny: parsing 360 uint16s and reducing
them to sector minima is ~60 µs against a 20 ms poll, an inertial burst is a
handful of numpy passes over a few dozen rows, and a pose is three floats.  All
are answered inline so that obstacle and attitude information is never stuck
behind a model in the frame queue — at 5 Hz, queueing a scan behind two 30 ms
frames would be most of its useful life.  The latest of each is also held on the
session and attached to the next frame, which is where fusion happens:
association by arrival time on the server's own clock, rather than by unrelated
client clocks.

The occupancy grid is the one stream that breaks that pattern, because it is the
one that is not perishable.  A grid arrives every few seconds, is decompressed
(a few ms for a real room, which is why the size limit is checked before the
allocation rather than after), and is then held for as long as the robot does not
replace it.  Its age is reported but does not disqualify it: a room does not
change in ten seconds, whereas the robot's pose in it does.

Three of the server's messages are not replies at all.  ``map_update``,
``pose_hint`` and ``found`` are produced by a processor — the Compare and
decision boxes of the system diagram — and travel back attached to whichever
frame they came out of; ``_send_result`` sends them first, stamping the
per-session sequence numbers that a worker thread has no business allocating,
and dropping any whose ``map_id`` no longer matches the map the robot is on.
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import zmq

from . import imu as imu_mod
from . import lidar as lidar_mod
from . import detect as detect_mod
from . import mission as mission_mod
from . import seek as seek_mod
from . import occupancy as occupancy_mod
from . import odometry as odom_mod
from . import processors, wire
from .config import Config
from .decode import DecodeError, SessionDecoder
from .pipeline import FrameQueue, ProcessedResult, WorkerPool
from .processors.base import Frame
from .snapshot import SnapshotWriter
from .waker import Waker

log = logging.getLogger("robocam.server")


def _identity_to_session_id(identity: bytes) -> str:
    """Human-readable session id, falling back to hex for binary identities."""
    try:
        text = identity.decode("utf-8")
    except UnicodeDecodeError:
        return identity.hex()
    if text.isprintable() and text:
        return text
    return identity.hex()


@dataclass
class Session:
    identity: bytes
    session_id: str
    client_id: str = ""
    codec: str = wire.CODEC_JPEG
    declared_width: int = 0
    declared_height: int = 0
    declared_fps: float = 0.0
    camera: str = ""
    created_ns: int = field(default_factory=wire.monotonic_ns)
    last_seen_ns: int = field(default_factory=wire.monotonic_ns)
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    frames_failed: int = 0
    bytes_received: int = 0
    last_seq: int = -1
    seq_gaps: int = 0
    decoder: SessionDecoder = field(default_factory=SessionDecoder)
    greeted: bool = False

    # -- LiDAR ------------------------------------------------------------
    # What the client declared in its hello, if anything.
    lidar_info: Dict[str, Any] = field(default_factory=dict)
    scans_received: int = 0
    scans_failed: int = 0
    # Latest good scan, kept for attaching to frames.  One slot, latest wins:
    # an older scan has no value once a newer one exists.
    last_scan: Optional[lidar_mod.Scan] = None
    last_scan_ns: int = 0
    # Interval between the last two scans, for the reported rate.  A device
    # spinning below its rated speed shows up here before it shows up in the
    # ranges, so it is worth carrying.
    scan_interval_ms: float = 0.0

    # -- IMU --------------------------------------------------------------
    # What the client declared in its hello, if anything.
    imu_info: Dict[str, Any] = field(default_factory=dict)
    imu_bursts_received: int = 0
    imu_samples_received: int = 0
    imu_failed: int = 0
    # Latest good burst, kept for attaching to frames.  One slot, latest wins.
    last_imu: Optional[imu_mod.ImuBatch] = None
    last_imu_ns: int = 0

    # -- odometry ---------------------------------------------------------
    odom_info: Dict[str, Any] = field(default_factory=dict)
    odom_received: int = 0
    odom_failed: int = 0
    last_odom: Optional[odom_mod.Odom] = None
    last_odom_ns: int = 0
    # The pose before that one.  Kept because "how far has the robot moved
    # since the last pose" is the number that says whether a frame gave the
    # reconstruction any new parallax, and it cannot be recovered afterwards.
    prev_odom: Optional[odom_mod.Odom] = None

    # -- occupancy grid ---------------------------------------------------
    map_info: Dict[str, Any] = field(default_factory=dict)
    maps_received: int = 0
    maps_failed: int = 0
    # The robot's map as this server currently understands it: the last full
    # grid, with every patch since applied.  One grid per session, shared by
    # every frame in flight, so nothing downstream may write to it in place.
    grid: Optional[occupancy_mod.Grid] = None
    grid_ns: int = 0

    # -- exits, phase, mission --------------------------------------------
    exits: List[Dict[str, Any]] = field(default_factory=list)
    exits_ns: int = 0
    exits_received: int = 0
    phase: str = wire.PHASE_EXPLORE
    mission: Dict[str, Any] = field(default_factory=dict)

    # -- is T1 finished? ---------------------------------------------------
    # How fast the map is still gaining observed cells.  Held per session
    # because it is a property of this robot's run, and sampled on every
    # uploaded grid rather than on a timer, so a robot that stops uploading
    # stops contributing evidence rather than appearing to have finished.
    coverage: seek_mod.CoverageTracker = field(
        default_factory=lambda: seek_mod.CoverageTracker(window_s=30.0))
    # Running mean of the comparison's agreement.  T1's product is the cloud,
    # and a cloud that never agreed with the grid is one T2 cannot search.
    agreement_sum: float = 0.0
    agreement_n: int = 0
    # Latched so that "t1 looks finished" is logged once rather than on every
    # grid for the rest of the run.
    t1_ready_logged: bool = False

    # -- the server's own outgoing streams --------------------------------
    # Sequence numbers for the three announcements.  Per session and owned
    # here, so that two workers cannot allocate the same one.
    map_update_seq: int = 0
    pose_hint_seq: int = 0
    found_seq: int = 0
    # When each was last sent, for the rate limits in config.  A costmap that
    # is rewritten at the reconstruction's rate costs the robot more than the
    # updates are worth.
    last_map_update_ns: int = 0
    last_pose_hint_ns: int = 0
    map_updates_sent: int = 0
    pose_hints_sent: int = 0
    founds_sent: int = 0

    @property
    def mean_agreement(self) -> Optional[float]:
        """Mean cloud/grid agreement so far, or None if none was ever measured.

        None and 0.0 are different answers and the distinction matters: never
        having compared is a configuration problem, and having compared and
        agreed with nothing is a placement problem.
        """
        return (self.agreement_sum / self.agreement_n) if self.agreement_n else None

    def touch(self) -> None:
        self.last_seen_ns = wire.monotonic_ns()

    def scan_hz(self) -> float:
        return 1000.0 / self.scan_interval_ms if self.scan_interval_ms > 0 else 0.0

    def fresh_scan(self, stale_after_ms: float) -> Tuple[Optional[lidar_mod.Scan], float]:
        """The last scan and its age, or (None, 0) if there is none or it is stale."""
        if self.last_scan is None:
            return None, 0.0
        age_ms = (wire.monotonic_ns() - self.last_scan_ns) / 1e6
        if stale_after_ms > 0 and age_ms > stale_after_ms:
            return None, age_ms
        return self.last_scan, age_ms

    def fresh_imu(self, stale_after_ms: float) -> Tuple[Optional[imu_mod.ImuBatch], float]:
        """The last burst and its age, or (None, 0) if there is none or it is stale."""
        if self.last_imu is None:
            return None, 0.0
        age_ms = (wire.monotonic_ns() - self.last_imu_ns) / 1e6
        if stale_after_ms > 0 and age_ms > stale_after_ms:
            return None, age_ms
        return self.last_imu, age_ms

    def fresh_odom(self, stale_after_ms: float) -> Tuple[Optional[odom_mod.Odom], float]:
        """The last pose and its age, or (None, 0) if there is none or it is stale.

        Stale means genuinely unusable rather than merely old: a pose is what
        places a cloud in the map, and placing it with a pose from 400 ms and
        half a metre ago puts a table through a wall.
        """
        if self.last_odom is None:
            return None, 0.0
        age_ms = (wire.monotonic_ns() - self.last_odom_ns) / 1e6
        if stale_after_ms > 0 and age_ms > stale_after_ms:
            return None, age_ms
        return self.last_odom, age_ms

    def grid_age_ms(self) -> float:
        if self.grid is None:
            return 0.0
        return (wire.monotonic_ns() - self.grid_ns) / 1e6

    def age_s(self) -> float:
        return (wire.monotonic_ns() - self.created_ns) / 1e9

    def idle_s(self) -> float:
        return (wire.monotonic_ns() - self.last_seen_ns) / 1e9


class StreamServer:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.ctx = zmq.Context.instance()
        self.sock: Optional[zmq.Socket] = None
        self.sessions: Dict[bytes, Session] = {}

        self.frames = FrameQueue(
            max_depth=config.queue.max_depth,
            drop_policy=config.queue.drop_policy,
        )
        self.results: "queue.Queue[ProcessedResult]" = queue.Queue(maxsize=256)
        self.pool: Optional[WorkerPool] = None
        self.waker: Optional[Waker] = None
        self.snapshots = SnapshotWriter(
            directory=config.snapshot.dir,
            every_n_frames=config.snapshot.every_n_frames,
            latest_only=config.snapshot.latest_only,
            jpeg_quality=config.snapshot.jpeg_quality,
            enabled=config.snapshot.enabled,
            lidar_overlay=config.snapshot.lidar_overlay and config.lidar.enabled,
            hfov_deg=config.lidar.camera_hfov_deg,
            mount_yaw_deg=config.lidar.mount_yaw_deg,
            fov_bins=config.lidar.fov_bins,
            imu_overlay=config.snapshot.imu_overlay and config.imu.enabled,
        )

        self._stop = threading.Event()
        self._frame_counter = 0
        # Rolling counters for the periodic throughput line.
        self._stat_t0 = time.monotonic()
        self._stat_frames = 0
        self._stat_bytes = 0
        self._stat_dropped = 0
        self._stat_latency_ms = 0.0
        self._stat_scans = 0
        self._stat_scan_bytes = 0
        self._stat_imu_bursts = 0
        self._stat_imu_samples = 0
        self._stat_imu_bytes = 0
        self._stat_map_bytes = 0
        self._stat_announcements = 0

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.setsockopt(zmq.RCVHWM, self.cfg.server.rcvhwm)
        self.sock.setsockopt(zmq.SNDHWM, self.cfg.server.sndhwm)
        # Fail loudly instead of silently discarding a reply to a peer that has
        # gone away — that is how we learn a session is dead.
        self.sock.setsockopt(zmq.ROUTER_MANDATORY, 1)
        # Do not block shutdown waiting to flush queued messages.
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(self.cfg.server.bind)

        self.snapshots.start()
        self.waker = Waker(self.ctx, f"inproc://robocam-wake-{id(self)}")

        name = self.cfg.processor.name
        options = dict(self.cfg.processor.options)

        def build_processor():
            # Each worker gets its own instance; each is told where the scanner
            # points, how the IMU is read, and — for the stages that compare a
            # reconstruction against the robot's map — where the camera is bolted
            # and which height slice counts as an obstacle.  Those last describe
            # the robot rather than the model, so they come from the server's
            # config and not from the processor's own options, where the two
            # could disagree.
            processor = processors.build(name, options)
            processor.configure(self.cfg.lidar, self.cfg.imu, self.cfg)
            return processor

        self.pool = WorkerPool(
            processor_factory=build_processor,
            frame_queue=self.frames,
            result_queue=self.results,
            workers=self.cfg.processor.workers,
            name=name,
            on_result_ready=self.waker.wake,
        )
        self.pool.start()

        log.info(
            "listening on %s | processor=%s workers=%d | queue depth=%d drop=%s | "
            "lidar=%s | imu=%s",
            self.cfg.server.bind,
            name,
            self.cfg.processor.workers,
            self.cfg.queue.max_depth,
            self.cfg.queue.drop_policy,
            (
                f"on (yaw {self.cfg.lidar.mount_yaw_deg:+.0f}°, "
                f"hfov {self.cfg.lidar.camera_hfov_deg:.0f}°, "
                f"stale >{self.cfg.lidar.stale_after_ms:.0f} ms)"
                if self.cfg.lidar.enabled else "off"
            ),
            (
                f"on (tilt warn {self.cfg.imu.tilt_warn_deg:.0f}°, "
                f"stale >{self.cfg.imu.stale_after_ms:.0f} ms)"
                if self.cfg.imu.enabled else "off"
            ),
        )
        log.info(
            "  odom=%s | map=%s | compare=%s | phase=%s%s",
            "on" if self.cfg.odom.enabled else "off",
            "on" if self.cfg.map.enabled else "off",
            (
                f"on (z {self.cfg.compare.z_min:.2f}-{self.cfg.compare.z_max:.2f} m, "
                f"camera at {self.cfg.compare.camera_z:.2f} m, "
                f"patches {'on' if self.cfg.map.send_updates else 'off'}, "
                f"hints {'on' if self.cfg.compare.send_hints else 'off'})"
                if self.cfg.compare.enabled else "off"
            ),
            self.cfg.mission.phase,
            f" | target={self.cfg.mission.target!r}" if self.cfg.mission.target else "",
        )
        log.info(
            "  seek=%s | t1 exit report=%s",
            (
                f"on (detector {self.cfg.seek.detector}, 1 frame in "
                f"{self.cfg.seek.detect_every_n}, grasp below "
                f"{self.cfg.seek.grasp_z_max:.2f} m, "
                f"keyframes {'on' if self.cfg.seek.keyframes else 'off'})"
                if self.cfg.seek.enabled else "off"
            ),
            (
                f"on (explored >{self.cfg.mission.t1_min_explored:.0%}, "
                f"agreement >{self.cfg.mission.t1_min_agreement:.2f})"
                if self.cfg.mission.t1_exit_report else "off"
            ),
        )
        if (self.cfg.seek.enabled and self.cfg.mission.phase == wire.PHASE_SEEK
                and not self.cfg.mission.target):
            # Worth saying at startup rather than at the first t2 frame: a
            # server started straight into t2 with no target looks like it is
            # working and never announces anything.
            log.warning(
                "starting in t2 with no target named. The decision stage will not "
                "search until one arrives -- pass --target, or send one in the "
                "robot's hello."
            )
        # These are addresses on *this* node, for a benchmark client running
        # here. They are deliberately not labelled as somewhere the robot can
        # dial: the robot is behind the lab router's NAT and reaches the server
        # only through the SSH bridge to nipg1 (see link/README.md). The banner
        # used to say "robot can connect to", which sent people to chase a
        # cluster address that was never reachable from the robot.
        port = self.cfg.server.bind.rsplit(":", 1)[-1]
        for addr in _local_addresses():
            log.info("  local clients can connect to tcp://%s:%s", addr, port)
        log.info("  the robot reaches this through the nipg1 bridge, not the above")

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if self.pool is not None:
            self.pool.stop()
            self.pool = None
        self.snapshots.stop()
        for session in self.sessions.values():
            session.decoder.close()
        self.sessions.clear()
        if self.waker is not None:
            self.waker.close()
            self.waker = None
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None

    # -- main loop --------------------------------------------------------

    def run(self) -> None:
        assert self.sock is not None, "call start() before run()"
        assert self.waker is not None
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        # A finished result wakes the loop immediately instead of waiting out
        # the poll timeout, which is otherwise the dominant source of latency.
        poller.register(self.waker.rx, zmq.POLLIN)
        last_housekeeping = time.monotonic()

        try:
            while not self._stop.is_set():
                events = dict(poller.poll(timeout=self.cfg.server.io_poll_ms))
                if self.waker.rx in events:
                    self.waker.drain()
                if self.sock in events:
                    # Drain what is already buffered before doing anything else,
                    # bounded so that sending results is never starved.
                    for _ in range(64):
                        try:
                            parts = self.sock.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        self._on_message(parts)

                self._flush_results()

                now = time.monotonic()
                if now - last_housekeeping >= 1.0:
                    self._reap_sessions()
                    self._log_stats()
                    last_housekeeping = now
        except KeyboardInterrupt:  # pragma: no cover - interactive
            log.info("interrupted")
        finally:
            self._flush_results()
            self.close()

    # -- receive path -----------------------------------------------------

    def _on_message(self, parts: List[bytes]) -> None:
        if len(parts) < 2:
            log.warning("ignoring malformed multipart message (%d frames)", len(parts))
            return
        identity, rest = parts[0], parts[1:]
        recv_ts_ns = wire.monotonic_ns()

        try:
            header, payload = wire.decode(rest)
        except wire.ProtocolError as exc:
            log.warning("protocol error from %s: %s", _identity_to_session_id(identity), exc)
            self._send(identity, wire.error(str(exc)))
            return

        msg_type = header.get("type")
        session = self.sessions.get(identity)

        if msg_type == wire.MSG_HELLO:
            self._on_hello(identity, header)
            return

        if session is None:
            # Tolerate a client that starts streaming without a handshake: the
            # link working matters more than the ceremony.  Log it once.
            session = self._create_session(identity, header)
            log.warning("session %s started streaming without hello", session.session_id)

        session.touch()

        if msg_type == wire.MSG_FRAME:
            self._on_frame(session, header, payload, recv_ts_ns)
        elif msg_type == wire.MSG_SCAN:
            self._on_scan(session, header, payload, recv_ts_ns)
        elif msg_type == wire.MSG_IMU:
            self._on_imu(session, header, payload, recv_ts_ns)
        elif msg_type == wire.MSG_ODOM:
            self._on_odom(session, header, recv_ts_ns)
        elif msg_type == wire.MSG_MAP:
            self._on_map(session, header, payload, recv_ts_ns)
        elif msg_type == wire.MSG_EXITS:
            self._on_exits(session, header, recv_ts_ns)
        elif msg_type == wire.MSG_PHASE:
            self._on_phase(session, header)
        elif msg_type == wire.MSG_PING:
            self._send(identity, wire.pong(header.get("nonce", 0), header.get("t_send_ns")))
        elif msg_type == wire.MSG_BYE:
            log.info("session %s said goodbye (%s)", session.session_id, header.get("reason", ""))
            self._drop_session(identity)
        else:
            log.warning("session %s sent unknown message type %r", session.session_id, msg_type)
            self._send(identity, wire.error(f"unknown message type {msg_type!r}"))

    def _on_hello(self, identity: bytes, header: Dict[str, Any]) -> None:
        peer_protocol = header.get("protocol")
        session = self._create_session(identity, header)
        session.greeted = True

        # A range rather than equality: everything added in v2 is additive, so a
        # v1 client is a client that simply never sends a map and never hears a
        # map_update.  What must still be refused is a version this server has
        # never heard of, which may mean something different by a field name it
        # thinks it recognises.
        if (not isinstance(peer_protocol, int)
                or not wire.MIN_PROTOCOL_VERSION <= peer_protocol <= wire.PROTOCOL_VERSION):
            msg = (f"client protocol {peer_protocol} is outside the supported range "
                   f"{wire.MIN_PROTOCOL_VERSION}..{wire.PROTOCOL_VERSION}")
            log.warning("session %s: %s", session.session_id, msg)
            self._send(
                identity,
                wire.welcome(
                    session.session_id,
                    self.cfg.processor.name,
                    accepted=False,
                    message=msg,
                    echo_t_send_ns=header.get("t_send_ns"),
                ),
            )
            self._drop_session(identity)
            return

        if peer_protocol != wire.PROTOCOL_VERSION:
            log.info(
                "session %s speaks protocol %d; this server is %d. The difference is "
                "additive (odometry, the occupancy grid, the exits and the server's "
                "announcements), so the session runs — without them.",
                session.session_id, peer_protocol, wire.PROTOCOL_VERSION,
            )

        codec = header.get("codec", wire.CODEC_JPEG)
        if codec not in wire.SUPPORTED_CODECS:
            msg = f"unsupported codec {codec!r}; server supports {', '.join(wire.SUPPORTED_CODECS)}"
            log.warning("session %s: %s", session.session_id, msg)
            self._send(
                identity,
                wire.welcome(
                    session.session_id,
                    self.cfg.processor.name,
                    accepted=False,
                    message=msg,
                    echo_t_send_ns=header.get("t_send_ns"),
                ),
            )
            self._drop_session(identity)
            return

        log.info(
            "session %s connected | client=%s camera=%s codec=%s %dx%d @%.1f fps | "
            "lidar=%s | imu=%s",
            session.session_id,
            session.client_id,
            session.camera or "-",
            session.codec,
            session.declared_width,
            session.declared_height,
            session.declared_fps,
            _describe_lidar(session.lidar_info),
            _describe_imu(session.imu_info),
        )
        self._send(
            identity,
            wire.welcome(
                session.session_id,
                self.cfg.processor.name,
                accepted=True,
                message="ready",
                server_info={
                    "host": socket.gethostname(),
                    "queue_depth": self.cfg.queue.max_depth,
                    "workers": self.cfg.processor.workers,
                    "processors_available": processors.available(),
                    # The client checks this before starting its scanner thread:
                    # streaming scans at a server that discards them wastes the
                    # robot's CPU and hides the misconfiguration.
                    "lidar": {
                        "enabled": self.cfg.lidar.enabled,
                        "encodings": list(wire.SUPPORTED_SCAN_ENCODINGS),
                        "mount_yaw_deg": self.cfg.lidar.mount_yaw_deg,
                        "camera_hfov_deg": self.cfg.lidar.camera_hfov_deg,
                        "stale_after_ms": self.cfg.lidar.stale_after_ms,
                    },
                    # Same bargain for the OpenCR: the robot is told what the
                    # server will do with bursts before it opens the board.
                    "imu": {
                        "enabled": self.cfg.imu.enabled,
                        "encodings": list(wire.SUPPORTED_IMU_ENCODINGS),
                        "fields": list(wire.IMU_FIELDS),
                        "stale_after_ms": self.cfg.imu.stale_after_ms,
                    },
                    # And for the two streams the robot's own software produces.
                    "odom": {
                        "enabled": self.cfg.odom.enabled,
                        "expect_frame": self.cfg.odom.expect_frame,
                        "stale_after_ms": self.cfg.odom.stale_after_ms,
                    },
                    "map": {
                        "enabled": self.cfg.map.enabled,
                        "encodings": list(wire.SUPPORTED_MAP_ENCODINGS),
                        "max_cells": self.cfg.map.max_cells,
                        # The robot needs all three before it can decide whether
                        # to spend the CPU serialising a grid every few seconds:
                        # a server that will not compare, or will compare but not
                        # send the result, is one there is no point uploading to.
                        "send_updates": self.cfg.map.send_updates and self.cfg.compare.enabled,
                        "merge": self.cfg.map.merge,
                    },
                    # What the robot must know to act on a pose_hint or a found:
                    # that they are coming at all, and which phase this server
                    # believes the session is in.
                    "compare": {
                        "enabled": self.cfg.compare.enabled,
                        "send_hints": self.cfg.compare.send_hints,
                        "z_slice": [self.cfg.compare.z_min, self.cfg.compare.z_max],
                    },
                    "mission": {
                        "phase": session.phase,
                        "target": session.mission.get("target", ""),
                        "rank_exits": self.cfg.mission.rank_exits,
                        "phases": list(wire.SUPPORTED_PHASES),
                    },
                    # Whether a `found` can ever arrive, and what it will mean.
                    # A robot whose seek behaviour tree waits on a monitor branch
                    # that no server is feeding waits forever and looks like it
                    # is searching, so this is worth stating before t2 rather
                    # than being inferred from silence.
                    "seek": {
                        "enabled": self.cfg.seek.enabled,
                        "detector": self.cfg.seek.detector if self.cfg.seek.enabled else "",
                        # The gripper envelope the `reachable` verdict is judged
                        # against, so the robot can explain a "too_high" without
                        # holding a second copy of a number that could disagree.
                        "grasp_z": [self.cfg.seek.grasp_z_min, self.cfg.seek.grasp_z_max],
                        "standoff_m": self.cfg.mission.approach_standoff_m,
                        # Announcing `found: false` after looking is what lets the
                        # robot stop waiting and start searching; a robot told
                        # this is 0 knows not to wait for one.
                        "absent_after_runs": self.cfg.seek.absent_after_runs,
                        "keyframes": self.cfg.seek.keyframes,
                        "basis": ["live", "memory", "absent"],
                    },
                },
                echo_t_send_ns=header.get("t_send_ns"),
            ),
        )

    def _create_session(self, identity: bytes, header: Dict[str, Any]) -> Session:
        old = self.sessions.pop(identity, None)
        if old is not None:
            old.decoder.close()
        session = Session(
            identity=identity,
            session_id=_identity_to_session_id(identity),
            client_id=str(header.get("client_id", "")),
            codec=str(header.get("codec", wire.CODEC_JPEG)),
            declared_width=int(header.get("width", 0) or 0),
            declared_height=int(header.get("height", 0) or 0),
            declared_fps=float(header.get("fps", 0) or 0),
            camera=str(header.get("camera", "")),
            lidar_info=dict(header.get("lidar") or {}),
            imu_info=dict(header.get("imu") or {}),
            odom_info=dict(header.get("odom") or {}),
            map_info=dict(header.get("map") or {}),
        )
        # The phase and the mission belong to the run, so the robot's hello wins
        # over the config; the config supplies the default for a client that has
        # no opinion, which is every v1 client.
        try:
            session.phase = mission_mod.check_phase(header.get("phase"))
        except mission_mod.MissionError:
            session.phase = self.cfg.mission.phase
        try:
            session.mission = mission_mod.normalise_mission(header.get("mission"))
        except mission_mod.MissionError as exc:
            log.warning("session %s sent an unusable mission: %s", session.session_id, exc)
            session.mission = {}
        if not session.mission.get("target") and self.cfg.mission.target:
            session.mission["target"] = self.cfg.mission.target
        self.sessions[identity] = session
        return session

    def _drop_session(self, identity: bytes) -> None:
        session = self.sessions.pop(identity, None)
        if session is not None:
            session.decoder.close()
            log.info(
                "session %s closed | %d frames, %d dropped, %d failed, "
                "%d scans (%d bad), %d imu samples (%d bad bursts), "
                "%d poses (%d bad), %d maps (%d bad), "
                "%d patches / %d hints / %d found sent | %.1f MB, %.0fs",
                session.session_id,
                session.frames_received,
                session.frames_dropped,
                session.frames_failed,
                session.scans_received,
                session.scans_failed,
                session.imu_samples_received,
                session.imu_failed,
                session.odom_received,
                session.odom_failed,
                session.maps_received,
                session.maps_failed,
                session.map_updates_sent,
                session.pose_hints_sent,
                session.founds_sent,
                session.bytes_received / 1e6,
                session.age_s(),
            )

    def _on_frame(self, session: Session, header: Dict[str, Any], payload: bytes, recv_ts_ns: int) -> None:
        seq = int(header.get("seq", -1))
        session.frames_received += 1
        session.bytes_received += len(payload)
        self._stat_frames += 1
        self._stat_bytes += len(payload)

        if session.last_seq >= 0 and seq != session.last_seq + 1:
            session.seq_gaps += 1
        session.last_seq = seq

        if len(payload) > self.cfg.server.max_payload_bytes:
            log.warning(
                "session %s: payload %d bytes exceeds limit %d",
                session.session_id, len(payload), self.cfg.server.max_payload_bytes,
            )
            session.frames_failed += 1
            self._send_failure(session, header, seq, len(payload), wire.REASON_DECODE_FAILED,
                               "payload too large", recv_ts_ns)
            return

        t0 = time.perf_counter()
        try:
            image = session.decoder.decode(header, payload)
        except DecodeError as exc:
            decode_ms = (time.perf_counter() - t0) * 1000.0
            session.frames_failed += 1
            reason = (
                wire.REASON_UNSUPPORTED_CODEC
                if "unsupported codec" in str(exc)
                else wire.REASON_DECODE_FAILED
            )
            log.warning("session %s: seq=%d decode failed: %s", session.session_id, seq, exc)
            self._send_failure(session, header, seq, len(payload), reason, str(exc),
                               recv_ts_ns, decode_ms=decode_ms)
            return
        decode_ms = (time.perf_counter() - t0) * 1000.0

        # Pair the frame with the most recent scan, if there is one recent
        # enough to still describe the same world.  Association is by arrival on
        # the server's clock: the two sensors timestamp on the robot's clock but
        # travel independently, and at 5 Hz the scan's own age dominates
        # anything the transport adds.
        scan, scan_age_ms = session.fresh_scan(self.cfg.lidar.stale_after_ms)
        # Same association, much tighter window: inertial data describes the
        # instant it was taken, and a 200 ms old attitude belongs to a robot
        # that may already have finished the turn.
        burst, imu_age_ms = session.fresh_imu(self.cfg.imu.stale_after_ms)
        # And the pose.  Tighter again than the inertial window in what it costs
        # to be wrong: this is what decides *where* everything the model produces
        # from this frame ends up on the robot's map.
        pose, odom_age_ms = session.fresh_odom(self.cfg.odom.stale_after_ms)

        self._frame_counter += 1
        self.snapshots.maybe_offer(session.session_id, seq, image, self._frame_counter,
                                   scan=scan, imu=burst)

        frame = Frame(
            seq=seq,
            session_id=session.session_id,
            image=image,
            header=header,
            recv_ts_ns=recv_ts_ns,
            decode_ms=decode_ms,
            payload_bytes=len(payload),
            scan=scan,
            scan_age_ms=scan_age_ms if scan is not None else 0.0,
            imu=burst,
            imu_age_ms=imu_age_ms if burst is not None else 0.0,
            odom=pose,
            odom_age_ms=odom_age_ms if pose is not None else 0.0,
            # The grid is handed over by reference and shared by every frame in
            # flight; see Frame.grid for why nothing may write to it in place.
            # No staleness cutoff: a map that is ten seconds old still describes
            # the room, which is the whole difference between it and a scan.
            grid=session.grid,
            grid_age_ms=session.grid_age_ms(),
            exits=list(session.exits),
            phase=session.phase,
            mission=dict(session.mission),
        )
        for evicted in self.frames.put(frame):
            session.frames_dropped += 1
            self._stat_dropped += 1
            self._send_dropped(session, evicted)

    # -- LiDAR path -------------------------------------------------------

    def _on_scan(self, session: Session, header: Dict[str, Any], payload: bytes, recv_ts_ns: int) -> None:
        """Parse, analyse and answer one revolution, all on the IO thread.

        Every scan gets exactly one reply for the same reason every frame does:
        the client sizes its in-flight window from outstanding replies, so a
        silently dropped scan would stall the LiDAR stream and nothing else.
        """
        seq = int(header.get("seq", -1))
        cfg = self.cfg.lidar

        if not cfg.enabled:
            self._send(session.identity, wire.scan_result(
                seq=seq, ok=False, reason=wire.REASON_LIDAR_DISABLED,
                payload_bytes=len(payload),
                encoding=str(header.get("encoding", "")),
                server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
                data={"error": "server has lidar.enabled: false"},
                t_capture_ns=header.get("t_capture_ns"),
                t_send_ns=header.get("t_send_ns"),
            ))
            return

        session.scans_received += 1
        session.bytes_received += len(payload)
        self._stat_scans += 1
        self._stat_scan_bytes += len(payload)

        t0 = time.perf_counter()
        try:
            scan = lidar_mod.decode_scan(header, payload, recv_ts_ns=recv_ts_ns)
        except lidar_mod.ScanError as exc:
            session.scans_failed += 1
            log.warning("session %s: scan seq=%d rejected: %s", session.session_id, seq, exc)
            self._send(session.identity, wire.scan_result(
                seq=seq, ok=False, reason=wire.REASON_BAD_SCAN,
                payload_bytes=len(payload),
                encoding=str(header.get("encoding", "")),
                parse_ms=(time.perf_counter() - t0) * 1000.0,
                server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
                data={"error": str(exc)},
                t_capture_ns=header.get("t_capture_ns"),
                t_send_ns=header.get("t_send_ns"),
            ))
            return

        if session.last_scan_ns:
            session.scan_interval_ms = (recv_ts_ns - session.last_scan_ns) / 1e6

        scan.summary = lidar_mod.analyse(
            scan,
            sectors=cfg.sectors,
            obstacle_m=cfg.obstacle_m,
            clear_m=cfg.clear_m,
            front_deg=cfg.front_deg,
            min_free_deg=cfg.min_free_deg,
            mount_yaw_deg=cfg.mount_yaw_deg,
            hz=session.scan_hz(),
        )
        parse_ms = (time.perf_counter() - t0) * 1000.0

        session.last_scan = scan
        session.last_scan_ns = recv_ts_ns

        self._send(session.identity, wire.scan_result(
            seq=seq, ok=True, reason=wire.REASON_OK,
            points=scan.count,
            payload_bytes=len(payload),
            encoding=str(header.get("encoding", "")),
            parse_ms=parse_ms,
            server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
            data=scan.summary,
            t_capture_ns=header.get("t_capture_ns"),
            t_send_ns=header.get("t_send_ns"),
        ))

    # -- IMU path ---------------------------------------------------------

    def _on_imu(self, session: Session, header: Dict[str, Any], payload: bytes, recv_ts_ns: int) -> None:
        """Parse, analyse and answer one burst of inertial samples.

        Every burst gets exactly one reply, for the same reason every frame and
        every scan does: the client sizes its in-flight window from outstanding
        replies, so a silently dropped burst would stall the IMU stream alone.
        """
        seq = int(header.get("seq", -1))
        cfg = self.cfg.imu

        if not cfg.enabled:
            self._send(session.identity, wire.imu_result(
                seq=seq, ok=False, reason=wire.REASON_IMU_DISABLED,
                payload_bytes=len(payload),
                encoding=str(header.get("encoding", "")),
                server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
                data={"error": "server has imu.enabled: false"},
                t_capture_ns=header.get("t_capture_ns"),
                t_send_ns=header.get("t_send_ns"),
            ))
            return

        session.imu_bursts_received += 1
        session.bytes_received += len(payload)
        self._stat_imu_bursts += 1
        self._stat_imu_bytes += len(payload)

        t0 = time.perf_counter()
        try:
            batch = imu_mod.decode_imu(header, payload, recv_ts_ns=recv_ts_ns)
        except imu_mod.ImuError as exc:
            session.imu_failed += 1
            log.warning("session %s: imu seq=%d rejected: %s", session.session_id, seq, exc)
            self._send(session.identity, wire.imu_result(
                seq=seq, ok=False, reason=wire.REASON_BAD_IMU,
                payload_bytes=len(payload),
                encoding=str(header.get("encoding", "")),
                parse_ms=(time.perf_counter() - t0) * 1000.0,
                server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
                data={"error": str(exc)},
                t_capture_ns=header.get("t_capture_ns"),
                t_send_ns=header.get("t_send_ns"),
            ))
            return

        session.imu_samples_received += batch.count
        self._stat_imu_samples += batch.count

        batch.summary = imu_mod.analyse(
            batch,
            still_gyro_dps=cfg.still_gyro_dps,
            still_accel_ms2=cfg.still_accel_ms2,
            tilt_warn_deg=cfg.tilt_warn_deg,
            shock_ms2=cfg.shock_ms2,
            gravity_tolerance_ms2=cfg.gravity_tolerance_ms2,
        )
        parse_ms = (time.perf_counter() - t0) * 1000.0

        session.last_imu = batch
        session.last_imu_ns = recv_ts_ns

        self._send(session.identity, wire.imu_result(
            seq=seq, ok=True, reason=wire.REASON_OK,
            samples=batch.count,
            payload_bytes=len(payload),
            encoding=str(header.get("encoding", "")),
            parse_ms=parse_ms,
            server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
            data=batch.summary,
            t_capture_ns=header.get("t_capture_ns"),
            t_send_ns=header.get("t_send_ns"),
        ))

    # -- odometry path ----------------------------------------------------

    def _on_odom(self, session: Session, header: Dict[str, Any], recv_ts_ns: int) -> None:
        """Parse and answer one pose.  No payload; the header is the message.

        The reply is thin because there is nothing to say about a single pose.
        What it does carry is the one thing the robot cannot determine from its
        own side: whether the server can actually use this pose — whether it is
        in the frame the uploaded grid is drawn in, and whether there is a grid
        at all.  "The server is not comparing anything" and "the server is
        comparing and finding nothing" are otherwise the same silence.
        """
        seq = int(header.get("seq", -1))
        cfg = self.cfg.odom

        if not cfg.enabled:
            self._send(session.identity, wire.odom_result(
                seq=seq, ok=False, reason=wire.REASON_ODOM_DISABLED,
                server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
                data={"error": "server has odom.enabled: false"},
                t_capture_ns=header.get("t_capture_ns"),
                t_send_ns=header.get("t_send_ns"),
            ))
            return

        session.odom_received += 1
        try:
            pose = odom_mod.decode_odom(header, recv_ts_ns=recv_ts_ns)
        except odom_mod.OdomError as exc:
            session.odom_failed += 1
            log.warning("session %s: odom seq=%d rejected: %s", session.session_id, seq, exc)
            self._send(session.identity, wire.odom_result(
                seq=seq, ok=False, reason=wire.REASON_BAD_ODOM,
                server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
                data={"error": str(exc)},
                t_capture_ns=header.get("t_capture_ns"),
                t_send_ns=header.get("t_send_ns"),
            ))
            return

        pose.summary = odom_mod.analyse(
            pose, previous=session.last_odom,
            still_speed_ms=cfg.still_speed_ms,
            still_yaw_rate_dps=cfg.still_yaw_rate_dps,
        )
        session.prev_odom = session.last_odom
        session.last_odom = pose
        session.last_odom_ns = recv_ts_ns

        data = dict(pose.summary)
        # The three preconditions for the comparison, reported rather than
        # implied.  A robot sending poses in "odom" against a grid in "map" gets
        # told so on every pose instead of discovering it as a map that never
        # improves.
        frame_ok = (not cfg.expect_frame) or pose.frame == cfg.expect_frame
        data["usable_for_compare"] = bool(
            self.cfg.compare.enabled and frame_ok and session.grid is not None
            and (session.grid is None or session.grid.frame == pose.frame)
        )
        if not frame_ok:
            data["frame_warning"] = (
                f"pose is in {pose.frame!r} but the server expects "
                f"{cfg.expect_frame!r}; dead reckoning cannot be compared "
                "against a SLAM grid"
            )
        elif session.grid is None:
            data["frame_warning"] = "no occupancy grid uploaded yet, nothing to compare against"
        elif session.grid.frame != pose.frame:
            data["frame_warning"] = (
                f"pose is in {pose.frame!r}, grid is in {session.grid.frame!r}"
            )

        self._send(session.identity, wire.odom_result(
            seq=seq, ok=True, reason=wire.REASON_OK,
            server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
            data=data,
            t_capture_ns=header.get("t_capture_ns"),
            t_send_ns=header.get("t_send_ns"),
        ))

    # -- occupancy grid path ----------------------------------------------

    def _on_map(self, session: Session, header: Dict[str, Any], payload: bytes,
                recv_ts_ns: int) -> None:
        """Parse and store one occupancy grid, or apply one patch of one.

        This is the only inline path that is not tiny — a 400x400 grid is 160 kB
        to decompress — and it is inline anyway because it happens every few
        seconds rather than every frame.  Queueing it behind the model would buy
        nothing and would mean a map arriving after the frames it should have
        been compared against.
        """
        seq = int(header.get("seq", -1))
        cfg = self.cfg.map

        if not cfg.enabled:
            self._send(session.identity, wire.map_result(
                seq=seq, ok=False, reason=wire.REASON_MAP_DISABLED,
                payload_bytes=len(payload),
                encoding=str(header.get("encoding", "")),
                server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
                data={"error": "server has map.enabled: false"},
                t_capture_ns=header.get("t_capture_ns"),
                t_send_ns=header.get("t_send_ns"),
            ))
            return

        session.maps_received += 1
        session.bytes_received += len(payload)
        self._stat_map_bytes += len(payload)

        t0 = time.perf_counter()
        try:
            grid = occupancy_mod.decode_map(
                header, payload, recv_ts_ns=recv_ts_ns, max_cells=cfg.max_cells,
            )
        except occupancy_mod.MapError as exc:
            session.maps_failed += 1
            reason = (wire.REASON_MAP_TOO_LARGE if "cell limit" in str(exc)
                      else wire.REASON_BAD_MAP)
            log.warning("session %s: map seq=%d rejected: %s", session.session_id, seq, exc)
            self._send(session.identity, wire.map_result(
                seq=seq, ok=False, reason=reason,
                payload_bytes=len(payload),
                encoding=str(header.get("encoding", "")),
                parse_ms=(time.perf_counter() - t0) * 1000.0,
                server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
                data={"error": str(exc)},
                t_capture_ns=header.get("t_capture_ns"),
                t_send_ns=header.get("t_send_ns"),
            ))
            return

        applied = 0
        if grid.full:
            session.grid = grid
        elif session.grid is None:
            # A patch with nothing to patch.  Refused rather than promoted to a
            # full grid: its x0/y0 place it inside a map this server has never
            # seen, and treating the window as the whole map would put every
            # later comparison at an offset nothing would ever detect.
            session.maps_failed += 1
            self._send(session.identity, wire.map_result(
                seq=seq, ok=False, reason=wire.REASON_BAD_MAP,
                payload_bytes=len(payload),
                encoding=str(header.get("encoding", "")),
                parse_ms=(time.perf_counter() - t0) * 1000.0,
                server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
                data={"error": "patch received before any full grid; send full: true first"},
                t_capture_ns=header.get("t_capture_ns"),
                t_send_ns=header.get("t_send_ns"),
            ))
            return
        elif session.grid.map_id != grid.map_id:
            session.maps_failed += 1
            self._send(session.identity, wire.map_result(
                seq=seq, ok=False, reason=wire.REASON_BAD_MAP,
                payload_bytes=len(payload),
                encoding=str(header.get("encoding", "")),
                parse_ms=(time.perf_counter() - t0) * 1000.0,
                server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
                data={"error": (f"patch is for map_id {grid.map_id!r}, the server holds "
                                f"{session.grid.map_id!r}; send the full grid after a reset")},
                t_capture_ns=header.get("t_capture_ns"),
                t_send_ns=header.get("t_send_ns"),
            ))
            return
        else:
            # The robot's own patches replace: they are its map, not an opinion
            # about it, and it may legitimately clear a cell it has re-observed.
            applied = occupancy_mod.apply_patch(
                session.grid, grid, merge=wire.MAP_MERGE_REPLACE,
            )
            session.grid.seq = grid.seq

        session.grid_ns = recv_ts_ns
        parse_ms = (time.perf_counter() - t0) * 1000.0

        held = session.grid
        held.summary = occupancy_mod.analyse(held)
        data = dict(held.summary)
        data["parse_ms"] = round(parse_ms, 2)
        if not grid.full:
            data["patch_applied_cells"] = applied
        pose, _ = session.fresh_odom(self.cfg.odom.stale_after_ms)
        data["compare_ready"] = bool(
            self.cfg.compare.enabled and pose is not None and pose.frame == held.frame
        )
        if not data["compare_ready"]:
            data["compare_blocked_by"] = (
                "compare.enabled is false" if not self.cfg.compare.enabled
                else "no fresh pose" if pose is None
                else f"pose is in {pose.frame!r}, grid is in {held.frame!r}"
            )

        verdict = self._evaluate_t1(session, held)
        if verdict is not None:
            data["t1_exit"] = verdict

        log.info(
            "session %s: map seq=%d %dx%d @%.3f m/cell, %.0f%% explored, "
            "%d occupied (%.1f kB on the wire, %.1f ms)",
            session.session_id, seq, held.width, held.height, held.resolution,
            100.0 * held.summary.get("explored_fraction", 0.0),
            held.summary.get("occupied", 0), len(payload) / 1000.0, parse_ms,
        )

        self._send(session.identity, wire.map_result(
            seq=seq, ok=True, reason=wire.REASON_OK,
            cells=held.size,
            payload_bytes=len(payload),
            encoding=str(header.get("encoding", "")),
            parse_ms=parse_ms,
            server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
            data=data,
            t_capture_ns=header.get("t_capture_ns"),
            t_send_ns=header.get("t_send_ns"),
        ))

    # -- is T1 finished? --------------------------------------------------

    def _note_agreement(self, session: Session, data: Dict[str, Any]) -> None:
        """Accumulate the comparison's health number off each frame's result.

        Read out of the processor's own payload rather than recomputed, because
        the comparison is the processor's product and a second implementation
        here could disagree with it.  Only comparisons that actually ran count:
        a frame with no grid or no pose produced no evidence about placement, and
        averaging its absence in as a zero would make a session that started
        before the robot's map arrived look permanently misplaced.
        """
        comp = data.get("compare") if isinstance(data, dict) else None
        if not isinstance(comp, dict) or not comp.get("placed"):
            return
        agreement = comp.get("agreement")
        if agreement is None:
            return
        session.agreement_sum += float(agreement)
        session.agreement_n += 1

    def _evaluate_t1(self, session: Session, grid: occupancy_mod.Grid) -> Optional[Dict[str, Any]]:
        """Answer "is the first scan finished?" and say what is still missing.

        A **recommendation**, never an action.  The server does not change phase
        on its own, for the same reason it does not decide which exit to drive
        to: the transition between exploring and seeking is a decision about the
        mission, whoever is running the mission owns it, and a server that
        switched by itself would also have to be argued with to end T1 early for
        a demo.  What it can do is measure the four things the robot cannot see
        from where it stands — how much of its own grid it has observed, whether
        that number is still moving, how many frontiers it still calls open, and
        whether the reconstruction it has been building is placed where the map
        is — and hand back the verdict with every test that produced it.
        """
        if not self.cfg.mission.t1_exit_report:
            return None

        session.coverage.observe(grid.size - int(grid.summary.get("unknown", 0)))
        criteria = seek_mod.T1Criteria(
            min_explored=self.cfg.mission.t1_min_explored,
            max_open_exits=self.cfg.mission.t1_max_open_exits,
            stall_cells_per_s=self.cfg.mission.t1_stall_cells_per_s,
            min_stall_window_s=self.cfg.mission.t1_min_stall_window_s,
            min_agreement=self.cfg.mission.t1_min_agreement,
            min_frames=self.cfg.mission.t1_min_frames,
            min_runtime_s=self.cfg.mission.t1_min_runtime_s,
        )
        verdict = seek_mod.t1_exit_criteria(
            grid, session.exits,
            criteria=criteria,
            coverage=session.coverage,
            mean_agreement=session.mean_agreement,
            frames=session.frames_processed,
            runtime_s=session.age_s(),
            min_exit_width_m=self.cfg.mission.min_exit_width_m,
        )
        if verdict["ready"] and session.phase == wire.PHASE_EXPLORE and not session.t1_ready_logged:
            session.t1_ready_logged = True
            log.info(
                "session %s: t1 looks finished -- %.0f%% explored, %d open exits, "
                "growth %s cells/s, agreement %s. Switch to t2 with a target when "
                "ready; this server will not switch on its own.",
                session.session_id,
                100.0 * verdict["tests"].get("explored_fraction", 0.0),
                verdict["tests"].get("open_exits", 0),
                verdict["tests"].get("growth_cells_per_s"),
                verdict["tests"].get("mean_agreement"),
            )
        elif not verdict["ready"]:
            # Un-latch, so a map that grows again after a plateau reports the
            # completion a second time rather than staying silent about it.
            session.t1_ready_logged = False
        return verdict

    # -- exits path -------------------------------------------------------

    def _on_exits(self, session: Session, header: Dict[str, Any], recv_ts_ns: int) -> None:
        """Validate the robot's exit candidates and answer with a ranking."""
        seq = int(header.get("seq", -1))
        try:
            candidates = mission_mod.decode_exits(header)
        except mission_mod.MissionError as exc:
            log.warning("session %s: exits seq=%d rejected: %s", session.session_id, seq, exc)
            self._send(session.identity, wire.exits_result(
                seq=seq, ok=False, reason=wire.REASON_BAD_EXITS,
                server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
                data={"error": str(exc)},
                t_capture_ns=header.get("t_capture_ns"),
                t_send_ns=header.get("t_send_ns"),
            ))
            return

        session.exits = candidates
        session.exits_ns = recv_ts_ns
        session.exits_received += 1

        if not self.cfg.mission.rank_exits:
            self._send(session.identity, wire.exits_result(
                seq=seq, ok=True, reason=wire.REASON_OK,
                server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
                data={"received": len(candidates), "ranked": False,
                      "note": "server has mission.rank_exits: false"},
                t_capture_ns=header.get("t_capture_ns"),
                t_send_ns=header.get("t_send_ns"),
            ))
            return

        pose, _ = session.fresh_odom(self.cfg.odom.stale_after_ms)
        ranked, chosen, method = mission_mod.rank_exits(
            candidates, pose,
            min_width_m=self.cfg.mission.min_exit_width_m,
            turn_cost_m_per_rad=self.cfg.mission.turn_cost_m_per_rad,
        )
        self._send(session.identity, wire.exits_result(
            seq=seq, ok=True, reason=wire.REASON_OK,
            ranked=ranked, chosen=chosen,
            server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
            data={"received": len(candidates), "method": method,
                  "phase": session.phase,
                  # Without a pose the ranking cannot measure distance and falls
                  # back to the robot's own priors; saying so beats returning an
                  # order that looks considered and is not.
                  "had_pose": pose is not None},
            t_capture_ns=header.get("t_capture_ns"),
            t_send_ns=header.get("t_send_ns"),
        ))

    # -- phase path -------------------------------------------------------

    def _on_phase(self, session: Session, header: Dict[str, Any]) -> None:
        """Move the session between t1 and t2, and say so back.

        Answered with the same message type and ``accepted`` set, which is what
        keeps the two ends from spending a minute in different phases after one
        lost message.  A rejected phase leaves the session where it was.
        """
        try:
            name = mission_mod.check_phase(header.get("phase"))
        except mission_mod.MissionError as exc:
            log.warning("session %s: %s", session.session_id, exc)
            self._send(session.identity, wire.phase(
                session.phase, reason=str(exc), accepted=False,
            ))
            return

        if "mission" in header:
            try:
                updated = mission_mod.normalise_mission(header.get("mission"))
            except mission_mod.MissionError as exc:
                log.warning("session %s sent an unusable mission: %s", session.session_id, exc)
            else:
                session.mission.update(updated)

        previous, session.phase = session.phase, name
        if previous != name:
            log.info("session %s: phase %s -> %s (%s)%s",
                     session.session_id, previous, name,
                     header.get("reason", "no reason given"),
                     f" target={session.mission['target']!r}"
                     if session.mission.get("target") else "")
        self._send(session.identity, wire.phase(
            name, reason=str(header.get("reason", "")),
            mission=session.mission, accepted=True,
        ))

    # -- send path --------------------------------------------------------

    def _flush_results(self) -> None:
        """Move everything the workers finished onto the socket."""
        while True:
            try:
                item = self.results.get_nowait()
            except queue.Empty:
                return
            self._send_result(item)

    def _send_result(self, item: ProcessedResult) -> None:
        session = self._session_by_id(item.session_id)
        if session is None:
            # Client disconnected while its frame was in flight.
            return
        session.frames_processed += 1
        self._note_agreement(session, item.data)

        # The processor's own products go first, so a robot acting on the result
        # already holds whatever they carried.
        if item.announcements:
            self._send_announcements(session, item)

        frame = item.frame
        img = frame.image
        server_ms = (wire.monotonic_ns() - frame.recv_ts_ns) / 1e6
        self._stat_latency_ms += server_ms

        header = wire.result(
            seq=item.seq,
            ok=item.ok,
            reason=item.reason,
            width=int(img.shape[1]),
            height=int(img.shape[0]),
            channels=int(img.shape[2]) if img.ndim == 3 else 1,
            dtype=str(img.dtype),
            nbytes=int(img.nbytes),
            payload_bytes=frame.payload_bytes,
            codec=str(frame.header.get("codec", "")),
            decode_ms=frame.decode_ms,
            process_ms=item.process_ms,
            queue_ms=item.queue_ms,
            server_ms=server_ms,
            processor=self.cfg.processor.name,
            data=item.data,
            t_capture_ns=frame.header.get("t_capture_ns"),
            t_send_ns=frame.header.get("t_send_ns"),
            scan_seq=frame.scan.seq if frame.scan is not None else None,
            scan_age_ms=frame.scan_age_ms if frame.scan is not None else None,
            imu_seq=frame.imu.seq if frame.imu is not None else None,
            imu_age_ms=frame.imu_age_ms if frame.imu is not None else None,
            odom_seq=frame.odom.seq if frame.odom is not None else None,
            odom_age_ms=frame.odom_age_ms if frame.odom is not None else None,
            map_seq=frame.grid.seq if frame.grid is not None else None,
            map_id=frame.grid.map_id if frame.grid is not None else None,
        )
        self._send(session.identity, header)

    def _send_announcements(self, session: Session, item: ProcessedResult) -> None:
        """Send the unsolicited messages one frame produced.

        Everything a worker cannot decide happens here, on the one thread that
        holds the session:

        * **Sequence numbers.** Per session, per stream, allocated here so two
          workers cannot allocate the same one.
        * **The operator's switches.** ``map.send_updates`` and
          ``compare.send_hints`` gate what leaves the server, independently of
          what the processor computed.  That split is deliberate: it lets you run
          the comparison and read what it *would* have sent, in the logs, without
          letting it write to the robot's costmap.
        * **Rate limits.** The reconstruction produces several clouds a second;
          the robot's costmap does not need rewriting at that rate, and a pose
          correction the robot has not had time to apply will only be measured
          again.
        * **Staleness of the map identity.** A patch whose ``map_id`` no longer
          matches the grid the robot is on describes coordinates in a map that no
          longer exists.  Dropping it here is cheap; applying it on the robot
          would corrupt the new map in a way nothing downstream could detect.
        """
        now = wire.monotonic_ns()
        current_map_id = session.grid.map_id if session.grid is not None else None

        for header, payload in item.announcements:
            mtype = header.get("type")

            if mtype == wire.MSG_MAP_UPDATE:
                if not (self.cfg.map.send_updates and self.cfg.compare.enabled):
                    log.debug("session %s: map_update suppressed by config", session.session_id)
                    continue
                if current_map_id is not None and header.get("map_id") != current_map_id:
                    log.info("session %s: dropping map_update for map_id %r; robot is on %r",
                             session.session_id, header.get("map_id"), current_map_id)
                    continue
                interval_ms = (now - session.last_map_update_ns) / 1e6
                if (session.last_map_update_ns
                        and interval_ms < self.cfg.map.min_update_interval_ms):
                    continue
                if int(header.get("cells_changed", 0)) < self.cfg.map.min_update_cells:
                    continue
                header["seq"] = session.map_update_seq
                header.setdefault("merge", self.cfg.map.merge)
                session.map_update_seq += 1
                session.last_map_update_ns = now
                session.map_updates_sent += 1

            elif mtype == wire.MSG_POSE_HINT:
                if not (self.cfg.compare.enabled and self.cfg.compare.send_hints):
                    continue
                if current_map_id is not None and header.get("map_id") != current_map_id:
                    continue
                interval_ms = (now - session.last_pose_hint_ns) / 1e6
                if (session.last_pose_hint_ns
                        and interval_ms < self.cfg.compare.min_hint_interval_ms):
                    continue
                header["seq"] = session.pose_hint_seq
                session.pose_hint_seq += 1
                session.last_pose_hint_ns = now
                session.pose_hints_sent += 1
                log.info("session %s: offering SLAM a correction of (%+.3f, %+.3f) m "
                         "from %d inliers -- advisory, %s",
                         session.session_id, header.get("dx", 0.0), header.get("dy", 0.0),
                         header.get("inliers", 0), header.get("method", "?"))

            elif mtype == wire.MSG_FOUND:
                # Not rate-limited and not gated: this is the end of the mission,
                # and a server that decided the target is in view has nothing to
                # gain by holding the message back.  It is logged at info because
                # it is the single most consequential thing this server sends.
                header["seq"] = session.found_seq
                session.found_seq += 1
                session.founds_sent += 1
                log.info("session %s: %s %r at (%.2f, %.2f) confidence %.2f -- %s",
                         session.session_id,
                         "FOUND" if header.get("found") else "did not find",
                         header.get("target", ""), header.get("x", 0.0),
                         header.get("y", 0.0), header.get("confidence", 0.0),
                         header.get("rationale", "") or header.get("decider", ""))

            else:
                log.warning("session %s: processor announced unknown type %r",
                            session.session_id, mtype)
                continue

            self._stat_announcements += 1
            self._send(session.identity, header, payload)

    def _send_dropped(self, session: Session, frame: Frame) -> None:
        """Acknowledge a frame the queue evicted.

        The client counts outstanding replies to decide when to send more, so
        every frame must produce exactly one result — including this one.
        """
        header = wire.result(
            seq=frame.seq,
            ok=False,
            reason=wire.REASON_DROPPED,
            payload_bytes=frame.payload_bytes,
            codec=str(frame.header.get("codec", "")),
            decode_ms=frame.decode_ms,
            server_ms=(wire.monotonic_ns() - frame.recv_ts_ns) / 1e6,
            processor=self.cfg.processor.name,
            t_capture_ns=frame.header.get("t_capture_ns"),
            t_send_ns=frame.header.get("t_send_ns"),
        )
        self._send(session.identity, header)

    def _send_failure(
        self,
        session: Session,
        frame_header: Dict[str, Any],
        seq: int,
        payload_bytes: int,
        reason: str,
        message: str,
        recv_ts_ns: int,
        decode_ms: float = 0.0,
    ) -> None:
        header = wire.result(
            seq=seq,
            ok=False,
            reason=reason,
            payload_bytes=payload_bytes,
            codec=str(frame_header.get("codec", "")),
            decode_ms=decode_ms,
            server_ms=(wire.monotonic_ns() - recv_ts_ns) / 1e6,
            processor=self.cfg.processor.name,
            data={"error": message},
            t_capture_ns=frame_header.get("t_capture_ns"),
            t_send_ns=frame_header.get("t_send_ns"),
        )
        self._send(session.identity, header)

    def _send(self, identity: bytes, header: Dict[str, Any], payload: bytes = b"") -> None:
        assert self.sock is not None
        head, body = wire.encode(header, payload)
        try:
            self.sock.send_multipart([identity, head, body], zmq.NOBLOCK)
        except zmq.Again:
            # Send buffer full: the peer is not draining. Dropping a result is
            # the right call — the next frame supersedes it anyway.
            log.debug("send buffer full for %s, dropping a result", _identity_to_session_id(identity))
        except zmq.ZMQError as exc:
            # ROUTER_MANDATORY reports EHOSTUNREACH once the peer is gone.
            log.info("peer %s unreachable (%s), dropping session",
                     _identity_to_session_id(identity), exc)
            self._drop_session(identity)

    # -- housekeeping -----------------------------------------------------

    def _session_by_id(self, session_id: str) -> Optional[Session]:
        for session in self.sessions.values():
            if session.session_id == session_id:
                return session
        return None

    def _reap_sessions(self) -> None:
        timeout = self.cfg.server.session_timeout_s
        if timeout <= 0:
            return
        for identity in [i for i, s in self.sessions.items() if s.idle_s() > timeout]:
            log.info("session %s timed out after %.0fs idle",
                     _identity_to_session_id(identity), timeout)
            self._drop_session(identity)

    def _log_stats(self) -> None:
        interval = self.cfg.logging.stats_interval_s
        if interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._stat_t0
        if elapsed < interval:
            return

        if self._stat_frames or self._stat_scans or self._stat_imu_bursts:
            fps = self._stat_frames / elapsed
            mbps = ((self._stat_bytes + self._stat_scan_bytes + self._stat_imu_bytes)
                    * 8) / elapsed / 1e6
            avg_ms = self._stat_latency_ms / max(self._stat_frames, 1)
            log.info(
                "%.1f fps | %.1f scans/s | %.0f imu/s%s%s | %.1f Mbit/s | "
                "%.1f ms server-side | %d dropped | %d session(s) | queue=%d",
                fps,
                self._stat_scans / elapsed,
                # Samples a second, not bursts: it is the number to compare
                # against the sensor's own rate when hunting a gap.
                self._stat_imu_samples / elapsed,
                self._nearest_obstacle_note(),
                self._attitude_note() + self._map_note(),
                mbps, avg_ms, self._stat_dropped, len(self.sessions), len(self.frames),
            )
        elif self.sessions:
            log.info("no frames in %.0fs | %d session(s) idle", elapsed, len(self.sessions))

        self._stat_t0 = now
        self._stat_frames = 0
        self._stat_bytes = 0
        self._stat_dropped = 0
        self._stat_latency_ms = 0.0
        self._stat_scans = 0
        self._stat_scan_bytes = 0
        self._stat_imu_bursts = 0
        self._stat_imu_samples = 0
        self._stat_imu_bytes = 0
        self._stat_map_bytes = 0
        self._stat_announcements = 0

    def _map_note(self) -> str:
        """What the server currently holds of the robot's map, and what it sent back.

        Two numbers, for the two questions this half of the system raises first:
        is there a map on the server at all (a robot that never uploads one looks
        identical to one whose uploads are being refused), and is anything coming
        back out of the comparison.
        """
        for session in self.sessions.values():
            if session.grid is None:
                continue
            note = " | map %dx%d %.0f%% known" % (
                session.grid.width, session.grid.height,
                100.0 * session.grid.summary.get("explored_fraction", 0.0),
            )
            if session.map_updates_sent or session.pose_hints_sent:
                note += " (%d patch, %d hint sent)" % (
                    session.map_updates_sent, session.pose_hints_sent)
            return note
        return ""

    def _nearest_obstacle_note(self) -> str:
        """The closest thing any session can currently see, for the stats line.

        Worth a few characters of log: it is the one number that tells you at a
        glance whether the LiDAR is producing plausible measurements or just
        producing messages.
        """
        best = None
        for session in self.sessions.values():
            summary = session.last_scan.summary if session.last_scan is not None else None
            if not summary or summary.get("nearest_m") is None:
                continue
            if best is None or summary["nearest_m"] < best[0]:
                best = (summary["nearest_m"], summary.get("nearest_deg", 0.0))
        if best is None:
            return ""
        return f" | nearest {best[0]:.2f} m @{best[1]:+.0f}°"

    def _attitude_note(self) -> str:
        """Tilt and yaw rate for the stats line, when any session has an IMU.

        The counterpart to the nearest-obstacle note: two numbers that say the
        inertial data is describing a real robot rather than merely arriving.
        A permanent 90° tilt on a robot standing on the floor is a mounting
        mistake, and it is visible here on the first stats line.
        """
        for session in self.sessions.values():
            summary = session.last_imu.summary if session.last_imu is not None else None
            if not summary or summary.get("tilt_deg") is None:
                continue
            note = " | tilt %.0f°" % summary["tilt_deg"]
            if summary.get("yaw_rate_dps") is not None:
                note += " yaw %+.0f°/s" % summary["yaw_rate_dps"]
            if summary.get("tilted"):
                note += " TILTED"
            if summary.get("shock"):
                note += " SHOCK"
            return note
        return ""


def _describe_lidar(info: Dict[str, Any]) -> str:
    """One-line summary of what a client said about its scanner."""
    if not info:
        return "none declared"
    model = info.get("model") or info.get("source") or "?"
    points = info.get("points")
    hz = info.get("hz")
    parts = [str(model)]
    if points:
        parts.append(f"{points} pts")
    if hz:
        parts.append(f"{float(hz):.1f} Hz")
    return " ".join(parts)


def _describe_imu(info: Dict[str, Any]) -> str:
    """One-line summary of what a client said about its IMU."""
    if not info:
        return "none declared"
    parts = [str(info.get("model") or info.get("source") or "?")]
    if info.get("transport"):
        parts.append(str(info["transport"]))
    if info.get("rate_hz"):
        parts.append(f"{float(info['rate_hz']):.0f} Hz")
    return " ".join(parts)


def _local_addresses() -> List[str]:
    """Best-effort list of non-loopback IPv4 addresses, for the startup banner."""
    addrs: List[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in addrs:
                addrs.append(ip)
    except OSError:
        pass
    if not addrs:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # no packet is actually sent
            addrs.append(s.getsockname()[0])
            s.close()
        except OSError:
            pass
    return addrs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robocam-server",
        description="Receive a video stream from the robot, process it, reply with results.",
    )
    default_cfg = Path(__file__).resolve().parent.parent / "config" / "server.yaml"
    p.add_argument("-c", "--config", default=str(default_cfg) if default_cfg.is_file() else None,
                   help="path to the YAML config file")
    p.add_argument("-b", "--bind", default=None,
                   help="override server.bind, e.g. tcp://0.0.0.0:5555")
    p.add_argument("-p", "--processor", default=None,
                   help="override processor.name")
    p.add_argument("--workers", type=int, default=None, help="override processor.workers")
    p.add_argument("--queue-depth", type=int, default=None, help="override queue.max_depth")
    p.add_argument("--no-snapshots", action="store_true", help="disable periodic frame dumps")
    p.add_argument("--no-lidar", action="store_true",
                   help="ignore scan messages (the robot is told, and stops sending)")
    p.add_argument("--no-imu", action="store_true",
                   help="ignore imu messages (the robot is told, and stops sending)")
    p.add_argument("--no-odom", action="store_true",
                   help="ignore odom messages. Nothing can be placed in the robot's "
                        "map frame without them, so this also disables compare.")
    p.add_argument("--no-map", action="store_true",
                   help="ignore the robot's occupancy grid, and with it the comparison")
    p.add_argument("--no-compare", action="store_true",
                   help="receive the map but do not compare the reconstruction against "
                        "it; useful for measuring the cost of the upload alone")
    p.add_argument("--no-map-updates", action="store_true",
                   help="run the comparison and log what it finds, but do not send "
                        "patches to the robot. The way to check what the server WOULD "
                        "write to the costmap before letting it.")
    p.add_argument("--pose-hints", action="store_true",
                   help="offer SLAM pose corrections (compare.send_hints). Off by "
                        "default: switch it on once agreement has been seen healthy.")
    p.add_argument("--camera-height", type=float, default=None,
                   help="override compare.camera_z, metres above the floor. Wrong here "
                        "puts every reconstructed surface at the wrong height.")
    p.add_argument("--phase", choices=list(wire.SUPPORTED_PHASES), default=None,
                   help="phase for sessions that do not declare one: t1 explore, t2 seek")
    p.add_argument("--target", default=None,
                   help="what t2 is looking for, in words. Free text: it is the query "
                        "the detector is given, so 'the red mug on the desk' beats "
                        "'mug'. Naming one here is how a mission is started at launch "
                        "time; a target in the robot's hello wins over it.")
    p.add_argument("--detector", default=None,
                   help=f"override seek.detector: {', '.join(detect_mod.available())}. "
                        "'colour' needs no weights and matches a colour word in the "
                        "target; 'owl' is open-vocabulary and is the one that seeks.")
    p.add_argument("--no-seek", action="store_true",
                   help="disable the decision stage. t2 then navigates without a target "
                        "search, which is what you want while fitting the detector.")
    p.add_argument("--grasp-height", type=float, default=None,
                   help="override seek.grasp_z_max, metres above the floor: the top of "
                        "the gripper's envelope. Anything higher is reported unreachable.")
    p.add_argument("--no-t1-report", action="store_true",
                   help="stop evaluating whether t1 is finished (mission.t1_exit_report)")
    p.add_argument("--mount-yaw", type=float, default=None,
                   help="override lidar.mount_yaw_deg: bearing the camera looks along")
    p.add_argument("--hfov", type=float, default=None,
                   help="override lidar.camera_hfov_deg, used to map bearings to columns")
    p.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    p.add_argument("--list-processors", action="store_true", help="print registered processors and exit")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_processors:
        for name in processors.available():
            print(name)
        return 0

    cfg = Config.load(args.config)
    if args.bind:
        cfg.server.bind = args.bind
    if args.processor:
        cfg.processor.name = args.processor
    if args.workers is not None:
        cfg.processor.workers = args.workers
    if args.queue_depth is not None:
        cfg.queue.max_depth = args.queue_depth
    if args.no_snapshots:
        cfg.snapshot.enabled = False
    if args.no_lidar:
        cfg.lidar.enabled = False
    if args.no_imu:
        cfg.imu.enabled = False
    if args.no_odom:
        cfg.odom.enabled = False
        # Not a separate switch to remember: a comparison needs a pose to place
        # the cloud with, and one that ran without would be comparing a cloud
        # against a map it has no reason to believe overlaps.
        cfg.compare.enabled = False
    if args.no_map:
        cfg.map.enabled = False
        cfg.compare.enabled = False
    if args.no_compare:
        cfg.compare.enabled = False
    if args.no_map_updates:
        cfg.map.send_updates = False
    if args.pose_hints:
        cfg.compare.send_hints = True
    if args.camera_height is not None:
        cfg.compare.camera_z = args.camera_height
    if args.phase:
        cfg.mission.phase = args.phase
    if args.target is not None:
        cfg.mission.target = args.target
    if args.detector is not None:
        if args.detector not in detect_mod.available():
            parser_error = (f"unknown detector {args.detector!r}; "
                            f"available: {', '.join(detect_mod.available())}")
            print(parser_error, file=sys.stderr)
            return 2
        cfg.seek.detector = args.detector
    if args.no_seek:
        cfg.seek.enabled = False
    if args.grasp_height is not None:
        cfg.seek.grasp_z_max = args.grasp_height
    if args.no_t1_report:
        cfg.mission.t1_exit_report = False
    if args.mount_yaw is not None:
        cfg.lidar.mount_yaw_deg = args.mount_yaw
    if args.hfov is not None:
        cfg.lidar.camera_hfov_deg = args.hfov
    if args.log_level:
        cfg.logging.level = args.log_level

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    server = StreamServer(cfg)

    def handle_signal(signum, _frame):
        log.info("received %s, shutting down", signal.Signals(signum).name)
        server.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        server.start()
    except Exception:
        log.exception("failed to start")
        server.close()
        return 1

    server.run()
    log.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
