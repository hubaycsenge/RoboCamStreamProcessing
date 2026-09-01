"""The stream server.

One thread owns the ZeroMQ socket and does nothing slow: receive, decode,
enqueue, and send back whatever the workers have finished.  Everything
expensive happens in the worker pool (see pipeline.py).

ZeroMQ sockets are not thread-safe, so results travel from the workers to the
IO thread through a plain ``queue.Queue`` which the IO loop drains after every
poll.  That keeps all socket calls on one thread without any locking.

The two non-camera sensors are the exception to "the IO thread does nothing
slow", and only because the work is genuinely tiny: parsing 360 uint16s and
reducing them to sector minima is ~60 µs against a 20 ms poll, and an inertial
burst is a handful of numpy passes over a few dozen rows.  Both are answered
inline so that obstacle and attitude information is never stuck behind a model
in the frame queue — at 5 Hz, queueing a scan behind two 30 ms frames would be
most of its useful life.  The last scan and the last burst are also held on the
session and attached to the next frame, which is where fusion happens:
association by arrival time on the server's own clock, rather than by unrelated
client clocks.
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
            # points and how the IMU is read before it sees a frame.
            processor = processors.build(name, options)
            processor.configure(self.cfg.lidar, self.cfg.imu)
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

        if peer_protocol != wire.PROTOCOL_VERSION:
            msg = f"client protocol {peer_protocol} != server protocol {wire.PROTOCOL_VERSION}"
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
        )
        self.sessions[identity] = session
        return session

    def _drop_session(self, identity: bytes) -> None:
        session = self.sessions.pop(identity, None)
        if session is not None:
            session.decoder.close()
            log.info(
                "session %s closed | %d frames, %d dropped, %d failed, "
                "%d scans (%d bad), %d imu samples (%d bad bursts), %.1f MB, %.0fs",
                session.session_id,
                session.frames_received,
                session.frames_dropped,
                session.frames_failed,
                session.scans_received,
                session.scans_failed,
                session.imu_samples_received,
                session.imu_failed,
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
        )
        self._send(session.identity, header)

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
                self._attitude_note(),
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
