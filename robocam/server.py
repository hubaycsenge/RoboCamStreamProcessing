"""The stream server.

One thread owns the ZeroMQ socket and does nothing slow: receive, decode,
enqueue, and send back whatever the workers have finished.  Everything
expensive happens in the worker pool (see pipeline.py).

ZeroMQ sockets are not thread-safe, so results travel from the workers to the
IO thread through a plain ``queue.Queue`` which the IO loop drains after every
poll.  That keeps all socket calls on one thread without any locking.
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
from typing import Any, Dict, List, Optional

import zmq

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

    def touch(self) -> None:
        self.last_seen_ns = wire.monotonic_ns()

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
        )

        self._stop = threading.Event()
        self._frame_counter = 0
        # Rolling counters for the periodic throughput line.
        self._stat_t0 = time.monotonic()
        self._stat_frames = 0
        self._stat_bytes = 0
        self._stat_dropped = 0
        self._stat_latency_ms = 0.0

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
        self.pool = WorkerPool(
            processor_factory=lambda: processors.build(name, options),
            frame_queue=self.frames,
            result_queue=self.results,
            workers=self.cfg.processor.workers,
            name=name,
            on_result_ready=self.waker.wake,
        )
        self.pool.start()

        log.info(
            "listening on %s | processor=%s workers=%d | queue depth=%d drop=%s",
            self.cfg.server.bind,
            name,
            self.cfg.processor.workers,
            self.cfg.queue.max_depth,
            self.cfg.queue.drop_policy,
        )
        for addr in _local_addresses():
            port = self.cfg.server.bind.rsplit(":", 1)[-1]
            log.info("  robot can connect to tcp://%s:%s", addr, port)

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
            "session %s connected | client=%s camera=%s codec=%s %dx%d @%.1f fps",
            session.session_id,
            session.client_id,
            session.camera or "-",
            session.codec,
            session.declared_width,
            session.declared_height,
            session.declared_fps,
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
        )
        self.sessions[identity] = session
        return session

    def _drop_session(self, identity: bytes) -> None:
        session = self.sessions.pop(identity, None)
        if session is not None:
            session.decoder.close()
            log.info(
                "session %s closed | %d frames, %d dropped, %d failed, %.1f MB, %.0fs",
                session.session_id,
                session.frames_received,
                session.frames_dropped,
                session.frames_failed,
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

        self._frame_counter += 1
        self.snapshots.maybe_offer(session.session_id, seq, image, self._frame_counter)

        frame = Frame(
            seq=seq,
            session_id=session.session_id,
            image=image,
            header=header,
            recv_ts_ns=recv_ts_ns,
            decode_ms=decode_ms,
            payload_bytes=len(payload),
        )
        for evicted in self.frames.put(frame):
            session.frames_dropped += 1
            self._stat_dropped += 1
            self._send_dropped(session, evicted)

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

        if self._stat_frames:
            fps = self._stat_frames / elapsed
            mbps = (self._stat_bytes * 8) / elapsed / 1e6
            avg_ms = self._stat_latency_ms / max(self._stat_frames, 1)
            log.info(
                "%.1f fps | %.1f Mbit/s | %.1f ms server-side | %d dropped | %d session(s) | queue=%d",
                fps, mbps, avg_ms, self._stat_dropped, len(self.sessions), len(self.frames),
            )
        elif self.sessions:
            log.info("no frames in %.0fs | %d session(s) idle", elapsed, len(self.sessions))

        self._stat_t0 = now
        self._stat_frames = 0
        self._stat_bytes = 0
        self._stat_dropped = 0
        self._stat_latency_ms = 0.0


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
