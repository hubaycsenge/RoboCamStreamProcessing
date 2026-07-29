"""End-to-end tests: a real server on a real socket, a real client.

These are the tests that would have caught a wire-format mistake, so they use
TCP on localhost rather than mocking the transport.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import zmq

from robocam import wire
from robocam.config import Config
from robocam.server import StreamServer

# The client is deployed to the robot as a standalone file, so it is not on the
# package path; add its directory explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))
import robocam_client  # noqa: E402


@pytest.fixture
def server():
    """A running server bound to an ephemeral localhost port."""
    cfg = Config.from_dict({
        "server": {"bind": "tcp://127.0.0.1:0", "session_timeout_s": 0},
        "processor": {"name": "stats", "options": {"brightness": True}},
        "snapshot": {"enabled": False},
        "logging": {"stats_interval_s": 0},
    })
    srv = StreamServer(cfg)
    srv.start()
    endpoint = srv.sock.getsockopt(zmq.LAST_ENDPOINT).decode()

    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    try:
        yield srv, endpoint
    finally:
        srv.stop()
        thread.join(timeout=5)


def jpeg(width: int, height: int) -> bytes:
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class RawPeer:
    """A bare DEALER socket, to exercise the protocol without the client."""

    def __init__(self, endpoint: str, identity: bytes = b"test-peer"):
        self.sock = zmq.Context.instance().socket(zmq.DEALER)
        self.sock.setsockopt(zmq.IDENTITY, identity)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(endpoint)

    def send(self, header, payload=b""):
        head, body = wire.encode(header, payload)
        self.sock.send_multipart([head, body])

    def recv(self, timeout_ms: int = 3000):
        if not self.sock.poll(timeout_ms):
            raise TimeoutError("no reply from server")
        parts = self.sock.recv_multipart()
        return json.loads(parts[0].decode())

    def close(self):
        self.sock.close(linger=0)


def test_handshake(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"handshake-peer")
    try:
        peer.send(wire.hello("robot-1", codec=wire.CODEC_JPEG, width=640, height=480, fps=30))
        reply = peer.recv()

        assert reply["type"] == wire.MSG_WELCOME
        assert reply["accepted"] is True
        assert reply["processor"] == "stats"
        assert "host" in reply["server"]
    finally:
        peer.close()


def test_frame_result_reports_actual_geometry(server):
    """The result must describe what the server decoded, not what the client claimed."""
    _, endpoint = server
    peer = RawPeer(endpoint, b"geometry-peer")
    try:
        peer.send(wire.hello("robot-1"))
        assert peer.recv()["type"] == wire.MSG_WELCOME

        payload = jpeg(640, 480)
        # Deliberately lie about the dimensions in the header.
        peer.send(wire.frame(seq=1, codec=wire.CODEC_JPEG, width=99, height=99,
                             t_capture_ns=12345), payload)
        result = peer.recv()

        assert result["type"] == wire.MSG_RESULT
        assert result["seq"] == 1
        assert result["ok"] is True
        assert result["reason"] == "ok"
        assert (result["width"], result["height"], result["channels"]) == (640, 480, 3)
        assert result["dtype"] == "uint8"
        assert result["nbytes"] == 640 * 480 * 3
        assert result["payload_bytes"] == len(payload)
        assert result["data"]["received"] is True
        assert result["data"]["shape"] == [480, 640, 3]
        # Client timestamps come back untouched for RTT computation.
        assert result["t_capture_ns"] == 12345
    finally:
        peer.close()


def test_raw_bgr_codec(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"raw-peer")
    try:
        peer.send(wire.hello("robot-1", codec=wire.CODEC_RAW_BGR))
        assert peer.recv()["accepted"] is True

        img = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        peer.send(wire.frame(seq=0, codec=wire.CODEC_RAW_BGR, width=160, height=120,
                             t_capture_ns=1), img.tobytes())
        result = peer.recv()

        assert result["ok"] is True
        assert (result["width"], result["height"]) == (160, 120)
    finally:
        peer.close()


def test_corrupt_payload_reports_failure_without_killing_the_session(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"corrupt-peer")
    try:
        peer.send(wire.hello("robot-1"))
        peer.recv()

        peer.send(wire.frame(seq=1, codec=wire.CODEC_JPEG, width=64, height=64,
                             t_capture_ns=1), b"this is not a jpeg")
        bad = peer.recv()
        assert bad["ok"] is False
        assert bad["reason"] == wire.REASON_DECODE_FAILED

        # The session must survive so a glitch does not drop the link.
        peer.send(wire.frame(seq=2, codec=wire.CODEC_JPEG, width=64, height=64,
                             t_capture_ns=2), jpeg(64, 64))
        good = peer.recv()
        assert good["ok"] is True
        assert good["seq"] == 2
    finally:
        peer.close()


def test_unsupported_codec_rejected_at_handshake(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"badcodec-peer")
    try:
        peer.send(wire.hello("robot-1", codec="vp9"))
        reply = peer.recv()
        assert reply["accepted"] is False
        assert "unsupported codec" in reply["message"]
    finally:
        peer.close()


def test_protocol_version_mismatch_rejected(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"oldproto-peer")
    try:
        header = wire.hello("robot-1")
        header["protocol"] = 999
        peer.send(header)
        reply = peer.recv()
        assert reply["accepted"] is False
        assert "protocol" in reply["message"]
    finally:
        peer.close()


def test_ping_pong(server):
    _, endpoint = server
    peer = RawPeer(endpoint, b"ping-peer")
    try:
        peer.send(wire.hello("robot-1"))
        peer.recv()
        peer.send(wire.ping(42))
        pong = peer.recv()
        assert pong["type"] == wire.MSG_PONG
        assert pong["nonce"] == 42
    finally:
        peer.close()


def test_every_frame_gets_exactly_one_result(server):
    """Including frames the queue drops — the client's flow control depends on it."""
    srv, endpoint = server
    peer = RawPeer(endpoint, b"count-peer")
    try:
        peer.send(wire.hello("robot-1"))
        peer.recv()

        n = 40
        payload = jpeg(320, 240)
        for seq in range(n):
            peer.send(wire.frame(seq=seq, codec=wire.CODEC_JPEG, width=320, height=240,
                                 t_capture_ns=seq), payload)

        seen = set()
        deadline = time.monotonic() + 10
        while len(seen) < n and time.monotonic() < deadline:
            try:
                msg = peer.recv(timeout_ms=1000)
            except TimeoutError:
                break
            if msg.get("type") == wire.MSG_RESULT:
                assert msg["seq"] not in seen, "duplicate result for a frame"
                seen.add(msg["seq"])

        assert seen == set(range(n)), f"missing results for {sorted(set(range(n)) - seen)}"
    finally:
        peer.close()


def test_real_client_against_real_server(server):
    """The deployable client file, talking to the server, with synthetic frames."""
    _, endpoint = server
    results = []

    client = robocam_client.RoboCamClient(
        server=endpoint,
        client_id="pytest-orin",
        jpeg_quality=70,
        max_inflight=2,
        on_result=results.append,
    )
    source = robocam_client.SyntheticSource(320, 240, fps=30)
    client.run(source, duration=2.0, status_every=0)

    ok = [r for r in results if r.get("ok")]
    assert ok, f"no successful results (got {len(results)} messages)"

    first = ok[0]
    assert first["width"] == 320
    assert first["height"] == 240
    assert first["channels"] == 3
    assert first["data"]["received"] is True
    # The synthetic pattern is not blank, and RTT was measured client-side.
    assert first["data"]["looks_blank"] is False
    assert first["rtt_ms"] > 0
    # JPEG should be well under the raw size.
    assert first["payload_bytes"] < first["nbytes"]
